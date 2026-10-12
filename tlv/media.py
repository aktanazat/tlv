"""Local media work: downloads (yt-dlp or direct), ffprobe, and ffmpeg cuts, stills, frame
samples, audio and contact sheets. Outputs are written under a ".partial" name and renamed
into place, so a failed run never leaves a broken file under the final name."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tlv import store

YT_DLP_FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"
# URL paths that name a media file are fetched as-is; anything else goes through yt-dlp.
DIRECT_MEDIA_PATH = re.compile(r"\.(mp4|m4v|mov|webm|mkv|avi|mpe?g|3gp)$", re.I)
# Contact sheets stay within this width so they can go back to an agent as one image.
SHEET_MAX_WIDTH = 1280
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


@dataclass(frozen=True)
class Range:
    start: float
    end: float


@dataclass(frozen=True)
class Probe:
    duration: float
    width: int
    height: int
    fps: float


def slug(text: str) -> str:
    """Lowercase ASCII words joined by "-", at most 48 characters."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words)[:48].strip("-") or "video"


def run(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        detail = "\n".join(done.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"{Path(command[0]).name} exited with {done.returncode}: {detail}")
    return done.stdout


def write_atomically(output: Path, write: Callable[[Path], object]) -> None:
    """Runs `write` on a ".partial" sibling of `output`, then renames it into place."""
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    try:
        write(partial)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def ffmpeg(args: list[str], output: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-abort_on", "empty_output", "-y", *args]
    write_atomically(output, lambda partial: run([*command, str(partial)]))


@dataclass(frozen=True)
class Download:
    path: Path
    title: str


def download(url: str) -> Download:
    """Downloads a URL into media/: a direct media file as-is, anything else with yt-dlp."""
    folder = store.data_dir("media")
    path_part = urllib.parse.urlparse(url).path
    if DIRECT_MEDIA_PATH.search(path_part):
        name = Path(urllib.parse.unquote(path_part)).name
        title, ext = Path(name).stem, Path(name).suffix.lower()
        digest = hashlib.sha1(url.encode()).hexdigest()[:8]
        path = folder / f"{slug(title)}-{digest}{ext}"
        if not path.exists():

            def fetch(partial: Path) -> None:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out)

            write_atomically(path, fetch)
        return Download(path, title)
    stdout = run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--no-progress",
            "-f",
            YT_DLP_FORMAT,
            "--merge-output-format",
            "mp4",
            "-o",
            str(folder / "%(extractor_key)s-%(id)s.%(ext)s"),
            "--print",
            "after_move:%(.{title,filepath})j",
            url,
        ]
    )
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"yt-dlp printed no downloaded file for {url}: {stdout.strip()[:300]}")
    info = json.loads(lines[-1])
    return Download(Path(info["filepath"]), info["title"])


def probe(path: Path) -> Probe:
    """Duration, frame size and frame rate of a video file."""
    stdout = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    info = json.loads(stdout)
    if not info.get("streams"):
        raise RuntimeError(f"{path} has no video stream")
    stream = info["streams"][0]
    num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return Probe(float(info["format"]["duration"]), int(stream["width"]), int(stream["height"]), fps)


def cut(source: Path, span: Range, output: Path) -> None:
    """Re-encodes [start, end] of a video into an mp4 that starts playing before it fully loads."""
    ffmpeg(
        [
            "-ss",
            f"{span.start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{span.end - span.start:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
        ],
        output,
    )


def still(source: Path, t: float, output: Path, width: int | None = None) -> None:
    """Saves the frame at `t` seconds as a JPEG, scaled to `width` pixels when given."""
    scale = ["-vf", f"scale={width}:-2"] if width else []
    ffmpeg(["-ss", f"{t:.3f}", "-i", str(source), "-frames:v", "1", *scale, "-q:v", "2", "-update", "1"], output)


def sample_frames(source: Path, every: float, width: int, folder: Path) -> list[tuple[float, Path]]:
    """One JPEG every `every` seconds, `width` pixels wide, named by time in milliseconds.
    Frame i shows time (i + 0.5) * every, the middle of its slot."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    pattern = folder / "f%06d.jpg"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"fps=1/{every}:start_time={every / 2},scale={width}:-2",
            "-q:v",
            "3",
            str(pattern),
        ]
    )
    frames = []
    for i, path in enumerate(sorted(folder.glob("f*.jpg"))):
        t = round((i + 0.5) * every, 3)
        named = folder / f"t{round(t * 1000):08d}.jpg"
        path.replace(named)
        frames.append((t, named))
    return frames


def audio(source: Path, output: Path, rate: int = 16000) -> None:
    """Mono 16-bit WAV of the whole video at `rate` Hz."""
    ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le"], output)


def contact_sheet(source: Path, span: Range, count: int, output: Path) -> list[float]:
    """Tiles `count` evenly spaced frames of [start, end] into one JPEG at most 1280 px wide,
    left to right, top to bottom. Returns the time of each tile."""
    info = probe(source)
    step = (span.end - span.start) / count
    times = [span.start + (i + 0.5) * step for i in range(count)]
    columns = count if count <= 3 else 2 if count == 4 else 3 if count <= 9 else 4
    tile_w = SHEET_MAX_WIDTH // columns // 2 * 2
    tile_h = max(2, round(tile_w * info.height / info.width / 2) * 2)
    if count == 1:
        graph = f"[0:v]scale={tile_w}:{tile_h},setsar=1[out]"
    else:
        tiles = ";".join(f"[{i}:v]scale={tile_w}:{tile_h},setsar=1[t{i}]" for i in range(count))
        layout = "|".join(f"{(i % columns) * tile_w}_{(i // columns) * tile_h}" for i in range(count))
        inputs = "".join(f"[t{i}]" for i in range(count))
        graph = f"{tiles};{inputs}xstack=inputs={count}:layout={layout}:fill=black[out]"
    args = [arg for t in times for arg in ("-ss", f"{t:.3f}", "-i", str(source))]
    ffmpeg([*args, "-filter_complex", graph, "-map", "[out]", "-frames:v", "1", "-q:v", "3", "-update", "1"], output)
    return times
