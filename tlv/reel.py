"""Highlight reels: the confirmed moments of a find cut into one mp4 on the beat of a song, each
clip opening with a lower third, and a sources file beside the mp4 naming where each clip came
from.

The song sets the grid. Its beats are found once: onset strength from how fast its spectrum
rises, the tempo from the lag at which that strength repeats, and the phase from the comb of
beats that lands on the most onsets. Each clip starts on a camera cut when one is near and lasts
a whole number of beats, and the song starts on its first beat, so every cut lands on a beat.
Each clip renders in its own ffmpeg process on the hardware video encoder, which leaves the CPU
and GPU to searches running beside it; a last pass joins the clips without re-encoding them and
lays the song under them."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sqlite3
import tempfile
import time
import unicodedata
import wave
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tlv import media, store
from tlv.agent.clock import QUARTERS, Reading, reading, reading_text
from tlv.agent.find import CLIP_MARGIN_S
from tlv.agent.sources import Line, screen_text
from tlv.timecode import format_range, format_timecode

Progress = Callable[[str], None] | None

MIN_BEATS = 2  # no clip is shorter than two beats
SNAP_SHOT_S = 0.5  # a clip edge this close to a camera cut moves onto it

# Beat finding: 64 ms spectra every 16 ms of the song at 16 kHz.
SAMPLE_RATE = 16000
WINDOW, HOP = 1024, 256
# An attack lifts the onset strength most in the frame whose window it enters, which is centered
# about a hop before the attack (measured on clicks).
ONSET_LAG_S = HOP / SAMPLE_RATE
TEMPO_BPM = (60.0, 200.0)
# A beat also repeats at half and twice its tempo. Tempos are weighted by a bell curve over
# octaves, LIKELY_SPREAD octaves wide around LIKELY_BPM, so the most usual reading wins.
LIKELY_BPM, LIKELY_SPREAD = 120.0, 0.7
# The first two minutes settle a steady tempo; a reel rarely runs longer.
ANALYZED_S = 120.0
MIN_SONG_S = 4.0
# How strongly onset strength must repeat at the beat (1 for a metronome; white noise, a held pad
# and game commentary measure under 0.08, drum loops mostly over 0.2) for the song to count as
# having a steady beat.
STEADY = 0.15
# Sound starts at the first sample this loud against the loudest one (-40 dB).
SOUND = 0.01

# Rendering.
FPS = 30
WIDTH, HEIGHT = 1280, 720
AUDIO_RATE = 48000
VIDEO_CODEC = ("-c:v", "h264_videotoolbox", "-b:v", "6M", "-g", "60")
RENDER_WORKERS = 3
GAME_VOLUME = 0.25  # the game's own sound under the song, which plays at 1
END_FADE_S = 1.0
TAIL_S = 0.5  # read this much past each clip, so the frame rate change never comes up short

# The lower third: the claim over a detail line, on a dark panel sliding in from the left.
HEADLINE_FONT = "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf"
DETAIL_FONT = "/System/Library/Fonts/Avenir Next Condensed.ttc"
DETAIL_FACE = 2  # Demi Bold
HEADLINE_PX, MIN_HEADLINE_PX, DETAIL_PX = 54, 36, 27
MARGIN, PANEL_H, ACCENT_W, PAD = 48, 104, 8, 22
# The text stops this wide, keeping the panel clear of the score bug broadcasts show bottom right.
TEXT_MAX_W = 780
LOWER_THIRD_S = 2.5  # the label starts sliding out here, or earlier in a shorter clip
SLIDE_IN_AT_S, SLIDE_S = 0.15, 0.35
HEADLINE_BASELINE, DETAIL_BASELINE = 60, 91
PANEL_COLOR, ACCENT_COLOR = (14, 16, 22, 220), (253, 185, 39, 255)
HEADLINE_COLOR, DETAIL_COLOR = (255, 255, 255, 255), (214, 219, 228, 255)


@dataclass(frozen=True)
class Beats:
    """A song's beat grid: its tempo and its first beat with sound, in seconds into the song,
    plus the song's length, which says when the reel has to loop it."""

    bpm: float
    first: float
    duration: float

    @property
    def period(self) -> float:
        return 60 / self.bpm


@dataclass(frozen=True)
class Moment:
    """A confirmed find result as the reel shows it: what happens (the claim), who (the named
    person, else what the find saw, such as a jersey number; empty when the claim names them),
    and the scoreboard nearest its middle in the frames its clip shows, if one is read there."""

    obs_id: str
    video_id: str
    start: float
    end: float
    confidence: float
    claim: str
    who: str
    clock: Reading | None


@dataclass(frozen=True)
class Video:
    """A video a reel cuts from: its local copy, length and camera cuts."""

    video_id: str
    title: str
    source: str
    path: Path
    duration: float
    shots: tuple[float, ...]
    has_audio: bool


@dataclass(frozen=True)
class Cut:
    """One clip of a reel: start..end of the moment's video, placed `at` seconds into the reel."""

    moment: Moment
    start: float
    end: float
    at: float


def saved_find(path: Path, pick: str | None) -> dict[str, Any]:
    """A find result saved as JSON: the output of `tlv find --json`, or an eval results file whose
    runs each hold one. `pick` names the find by run id, query or claim; a file of one find needs
    no name."""
    data = json.loads(path.read_text())
    runs = data["runs"] if "runs" in data else [{"id": data["claim"], "query": data["claim"], "output": data}]
    finds = [run for run in runs if run["output"]]
    if pick is None and len(finds) == 1:
        return finds[0]["output"]
    for run in finds:
        if pick is not None and pick.lower() in {
            run["id"].lower(),
            run["query"].lower(),
            run["output"]["claim"].lower(),
        }:
            return run["output"]
    names = ", ".join(run["id"] for run in finds)
    if pick is None:
        raise ValueError(f"{path} holds {len(finds)} finds: name one of {names}")
    raise ValueError(f'No find "{pick}" in {path}: name one of {names}')


# The beat grid


def beats(song: Path, bpm: float | None = None) -> Beats:
    """The beat grid of a song: its tempo, or `bpm` when given, and its first beat with sound.
    Raises ValueError when the song has no steady beat, so the caller can pass its tempo."""
    with tempfile.TemporaryDirectory(dir=store.data_dir("work")) as work:
        wav = Path(work) / "song.wav"
        media.audio(song, wav, SAMPLE_RATE)
        with wave.open(str(wav)) as audio:
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), np.int16)
    duration = len(pcm) / SAMPLE_RATE
    if duration < MIN_SONG_S:
        raise ValueError(f"{song.name} is {duration:.1f} s long, too short to find its beat")
    samples = pcm[: int(ANALYZED_S * SAMPLE_RATE)].astype(np.float32) / 32768
    onsets = _onsets(samples)
    rate = SAMPLE_RATE / HOP
    period = 60 / bpm if bpm else _period(onsets, rate) / rate
    phase = _phase(onsets, period * rate) / rate + ONSET_LAG_S
    loudness = np.abs(samples)
    sound = int(np.argmax(loudness >= SOUND * loudness.max())) / SAMPLE_RATE
    # The first beat no more than a quarter beat before the sound starts.
    first = phase + math.ceil((sound - phase) / period - 0.25) * period
    return Beats(bpm=60 / period, first=max(0.0, first), duration=duration)


def _onsets(samples: np.ndarray) -> np.ndarray:
    """Onset strength per frame: how much the log spectrum rose since the frame before, summed over
    frequencies. Drum hits and note attacks rise sharply; held notes do not rise at all."""
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(samples, WINDOW // 2), WINDOW)[::HOP]
    spectrum = np.log1p(1000 * np.abs(np.fft.rfft(frames * np.hanning(WINDOW).astype(np.float32), axis=1)))
    return np.concatenate([[0.0], np.maximum(np.diff(spectrum, axis=0), 0).sum(axis=1)])


def _period(onsets: np.ndarray, rate: float) -> float:
    """Frames per beat. The autocorrelation of onset strength peaks at every lag the music repeats
    at. Between 60 and 200 bpm, the beat is the lag where it is strongest plus half as strong at
    twice the lag (a beat repeats every two beats; an off-beat pattern often does not), weighted
    toward LIKELY_BPM. On 416 Apple Loops drum loops played for 30 s this reads 69% of tempos
    exactly (the strongest peak alone: 54%) and 13% at half or twice the tempo, which still cuts
    on beats. The beat's multiples then pin it down: the peak near 32 beats, divided by 32, is 32
    times finer than the peak near one beat, which keeps cuts on the beat a minute into a reel."""
    centered = onsets - onsets.mean()
    n = len(centered)
    power = np.abs(np.fft.rfft(centered, 2 * n)) ** 2
    correlation = np.fft.irfft(power)[:n] / np.arange(n, 0, -1)
    correlation /= correlation[0]
    lags = np.arange(int(rate * 60 / TEMPO_BPM[1]), int(rate * 60 / TEMPO_BPM[0]) + 1)
    likely = np.exp(-0.5 * (np.log2(rate * 60 / lags / LIKELY_BPM) / LIKELY_SPREAD) ** 2)
    lag = int(lags[np.argmax((correlation[lags] + 0.5 * correlation[2 * lags]) * likely)])
    if correlation[lag] < STEADY:
        raise ValueError(f"no steady beat in the music (pulse {correlation[lag]:.2f}); pass --bpm")
    period = _peak(correlation, lag)
    multiple = 1
    while (2 * multiple + 1) * period < n / 2:
        multiple *= 2
        reach = round(period / 4)
        near = round(multiple * period)
        lag = near - reach + int(np.argmax(correlation[near - reach : near + reach + 1]))
        period = _peak(correlation, lag) / multiple
    return period


def _peak(values: np.ndarray, i: int) -> float:
    """Where the parabola through values[i - 1 : i + 2] peaks: the maximum between samples."""
    left, middle, right = values[i - 1], values[i], values[i + 1]
    bend = left - 2 * middle + right
    return i + (0.5 * (left - right) / bend if bend < 0 else 0.0)


def _phase(onsets: np.ndarray, period: float) -> float:
    """Frames from the start of the song to its first beat: the offset at which a comb of beats one
    period apart gathers the most onset strength."""
    smooth = np.convolve(onsets, [0.25, 0.5, 0.25], mode="same")
    count = int(len(onsets) / period) - 1
    offsets = np.arange(0, period, 0.25)
    at = offsets[:, None] + period * np.arange(count)
    return float(offsets[np.argmax(np.interp(at, np.arange(len(onsets)), smooth).sum(axis=1))])


# The plan


def moment(found: dict[str, Any], person: str | None, screen: list[Line]) -> Moment:
    """A confirmed find result as a reel clip. Who is `person` (the name of the entity the find
    matched), else what the find saw, with a leading jersey number written "#30". The clock is the
    scoreboard reading nearest the moment's middle in `screen`, the on-screen text of the frames
    its clip shows: the clip can open on the play before a camera cut, and a stat line can cover
    the scoreboard while the moment plays."""
    middle = (found["start"] + found["end"]) / 2
    readings = [r for line in screen if (r := reading(line)) is not None]
    clock = min(readings, key=lambda r: abs(r.t - middle), default=None)
    who = person or re.sub(r"^(\d{1,2})\b", r"#\1", found.get("who") or "")
    if who.lower() in found["claim"].lower():
        who = ""
    return Moment(
        found["id"], found["video_id"], found["start"], found["end"], found["confidence"], found["claim"], who, clock
    )


def _period_rank(period: str) -> int:
    """1 to 4 for quarters, then overtimes in order: OT, 2OT, 3OT..."""
    return int(period) if period in QUARTERS else 4 + int(period.removesuffix("OT") or 1)


def plan(moments: list[Moment], videos: Mapping[str, Video], grid: Beats | None, order: str) -> list[Cut]:
    """Places each moment's clip (see `_span`) in the reel, in `order`: found (the find's order),
    game (by period and game clock; moments without a readable clock last) or video (videos in the
    order they were found, each in time order), each clip starting where the one before ends."""
    cuts: list[Cut] = []
    at = 0.0
    for m in _ordered(moments, order):
        start, end = _span(m.start, m.end, videos[m.video_id], grid)
        cuts.append(Cut(m, start, end, at))
        at += end - start
    return cuts


def _span(start: float, end: float, video: Video, grid: Beats | None) -> tuple[float, float]:
    """What the clip of a moment from `start` to `end` shows of `video`: a second either side of
    it, each edge moved onto a camera cut within SNAP_SHOT_S. With a song it then lasts a whole
    number of beats, at least MIN_BEATS, so the next clip starts on a beat; a clip ending on a
    camera cut loses its part beat rather than show the next shot."""
    lead = max(0.0, start - CLIP_MARGIN_S)
    snapped = _snap(lead, video.shots)
    first = lead if snapped is None else snapped
    tail = min(video.duration, end + CLIP_MARGIN_S)
    on_cut = _snap(tail, video.shots)
    length = (tail if on_cut is None else on_cut) - first
    if grid is not None:
        whole = round(length / grid.period) if on_cut is None else math.floor(length / grid.period + 1e-6)
        length = max(MIN_BEATS, whole) * grid.period
        first = max(0.0, min(first, video.duration - length))
    return first, first + length


def _snap(t: float, shots: tuple[float, ...]) -> float | None:
    """The camera cut within SNAP_SHOT_S of `t`, if there is one."""
    near = min(shots, key=lambda shot: abs(shot - t), default=None)
    return near if near is not None and abs(near - t) <= SNAP_SHOT_S else None


def _ordered(moments: list[Moment], order: str) -> list[Moment]:
    if order == "game":
        return sorted(moments, key=_game_time)
    if order == "video":
        rank = {video_id: i for i, video_id in enumerate(dict.fromkeys(m.video_id for m in moments))}
        return sorted(moments, key=lambda m: (rank[m.video_id], m.start))
    return moments


def _game_time(m: Moment) -> tuple[bool, int, float]:
    if m.clock is None:
        return True, 0, 0.0
    return False, _period_rank(m.clock.period), -m.clock.seconds


# The lower third


def lower_third(m: Moment, title: str, path: Path) -> None:
    """Draws a clip's lower third as a PNG the size of its panel: the claim in capitals over who,
    when in the game and the video's title, on a dark panel behind an accent bar. Text too wide
    for TEXT_MAX_W shrinks (the headline) or loses the title's later parts after " | ", then ends
    in an ellipsis."""
    headline_text = _drawable(m.claim).upper()
    size = HEADLINE_PX
    headline = ImageFont.truetype(HEADLINE_FONT, size)
    while headline.getlength(headline_text) > TEXT_MAX_W and size > MIN_HEADLINE_PX:
        size -= 2
        headline = ImageFont.truetype(HEADLINE_FONT, size)
    headline_text = _ellipsized(headline_text, headline)
    detail = ImageFont.truetype(DETAIL_FONT, DETAIL_PX, index=DETAIL_FACE)
    when = reading_text(m.clock) if m.clock else ""
    detail_text = _drawable(" · ".join(part for part in (m.who, when, title) if part))
    if detail.getlength(detail_text) > TEXT_MAX_W:
        detail_text = _drawable(" · ".join(part for part in (m.who, when, title.split(" | ")[0]) if part))
    detail_text = _ellipsized(detail_text, detail)
    text_w = max(headline.getlength(headline_text), detail.getlength(detail_text))
    panel = Image.new("RGBA", (ACCENT_W + 2 * PAD + math.ceil(text_w), PANEL_H), PANEL_COLOR)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, ACCENT_W - 1, PANEL_H - 1), fill=ACCENT_COLOR)
    draw.text((ACCENT_W + PAD, HEADLINE_BASELINE), headline_text, font=headline, fill=HEADLINE_COLOR, anchor="ls")
    draw.text((ACCENT_W + PAD, DETAIL_BASELINE), detail_text, font=detail, fill=DETAIL_COLOR, anchor="ls")
    panel.save(path)


def _drawable(text: str) -> str:
    """The text without emoji and other pictographs, which the label fonts cannot draw."""
    kept = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cf") and not "\ufe00" <= c <= "\ufe0f")
    return " ".join(kept.split())


def _ellipsized(text: str, font: ImageFont.FreeTypeFont) -> str:
    """The text cut to TEXT_MAX_W pixels, ending in an ellipsis where it was cut."""
    if font.getlength(text) <= TEXT_MAX_W:
        return text
    while font.getlength(text + "…") > TEXT_MAX_W:
        text = text[:-1]
    return text.rstrip() + "…"


# Rendering


def render(
    cuts: list[Cut], videos: Mapping[str, Video], song: Path | None, grid: Beats | None, out: Path, mute_video: bool
) -> None:
    """Renders the reel to `out`. Each clip renders in its own ffmpeg process, RENDER_WORKERS at a
    time, to one frame size, frame rate and audio format and to an exact number of frames, so the
    clips add up to the beat grid; a last pass joins them without re-encoding the video and lays
    the song, from its first beat, over the game's own sound."""
    ends = [round(cut.at * FPS) for cut in cuts] + [round((cuts[-1].at + cuts[-1].end - cuts[-1].start) * FPS)]
    frames = [end - start for start, end in itertools.pairwise(ends)]
    with tempfile.TemporaryDirectory(dir=store.data_dir("work")) as work:
        folder = Path(work)

        def clip(i: int) -> Path:
            cut = cuts[i]
            label, part = folder / f"{i}.png", folder / f"{i}.mov"
            lower_third(cut.moment, videos[cut.moment.video_id].title, label)
            _clip(cut, videos[cut.moment.video_id], frames[i], label, not mute_video, i == len(cuts) - 1, part)
            return part

        with ThreadPoolExecutor(RENDER_WORKERS) as pool:
            parts = list(pool.map(clip, range(len(cuts))))
        listing = folder / "clips.txt"
        listing.write_text("".join(f"file '{p}'\nduration {n / FPS:.6f}\n" for p, n in zip(parts, frames, strict=True)))
        _join(listing, sum(frames) / FPS, song, grid, not mute_video, out)


def _clip(cut: Cut, video: Video, frames: int, label: Path, game_sound: bool, last: bool, out: Path) -> None:
    """Renders one clip: the video fit into WIDTH x HEIGHT at FPS and held to exactly `frames`
    frames (repeating the last one if the video ends first), with the lower third sliding in and
    out, and the game's sound, or silence for a video without sound, held to the same length."""
    seconds = frames / FPS
    leave = min(LOWER_THIRD_S, seconds - SLIDE_S)
    slide = f"pow(1-clip((t-{SLIDE_IN_AT_S})/{SLIDE_S},0,1),3)+pow(clip((t-{leave:.3f})/{SLIDE_S},0,1),3)"
    graph = (
        f"[0:v]fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:-1:-1,"
        f"setsar=1,tpad=stop=-1:stop_mode=clone,trim=end_frame={frames},setpts=PTS-STARTPTS[base];"
        f"[base][1:v]overlay=x='{MARGIN}-({MARGIN}+w)*({slide})':y=H-h-{MARGIN}:eval=frame"
        f":enable='lt(t,{leave + SLIDE_S:.3f})',format=yuv420p"
    )
    if last:
        fade = min(END_FADE_S, seconds / 2)
        graph += f",fade=t=out:st={seconds - fade:.3f}:d={fade:.3f}"
    graph += "[v]"
    inputs = ["-ss", f"{cut.start:.3f}", "-t", f"{cut.end - cut.start + TAIL_S:.3f}", "-i", str(video.path)]
    inputs += ["-i", str(label)]
    outputs = ["-map", "[v]", *VIDEO_CODEC]
    if game_sound:
        samples = frames * AUDIO_RATE // FPS
        if not video.has_audio:
            inputs += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        graph += (
            f";[{0 if video.has_audio else 2}:a]aresample={AUDIO_RATE},aformat=sample_fmts=s16:channel_layouts=stereo,"
            f"atrim=end_sample={samples},asetpts=N/SR/TB,apad=whole_len={samples}[a]"
        )
        outputs += ["-map", "[a]", "-c:a", "pcm_s16le"]
    media.ffmpeg([*inputs, "-filter_complex", graph, *outputs], out)


def _join(listing: Path, seconds: float, song: Path | None, grid: Beats | None, game_sound: bool, out: Path) -> None:
    """Joins the rendered clips listed in `listing` into `out` without re-encoding their video, and
    mixes the sound: the song with the game's sound under it, either alone, or none at all."""
    inputs = ["-f", "concat", "-safe", "0", "-i", str(listing)]
    fade = min(END_FADE_S, seconds / 2)
    fade_out = f"afade=t=out:st={seconds - fade:.3f}:d={fade:.3f}"
    if song is not None and grid is not None:
        inputs += ["-i", str(song)]
        music = f"[1:a]{_music(grid, seconds)}"
        mix = f"[m];[m][0:a]amix=inputs=2:duration=first:weights='1 {GAME_VOLUME}'," if game_sound else ","
        graph: str | None = f"{music}{mix}{fade_out}[a]"
    else:
        graph = f"[0:a]{fade_out}[a]" if game_sound else None
    sound = ["-filter_complex", graph, "-map", "[a]", "-c:a", "aac", "-b:a", "192k"] if graph else []
    media.ffmpeg([*inputs, *sound, "-map", "0:v", "-c:v", "copy", "-movflags", "+faststart", "-f", "mp4"], out)


def _music(grid: Beats, seconds: float) -> str:
    """The filters that play the song from its first beat for `seconds`. A song that ends sooner
    loops whole bars (four beats) from its first beat, so the beat carries on through the loop."""
    start = round(grid.first * AUDIO_RATE)
    left = grid.duration - grid.first
    if left >= seconds:
        loop = ""
        end = start + math.ceil(seconds * AUDIO_RATE)
    else:
        beats_left = math.floor(left / grid.period + 1e-6)
        size = round((beats_left // 4 * 4 or beats_left) * grid.period * AUDIO_RATE)
        loop = f"aloop=loop=-1:size={size},asetpts=N/SR/TB,atrim=duration={seconds:.6f},"
        end = start + size
    return (
        f"aresample={AUDIO_RATE},aformat=channel_layouts=stereo,atrim=start_sample={start}:end_sample={end},"
        f"asetpts=N/SR/TB,{loop}afade=t=in:d=0.01"
    )


# The sources file


def sources(cuts: list[Cut], videos: Mapping[str, Video]) -> list[dict[str, Any]]:
    """Where each clip of a reel comes from: its place in the reel, its video and range there, and
    the find result it shows."""
    listed = []
    for cut in cuts:
        m, video = cut.moment, videos[cut.moment.video_id]
        reel_end = cut.at + cut.end - cut.start
        listed.append(
            {
                "reel_start": round(cut.at, 2),
                "reel_end": round(reel_end, 2),
                "reel_timecode": format_range(cut.at, reel_end),
                "video_id": m.video_id,
                "title": video.title,
                "source": video.source,
                "start": round(cut.start, 2),
                "end": round(cut.end, 2),
                "timecode": format_range(cut.start, cut.end),
                "obs_id": m.obs_id,
                "moment": format_range(m.start, m.end),
                "confidence": m.confidence,
                "claim": m.claim,
                "who": m.who or None,
                "game_clock": reading_text(m.clock) if m.clock else None,
            }
        )
    return listed


def sources_markdown(
    claim: str, reel: Path, listed: list[dict[str, Any]], song: Path | None, grid: Beats | None
) -> str:
    """The sources as a list to paste under the reel wherever it is posted: each clip's place in
    the reel, then where it comes from."""
    music = f"; music: {song.name} at {grid.bpm:.1f} bpm" if song and grid else ""
    lines = [
        f"# Sources: {claim}",
        "",
        f"{reel.name}: {len(listed)} clips, {format_timecode(listed[-1]['reel_end'])}{music}",
        "",
    ]
    for n, clip in enumerate(listed, 1):
        lines.append(f"{n}. {clip['reel_timecode']}: {clip['title']}")
        lines.append(f"   {clip['source']} at {clip['timecode']} (video_id {clip['video_id']})")
        shows = " · ".join(part for part in (clip["claim"], clip["who"], clip["game_clock"]) if part)
        lines.append(f"   {shows} (find result {clip['obs_id']}, confidence {clip['confidence']:.2f})")
    return "\n".join(lines) + "\n"


# The reel


def make(
    db: sqlite3.Connection,
    claim: str,
    confirmed: list[dict[str, Any]],
    rows: Mapping[str, sqlite3.Row],
    song: Path | None,
    out: Path | None,
    order: str,
    bpm: float | None,
    mute_video: bool,
    progress: Progress,
) -> dict[str, Any]:
    """Cuts the confirmed results of a find into a reel at `out` (by default a name in clips/ that
    changes when the cuts do) with its sources beside it as .sources.md and .sources.json. `rows`
    holds the videos table row of every video the results come from."""
    started = time.monotonic()
    say = progress or (lambda _message: None)
    names = dict(db.execute("SELECT entity_id, name FROM entities WHERE name IS NOT NULL").fetchall())
    videos = {video_id: _video(db, row) for video_id, row in rows.items()}
    grid = None
    if song is not None:
        grid = beats(song, bpm)
        say(f"music: {grid.bpm:.1f} bpm, first beat {grid.first:.2f} s into {song.name}")
    moments = []
    for result in confirmed:
        shown = _span(result["start"], result["end"], videos[result["video_id"]], grid)
        moments.append(moment(result, names.get(result["entity_id"]), screen_text(db, result["video_id"], *shown)))
    cuts = plan(moments, videos, grid, order)
    if out is None:
        key = json.dumps(
            [[c.moment.video_id, round(c.start, 2), round(c.end, 2)] for c in cuts] + [str(song), mute_video]
        )
        out = store.data_dir("clips") / f"reel-{media.slug(claim)}-{hashlib.sha1(key.encode()).hexdigest()[:10]}.mp4"
    say(f"rendering {len(cuts)} clips")
    render(cuts, videos, song, grid, out, mute_video)
    listed = sources(cuts, videos)
    sources_json, sources_md = out.with_suffix(".sources.json"), out.with_suffix(".sources.md")
    media.write_atomically(
        sources_json, lambda p: p.write_text(json.dumps(listed, indent=2, ensure_ascii=False) + "\n")
    )
    media.write_atomically(sources_md, lambda p: p.write_text(sources_markdown(claim, out, listed, song, grid)))
    return {
        "path": str(out),
        "sources_path": str(sources_md),
        "sources_json_path": str(sources_json),
        "claim": claim,
        "music": str(song) if song else None,
        "bpm": round(grid.bpm, 2) if grid else None,
        "duration": listed[-1]["reel_end"],
        "cuts": listed,
        "seconds": round(time.monotonic() - started, 2),
    }


def _video(db: sqlite3.Connection, row: sqlite3.Row) -> Video:
    path = Path(row["local_path"])
    shots = tuple(
        t for (t,) in db.execute("SELECT start FROM shots WHERE video_id = ? ORDER BY start", (row["video_id"],))
    )
    return Video(row["video_id"], row["title"], row["source"], path, row["duration"], shots, media.has_audio(path))
