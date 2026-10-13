"""On-screen text, faces and bodies from Apple Vision, through the tlv-vision helper in swift/.

The helper is compiled with swiftc on first use. It runs outside the model host: Vision's models
ship with macOS and stay small (172 MB peak footprint over a whole 720p video), so they never
compete with the MLX models for the memory the guard protects. It takes its own lock instead,
since two helpers at once contend for the Neural Engine and one fails with an e5rt error.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tlv import guard, media, store

VERSION = 1
SOURCE = Path(__file__).resolve().parents[2] / "swift" / "tlv-vision.swift"
BINARY = SOURCE.parent / ".build" / "tlv-vision"
# Text is read on every second sampled frame, once a second: on-screen text stays up for seconds.
TEXT_EVERY = 2
FACE_KEYS = ("landmarks", "roll", "yaw", "quality")

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TextLine:
    """One line of on-screen text. Boxes are x, y, w, h normalized to the image, top-left origin."""

    text: str
    conf: float
    box: Box


@dataclass(frozen=True)
class Detection:
    """A face or a body. A face's extra holds what Vision found of landmarks, roll, yaw (radians)
    and capture quality (0..1); landmarks are five [x, y] points in image coordinates: the eye
    with the smaller x, the other eye, the nose tip, the mouth corner with the smaller x, the other."""

    kind: Literal["face", "body"]
    box: Box
    conf: float
    extra: dict[str, Any]


def read_text(paths: Sequence[str | Path]) -> list[list[TextLine]]:
    """The lines of text in each image file."""
    return [_lines(seen) for seen in _images(paths, text=True, detect=False)]


def detect(paths: Sequence[str | Path]) -> list[list[Detection]]:
    """The faces and bodies in each image file."""
    return [_detections(seen) for seen in _images(paths, text=False, detect=True)]


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    """Reads text once a second and finds faces and bodies twice a second, on full-resolution
    frames at exactly the times tlv.index.frames sampled, so every row joins a stored frame by t."""
    video_id = video["video_id"]
    times = [row["t"] for row in db.execute("SELECT t FROM frames WHERE video_id = ? ORDER BY t", (video_id,))]
    samples = [{"t": t, "text": i % TEXT_EVERY == 0, "detect": True} for i, t in enumerate(times)]
    ocr: list[tuple[Any, ...]] = []
    detections: list[tuple[Any, ...]] = []
    for done, seen in enumerate(run_helper("video", {"path": video["local_path"], "samples": samples}), 1):
        t = seen["t"]
        ocr += [(video_id, t, line.text, *line.box, line.conf) for line in _lines(seen)]
        detections += [
            (video_id, t, found.kind, *found.box, found.conf, json.dumps(found.extra) if found.extra else None)
            for found in _detections(seen)
        ]
        if progress and done % 240 == 0:
            progress(f"vision: {done} of {len(samples)} frames")
    with store.transaction(db):
        db.execute("DELETE FROM ocr WHERE video_id = ?", (video_id,))
        db.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))
        db.executemany("INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ocr)
        db.executemany(
            "INSERT INTO detections (video_id, t, kind, x, y, w, h, conf, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            detections,
        )


def run_helper(mode: str, job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Runs one tlv-vision job ("images", "video" or "sounds"), yielding result lines as they come."""
    with (
        store.lock("tlv-vision"),
        tempfile.TemporaryFile("w+") as errors,
        subprocess.Popen(
            [str(helper()), mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True
        ) as child,
    ):
        child.stdin.write(json.dumps(job))
        child.stdin.close()
        yield from (json.loads(line) for line in child.stdout)
        if child.wait():
            errors.seek(0)
            raise RuntimeError(f"tlv-vision {mode} exited with {child.returncode}: {errors.read().strip()[-600:]}")


def helper() -> Path:
    """The compiled tlv-vision, built when missing or older than its source."""
    with store.lock("tlv-vision-build"):
        if not BINARY.exists() or BINARY.stat().st_mtime < SOURCE.stat().st_mtime:
            BINARY.parent.mkdir(exist_ok=True)
            media.write_atomically(BINARY, lambda partial: media.run(["swiftc", "-O", "-o", str(partial), str(SOURCE)]))
    return BINARY


def _images(paths: Sequence[str | Path], text: bool, detect: bool) -> list[dict[str, Any]]:
    if not paths:
        return []
    return list(run_helper("images", {"paths": [str(path) for path in paths], "text": text, "detect": detect}))


def _lines(seen: dict[str, Any]) -> list[TextLine]:
    return [TextLine(line["text"], line["conf"], tuple(line["box"])) for line in seen.get("text", [])]


def _detections(seen: dict[str, Any]) -> list[Detection]:
    faces = [
        Detection("face", tuple(face["box"]), face["conf"], {key: face[key] for key in FACE_KEYS if key in face})
        for face in seen.get("faces", [])
    ]
    return faces + [Detection("body", tuple(body["box"]), body["conf"], {}) for body in seen.get("bodies", [])]
