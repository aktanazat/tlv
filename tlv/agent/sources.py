"""What the index holds about a time range of one video, read for prompts and evidence: the
video itself, its sampled frames, the words spoken, the text on screen and the sound events."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Video:
    video_id: str
    title: str
    source: str
    local_path: str
    duration: float


@dataclass(frozen=True)
class Frame:
    t: float
    path: Path


@dataclass(frozen=True)
class Line:
    """One timed line of transcript, on-screen text or sound."""

    start: float
    end: float
    text: str


def fit(lines: list[Line], chars: int) -> list[Line]:
    """The lines, or every k-th of them spread over the stretch when together they run past `chars`."""
    total = sum(len(line.text) for line in lines)
    if total <= chars:
        return lines
    return lines[:: math.ceil(total / chars)]


def videos(db: sqlite3.Connection, video_ids: Iterable[str] | None = None) -> list[Video]:
    """The named videos (every video when None), in the order asked. An unknown id raises."""
    rows = db.execute("SELECT video_id, title, source, local_path, duration FROM videos ORDER BY added_at").fetchall()
    found = {row["video_id"]: Video(*row) for row in rows}
    if video_ids is None:
        return list(found.values())
    wanted = list(dict.fromkeys(video_ids))
    missing = [v for v in wanted if v not in found]
    if missing:
        named = ", ".join(f'"{v}"' for v in missing)
        raise LookupError(f"No video {named} in this index. video_list (tlv list) shows the ids.")
    return [found[v] for v in wanted]


def video(db: sqlite3.Connection, video_id: str) -> Video:
    return videos(db, [video_id])[0]


def frames(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[Frame]:
    """Sampled frames with start <= t <= end, in time order."""
    rows = db.execute(
        "SELECT t, path FROM frames WHERE video_id = ? AND t BETWEEN ? AND ? ORDER BY t", (video_id, start, end)
    ).fetchall()
    return [Frame(row["t"], Path(row["path"])) for row in rows]


def _segments(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[list[Line]]:
    """The words spoken inside the range, one list per transcript segment, in time order."""
    rows = db.execute(
        "SELECT words FROM speech WHERE video_id = ? AND start < ? AND end > ? ORDER BY start", (video_id, end, start)
    ).fetchall()
    segments = [
        [
            Line(w["s"], w["e"], w["w"].strip())
            for w in json.loads(row["words"])
            if w["e"] > start and w["s"] < end and w["w"].strip()
        ]
        for row in rows
    ]
    return [segment for segment in segments if segment]


def speech(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[Line]:
    """Transcript segments cut to the words spoken inside the range."""
    return [Line(s[0].start, s[-1].end, " ".join(w.text for w in s)) for s in _segments(db, video_id, start, end)]


def spoken_words(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[Line]:
    """Each word spoken inside the range as a line of its own, in time order."""
    return [word for segment in _segments(db, video_id, start, end) for word in segment]


def screen_text(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[Line]:
    """On-screen text, one line per sampled frame in reading order; a frame showing the same text
    as the frame before it is left out."""
    rows = db.execute(
        "SELECT t, text FROM ocr WHERE video_id = ? AND t BETWEEN ? AND ? ORDER BY t, y, x", (video_id, start, end)
    ).fetchall()
    by_time: dict[float, list[str]] = {}
    for row in rows:
        by_time.setdefault(row["t"], []).append(row["text"].strip())
    lines: list[Line] = []
    for t, texts in by_time.items():
        text = " | ".join(part for part in texts if part)
        if text and (not lines or lines[-1].text != text):
            lines.append(Line(t, t, text))
    return lines


def sounds(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[Line]:
    rows = db.execute(
        "SELECT start, end, label FROM sounds WHERE video_id = ? AND start < ? AND end > ? ORDER BY start",
        (video_id, end, start),
    ).fetchall()
    return [Line(row["start"], row["end"], row["label"]) for row in rows]


def captions(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[Line]:
    """The captioner's description of each span touching the range."""
    rows = db.execute(
        "SELECT start, end, caption FROM spans WHERE video_id = ? AND start < ? AND end > ? AND caption IS NOT NULL"
        " ORDER BY start",
        (video_id, end, start),
    ).fetchall()
    return [Line(row["start"], row["end"], row["caption"]) for row in rows]


def mostly_replay(db: sqlite3.Connection, video_id: str, start: float, end: float) -> bool:
    """Whether the captioner called at least half of the range a replay."""
    rows = db.execute(
        "SELECT start, end FROM spans WHERE video_id = ? AND replay = 1 AND start < ? AND end > ?",
        (video_id, end, start),
    ).fetchall()
    flagged = sum(min(row["end"], end) - max(row["start"], start) for row in rows)
    return flagged >= 0.5 * (end - start)


def person_names(db: sqlite3.Connection) -> list[str]:
    """The names of known people, for resolving @Name mentions."""
    return [row["name"] for row in db.execute("SELECT DISTINCT name FROM entities WHERE name IS NOT NULL")]
