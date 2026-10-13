"""Audio events (speech, cheering, music, ball bounces ...) from Apple's built-in sound classifier.

The classifier labels one-second windows every half second; windows of one label that overlap or
touch merge into one event that keeps the highest confidence.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tlv import guard, store
from tlv.index import speech, vision

VERSION = 1
WINDOW_S = 1.0
OVERLAP = 0.5
TOP_LABELS = 3
MIN_CONF = 0.3


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    label: str
    conf: float


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    video_id = video["video_id"]
    audio = speech.audio_file(video)
    job = {"path": str(audio), "window": WINDOW_S, "overlap": OVERLAP, "top": TOP_LABELS, "min_conf": MIN_CONF}
    events = merge(vision.run_helper("sounds", job)) if audio else []
    with store.transaction(db):
        db.execute("DELETE FROM sounds WHERE video_id = ?", (video_id,))
        db.executemany(
            "INSERT INTO sounds (video_id, start, end, label, conf) VALUES (?, ?, ?, ?, ?)",
            [(video_id, event.start, event.end, event.label, event.conf) for event in events],
        )


def merge(windows: Iterable[dict[str, Any]]) -> list[Event]:
    """Events from classifier windows ({"start", "end", "labels": [{"label", "conf"}]}), by start time."""
    by_label: dict[str, list[tuple[float, float, float]]] = {}
    for window in windows:
        for found in window["labels"]:
            by_label.setdefault(found["label"], []).append((window["start"], window["end"], found["conf"]))
    events = []
    for label, heard in by_label.items():
        heard.sort()
        start, end, conf = heard[0]
        for next_start, next_end, next_conf in heard[1:]:
            if next_start > end:
                events.append(Event(start, end, label, conf))
                start, end, conf = next_start, next_end, next_conf
            else:
                end, conf = max(end, next_end), max(conf, next_conf)
        events.append(Event(start, end, label, conf))
    return sorted(events, key=lambda event: (event.start, event.label))
