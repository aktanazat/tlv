"""Camera cuts, from ffmpeg's scene-change score on a small copy of every frame."""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Iterable

from tlv import guard, media, store

VERSION = 1
# ffmpeg's own default for scdet. Checked by eye on the 76ers video: scores of 12 and up were
# cuts, fast pans and slow-motion replays reached 8 to 9.
THRESHOLD = 10.0
# A dissolve or a flash scores high on several frames in a row; the strongest of those counts.
MIN_SHOT_S = 0.5
SCORE = re.compile(r"lavfi\.scd\.score=([\d.]+)\nlavfi\.scd\.time=([\d.]+)")


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    video_id = video["video_id"]
    found = shots(scores(video["local_path"]), video["duration"])
    with store.transaction(db):
        db.execute("DELETE FROM shots WHERE video_id = ?", (video_id,))
        db.executemany(
            "INSERT INTO shots (video_id, idx, start, end) VALUES (?, ?, ?, ?)",
            [(video_id, idx, start, end) for idx, (start, end) in enumerate(found)],
        )


def scores(path: str) -> list[tuple[float, float]]:
    """(time, scene-change score 0..100) for every frame of the video."""
    printed = media.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-an",
            "-vf",
            "scale=256:-2,scdet=threshold=0,metadata=mode=print:file=-",
            "-f",
            "null",
            "-",
        ]
    )
    return [(float(t), float(score)) for score, t in SCORE.findall(printed)]


def shots(scores: Iterable[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """(start, end) of each shot, covering [0, duration]: a cut wherever the score reaches
    THRESHOLD, and only the strongest of cuts closer together than MIN_SHOT_S."""
    cuts = [(0.0, math.inf)]
    for t, score in scores:
        if score < THRESHOLD:
            continue
        if t - cuts[-1][0] >= MIN_SHOT_S:
            cuts.append((t, score))
        elif score > cuts[-1][1]:
            cuts[-1] = (t, score)
    starts = [0.0] + [round(t, 3) for t, _ in cuts[1:] if duration - t >= MIN_SHOT_S]
    return list(zip(starts, [*starts[1:], duration], strict=True))
