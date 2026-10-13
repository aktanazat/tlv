"""Frames every half second, saved as JPEGs and embedded with SigLIP 2 for visual search."""

from __future__ import annotations

import bisect
import math
import sqlite3
from pathlib import Path

from tlv import embed, guard, media, store

VERSION = 1
EVERY_S = 0.5
WIDTH = 512


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    video_id = video["video_id"]
    sampled = media.sample_frames(Path(video["local_path"]), EVERY_S, WIDTH, store.data_dir("frames") / video_id)
    vectors = embed.images([path for _, path in sampled], progress)
    starts = [
        row["start"] for row in db.execute("SELECT start FROM shots WHERE video_id = ? ORDER BY idx", (video_id,))
    ]
    rows = [
        (video_id, t, str(path), bisect.bisect_right(starts, t) - 1, store.to_blob(vector))
        for (t, path), vector in zip(sampled, vectors, strict=True)
    ]
    with store.transaction(db):
        db.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
        db.executemany("INSERT INTO frames (video_id, t, path, shot_idx, embedding) VALUES (?, ?, ?, ?, ?)", rows)


def times(duration: float) -> list[float]:
    """The middle of every EVERY_S slot of a video `duration` seconds long: the grid `run` samples
    on, given without decoding the video."""
    return [round((i + 0.5) * EVERY_S, 3) for i in range(math.ceil(duration / EVERY_S - 0.5))]
