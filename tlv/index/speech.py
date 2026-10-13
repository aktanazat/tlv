"""The transcript: Whisper segments with word times, from the video's audio track."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tlv import guard, media, store

VERSION = 1


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    video_id = video["video_id"]
    with guard.ModelProcess("tlv.models.whisper", progress=progress) as whisper:
        result = whisper.call({"audio": str(audio_file(video))})
    rows = [
        (video_id, segment["start"], segment["end"], segment["text"], json.dumps(segment["words"]))
        for segment in result["segments"]
        if segment["text"]
    ]
    with store.transaction(db):
        db.execute("DELETE FROM speech WHERE video_id = ?", (video_id,))
        db.executemany("INSERT INTO speech (video_id, start, end, text, words) VALUES (?, ?, ?, ?, ?)", rows)


def audio_file(video: sqlite3.Row) -> Path:
    """The video's audio as a 16 kHz mono WAV under work/<video_id>/, extracted once and shared
    by speech and sounds."""
    path = store.data_dir("work") / video["video_id"] / "audio.wav"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        media.audio(Path(video["local_path"]), path)
    return path
