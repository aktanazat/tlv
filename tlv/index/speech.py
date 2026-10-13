"""The transcript: Whisper segments with word times, from the video's audio track."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tlv import guard, media, store

VERSION = 1


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    video_id = video["video_id"]
    audio = audio_file(video)
    segments = []
    if audio:
        with guard.ModelProcess("tlv.models.whisper", progress=progress) as whisper:
            segments = whisper.call({"audio": str(audio)})["segments"]
    rows = [
        (video_id, segment["start"], segment["end"], segment["text"], json.dumps(segment["words"]))
        for segment in segments
        if segment["text"]
    ]
    with store.transaction(db):
        db.execute("DELETE FROM speech WHERE video_id = ?", (video_id,))
        db.executemany("INSERT INTO speech (video_id, start, end, text, words) VALUES (?, ?, ?, ?, ?)", rows)


def audio_file(video: sqlite3.Row) -> Path | None:
    """The video's audio as a 16 kHz mono WAV under work/<video_id>/, extracted once and shared
    by speech and sounds, which run at the same time; None when the video has no sound track, as
    screen recordings often do."""
    source = Path(video["local_path"])
    if not media.has_audio(source):
        return None
    path = store.data_dir("work") / video["video_id"] / "audio.wav"
    with store.lock(f"audio-{video['video_id']}"):
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            media.audio(source, path)
    return path
