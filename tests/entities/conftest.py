"""A private $TLV_HOME, and helpers that store videos and appearances as index_video leaves them."""

import json

import pytest

from tlv import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("TLV_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TLV_INDEX", raising=False)
    return store.connect()


@pytest.fixture
def video(db, tmp_path):
    """Adds a video row and returns its id: video(video_id)."""

    def add(video_id: str) -> str:
        db.execute(
            "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
            " VALUES (?, ?, ?, ?, 60.0, 1280, 720, 30, ?, 'ready')",
            (video_id, video_id, f"{video_id}.mp4", str(tmp_path / f"{video_id}.mp4"), store.now()),
        )
        return video_id

    return add


@pytest.fixture
def appearance(db, tmp_path):
    """Stores an appearance and returns its id: appearance(id, video_id, kind, start, end, vector,
    height=0.2, clues=(), body=None, quality=0.5). `height` is the box's share of the frame height;
    `quality`, a face's capture quality, is sharp by default and ignored for bodies."""

    def add(appearance_id, video_id, kind, start, end, vector, height=0.2, clues=(), body=None, quality=0.5):
        db.execute(
            "INSERT INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path, embedding,"
            " clues, body, quality) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                appearance_id,
                video_id,
                kind,
                start,
                end,
                start,
                json.dumps([0.4, 0.1, height / 2, height]),
                str(tmp_path / f"{appearance_id}.jpg"),
                store.to_blob(vector),
                json.dumps(list(clues)),
                body,
                quality if kind == "face" else None,
            ),
        )
        return appearance_id

    return add
