"""A private $TLV_HOME, and a video row to index."""

import pytest

from tlv import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("TLV_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TLV_INDEX", raising=False)
    return store.connect()


@pytest.fixture
def video(db, tmp_path):
    """An 8 s video row; its file is never read by the fakes these tests run."""
    db.execute(
        "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
        " VALUES ('v_test', 'Test', 'test.mp4', ?, 8.0, 1280, 720, 30, ?, 'queued')",
        (str(tmp_path / "test.mp4"), store.now()),
    )
    return db.execute("SELECT * FROM videos WHERE video_id = 'v_test'").fetchone()
