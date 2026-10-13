"""A private $TLV_HOME with one short synthetic video in the default index."""

import subprocess

import pytest

from tlv import store

DURATION = 10.0


@pytest.fixture(scope="session")
def clip_file(tmp_path_factory):
    """A 10 s, 320x240, 25 fps test pattern with a tone: deterministic and fast to cut."""
    path = tmp_path_factory.mktemp("media") / "pattern.mp4"
    command = (
        f"ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc=duration={DURATION}:size=320x240:rate=25 "
        f"-f lavfi -i sine=frequency=440:duration={DURATION} -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest"
    )
    subprocess.run([*command.split(), str(path)], check=True)
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TLV_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TLV_INDEX", raising=False)
    return tmp_path / "home"


def _add_video(index, video_id, title, local_path):
    with store.transaction(store.connect(index)) as db:
        db.execute(
            "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status) "
            "VALUES (?, ?, ?, ?, ?, 320, 240, 25, ?, 'ready')",
            (video_id, title, f"https://example.com/{video_id}.mp4", str(local_path), DURATION, store.now()),
        )


@pytest.fixture
def add_video(home):
    """Adds a row to the videos table of an index: add_video(index, video_id, title, local_path)."""
    return _add_video


@pytest.fixture
def video(add_video, clip_file):
    """The id of the synthetic video, added to the default index."""
    add_video("tlv", "Youtube-pattern01", "Test Pattern", clip_file)
    return "Youtube-pattern01"
