"""A video without a sound track, as screen recordings often are, indexes with no speech and no
sounds, and loads no speech model for nothing."""

import subprocess

from tlv import guard
from tlv.index import sounds, speech


def test_a_video_without_sound_has_no_speech_or_sounds_and_loads_no_model(db, video, monkeypatch):
    lavfi = f"testsrc=duration={video['duration']}:size=320x180:rate=10"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", lavfi, video["local_path"]], check=True)

    def no_model(*args, **kwargs):
        raise AssertionError("a model was loaded for a video with no sound")

    monkeypatch.setattr(guard, "ModelProcess", no_model)
    speech.run(db, video, None)
    sounds.run(db, video, None)
    counts = [
        db.execute(f"SELECT count(*) FROM {table} WHERE video_id = 'v_test'").fetchone()[0]
        for table in ("speech", "sounds")
    ]
    assert counts == [0, 0]
