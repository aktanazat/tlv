"""Indexing bookkeeping: a stage reruns when its version changes or a stage it stands on reran, a
video is ready only when every stage is current, and one file is one video."""

import shutil
import subprocess

import pytest

from tlv.index import pipeline


@pytest.fixture
def ran(monkeypatch):
    """Replaces every stage with one that only records its name."""
    names = []
    for name, module in pipeline.MODULES.items():
        attribute = "index_video" if name == "entities" else "run"
        monkeypatch.setattr(module, attribute, lambda *args, name=name, **kwargs: names.append(name))
    return names


@pytest.mark.parametrize(
    ("bumped", "kwargs", "expected"),
    [
        (None, {}, []),
        ("frames", {}, ["frames", "vision", "spans", "entities"]),
        ("speech", {}, ["speech", "spans", "entities"]),
        ("entities", {}, ["entities"]),
        (None, {"only": "sounds", "force": True}, ["sounds"]),
        (None, {"force": True}, list(pipeline.STAGES)),
    ],
    ids=["unchanged", "frames version", "speech version", "entities version", "one forced", "all forced"],
)
def test_reindexing_reruns_changed_stages_and_those_built_on_them(video, ran, monkeypatch, bumped, kwargs, expected):
    pipeline.index("v_test")
    ran.clear()
    if bumped:
        monkeypatch.setattr(pipeline.MODULES[bumped], "VERSION", pipeline.MODULES[bumped].VERSION + 1)
    pipeline.index("v_test", **kwargs)
    assert ran == expected


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        ([{}], ("ready", [])),
        ([{"only": ["shots", "speech", "sounds"]}], ("queued", ["frames", "vision", "spans", "entities"])),
        ([{}, {"only": "sounds", "force": True}], ("queued", ["spans", "entities"])),
    ],
    ids=["all stages", "some stages", "an input rerun alone"],
)
def test_a_video_is_ready_only_when_every_stage_is_current(video, ran, runs, expected):
    for kwargs in runs:
        pipeline.index("v_test", **kwargs)
    status = pipeline.status("v_test")
    assert (
        status["status"],
        [name for name, stage in status["stages"].items() if stage["state"] != "done"],
    ) == expected


@pytest.mark.parametrize(
    ("only", "message"),
    [
        ("vision", "Run frames first"),
        (["spans", "vision"], "Run frames, shots, sounds, speech first"),
        ("faces", "Unknown stage faces"),
    ],
)
def test_a_stage_is_refused_without_its_inputs(video, ran, only, message):
    with pytest.raises(ValueError, match=message):
        pipeline.index("v_test", only=only)
    assert ran == []


def test_a_failed_stage_fails_the_video_and_reruns_next_time(video, ran, monkeypatch):
    def no_audio(db, video, progress):
        raise RuntimeError("no audio track")

    recorder = pipeline.MODULES["speech"].run
    monkeypatch.setattr(pipeline.MODULES["speech"], "run", no_audio)
    with pytest.raises(RuntimeError, match="no audio track"):
        pipeline.index("v_test")
    failed = pipeline.status("v_test")
    assert (failed["status"], failed["error"], failed["stages"]["speech"]["state"]) == (
        "failed",
        "RuntimeError: no audio track",
        "pending",
    )
    monkeypatch.setattr(pipeline.MODULES["speech"], "run", recorder)
    pipeline.index("v_test")
    assert pipeline.status("v_test")["status"] == "ready"


def test_the_same_file_added_from_two_places_is_one_video(db, tmp_path):
    first = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10", str(first)],
        check=True,
    )
    second = tmp_path / "elsewhere" / "renamed.mp4"
    second.parent.mkdir()
    shutil.copyfile(first, second)
    added = [pipeline.add(str(path), wait=False) for path in (first, second)]
    assert [(r["video_id"], r["already_indexed"]) for r in added] == [
        (added[0]["video_id"], False),
        (added[0]["video_id"], True),
    ]
    assert db.execute("SELECT count(*) FROM videos").fetchone()[0] == 1
