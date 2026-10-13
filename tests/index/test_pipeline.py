"""Indexing bookkeeping: a stage reruns when its version changes or a stage it stands on reran,
starts once those are done, runs beside the stages it does not stand on, a video is ready only
when every stage is current, one file is one video, and indexing's model jobs are background
work."""

import json
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from tlv import guard
from tlv.index import pipeline


@pytest.fixture
def ran(monkeypatch):
    """Replaces every stage with one that only records its name."""
    names = []
    for name, module in pipeline.MODULES.items():
        attribute = "index_video" if name == "entities" else "run"
        monkeypatch.setattr(module, attribute, lambda *args, name=name, **kwargs: names.append(name))
    return names


@pytest.fixture
def model_requests(monkeypatch):
    """The requests a stand-in model host receives; it answers each with an empty result. Its
    folder is short because a Unix socket path holds at most 104 bytes."""
    run = Path(tempfile.mkdtemp(prefix="tlv-"))
    monkeypatch.setenv("TLV_MODEL_HOST", str(run / "models.sock"))
    requests = []
    stop = threading.Event()

    def answer(conn):
        with conn, conn.makefile("r", encoding="utf-8") as lines:
            for line in lines:
                requests.append(json.loads(line))
                conn.sendall(b'{"result": {}}\n')

    def serve(server):
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            conn.settimeout(None)
            threading.Thread(target=answer, args=(conn,), daemon=True).start()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(run / "models.sock"))
        server.listen(16)
        server.settimeout(0.1)
        serving = threading.Thread(target=serve, args=(server,))
        serving.start()
        yield requests
        stop.set()
        serving.join()
    shutil.rmtree(run)


def test_every_stage_sends_its_model_jobs_as_background_work(video, model_requests, monkeypatch):
    """The model host runs a job someone is waiting on, such as a search's, before background
    work. Each stage runs in a thread of its own, and every one of them must say so."""

    def stage(*args, **kwargs):
        with guard.ModelProcess("tlv.models.stand_in") as model:
            model.call({})

    for name, module in pipeline.MODULES.items():
        monkeypatch.setattr(module, "index_video" if name == "entities" else "run", stage)
    pipeline.index("v_test")
    # each stage loads its model, then sends one job
    assert [request.get("background") for request in model_requests] == [True] * 2 * len(pipeline.STAGES)


@pytest.mark.parametrize(
    ("bumped", "kwargs", "expected"),
    [
        (None, {}, []),
        ("frames", {}, ["frames", "spans", "entities"]),
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
    assert sorted(ran) == sorted(expected)


def test_a_stage_starts_only_once_the_stages_it_stands_on_are_done(video, ran):
    said = []
    pipeline.index("v_test", progress=said.append)
    started = {line.split(":")[0]: i for i, line in enumerate(said) if line.endswith(": running")}
    done = {line.split(":")[0]: i for i, line in enumerate(said) if ": done in " in line}
    assert {name: [need for need in pipeline.INPUTS[name] if done[need] > started[name]] for name in started} == {
        name: [] for name in pipeline.STAGES
    }


def test_stages_that_stand_on_nothing_run_at_the_same_time(video, ran, monkeypatch):
    # Each waits for the other three; run one at a time, the first gives up and fails the run.
    together = threading.Barrier(4, timeout=10)
    for name in ("shots", "speech", "vision", "sounds"):
        monkeypatch.setattr(pipeline.MODULES[name], "run", lambda *args, **kwargs: together.wait())
    pipeline.index("v_test")
    assert pipeline.status("v_test")["status"] == "ready"


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
        ("frames", "Run shots first"),
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
