"""Models run in the shared model host: each held to its cap, kept loaded between runs, and let go
when idle, over the held-memory cap, or when the Mac is short of memory."""

import contextlib
import os
import re
import shutil
import signal
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

from tlv import guard, host

HOG = textwrap.dedent(
    """
    import os
    import time
    import numpy as np
    from tlv.models.protocol import serve

    held = []

    def load():
        for _ in range(int(os.environ.get("HOG_LOAD_MB", "0")) // 50):
            held.append(np.ones(50 * 1024 * 1024, dtype=np.uint8))

        def handle(job):
            began = time.time()
            for _ in range(job.get("mb", 0) // 50):
                held.append(np.ones(50 * 1024 * 1024, dtype=np.uint8))
                time.sleep(0.02)
            if job.get("fail"):
                raise ValueError("bad frame")
            time.sleep(job.get("sleep", 0))
            if job.get("hold"):  # runs until the test deletes this file
                open(job["hold"], "w").close()
                while os.path.exists(job["hold"]):
                    time.sleep(0.01)
            return {"held_mb": 50 * len(held), "pid": os.getpid(), "began": began, "ended": time.time()}
        return handle

    serve(load)
    """
)

MB = 1024 * 1024


def _headroom_script(path, free_pct, swap_mb=0):
    host = f'{{"pressure_free_pct": "{free_pct}", "swap_used_mb": {swap_mb}}}'
    path.write_text(f"#!/bin/sh\necho '{{\"host\": {host}}}'\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def hog(tmp_path, monkeypatch):
    """The hog model's env, served by a private model host that is stopped afterwards. The host's
    folder is short because a Unix socket path holds at most 104 bytes, which tmp_path exceeds."""
    run = Path(tempfile.mkdtemp(prefix="tlv-"))
    monkeypatch.setenv("TLV_MODEL_HOST", str(run / "models.sock"))
    monkeypatch.setenv("TLV_MODEL_IDLE_S", "60")
    monkeypatch.setenv("TLV_PRESSURE_CHECK_S", "0.2")
    monkeypatch.setenv("TLV_HEADROOM_CMD", _headroom_script(tmp_path / "ample", 60))
    (tmp_path / "hog.py").write_text(HOG)
    yield {"PYTHONPATH": str(tmp_path)}
    log = run / "models.log"
    for pid in re.findall(r"model host (\d+) serving", log.read_text() if log.exists() else ""):
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(pid), signal.SIGTERM)
    shutil.rmtree(run)


def _run(env, job=None, cap_mb=600, progress=None):
    with guard.ModelProcess("hog", cap_bytes=cap_mb * MB, env=env, progress=progress) as model:
        return model.call(job or {})


@pytest.mark.parametrize(
    ("job", "cap_mb", "outcome"),
    [
        ({"mb": 100}, 600, "ok"),
        ({"mb": 2000}, 400, "over the 0.4 GB cap"),
        ({"fail": True}, 600, "ValueError: bad frame"),
    ],
)
def test_model_child_is_held_to_its_cap(hog, job, cap_mb, outcome):
    if outcome == "ok":
        assert _run(hog, job, cap_mb)["held_mb"] == 100
        return
    with pytest.raises(guard.ModelError, match=outcome) as error:
        _run(hog, job, cap_mb)
    used = re.search(r"used (\d+\.\d) GB", str(error.value))
    assert not used or float(used.group(1)) < 1.2, "the watchdog let the child run far past its cap"


def test_a_model_stays_loaded_between_runs(hog):
    first = _run(hog)["pid"]
    with pytest.raises(guard.ModelError, match="bad frame"):
        _run(hog, {"fail": True})
    assert _run(hog)["pid"] == first


def test_a_loaded_model_is_replaced_once_tlv_code_changes(hog):
    """A child runs the code it started with; after an edit or update the next run gets a fresh one."""
    first = _run(hog)["pid"]
    os.utime(Path(guard.__file__).with_name("models") / "protocol.py")  # what saving the file does
    assert _run(hog)["pid"] != first


@pytest.mark.parametrize(
    ("setting", "job", "open_between", "let_go"),
    [
        pytest.param({"TLV_MODEL_IDLE_S": "1"}, {}, False, True, id="unused past the idle time"),
        pytest.param({"TLV_MODEL_IDLE_S": "1"}, {}, True, False, id="idle but held open by its caller"),
        pytest.param({"TLV_HELD_MEMORY_GB": "0.25"}, {"mb": 300}, False, True, id="over the held-memory cap"),
        pytest.param({"headroom": (10, 0)}, {}, False, True, id="free memory under the floor"),
        pytest.param({"headroom": (10, 0)}, {}, True, False, id="short of memory but held open by its caller"),
        # 2026-10-12 9:20pm: swap at 12.3 of 13.3 GB, yet host-headroom's free share read 33%
        pytest.param({"headroom": (33, 12600)}, {}, False, True, id="swap past 8 GB while free memory reads fine"),
        pytest.param({"headroom": (60, 7000)}, {}, False, False, id="ample free memory and swap under 8 GB"),
    ],
)
def test_models_are_let_go_when_idle_over_the_cap_or_short_of_memory(
    hog, monkeypatch, tmp_path, setting, job, open_between, let_go
):
    for name, value in setting.items():
        if name == "headroom":
            monkeypatch.setenv("TLV_HEADROOM_CMD", _headroom_script(tmp_path / "headroom", *value))
        else:
            monkeypatch.setenv(name, value)
    with guard.ModelProcess("hog", cap_bytes=600 * MB, env=hog) as model:
        first = model.call(job)["pid"]
        if open_between:
            time.sleep(2.5)
            again = model.call({})["pid"]
    if not open_between:
        time.sleep(2.5)
        again = _run(hog)["pid"]
    assert (again != first) == let_go


def test_while_memory_is_short_runs_hold_one_model_at_a_time(hog, monkeypatch, tmp_path):
    """The last memory reading holds until the next one, so between readings a model its run has
    closed is let go before another loads rather than staying beside it."""
    monkeypatch.setenv("TLV_HEADROOM_CMD", _headroom_script(tmp_path / "headroom", 33, 12600))
    monkeypatch.setenv("TLV_PRESSURE_CHECK_S", "600")  # one reading for the whole test
    _run(hog)
    kept = _run(hog)["pid"]
    _run({**hog, "TLV_OTHER": "another model"})
    with pytest.raises(ProcessLookupError):
        os.kill(kept, 0)


def test_an_idle_host_exits(hog, monkeypatch):
    monkeypatch.setenv("TLV_MODEL_IDLE_S", "1")
    first = _run(hog)["pid"]
    deadline = time.monotonic() + 10
    while guard.socket_path().exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not guard.socket_path().exists(), "the host stayed up with nothing loaded or asked for"
    assert _run(hog)["pid"] != first, "the next run starts a new host and loads the model again"


def test_runs_wait_their_turn(hog):
    """A run asked for while another's job is running waits for it instead of running beside it."""
    results = {}
    waits = []
    _run(hog)  # load the model first, so the second run below waits on the job, not the load
    thread = threading.Thread(target=lambda: results.update(first=_run(hog, {"sleep": 1.5})))
    thread.start()
    time.sleep(0.5)
    second = _run(hog, progress=waits.append)
    thread.join()
    assert second["began"] >= results["first"]["ended"]
    assert any("waiting for another model run" in w for w in waits)


def _queued_behind_a_running_job(hog, monkeypatch, tmp_path, order):
    """Runs one job per letter of `order` on model a or b, queued in that order while a job on a
    holds the host, with the held-memory cap fitting one model; returns their results in order."""
    monkeypatch.setenv("TLV_HELD_MEMORY_GB", "0.3")
    a = {**hog, "HOG_LOAD_MB": "200"}  # about 0.22 GB a model, so two are over the cap
    envs = {"a": a, "b": {**a, "TLV_OTHER": "another model"}}
    hold = tmp_path / "hold"
    results = [{}] * len(order)
    queued = [threading.Event() for _ in order]

    def waiting(event):
        def progress(message):
            if message.startswith("waiting for another model run"):
                event.set()

        return progress

    def call(i, model):
        results[i] = model.call({})

    with contextlib.ExitStack() as stack:

        def open_model(name, progress=None):
            model = guard.ModelProcess("hog", cap_bytes=600 * MB, env=envs[name], progress=progress)
            return stack.enter_context(model)

        holder = open_model("a")
        callers = [open_model(name, waiting(event)) for name, event in zip(order, queued, strict=True)]
        threads = [threading.Thread(target=holder.call, args=({"hold": str(hold)},))]
        threads[0].start()
        deadline = time.monotonic() + 10
        while not hold.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        for i, caller in enumerate(callers):
            threads.append(threading.Thread(target=call, args=(i, caller)))
            threads[-1].start()
            assert queued[i].wait(10), f"job {i} ran instead of waiting behind the held job"
        hold.unlink()
        for thread in threads:
            thread.join()
    return results


def test_queued_jobs_on_a_loaded_model_go_first_so_models_are_not_reloaded_in_turn(hog, monkeypatch, tmp_path):
    """2026-10-13: callers on different models took turns, and with room for one model each turn
    unloaded the other's. Each process that ran a job is one load of its model."""
    order = "ab" * 6
    results = _queued_behind_a_running_job(hog, monkeypatch, tmp_path, order)
    loads = len({result["pid"] for result in results})
    assert loads <= 1 + len(order) // (host.MAX_OVERTAKEN + 1), f"{loads} loads for {len(order)} jobs"


def test_a_job_on_another_model_is_passed_over_at_most_a_few_times(hog, monkeypatch, tmp_path):
    results = _queued_behind_a_running_job(hog, monkeypatch, tmp_path, "b" + "a" * 5)
    ahead = sum(result["began"] < results[0]["began"] for result in results[1:])
    assert ahead <= host.MAX_OVERTAKEN, f"{ahead} later jobs went ahead of the first"
