"""The memory guard stops a model child that grows past its cap, and passes normal jobs."""

import textwrap

import pytest

from tlv import guard

HOG = textwrap.dedent(
    """
    import time
    import numpy as np
    from tlv.models.protocol import serve

    held = []

    def load():
        def handle(job):
            for _ in range(job["mb"] // 50):
                held.append(np.ones(50 * 1024 * 1024, dtype=np.uint8))
                time.sleep(0.02)
            if job.get("fail"):
                raise ValueError("bad frame")
            time.sleep(2)
            return {"held_mb": 50 * len(held)}
        return handle

    serve(load)
    """
)

MB = 1024 * 1024


@pytest.fixture
def hog(tmp_path, monkeypatch):
    monkeypatch.setenv("TLV_HOME", str(tmp_path / "home"))
    (tmp_path / "hog.py").write_text(HOG)
    return {"PYTHONPATH": str(tmp_path)}


@pytest.mark.parametrize(
    ("job", "cap_mb", "outcome"),
    [
        ({"mb": 100}, 600, "ok"),
        ({"mb": 2000}, 400, "over the 0.4 GB cap"),
        ({"mb": 0, "fail": True}, 600, "ValueError: bad frame"),
    ],
)
def test_model_child_is_held_to_its_cap(hog, job, cap_mb, outcome):
    with guard.ModelProcess("hog", cap_bytes=cap_mb * MB, env=hog) as child:
        if outcome == "ok":
            assert child.call(job) == {"held_mb": 100}
            assert child.peak < cap_mb * MB
            return
        with pytest.raises(guard.ModelError, match=outcome):
            child.call(job)
        assert child.peak < 1200 * MB, "the watchdog let the child run far past its cap"


def test_runs_wait_their_turn_for_the_model_lock(hog):
    """A second run started while one holds the model lock waits instead of loading beside it."""
    waits = []
    with guard.ModelProcess("hog", cap_bytes=600 * MB, env=hog):
        second = guard.ModelProcess("hog", cap_bytes=600 * MB, env=hog, progress=waits.append)
        import threading

        entered = threading.Event()

        def start():
            with second:
                entered.set()

        thread = threading.Thread(target=start)
        thread.start()
        assert not entered.wait(1.5), "second model run started while the first held the lock"
    assert entered.wait(30)
    thread.join()
    assert any("waiting for another model run" in w for w in waits)
