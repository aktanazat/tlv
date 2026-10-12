"""Memory guard for model runs, so a model can never swamp the Mac.

Every model (vision-language, speech, embeddings, faces) runs in a child process started by
`ModelProcess`, which:

1. takes the machine-wide "model" lock, so only one model is loaded at a time across every
   tlv process and agent;
2. waits until the kernel's memory pressure allows the run: never at critical, and at warn
   only when the reclaimable memory covers the cap;
3. watches the child's physical footprint (the number Activity Monitor shows, GPU buffers
   included) four times a second and kills it past the cap, or when pressure turns critical
   for several seconds while the child holds real memory.

The cap is TLV_MODEL_MEMORY_GB (default 6; the 4B vision-language model peaks near 4.6 GB on
six frames). The child also gets 80% of the cap as its MLX memory limit, so it frees caches
before the watchdog has to act.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from tlv import store

GB = 1024**3
NORMAL, WARN, CRITICAL = 1, 2, 4
CAP_BYTES = int(float(os.environ.get("TLV_MODEL_MEMORY_GB", "6")) * GB)
HEADROOM_WAIT_S = float(os.environ.get("TLV_HEADROOM_WAIT_S", "600"))
WATCH_EVERY_S = 0.25
# Critical pressure this long while the child holds at least CRITICAL_HOLD_BYTES stops it.
CRITICAL_GRACE_S = 5.0
CRITICAL_HOLD_BYTES = GB

Progress = Callable[[str], None]

_libc = ctypes.CDLL(None)
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_RUSAGE_INFO_V2 = 2
_PHYS_FOOTPRINT_OFFSET = 72  # struct rusage_info_v2: uuid[16] then 7 uint64 before ri_phys_footprint


class HeadroomError(RuntimeError):
    """The Mac stayed too short of memory to start a model run."""


class ModelError(RuntimeError):
    """A model child died, was stopped by the watchdog, or reported a failed job."""


def pressure_level() -> int:
    """The kernel's memory pressure: 1 normal, 2 warn, 4 critical."""
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if _libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value), ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), "sysctl kern.memorystatus_vm_pressure_level failed")
    return value.value


def reclaimable_bytes() -> int:
    """Free, inactive, speculative and purgeable memory: what a new process can take without
    making the kernel compress or swap someone else's pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^Pages ([\w ]+):\s+(\d+)\.", out, re.M)}
    return page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))


def footprint(pid: int) -> int:
    """Physical footprint of a process in bytes, or 0 once it has exited."""
    buffer = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, buffer) != 0:
        return 0
    return int.from_bytes(buffer.raw[_PHYS_FOOTPRINT_OFFSET : _PHYS_FOOTPRINT_OFFSET + 8], "little")


def headroom(need: int) -> str | None:
    """None when a run needing `need` bytes may start now, else why not."""
    level = pressure_level()
    spare = reclaimable_bytes()
    if level >= CRITICAL:
        return f"memory pressure is critical ({spare / GB:.1f} GB reclaimable)"
    if level >= WARN and spare < need:
        return (
            f"memory pressure is elevated and only {spare / GB:.1f} GB is reclaimable; the run needs {need / GB:.1f} GB"
        )
    return None


def wait_for_headroom(need: int, progress: Progress | None = None, wait_s: float = HEADROOM_WAIT_S) -> None:
    deadline = time.monotonic() + wait_s
    reason = headroom(need)
    told = False
    while reason:
        if time.monotonic() >= deadline:
            raise HeadroomError(f"Not starting a model run: {reason}. Close some apps or try again later.")
        if progress and not told:
            progress(f"waiting for memory: {reason}")
            told = True
        time.sleep(5)
        reason = headroom(need)


class ModelProcess:
    """A model loaded in a guarded child process, answering JSON jobs one at a time.

    with ModelProcess("tlv.models.vlm", progress=print) as vlm:
        answer = vlm.call({"images": [...], "prompt": "..."})
    """

    def __init__(
        self,
        module: str,
        *,
        cap_bytes: int = CAP_BYTES,
        progress: Progress | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.module = module
        self.cap = cap_bytes
        self.progress = progress
        self.env = env or {}
        self.peak = 0
        self._child: subprocess.Popen[str] | None = None
        self._stopped: str | None = None
        self._stderr: deque[str] = deque(maxlen=20)
        self._lock_cm: Any = None

    def __enter__(self) -> ModelProcess:
        name = self.module.rsplit(".", 1)[-1]
        waiting = f"waiting for another model run to finish before {name}"
        self._lock_cm = store.lock("model", waiting=lambda: self._say(waiting))
        self._lock_cm.__enter__()
        try:
            wait_for_headroom(self.cap, self.progress)
            env = {**os.environ, **self.env, "TLV_MEMORY_LIMIT_BYTES": str(int(self.cap * 0.8))}
            self._child = subprocess.Popen(
                [sys.executable, "-m", self.module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            threading.Thread(target=self._drain_stderr, daemon=True).start()
            threading.Thread(target=self._watch, daemon=True).start()
            ready = self._receive()
            if not ready.get("ready"):
                raise ModelError(f"{name} did not start: {ready}")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def call(self, job: dict[str, Any]) -> dict[str, Any]:
        child = self._require_child()
        try:
            child.stdin.write(json.dumps(job) + "\n")
            child.stdin.flush()
        except BrokenPipeError:
            raise self._death() from None
        result = self._receive()
        if "error" in result:
            raise ModelError(f"{self.module} failed a job: {result['error']}")
        return result

    def __exit__(self, *exc: object) -> None:
        child, self._child = self._child, None
        if child:
            if child.stdin:
                child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if self._lock_cm:
            self._lock_cm.__exit__(None, None, None)
            self._lock_cm = None

    def _require_child(self) -> subprocess.Popen[str]:
        if not self._child:
            raise ModelError(f"{self.module} is not running")
        return self._child

    def _receive(self) -> dict[str, Any]:
        line = self._require_child().stdout.readline()
        if not line:
            raise self._death()
        return json.loads(line)

    def _death(self) -> ModelError:
        child = self._require_child()
        child.wait()
        if self._stopped:
            return ModelError(self._stopped)
        tail = "\n".join(self._stderr) or "no output"
        return ModelError(f"{self.module} exited with code {child.returncode}:\n{tail}")

    def _drain_stderr(self) -> None:
        child = self._child
        if not child or not child.stderr:
            return
        for line in child.stderr:
            self._stderr.append(line.rstrip())

    def _watch(self) -> None:
        child = self._child
        if not child:
            return
        critical_since: float | None = None
        while child.poll() is None:
            used = footprint(child.pid)
            self.peak = max(self.peak, used)
            if used > self.cap:
                over = f"{self.module} used {used / GB:.1f} GB, over the {self.cap / GB:.1f} GB cap"
                self._stop(child, f"{over}, so it was stopped")
                return
            if pressure_level() >= CRITICAL and used >= CRITICAL_HOLD_BYTES:
                critical_since = critical_since or time.monotonic()
                if time.monotonic() - critical_since >= CRITICAL_GRACE_S:
                    held = f"{self.module} held {used / GB:.1f} GB"
                    self._stop(child, f"memory pressure turned critical while {held}, so it was stopped")
                    return
            else:
                critical_since = None
            time.sleep(WATCH_EVERY_S)

    def _stop(self, child: subprocess.Popen[str], reason: str) -> None:
        self._stopped = reason
        child.kill()
        self._say(reason)

    def _say(self, message: str) -> None:
        if self.progress:
            self.progress(message)
