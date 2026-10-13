"""Memory guard for model runs, so a model can never swamp the Mac.

Every model (vision-language, speech, embeddings, faces) runs in a guarded child process
(`Child`) owned by the model host (tlv.host), one host per user shared by every tlv process,
agent and $TLV_HOME. Callers reach it through `ModelProcess`, which starts the host on first
use. The host runs one job at a time and keeps models loaded between runs; tlv.host says when it
lets them go.
Before a model loads, the host waits until the kernel's memory pressure allows it: never at
critical, and at warn only when the reclaimable memory covers the cap. While a child lives, its
watchdog reads the child's physical footprint (the number Activity Monitor shows, GPU buffers
included) four times a second and kills it past the cap, or when pressure turns critical for
several seconds while the child holds real memory.

The cap is TLV_MODEL_MEMORY_GB (default 6; the 4B vision-language model peaks near 5.7 GB on
twelve 448 px frames). The child also gets 80% of the cap as its MLX memory limit, so it frees
caches before the watchdog has to act.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

GB = 1024**3
NORMAL, WARN, CRITICAL = 1, 2, 4
CAP_BYTES = int(float(os.environ.get("TLV_MODEL_MEMORY_GB", "6")) * GB)
HEADROOM_WAIT_S = float(os.environ.get("TLV_HEADROOM_WAIT_S", "600"))
WATCH_EVERY_S = 0.25
# Critical pressure this long while the child holds at least CRITICAL_HOLD_BYTES stops it.
CRITICAL_GRACE_S = 5.0
CRITICAL_HOLD_BYTES = GB
HOST_START_S = 30.0  # how long a caller waits for a host it started to accept connections
# Variables that pick a model checkpoint (TLV_VLM_MODEL, ...). A caller's values travel with each
# request, so a model loaded for one setting is never handed to a caller asking for another.
MODEL_VARIABLE = re.compile(r"TLV_\w+_MODEL")

Progress = Callable[[str], None]

_libc = ctypes.CDLL(None)
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
_RUSAGE_INFO_V2 = 2
_PHYS_FOOTPRINT_OFFSET = 72  # struct rusage_info_v2: uuid[16] then 7 uint64 before ri_phys_footprint


class HeadroomError(RuntimeError):
    """The Mac stayed too short of memory to start a model run."""


class ModelError(RuntimeError):
    """A model child died, was stopped by the watchdog, or reported a failed job."""


class _HostGone(ModelError):
    """The model host closed the connection without answering."""


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


def socket_path() -> Path:
    """The model host's socket; its lock and log sit beside it. One host serves every $TLV_HOME of
    this user, so the memory it holds is bounded machine-wide. TLV_MODEL_HOST names another socket,
    for a host of its own, as the tests use."""
    path = Path(os.environ.get("TLV_MODEL_HOST") or Path.home() / ".local/share/tlv/run/models.sock")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class ModelProcess:
    """A model in the model host, answering JSON jobs one at a time.

    with ModelProcess("tlv.models.vlm", progress=print) as vlm:
        answer = vlm.call({"images": [...], "prompt": "..."})

    Entering loads the model unless the host already holds it; leaving lets the host keep it.
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
        self.env = {**{k: v for k, v in os.environ.items() if MODEL_VARIABLE.fullmatch(k)}, **(env or {})}
        self._conn: socket.socket | None = None
        self._lines: TextIO | None = None

    def __enter__(self) -> ModelProcess:
        try:
            try:
                self._load()
            except _HostGone:
                # A host exiting for idleness can drop a connection it had not yet served; a
                # second connection reaches the host that replaced it.
                self._load()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def call(self, job: dict[str, Any]) -> dict[str, Any]:
        return self._ask(job)

    def __exit__(self, *exc: object) -> None:
        lines, conn, self._lines, self._conn = self._lines, self._conn, None, None
        if lines:
            lines.close()
        if conn:
            conn.close()

    def _load(self) -> None:
        self.__exit__(None, None, None)
        self._conn = _connect()
        self._lines = self._conn.makefile("r", encoding="utf-8")
        self._ask(None)

    def _ask(self, job: dict[str, Any] | None) -> dict[str, Any]:
        if not self._conn or not self._lines:
            raise ModelError(f"{self.module} is not running")
        request = {"module": self.module, "cap": self.cap, "env": self.env, "job": job}
        try:
            self._conn.sendall((json.dumps(request) + "\n").encode())
            for line in self._lines:
                reply = json.loads(line)
                if "progress" in reply:
                    if self.progress:
                        self.progress(reply["progress"])
                elif "error" in reply:
                    raise (HeadroomError if reply.get("kind") == "headroom" else ModelError)(reply["error"])
                else:
                    return reply["result"]
        except (BrokenPipeError, ConnectionResetError):
            pass
        log = socket_path().with_name("models.log")
        raise _HostGone(f"the model host stopped before answering {self.module}; see {log}")


def _connect() -> socket.socket:
    """A connection to this $TLV_HOME's model host, starting the host when none is running."""
    path = str(socket_path())
    started = False
    deadline = time.monotonic() + HOST_START_S
    while True:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.connect(path)
            return conn
        except (FileNotFoundError, ConnectionRefusedError):
            conn.close()
        if not started:
            # The host runs in its own session, so it outlives this process and ignores its
            # terminal's Ctrl-C. Two callers starting hosts together is fine: the second host
            # finds the first's lock and leaves.
            with socket_path().with_name("models.log").open("a") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "tlv.host"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            started = True
        elif time.monotonic() >= deadline:
            raise ModelError(f"the model host did not start; see {socket_path().with_name('models.log')}")
        time.sleep(0.05)


class Child:
    """One model loaded in a guarded child process (`python -m <module>`, which calls
    tlv.models.protocol.serve), answering JSON jobs one at a time. Its watchdog stops it past
    its cap, or when memory pressure stays critical while it holds real memory."""

    def __init__(self, module: str, cap: int, env: dict[str, str], progress: Progress | None = None) -> None:
        self.module = module
        self.cap = cap
        self.env = env
        self.progress = progress
        self._child: subprocess.Popen[str] | None = None
        self._stopped: str | None = None
        self._stderr: deque[str] = deque(maxlen=20)

    @property
    def pid(self) -> int:
        return self._require_child().pid

    def alive(self) -> bool:
        return self._child is not None and self._child.poll() is None

    def start(self) -> None:
        name = self.module.rsplit(".", 1)[-1]
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
        try:
            ready = self._receive()
            if not ready.get("ready"):
                raise ModelError(f"{name} did not start: {ready}")
        except BaseException:
            self.close()
            raise

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

    def close(self) -> None:
        child = self._child
        if not child:
            return
        if child.stdin:
            child.stdin.close()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

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
        progress = self.progress
        if progress:
            progress(reason)
