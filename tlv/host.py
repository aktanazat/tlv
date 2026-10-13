"""The model host: one background process per user that keeps models loaded between runs.

tlv.guard.ModelProcess reaches it over a Unix socket (~/.local/share/tlv/run/models.sock, or
TLV_MODEL_HOST) and starts it when none is running; it logs to models.log beside the socket, so
every $TLV_HOME shares one host and one memory budget. The host

1. runs one job at a time across every tlv process and agent, loading a model in a guarded child
   (tlv.guard.Child) the first time it is asked for, and tells a caller who has to wait;
2. keeps a model loaded until it has gone unused for TLV_MODEL_IDLE_S (default 600 s);
3. holds at most TLV_HELD_MEMORY_GB (default 8) in the models it keeps: after every job it
   unloads the least recently used ones until the rest fit;
4. backs off when the Mac is short of memory. Every model no caller has open is unloaded when
   ~/memory/bin/host-headroom (TLV_HEADROOM_CMD), read every TLV_PRESSURE_CHECK_S (default 30 s)
   while such a model is loaded, puts the share of free memory under TLV_HOLD_MIN_FREE_PCT
   (default 25); when it cannot be read, the kernel's pressure level decides. Before loading a
   model the host unloads the others rather than wait for memory it holds itself, and the
   guard's watchdog still stops any model, open or not, when pressure turns critical;
5. exits once it has held nothing and been asked nothing for TLV_MODEL_IDLE_S.

A request is one JSON line, {"module", "cap", "env", "job"}, where a null job only loads the
model. The replies are any number of {"progress": text}, then {"result": {...}} or
{"error": text, "kind": "model" | "headroom"}. A connection is one caller's session with one
model: the model counts as open until the connection closes.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tlv import guard

IDLE_S = float(os.environ.get("TLV_MODEL_IDLE_S", "600"))
HELD_BYTES = int(float(os.environ.get("TLV_HELD_MEMORY_GB", "8")) * guard.GB)
PRESSURE_CHECK_S = float(os.environ.get("TLV_PRESSURE_CHECK_S", "30"))
MIN_FREE_PCT = float(os.environ.get("TLV_HOLD_MIN_FREE_PCT", "25"))
HEADROOM_CMD = os.environ.get("TLV_HEADROOM_CMD") or str(Path.home() / "memory/bin/host-headroom")
HEADROOM_TIMEOUT_S = 20.0
TICK_S = 0.5
LOCK_WAIT_S = 5.0  # a host on its way out releases its lock well within this

Key = tuple[str, int, tuple[tuple[str, str], ...]]  # module, cap, the caller's model variables


@dataclass
class Resident:
    child: guard.Child
    used: float  # time.monotonic() when it last loaded or finished a job


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", file=sys.stderr, flush=True)


def _name(key: Key) -> str:
    return key[0].rsplit(".", 1)[-1]


def _send(conn: socket.socket, message: dict[str, Any]) -> None:
    conn.sendall((json.dumps(message) + "\n").encode())


class Host:
    def __init__(self) -> None:
        self.jobs = threading.Lock()  # held for every load, job and unload
        self.residents: dict[Key, Resident] = {}  # changed only under `jobs`
        self.state = threading.Lock()  # guards the counts below
        self.clients = 0
        self.open: Counter[Key] = Counter()  # connections holding each model
        self.asked = time.monotonic()
        self.next_check = 0.0
        self.headroom_unreadable = False

    def accepted(self) -> None:
        with self.state:
            self.clients += 1
            self.asked = time.monotonic()

    def serve(self, conn: socket.socket) -> None:
        """Answers one connection's requests in order until it closes."""
        held: Key | None = None

        def say(message: str) -> None:
            with contextlib.suppress(OSError):  # the caller left; the job still finishes
                _send(conn, {"progress": message})

        try:
            with conn, conn.makefile("r", encoding="utf-8") as lines:
                for line in lines:
                    request = json.loads(line)
                    key: Key = (request["module"], int(request["cap"]), tuple(sorted(request["env"].items())))
                    if held is None:
                        held = key
                        with self.state:
                            self.open[key] += 1
                    _send(conn, self.answer(key, request["job"], say))
        except OSError:
            pass  # the caller went away
        finally:
            with self.state:
                self.clients -= 1
                if held is not None:
                    self.open[held] -= 1

    def answer(self, key: Key, job: dict[str, Any] | None, say: guard.Progress) -> dict[str, Any]:
        try:
            return {"result": self.run(key, job, say)}
        except guard.HeadroomError as error:
            return {"error": str(error), "kind": "headroom"}
        except guard.ModelError as error:
            return {"error": str(error), "kind": "model"}
        except Exception as error:  # the host's boundary: a bug fails this request, not the host
            log(traceback.format_exc())
            return {"error": f"the model host failed: {type(error).__name__}: {error}", "kind": "model"}

    def run(self, key: Key, job: dict[str, Any] | None, say: guard.Progress) -> dict[str, Any]:
        if not self.jobs.acquire(blocking=False):
            say(f"waiting for another model run to finish before {_name(key)}")
            self.jobs.acquire()
        try:
            with self.state:
                self.asked = time.monotonic()
            resident = self._load(key, say)
            if job is None:
                return {}
            resident.child.progress = say
            try:
                return resident.child.call(job)
            finally:
                resident.child.progress = None
                resident.used = time.monotonic()
                if not resident.child.alive():
                    del self.residents[key]
                self._fit(key)
        finally:
            self.jobs.release()

    def tick(self) -> bool:
        """Lets go of models that are idle, stopped, or held while memory is short. True once the
        host has held nothing and been asked nothing for IDLE_S."""
        if not self.jobs.acquire(blocking=False):
            return False  # a job is running
        try:
            now = time.monotonic()
            with self.state:
                open_keys = {key for key, n in self.open.items() if n > 0}
                clients, asked = self.clients, self.asked
            for key, resident in list(self.residents.items()):
                if not resident.child.alive():
                    del self.residents[key]
                    log(f"dropped {_name(key)}: its process had stopped")
                elif key not in open_keys and now - resident.used >= IDLE_S:
                    self._unload(key, f"unused for {IDLE_S:g} s")
            closed = [key for key in self.residents if key not in open_keys]
            short = self._short(now) if closed else None
            for key in closed if short else ():
                self._unload(key, short)
            return not self.residents and clients == 0 and now - asked >= IDLE_S
        finally:
            self.jobs.release()

    def _load(self, key: Key, say: guard.Progress) -> Resident:
        resident = self.residents.get(key)
        if resident and resident.child.alive():
            return resident
        self.residents.pop(key, None)
        module, cap, env = key
        while self.residents and (reason := guard.headroom(cap)):
            self._unload(self._least_recent(), f"loading {_name(key)} needs memory, and {reason}")
        guard.wait_for_headroom(cap, say)
        child = guard.Child(module, cap, dict(env), say)
        began = time.monotonic()
        child.start()
        child.progress = None
        log(f"loaded {_name(key)} in {time.monotonic() - began:.1f} s, {guard.footprint(child.pid) / guard.GB:.1f} GB")
        resident = self.residents[key] = Resident(child, time.monotonic())
        return resident

    def _fit(self, keep: Key) -> None:
        """Unloads models, least recently used first and `keep` last, until the rest fit HELD_BYTES."""
        for key in sorted(self.residents, key=lambda k: (k == keep, self.residents[k].used)):
            held = sum(guard.footprint(r.child.pid) for r in self.residents.values())
            if held <= HELD_BYTES:
                return
            self._unload(key, f"the models held {held / guard.GB:.1f} GB, over the {HELD_BYTES / guard.GB:g} GB cap")

    def _least_recent(self) -> Key:
        return min(self.residents, key=lambda k: self.residents[k].used)

    def _unload(self, key: Key, why: str) -> None:
        resident = self.residents.pop(key)
        held = guard.footprint(resident.child.pid)
        resident.child.close()
        log(f"unloaded {_name(key)} ({held / guard.GB:.1f} GB): {why}")

    def _short(self, now: float) -> str | None:
        """Why the Mac is too short of memory to keep idle models, or None."""
        if now < self.next_check:
            return None
        self.next_check = now + PRESSURE_CHECK_S
        try:
            out = subprocess.run(
                [HEADROOM_CMD, "--json", "--sample", "0"],
                capture_output=True,
                text=True,
                check=True,
                timeout=HEADROOM_TIMEOUT_S,
            ).stdout
            free = float(json.loads(out)["host"]["pressure_free_pct"])
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
            if not self.headroom_unreadable:
                log(f"cannot read {HEADROOM_CMD} ({type(error).__name__}: {error}); using kernel pressure instead")
                self.headroom_unreadable = True
            return "the kernel reports memory pressure" if guard.pressure_level() >= guard.WARN else None
        if free < MIN_FREE_PCT:
            return f"host-headroom puts free memory at {free:g}%, under {MIN_FREE_PCT:g}%"
        return None


def _take(handle: Any) -> bool:
    deadline = time.monotonic() + LOCK_WAIT_S
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


def main() -> None:
    with guard.socket_path().with_name("models.lock").open("a") as lock:
        if not _take(lock):
            return  # another host serves this socket
        path = guard.socket_path()
        path.unlink(missing_ok=True)
        host = Host()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            server.listen(16)
            server.settimeout(TICK_S)
            log(f"model host {os.getpid()} serving {path}")
            while True:
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    pass
                else:
                    conn.settimeout(None)
                    host.accepted()
                    threading.Thread(target=host.serve, args=(conn,), daemon=True).start()
                if host.tick():
                    break
            path.unlink(missing_ok=True)
        log(f"model host {os.getpid()} exiting: nothing loaded or asked for in {IDLE_S:g} s")


if __name__ == "__main__":
    main()
