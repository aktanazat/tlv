"""Adding videos and indexing them: each stage starts once the stages it stands on are done, and
is redone only when its version changes or a stage it stands on reran.

A stage replaces its own rows for the video in one transaction and is recorded in `stages` only
after it finishes, so an interrupted run leaves every stage either done or pending.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import queue
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

from tlv import entities, guard, media, store
from tlv.index import frames, shots, sounds, spans, speech, vision

# The stages, in an order that puts every stage after those it stands on.
MODULES: dict[str, ModuleType] = {
    "shots": shots,
    "frames": frames,
    "speech": speech,
    "vision": vision,
    "sounds": sounds,
    "spans": spans,
    "entities": entities,
}
STAGES = tuple(MODULES)
# The earlier stages each stage's rows are built from; when one reruns, these go stale.
INPUTS: dict[str, tuple[str, ...]] = {
    "shots": (),
    "frames": ("shots",),
    "speech": (),
    "vision": (),
    "sounds": (),
    "spans": ("shots", "frames", "speech", "vision", "sounds"),
    "entities": ("shots", "frames", "speech", "vision", "sounds", "spans"),
}
URL = re.compile(r"https?://", re.I)
# The tables with a row per video fact; deleting the video's row cascades to the rest.
VIDEO_TABLES = (
    "videos",
    "stages",
    "shots",
    "frames",
    "speech",
    "ocr",
    "detections",
    "sounds",
    "spans",
    "appearances",
    "observations",
)


class AddResult(TypedDict):
    video_id: str
    title: str
    duration: float
    status: str
    local_path: str
    already_indexed: bool


class RemoveResult(TypedDict):
    video_id: str
    title: str
    rows: dict[str, int]  # rows deleted per table
    people: dict[str, int]  # "removed": seen only in this video; "kept": also seen elsewhere
    files: list[str]  # files and folders deleted
    kept_for: list[str]  # other indexes holding the same video, whose files stay


def add(source: str, title: str | None = None, wait: bool = True, progress: guard.Progress | None = None) -> AddResult:
    """Downloads a URL or copies a file into media/, probes it and records it as a video, then
    indexes it when `wait`. The same content added again, from anywhere, is the same video."""
    db = store.connect()
    local, found_title = _fetch(db, source)
    video_id = _video_id(local)
    already = db.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,)).fetchone() is not None
    if not already:
        info = media.probe(local)
        db.execute(
            "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued')",
            (
                video_id,
                title or found_title,
                source,
                str(local),
                info.duration,
                info.width,
                info.height,
                info.fps,
                store.now(),
            ),
        )
    if wait:
        index(video_id, progress=progress)
    video = _video(db, video_id)
    return AddResult(
        video_id=video_id,
        title=video["title"],
        duration=video["duration"],
        status=video["status"],
        local_path=video["local_path"],
        already_indexed=already,
    )


def index(
    video_id: str, only: str | Iterable[str] | None = None, force: bool = False, progress: guard.Progress | None = None
) -> None:
    """Runs the stages (all, or those named in `only`), skipping any already current unless
    `force`; a stage built on one that reruns reruns too. Its model jobs run in the background:
    the model host runs any job someone is waiting on, such as a search's, first."""
    wanted = _wanted(only)
    db = store.connect()

    def say(message: str) -> None:
        if progress:
            progress(message)

    with (
        guard.background(),
        store.lock(f"index-{video_id}", waiting=lambda: say(f"waiting for another index run of {video_id}")),
    ):
        video = _video(db, video_id)
        missing = {need for name in wanted for need in INPUTS[name] if need not in wanted}
        missing = sorted(need for need in missing if not _current(db, video_id, need))
        if missing:
            raise ValueError(f"Run {', '.join(missing)} first: the stages asked for are built on them")
        todo: list[str] = []
        for name in STAGES:
            if name in wanted and (force or not _current(db, video_id, name) or set(INPUTS[name]) & set(todo)):
                todo.append(name)
        _set_status(db, video_id, "indexing")
        try:
            _run_stages(db, video, todo, progress, force)
        except Exception as error:
            _set_status(db, video_id, "failed", f"{type(error).__name__}: {error}")
            raise
        except BaseException:
            _set_status(db, video_id, "queued")
            raise
        _set_status(db, video_id, "ready" if all(_current(db, video_id, name) for name in STAGES) else "queued")


def status(video_id: str) -> dict[str, Any]:
    """The video, each stage's state ("done", "stale" or "pending"), version and seconds, and
    its row counts."""
    db = store.connect()
    video = _video(db, video_id)
    recorded = {row["stage"]: row for row in db.execute("SELECT * FROM stages WHERE video_id = ?", (video_id,))}
    stages = {}
    for name in STAGES:
        row = recorded.get(name)
        if row is None:
            stages[name] = {"state": "pending", "version": None, "seconds": None, "done_at": None}
            continue
        state = "done" if row["version"] == MODULES[name].VERSION else "stale"
        stages[name] = {
            "state": state,
            "version": row["version"],
            "seconds": round(row["seconds"], 1),
            "done_at": row["done_at"],
        }
    return {
        "video_id": video_id,
        "title": video["title"],
        "source": video["source"],
        "local_path": video["local_path"],
        "local_exists": Path(video["local_path"]).exists(),
        "duration": video["duration"],
        "status": video["status"],
        "error": video["error"],
        "stages": stages,
        "counts": _counts(db, video_id),
    }


def remove(video_id: str, progress: guard.Progress | None = None) -> RemoveResult:
    """Removes a video from the index. One transaction deletes every row about it and the people
    seen only in it; people also seen in other videos keep those appearances and the owner's
    corrections (tlv.entities.leaving). Then its files under $TLV_HOME go: the media copy, unless
    an index lists it for another video, and its frames, crops and work folders, unless another
    index holds the same video (they are named by its content, so that index uses them too).
    Adding the same file again indexes it anew."""
    db = store.connect()

    def say(message: str) -> None:
        if progress:
            progress(message)

    with store.lock(f"index-{video_id}", waiting=lambda: say(f"waiting for the index run of {video_id} to end")):
        video = _video(db, video_id)
        rows = {
            table: db.execute(f"SELECT count(*) FROM {table} WHERE video_id = ?", (video_id,)).fetchone()[0]
            for table in VIDEO_TABLES
        }
        people = _people(db, video_id)
        kept_for, media_used = _held_elsewhere(db, video)
        with entities.leaving(db, video_id, waiting=lambda: say("waiting for another entities run")):
            _drop_replays(db, video_id)
            db.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        files = _files(video, kept_for, media_used)
        for path in files:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return RemoveResult(
        video_id=video_id,
        title=video["title"],
        rows=rows,
        people=people,
        files=[str(path) for path in files],
        kept_for=kept_for,
    )


def _people(db: sqlite3.Connection, video_id: str) -> dict[str, int]:
    """How many people the video's appearances belong to: seen only in it, or also elsewhere."""
    seen_in: dict[str, set[str | None]] = {}
    for row in db.execute(
        "SELECT m.entity_id, a.video_id FROM entity_members m JOIN appearances a USING (appearance_id)"
    ):
        seen_in.setdefault(row["entity_id"], set()).add(row["video_id"])
    theirs = [videos for videos in seen_in.values() if video_id in videos]
    only = sum(1 for videos in theirs if videos == {video_id})
    return {"removed": only, "kept": len(theirs) - only}


def _held_elsewhere(db: sqlite3.Connection, video: sqlite3.Row) -> tuple[list[str], bool]:
    """The other indexes holding the same video, and whether any index lists its media file for
    another video."""
    video_id, local = video["video_id"], video["local_path"]
    used = db.execute("SELECT 1 FROM videos WHERE local_path = ? AND video_id != ?", (local, video_id)).fetchone()
    holding = []
    for name in store.indexes():
        if name == store.index_name():
            continue
        with contextlib.closing(store.connect(name)) as other:
            if other.execute("SELECT 1 FROM videos WHERE video_id = ?", (video_id,)).fetchone():
                holding.append(name)
            used = used or other.execute("SELECT 1 FROM videos WHERE local_path = ?", (local,)).fetchone()
    return holding, used is not None


def _files(video: sqlite3.Row, kept_for: list[str], media_used: bool) -> list[Path]:
    """The video's files and folders under $TLV_HOME that no other index or video uses."""
    home, video_id = store.home(), video["video_id"]
    local = Path(video["local_path"])
    found = [local] if local.is_relative_to(home / "media") and not media_used else []
    if not kept_for:
        found += [home / "frames" / video_id, home / "crops" / video_id, home / "work" / video_id]
        found.append(home / "work" / "entities" / video_id)
    return [path for path in found if path.exists()]


def _drop_replays(db: sqlite3.Connection, video_id: str) -> None:
    """Takes the video out of the replays other videos' verified observations list."""
    changes = []
    for row in db.execute(
        "SELECT obs_id, evidence FROM observations WHERE kind = 'verified' AND video_id != ?", (video_id,)
    ):
        evidence = json.loads(row["evidence"])
        replays = [replay for replay in evidence.get("replays", []) if replay["video_id"] != video_id]
        if replays != evidence.get("replays", []):
            changes.append((json.dumps({**evidence, "replays": replays}, ensure_ascii=False), row["obs_id"]))
    db.executemany("UPDATE observations SET evidence = ? WHERE obs_id = ?", changes)


def _counts(db: sqlite3.Connection, video_id: str) -> dict[str, int]:
    queries = {
        "shots": "SELECT count(*) FROM shots WHERE video_id = ?",
        "frames": "SELECT count(*) FROM frames WHERE video_id = ?",
        "frames_embedded": "SELECT count(*) FROM frames WHERE video_id = ? AND embedding IS NOT NULL",
        "speech": "SELECT count(*) FROM speech WHERE video_id = ?",
        "words": "SELECT coalesce(sum(json_array_length(words)), 0) FROM speech WHERE video_id = ?",
        "ocr": "SELECT count(*) FROM ocr WHERE video_id = ?",
        "detections": "SELECT count(*) FROM detections WHERE video_id = ?",
        "sounds": "SELECT count(*) FROM sounds WHERE video_id = ?",
        "spans": "SELECT count(*) FROM spans WHERE video_id = ?",
        "spans_captioned": "SELECT count(*) FROM spans WHERE video_id = ? AND caption IS NOT NULL",
        "observations": "SELECT count(*) FROM observations WHERE video_id = ?",
    }
    return {name: db.execute(query, (video_id,)).fetchone()[0] for name, query in queries.items()}


def _fetch(db: sqlite3.Connection, source: str) -> tuple[Path, str]:
    """The local media file for a source, and a title for it."""
    known = db.execute("SELECT local_path, title FROM videos WHERE source = ?", (source,)).fetchone()
    if known and Path(known["local_path"]).exists():
        return Path(known["local_path"]), known["title"]
    if URL.match(source):
        download = media.download(source)
        return download.path, download.title
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No video file at {path}")
    folder = store.data_dir("media")
    if path.parent == folder:
        return path, path.stem
    copy = folder / f"{media.slug(path.stem)}-{_video_id(path)[2:]}{path.suffix.lower()}"
    if not copy.exists():
        media.write_atomically(copy, lambda partial: shutil.copyfile(path, partial))
    return copy, path.stem


def _video_id(path: Path) -> str:
    """Names a video by its content, so a file added twice, from anywhere, is one video."""
    with path.open("rb") as handle:
        return "v_" + hashlib.file_digest(handle, "sha1").hexdigest()[:10]


def _video(db: sqlite3.Connection, video_id: str) -> sqlite3.Row:
    video = db.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if video is None:
        raise LookupError(f"No video with id {video_id}")
    return video


def _wanted(only: str | Iterable[str] | None) -> set[str]:
    if only is None:
        return set(STAGES)
    wanted = {only} if isinstance(only, str) else set(only)
    unknown = wanted - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stage {', '.join(sorted(unknown))}; the stages are {', '.join(STAGES)}")
    return wanted


def _current(db: sqlite3.Connection, video_id: str, name: str) -> bool:
    row = db.execute("SELECT version FROM stages WHERE video_id = ? AND stage = ?", (video_id, name)).fetchone()
    return row is not None and row["version"] == MODULES[name].VERSION


def _run_stages(
    db: sqlite3.Connection, video: sqlite3.Row, todo: list[str], progress: guard.Progress | None, force: bool
) -> None:
    """Runs each stage of `todo` in a thread of its own as soon as the stages it stands on have
    finished, so stages that use different parts of the Mac overlap: ffmpeg on the CPU, Apple
    Vision on the Neural Engine, the models on the GPU (the model host runs one job at a time).
    A stage is recorded as it finishes. After a failure no stage starts; those running finish
    and are recorded, and the first error is raised. The threads are daemons, so an interrupt
    ends the run at once, as it did when one stage ran in the foreground. Each runs in a copy of
    the caller's context, so its model jobs stay in the background."""
    finished: queue.SimpleQueue[tuple[str, float, BaseException | None]] = queue.SimpleQueue()

    def run(name: str) -> None:
        began = time.monotonic()
        try:
            with contextlib.closing(store.connect()) as own:
                _run(name, own, video, progress, force)
        except BaseException as error:
            finished.put((name, time.monotonic() - began, error))
        else:
            finished.put((name, time.monotonic() - began, None))

    waiting, running = list(todo), set()
    failure: BaseException | None = None
    while running or (waiting and failure is None):
        if failure is None:
            for name in [name for name in waiting if not set(INPUTS[name]) & {*waiting, *running}]:
                waiting.remove(name)
                running.add(name)
                if progress:
                    progress(f"{name}: running")
                threading.Thread(
                    target=contextvars.copy_context().run, args=(run, name), name=f"tlv-index-{name}", daemon=True
                ).start()
        name, seconds, error = finished.get()
        running.remove(name)
        if error is not None:
            failure = failure or error
            continue
        _record(db, video["video_id"], name, seconds)
        if progress:
            progress(f"{name}: done in {seconds:.1f} s")
    if failure is not None:
        raise failure


def _run(name: str, db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None, force: bool) -> None:
    if name == "entities":
        entities.index_video(db, video["video_id"], progress)
    elif name == "spans":
        spans.run(db, video, progress, recaption=force)
    else:
        MODULES[name].run(db, video, progress)


def _record(db: sqlite3.Connection, video_id: str, name: str, seconds: float) -> None:
    """Marks a stage done at its current version, and forgets every stage built on it, so they rerun."""
    built_on = [stage for stage in STAGES if name in INPUTS[stage]]
    with store.transaction(db):
        db.execute(
            "INSERT OR REPLACE INTO stages (video_id, stage, version, seconds, done_at) VALUES (?, ?, ?, ?, ?)",
            (video_id, name, MODULES[name].VERSION, seconds, store.now()),
        )
        db.executemany("DELETE FROM stages WHERE video_id = ? AND stage = ?", [(video_id, stage) for stage in built_on])


def _set_status(db: sqlite3.Connection, video_id: str, state: str, error: str | None = None) -> None:
    db.execute("UPDATE videos SET status = ?, error = ? WHERE video_id = ?", (state, error, video_id))
