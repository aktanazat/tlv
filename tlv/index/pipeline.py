"""Adding videos and indexing them: stages run in order, and each is redone only when its version
changes or a stage it stands on reran.

A stage replaces its own rows for the video in one transaction and is recorded in `stages` only
after it finishes, so an interrupted run leaves every stage either done or pending.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

from tlv import entities, guard, media, store
from tlv.index import frames, shots, sounds, spans, speech, vision

# The stages, in the order they run.
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
    "vision": ("frames",),
    "sounds": (),
    "spans": ("shots", "frames", "speech", "vision", "sounds"),
    "entities": ("shots", "frames", "speech", "vision", "sounds", "spans"),
}
URL = re.compile(r"https?://", re.I)


class AddResult(TypedDict):
    video_id: str
    title: str
    duration: float
    status: str
    local_path: str
    already_indexed: bool


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
    """Runs the stages in order (all, or those named in `only`), skipping any already current
    unless `force`."""
    wanted = _wanted(only)
    db = store.connect()

    def say(message: str) -> None:
        if progress:
            progress(message)

    with store.lock(f"index-{video_id}", waiting=lambda: say(f"waiting for another index run of {video_id}")):
        video = _video(db, video_id)
        missing = {need for name in wanted for need in INPUTS[name] if need not in wanted}
        missing = sorted(need for need in missing if not _current(db, video_id, need))
        if missing:
            raise ValueError(f"Run {', '.join(missing)} first: the stages asked for are built on them")
        _set_status(db, video_id, "indexing")
        try:
            for name in STAGES:
                if name not in wanted or (not force and _current(db, video_id, name)):
                    continue
                say(f"{name}: running")
                began = time.monotonic()
                _run(name, db, video, progress, force)
                _record(db, video_id, name, time.monotonic() - began)
                say(f"{name}: done in {time.monotonic() - began:.1f} s")
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
