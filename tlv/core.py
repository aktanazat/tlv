"""The tlv operations behind the CLI and the MCP server. Each operation takes the index name
first and returns a JSON-ready dict; its *_text function renders that dict as a few short lines.

list, status, cut, frames and timeline read the store and cut media here. The rest are thin
calls into the indexer (tlv.index.pipeline), the people module (tlv.entities) and the agent
(tlv.agent). Those are imported where used, so quick commands start without loading them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path
from typing import Any, Literal

from tlv import media, store
from tlv.media import Range
from tlv.timecode import format_range, format_timecode

Progress = Callable[[str], None] | None
Frame = Literal["mid", "sheet", "none"]

FRAMES: tuple[Frame, ...] = ("mid", "sheet", "none")
MODALITIES = ("visual", "audio", "transcription")
SHEET_FRAMES = 6
STILL_WIDTH = 960
INDEX_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def default_index() -> str:
    return os.environ.get("TLV_INDEX") or "tlv"


def _r(seconds: float) -> float:
    return round(seconds, 2)


def _is_url(text: str) -> bool:
    return re.match(r"https?://", text, re.I) is not None


def _checked(index: str) -> str:
    if not INDEX_NAME.fullmatch(index):
        raise ValueError(f'"{index}" is not an index name: use letters, digits, ".", "_" and "-"')
    return index


class _IndexTurns:
    """The indexer and the agent open the database with store.connect(), which reads $TLV_INDEX,
    so the index is chosen process-wide: calls on one index run together, and a call on another
    index waits until they finish."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._name = ""
        self._users = 0
        self._saved: str | None = None

    @contextmanager
    def use(self, index: str) -> Iterator[None]:
        with self._changed:
            self._changed.wait_for(lambda: self._users == 0 or self._name == index)
            if self._users == 0:
                self._name, self._saved = index, os.environ.get("TLV_INDEX")
                os.environ["TLV_INDEX"] = index
            self._users += 1
        try:
            yield
        finally:
            with self._changed:
                self._users -= 1
                if self._users == 0:
                    if self._saved is None:
                        os.environ.pop("TLV_INDEX", None)
                    else:
                        os.environ["TLV_INDEX"] = self._saved
                    self._changed.notify_all()


_turns = _IndexTurns()


def _selected(index: str) -> AbstractContextManager[None]:
    """Makes `index` the database the indexer and the agent open, for the duration of a call."""
    return _turns.use(_checked(index))


@contextmanager
def _db(index: str) -> Iterator[sqlite3.Connection]:
    with closing(store.connect(_checked(index))) as db:
        yield db


def _video(db: sqlite3.Connection, video_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if row is None:
        raise LookupError(f'No video "{video_id}" in this index. video_list (tlv list) shows the ids.')
    return row


def _local_file(video: sqlite3.Row) -> Path:
    path = Path(video["local_path"])
    if not path.exists():
        raise FileNotFoundError(
            f'The local file of "{video["title"]}" ({path}) is missing. '
            f'Re-add its source to restore it: video_add source="{video["source"]}" (tlv add "{video["source"]}").'
        )
    return path


def _clamp(video: sqlite3.Row, start: float, end: float) -> Range:
    """The requested range inside the video; an empty range is an error."""
    duration = video["duration"]
    clamped = Range(max(0.0, start), min(end, duration) if duration > 0 else end)
    if not clamped.end > clamped.start:
        raise ValueError(
            f'Empty range {format_range(start, end)} for "{video["title"]}" ({format_timecode(duration)} long)'
        )
    return clamped


def _stem(video: sqlite3.Row, span: Range) -> str:
    """File name stem for media cut from a range of a video."""
    return f"{media.slug(video['title'])}-{video['video_id'][-6:]}_{span.start:.1f}-{span.end:.1f}"


def _span(span: Range) -> dict[str, Any]:
    return {"start": _r(span.start), "end": _r(span.end), "timecode": format_range(span.start, span.end)}


# video_add and video_index


def video_add(
    index: str, source: str, title: str | None = None, wait: bool = True, progress: Progress = None
) -> dict[str, Any]:
    """Adds a video and indexes it. Without `wait` it returns once the video is recorded and a
    detached `tlv index` process does the indexing, logging to `index_log`."""
    from tlv.index import pipeline

    with _selected(index):
        result = pipeline.add(source, title=title, wait=wait, progress=progress)
    if wait or result["already_indexed"]:
        return dict(result)
    return {**result, "index_log": str(_index_detached(index, result["video_id"]))}


def _index_detached(index: str, video_id: str) -> Path:
    """Starts `tlv index` for the video in its own session, so it outlives this call."""
    log = store.data_dir("work") / "logs" / f"{index}-{video_id}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("ab") as out:
        subprocess.Popen(
            [sys.executable, "-m", "tlv.cli", "index", video_id, "--index", index],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return log


def add_text(r: dict[str, Any]) -> str:
    lines = [
        f'{"already indexed" if r["already_indexed"] else "added"}: "{r["title"]}" '
        f"({format_timecode(r['duration'])}) video_id={r['video_id']} status={r['status']}",
        f"local: {r['local_path']}",
    ]
    if "index_log" in r:
        lines.append(f"indexing in the background (tlv status {r['video_id']}); log: {r['index_log']}")
    return "\n".join(lines)


def video_index(
    index: str, video_id: str, stages: list[str] | None = None, force: bool = False, progress: Progress = None
) -> dict[str, Any]:
    from tlv.index import pipeline

    with _selected(index):
        pipeline.index(video_id, only=stages, force=force, progress=progress)
    return video_status(index, video_id)


# video_list and video_status


def video_list(index: str) -> dict[str, Any]:
    with _db(index) as db:
        rows = db.execute("SELECT video_id, title, duration, status, local_path FROM videos ORDER BY added_at")
        videos = [
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "duration": _r(row["duration"]),
                "status": row["status"],
                "local_path": row["local_path"],
                "local_exists": Path(row["local_path"]).exists(),
            }
            for row in rows
        ]
    return {"index": index, "videos": videos}


def list_text(r: dict[str, Any]) -> str:
    if not r["videos"]:
        return f'no videos in index "{r["index"]}"'
    return "\n".join(
        f'{v["video_id"]}  {v["status"]}  {format_timecode(v["duration"])}  "{v["title"]}"'
        + ("" if v["local_exists"] else "  (local file missing)")
        for v in r["videos"]
    )


def video_status(index: str, video_id: str) -> dict[str, Any]:
    """The video, each indexing stage's state, version and seconds, and its row counts."""
    from tlv.index import pipeline

    with _selected(index):
        return pipeline.status(video_id)


def status_text(r: dict[str, Any]) -> str:
    def stage(name: str, s: dict[str, Any]) -> str:
        return f"{name} {s['seconds']:.1f} s" if s["state"] == "done" else f"{name} ({s['state']})"

    lines = [f'"{r["title"]}" video_id={r["video_id"]} index: {r["status"]} ({format_timecode(r["duration"])})']
    if r["error"]:
        lines.append(f"error: {r['error']}")
    lines.append(f"local: {r['local_path']}" + ("" if r["local_exists"] else " (missing)"))
    lines.append("stages: " + ", ".join(stage(name, s) for name, s in r["stages"].items()))
    lines.append("rows: " + ", ".join(f"{table} {n}" for table, n in r["counts"].items()))
    return "\n".join(lines)


# video_search, video_ask and video_find


def video_search(
    index: str,
    query: str,
    then: list[str] = (),
    modalities: list[str] | None = None,
    max_gap: float = 12,
    limit: int = 10,
    video_ids: list[str] | None = None,
    clock: str | None = None,
) -> dict[str, Any]:
    """`modalities` limits the channels searched: visual, audio, transcription (default all).
    `clock` limits the search to where the scoreboard shows that game time ("4th", "last 2 min")."""
    from tlv import agent

    shown = _clock_within(index, video_ids, clock) if clock else None
    with _selected(index):
        result = agent.search(
            query,
            then=list(then),
            limit=limit,
            video_ids=video_ids,
            max_gap=max_gap,
            modalities=modalities,
            within=shown["ranges"] if shown else None,
        )
    return {**result, "clock": shown} if shown else result


def search_text(r: dict[str, Any]) -> str:
    asked = ", then ".join([r["query"], *r["then"]])
    if not r["results"]:
        return "\n".join([*_clock_lines(r), f'no results for "{asked}"'])
    lines = [*_clock_lines(r), f'{len(r["results"])} results for "{asked}":']
    for i, hit in enumerate(r["results"], 1):
        lines.append(
            f'{i}. {hit["timecode"]}  "{hit["title"]}"  video_id={hit["video_id"]}  {hit["kind"]} score {hit["score"]}'
        )
        lines.extend(f"   {step['timecode']}  {step['step']} (rank {step['rank']})" for step in hit.get("steps", ()))
    return "\n".join(lines)


def video_ask(
    index: str,
    video_id: str,
    question: str,
    start: float | None = None,
    end: float | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from tlv import agent

    with _selected(index):
        return agent.ask(video_id, question, start=start, end=end, json_schema=json_schema)


def ask_text(r: dict[str, Any]) -> str:
    where = f'{r["timecode"]} of "{r["title"]}"' if r.get("timecode") else f'"{r["title"]}"'
    body = json.dumps(r["json"], indent=2, ensure_ascii=False) if "json" in r else r.get("answer", "")
    return f"{where}{' (answer cut off at the token limit)' if r['truncated'] else ''}:\n{body}"


def video_find(
    index: str,
    query: str,
    then: list[str] = (),
    max_candidates: int = 8,
    pad: float = 3,
    min_confidence: float = 0.6,
    cut: bool = True,
    video_ids: list[str] | None = None,
    clock: str | None = None,
    progress: Progress = None,
) -> dict[str, Any]:
    """`clock` checks only where the scoreboard shows that game time ("4th", "last 2 min"). The
    find is kept as the index's last, for video_more to follow up on."""
    request = {
        "query": query,
        "then": list(then),
        "max_candidates": max_candidates,
        "pad": pad,
        "min_confidence": min_confidence,
        "cut": cut,
        "video_ids": video_ids,
        "clock": clock,
        "no_replays": False,
    }
    result = _found(index, request, progress)
    _keep_find(index, request, result)
    return result


def find_text(r: dict[str, Any]) -> str:
    counts = r["counts"]
    lines = [
        *_clock_lines(r),
        f'confirmed {counts["confirmed"]} of {counts["candidates"]} candidates for "{r["claim"]}"',
    ]
    for o in r["confirmed"]:
        lines.append(
            f'{o["id"]}  {o["timecode"]}  "{o["title"]}"  video_id={o["video_id"]}  confidence {o["confidence"]}'
        )
        if o.get("who"):
            lines.append(f"   who: {o['who']}")
        evidence = o["evidence"]
        lines.append(f"   evidence: {evidence['summary']}")
        lines.extend(f'   said: {format_range(s["start"], s["end"])}  "{s["text"]}"' for s in evidence["transcript"])
        lines.extend(f"   on screen: {format_timecode(t['t'])}  {t['text']}" for t in evidence["ocr"])
        lines.extend(f"   heard: {format_range(s['start'], s['end'])}  {s['label']}" for s in evidence["sounds"])
        lines.extend(f"   frame: {format_timecode(f['t'])}  {f['path']}" for f in evidence["frames"])
        lines.extend(f"   {s['timecode']}  {s['step']}" for s in o["steps"])
        lines.extend(
            f"   replay: {p['timecode']}  video_id={p['video_id']}  {p['reason']}" for p in o.get("replays", ())
        )
        if o.get("clip_path"):
            lines.append(f"   clip: {o['clip_path']}")
        if o.get("frame_path"):
            tiles = ", ".join(format_timecode(t) for t in o.get("frame_times", ()))
            lines.append(f"   frames: {o['frame_path']} (tiles at {tiles})")
        if o.get("cut_error"):
            lines.append(f"   not cut: {o['cut_error']}")
    lines.extend(
        f"rejected {format_range(x['start'], x['end'])} video_id={x['video_id']}: {x['reason']}" for x in r["rejected"]
    )
    return "\n".join(lines)


# video_more, video_at, and the game clock windows of find and search


def _last_find(index: str) -> Path:
    folder = store.data_dir("work") / "sessions"
    folder.mkdir(exist_ok=True)
    return folder / f"find-{_checked(index)}.json"


def _keep_find(index: str, request: dict[str, Any], result: dict[str, Any]) -> None:
    """Saves a find as the index's last, what was asked beside what it found, for video_more."""
    session = json.dumps({"request": request, "result": result}, ensure_ascii=False)
    media.write_atomically(_last_find(index), lambda partial: partial.write_text(session))


def _clock_shown(index: str, video_ids: list[str] | None, clock: str) -> dict[str, Any]:
    """A window of game time and where the videos show it, as a result's "clock": the window, the
    stretches of each video inside it, and the videos whose scoreboard was never read."""
    from tlv.agent import clock as game_clock
    from tlv.agent import sources

    window = game_clock.Window.parse(clock)
    with _db(index) as db:
        coverage = game_clock.coverage(db, sources.videos(db, video_ids), window)
    ranges = {v: [(_r(start), _r(end)) for start, end in spans] for v, spans in coverage.ranges.items()}
    return {"window": str(window), "ranges": ranges, "unread": coverage.unread}


def _clock_within(index: str, video_ids: list[str] | None, clock: str) -> dict[str, Any]:
    """`_clock_shown` for a search or find to keep within; a window no video shows is an error,
    raised before anything is searched."""
    shown = _clock_shown(index, video_ids, clock)
    if not shown["ranges"]:
        unread = f"; {len(shown['unread'])} of the videos show no game clock" if shown["unread"] else ""
        raise LookupError(f"no video shows {shown['window']} on its scoreboard{unread}")
    return shown


def _clock_lines(r: dict[str, Any]) -> list[str]:
    if "clock" not in r:
        return []
    shown = r["clock"]
    seconds = sum(end - start for spans in shown["ranges"].values() for start, end in spans)
    videos = len(shown["ranges"])
    lines = [f"game clock {shown['window']}: {format_timecode(seconds)} of {videos} video{'s' * (videos != 1)}"]
    if shown["unread"]:
        lines.append(f"no game clock read in {', '.join(shown['unread'])}")
    return lines


def _replayed(db: sqlite3.Connection, clips: list[dict[str, Any]]) -> set[str]:
    """The ids of the clips the captioner calls mostly replay."""
    from tlv.agent import sources

    return {c["id"] for c in clips if sources.mostly_replay(db, c["video_id"], c["start"], c["end"])}


def _found(index: str, request: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Runs the find a request describes: within its game clock window, without the clips that
    are mostly replay when it says so."""
    from tlv import agent
    from tlv.agent import followup

    shown = _clock_within(index, request["video_ids"], request["clock"]) if request["clock"] else None
    with _selected(index):
        result = agent.find(
            request["query"],
            then=request["then"],
            max_candidates=request["max_candidates"],
            pad=request["pad"],
            min_confidence=request["min_confidence"],
            cut=request["cut"],
            video_ids=request["video_ids"],
            progress=progress,
            within=shown["ranges"] if shown else None,
        )
    if request["no_replays"]:
        with _db(index) as db:
            replayed = _replayed(db, result["confirmed"])
        result = followup.narrowed(result, followup.Narrow(no_replays=True), {}, replayed)
    return {**result, "clock": shown} if shown else result


def video_more(index: str, instruction: str, progress: Progress = None) -> dict[str, Any]:
    """Follows up on the index's last find, in words. "now only fourth quarter", "without the
    replays", "only the warriors game" or "drop obs_1a2b3c" keep some of the clips it confirmed,
    read from the index with no model and no new search; "same but steph" reruns it for someone
    else. Rules read those phrasings; the model reads other ways to name a person, a game time or
    a video. Each follow-up builds on the one before: the result is kept as the last find."""
    from tlv.agent import clock as game_clock
    from tlv.agent import followup, sources
    from tlv.agent.find import plain_texts
    from tlv.agent.model import Vlm

    began = time.perf_counter()
    path = _last_find(index)
    if not path.exists():
        raise LookupError(f'no find to follow up in index "{index}": run tlv find first')
    session = json.loads(path.read_text())
    request, result = session["request"], session["result"]
    with _db(index) as db:
        said, *steps = plain_texts(db, [instruction, request["query"], *request["then"]])
        names = sources.person_names(db)
        titles = {video.video_id: video.title for video in sources.videos(db)}
    change = followup.parse(said, steps, names, titles)
    read_by = "rules"
    if change is None:
        with Vlm(progress) as vlm:
            change = followup.with_model(vlm, said, steps, names, titles)
        read_by = "model"
    if change is None:
        raise ValueError(
            f'could not tell what to change from "{instruction}": say "now only the 4th quarter", "without the '
            'replays", "only the <title words> game", "drop <clip id>" or "same but <name>"'
        )
    narrow, clock, video_ids = change.narrow, request["clock"], request["video_ids"]
    if narrow.clock:
        clock = str(game_clock.Window.parse(clock).narrowed_to(narrow.clock) if clock else narrow.clock)
    if narrow.video_ids is not None:
        video_ids = sorted(narrow.video_ids & set(video_ids or narrow.video_ids))
        if not video_ids:
            raise ValueError(f"the last find did not search {', '.join(sorted(narrow.video_ids))}")
    no_replays = request["no_replays"] or narrow.no_replays
    request = {**request, "clock": clock, "video_ids": video_ids, "no_replays": no_replays}
    before = len(result["confirmed"])
    if change.swap:
        query, *then = followup.swapped(steps, change.swap)
        request = {**request, "query": query, "then": then}
        result = _found(index, request, progress)
    else:
        clips = result["confirmed"]
        shown = _clock_shown(index, sorted({c["video_id"] for c in clips}), clock) if narrow.clock else None
        with _db(index) as db:
            replayed = _replayed(db, clips) if narrow.no_replays else set()
        result = followup.narrowed(result, narrow, shown["ranges"] if shown else {}, replayed)
        result = {**result, "clock": shown} if shown else result
    _keep_find(index, request, result)
    done = {
        "instruction": instruction,
        "read_by": read_by,
        "reran": change.swap is not None,
        "query": ", then ".join([request["query"], *request["then"]]),
        "clock": request["clock"],
        "video_ids": request["video_ids"],
        "no_replays": request["no_replays"],
        "dropped": [] if change.swap else sorted(narrow.drop),
        "before": before,
        "ms": round((time.perf_counter() - began) * 1000, 1),
    }
    return {**result, "followup": done}


def more_text(r: dict[str, Any]) -> str:
    f = r["followup"]
    asked = [
        *([f"game clock {f['clock']}"] if f["clock"] else []),
        *([f"videos {', '.join(f['video_ids'])}"] if f["video_ids"] else []),
        *(["without replays"] if f["no_replays"] else []),
        *([f"without {', '.join(f['dropped'])}"] if f["dropped"] else []),
    ]
    took = f"{f['ms']:.0f} ms" if f["ms"] < 1000 else f"{f['ms'] / 1000:.1f} s"
    if f["reran"]:
        head = f'reran the find for "{f["query"]}"'
    else:
        head = f"kept {r['counts']['confirmed']} of {f['before']} clips"
    read = "no model" if f["read_by"] == "rules" and not f["reran"] else f"instruction read by {f['read_by']}"
    return f"{head} ({'; '.join([*asked, took, read])})\n{find_text(r)}"


AT_CLIP_S = 6.0  # seconds of the clip video_at cuts around each moment


def video_at(index: str, video_id: str, clock: str, cut: bool = False) -> dict[str, Any]:
    """Every stretch of a video whose scoreboard shows a game time ("4th 1:12") or a window of it
    ("Q4 2:00-1:00", "OT"): the live play and each replay that keeps the scoreboard on screen,
    each with a still of its frame nearest the asked time and, with `cut`, a clip around that
    frame. Reads the on-screen text in the index; no model runs."""
    from tlv.agent import clock as game_clock

    window = game_clock.Window.parse(clock)
    with _db(index) as db:
        video = _video(db, video_id)
        readings = game_clock.readings(db, video_id)
    hits = [
        {
            "t": _r(shown.best.t),
            "timecode": format_timecode(shown.best.t),
            "start": _r(shown.start),
            "end": _r(shown.end),
            "shows": game_clock.reading_text(shown.best),
            **_at_media(video, shown.best.t, cut),
        }
        for shown in game_clock.showings(readings, window)
    ]
    return {
        "video_id": video_id,
        "title": video["title"],
        "clock": str(window),
        "readings": len(readings),
        "hits": hits,
    }


def _at_media(video: sqlite3.Row, t: float, cut: bool) -> dict[str, str]:
    """The still at `t` and, with `cut`, an AT_CLIP_S clip around it."""
    source = _local_file(video)
    still = store.data_dir("frames") / f"{media.slug(video['title'])}-{video['video_id'][-6:]}_{t:.1f}_at.jpg"
    media.still(source, t, still, STILL_WIDTH)
    if not cut:
        return {"frame_path": str(still)}
    span = _clamp(video, t - AT_CLIP_S / 2, t + AT_CLIP_S / 2)
    clip = store.data_dir("clips") / f"{_stem(video, span)}.mp4"
    media.cut(source, span, clip)
    return {"frame_path": str(still), "clip_path": str(clip)}


def at_text(r: dict[str, Any]) -> str:
    if not r["readings"]:
        return f'no game clock read in "{r["title"]}"'
    if not r["hits"]:
        return f'no frame of "{r["title"]}" shows {r["clock"]} (the game clock is read in {r["readings"]} frames)'
    count = len(r["hits"])
    lines = [f'"{r["title"]}" shows {r["clock"]} in {count} stretch{"es" * (count != 1)}:']
    for hit in r["hits"]:
        lines.append(f"{hit['timecode']}  {hit['shows']}  on screen {format_range(hit['start'], hit['end'])}")
        lines.append(f"   frame: {hit['frame_path']}")
        if "clip_path" in hit:
            lines.append(f"   clip: {hit['clip_path']}")
    return "\n".join(lines)


# video_cut, video_frames and video_timeline


def video_cut(index: str, video_id: str, start: float, end: float, frame: Frame = "mid") -> dict[str, Any]:
    """Cuts [start, end] from the local copy, with its middle frame or a contact sheet."""
    with _db(index) as db:
        video = _video(db, video_id)
    span = _clamp(video, start, end)
    source = _local_file(video)
    stem = _stem(video, span)
    clip_path = store.data_dir("clips") / f"{stem}.mp4"
    media.cut(source, span, clip_path)
    result = {"video_id": video_id, "title": video["title"], **_span(span), "clip_path": str(clip_path)}
    if frame == "sheet":
        sheet_path = store.data_dir("frames") / f"{stem}_sheet.jpg"
        times = media.contact_sheet(source, span, SHEET_FRAMES, sheet_path)
        return {**result, "frame_path": str(sheet_path), "frame_times": [_r(t) for t in times]}
    if frame == "mid":
        middle = (span.start + span.end) / 2
        still_path = store.data_dir("frames") / f"{stem}_mid.jpg"
        media.still(source, middle, still_path, STILL_WIDTH)
        return {**result, "frame_path": str(still_path), "frame_times": [_r(middle)]}
    return result


def cut_text(r: dict[str, Any]) -> str:
    lines = [f'{r["timecode"]} of "{r["title"]}"', f"clip: {r['clip_path']}"]
    if r.get("frame_path"):
        lines.append(f"frame: {r['frame_path']}")
    return "\n".join(lines)


def video_frames(index: str, video_id: str, start: float, end: float, count: int = SHEET_FRAMES) -> dict[str, Any]:
    """A contact sheet of `count` evenly spaced frames of [start, end], without cutting a clip."""
    if not 1 <= count <= 12:
        raise ValueError(f"count must be a whole number from 1 to 12, got {count}")
    with _db(index) as db:
        video = _video(db, video_id)
    span = _clamp(video, start, end)
    sheet_path = store.data_dir("frames") / f"{_stem(video, span)}_sheet{count}.jpg"
    times = media.contact_sheet(_local_file(video), span, count, sheet_path)
    return {
        "video_id": video_id,
        "title": video["title"],
        **_span(span),
        "sheet_path": str(sheet_path),
        "times": [_r(t) for t in times],
    }


def frames_text(r: dict[str, Any]) -> str:
    return "\n".join(
        [
            f'{r["timecode"]} of "{r["title"]}"',
            f"sheet: {r['sheet_path']}",
            f"tiles (left to right, top to bottom): {', '.join(format_timecode(t) for t in r['times'])}",
        ]
    )


def video_timeline(index: str, video_id: str, start: float, end: float) -> dict[str, Any]:
    """Everything indexed about [start, end] of a video: what was said, written on screen and
    heard, the captions of the spans it touches, and who is visible. The evidence behind a claim,
    read the way the agent reads it."""
    from tlv import entities
    from tlv.agent import sources

    with _db(index) as db:
        video = _video(db, video_id)
        span = _clamp(video, start, end)
        between = (video_id, span.start, span.end)
        sounds = db.execute(
            "SELECT start, end, label, conf FROM sounds WHERE video_id = ? AND end > ? AND start < ? ORDER BY start",
            between,
        )
        captions = db.execute(
            "SELECT span_id, start, end, caption, replay FROM spans WHERE video_id = ? AND end > ? AND start < ? "
            "AND caption IS NOT NULL ORDER BY start",
            between,
        )
        return {
            "video_id": video_id,
            "title": video["title"],
            **_span(span),
            "speech": [{"start": _r(s.start), "end": _r(s.end), "text": s.text} for s in sources.speech(db, *between)],
            "screen_text": [{"t": _r(line.start), "text": line.text} for line in sources.screen_text(db, *between)],
            "sounds": [{**dict(row), "start": _r(row["start"]), "end": _r(row["end"])} for row in sounds],
            "captions": [
                {
                    **dict(row),
                    "start": _r(row["start"]),
                    "end": _r(row["end"]),
                    "replay": None if row["replay"] is None else bool(row["replay"]),
                }
                for row in captions
            ],
            "people": entities.present(db, *between),
        }


def _person(entity: dict[str, Any]) -> str:
    return f"@{entity['name']}" if entity["name"] else f"(unnamed {entity['entity_id']})"


def timeline_text(r: dict[str, Any]) -> str:
    def section(title: str, items: list[str]) -> list[str]:
        return [f"{title}:", *(f"  {item}" for item in items)] if items else [f"{title}: none"]

    def at(item: dict[str, Any]) -> str:
        return format_range(item["start"], item["end"])

    def caption(c: dict[str, Any]) -> str:
        replay = {True: "  (replay)", False: "", None: ""}[c["replay"]]
        return f"{at(c)}  {c['caption']}{replay}"

    return "\n".join(
        [
            f'{r["timecode"]} of "{r["title"]}" video_id={r["video_id"]}',
            *section("speech", [f'{at(s)}  "{s["text"]}"' for s in r["speech"]]),
            *section("on screen", [f"{format_timecode(t['t'])}  {t['text']}" for t in r["screen_text"]]),
            *section("sounds", [f"{at(s)}  {s['label']} ({s['conf']:.2f})" for s in r["sounds"]]),
            *section("captions", [caption(c) for c in r["captions"]]),
            *section("people", [f"{at(p)}  {_person(p)}  {p['entity_id']}" for p in r["people"]]),
        ]
    )


# People


def _fetch_image(url: str) -> Path:
    """Downloads a reference image URL once into work/references/."""
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    folder = store.data_dir("work") / "references"
    folder.mkdir(exist_ok=True)
    path = folder / f"{hashlib.sha1(url.encode()).hexdigest()[:12]}{Path(name).suffix.lower() or '.jpg'}"
    if not path.exists():

        def fetch(partial: Path) -> None:
            request = urllib.request.Request(url, headers={"User-Agent": media.USER_AGENT})
            with urllib.request.urlopen(request) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)

        media.write_atomically(path, fetch)
    return path


def entity_add(
    index: str, name: str, images: list[str] = (), video_id: str | None = None, times: list[float] = ()
) -> dict[str, Any]:
    """Teaches a person from reference images (paths or URLs) or from frames of an indexed video."""
    from tlv import entities

    name = name.strip().removeprefix("@").strip()
    if not name:
        raise ValueError("An entity needs a name")
    if not images and not times:
        raise ValueError("Give reference images, or a video_id and times to grab frames from")
    files = [Path(image).expanduser().resolve() for image in images if not _is_url(image)]
    missing = [str(file) for file in files if not file.exists()]
    if missing:
        raise FileNotFoundError(f"Image not found: {', '.join(missing)}")
    files += [_fetch_image(image) for image in images if _is_url(image)]
    with _db(index) as db:
        return entities.add_reference(db, name, images=files, frames=[(video_id, t) for t in times])


def entity_add_text(r: dict[str, Any]) -> str:
    return (
        f"{'extended' if r['extended'] else 'created'} @{r['name']} ({r['entity_id']}) with {r['added_images']} "
        f"images; now {r['appearances']} appearances in {r['videos']} videos"
    )


def video_entities(index: str) -> dict[str, Any]:
    from tlv import entities

    with _db(index) as db:
        return {"index": index, "entities": entities.entities(db)}


def _entity_line(e: dict[str, Any]) -> str:
    return f"{_person(e)}  {e['entity_id']}  {e['appearances']} appearances in {e['videos']} videos  {e['crop_path']}"


def entities_text(r: dict[str, Any]) -> str:
    if not r["entities"]:
        return f'no people in index "{r["index"]}" yet'
    return "\n".join(_entity_line(e) for e in r["entities"])


def entity_name(index: str, target: str, name: str) -> dict[str, Any]:
    """Names the person an entity or appearance id belongs to; the name survives re-clustering."""
    from tlv import entities

    with _db(index) as db:
        return {"entities": entities.name(db, target, name.strip().removeprefix("@").strip())}


def entity_same(index: str, a: str, b: str) -> dict[str, Any]:
    """Records that two entity or appearance ids are one person, and merges them."""
    from tlv import entities

    with _db(index) as db:
        return {"entities": entities.same(db, a, b)}


def entity_different(index: str, a: str, b: str) -> dict[str, Any]:
    """Records that two entity or appearance ids are different people, and splits them."""
    from tlv import entities

    with _db(index) as db:
        return {"entities": entities.different(db, a, b)}


def corrected_text(r: dict[str, Any]) -> str:
    return "\n".join(_entity_line(e) for e in r["entities"])


def person(index: str, who: str, strip: bool = False, video_ids: list[str] | None = None) -> dict[str, Any]:
    """One person across the videos, with no reference photos: each stretch on screen with its
    clearest crop, seconds on screen per video, where the name is said or shown, and the moments a
    find verified about them. Reads the index only; no model runs. With `strip`, also a filmstrip
    still of one crop per stretch."""
    from tlv.entities import timeline

    with _db(index) as db:
        found = timeline.timeline(db, who, video_ids or ())
    involved = {item.video_id for item in (*found.presences, *found.mentions, *found.verified)}
    result: dict[str, Any] = {
        "entity_ids": list(found.entity_ids),
        "name": found.name,
        "seconds": _r(sum(found.seconds.values())),
        "videos": [
            {"video_id": v.video_id, "title": v.title, "seconds": _r(found.seconds.get(v.video_id, 0.0))}
            for v in found.videos
            if v.video_id in involved
        ],
        "presences": [
            {
                "video_id": p.video_id,
                **_span(Range(p.start, p.end)),
                "seconds": _r(p.seen),
                "best_t": _r(p.best_t),
                "crop_path": p.crop_path,
            }
            for p in found.presences
        ],
        "mentions": [
            {"video_id": m.video_id, "kind": m.kind, **_span(Range(m.start, m.end)), "text": m.text}
            for m in found.mentions
        ],
        "verified": [
            {
                "video_id": m.video_id,
                **_span(Range(m.start, m.end)),
                "claims": list(m.claims),
                "obs_ids": list(m.obs_ids),
            }
            for m in found.verified
        ],
    }
    if strip:
        out = store.data_dir("frames") / f"person-{found.entity_ids[0]}.jpg"
        result["strip_path"] = str(timeline.strip(found, out)) if found.presences else None
    return result


def person_text(r: dict[str, Any]) -> str:
    events: dict[str, list[tuple[float, str, str]]] = {}
    for p in r["presences"]:
        events.setdefault(p["video_id"], []).append((p["start"], p["timecode"], f"on screen {p['seconds']:.1f} s"))
    for m in r["mentions"]:
        said = m["kind"] == "speech"
        when = format_timecode(m["start"]) if said or m["start"] == m["end"] else m["timecode"]
        events.setdefault(m["video_id"], []).append((m["start"], when, f'{"said" if said else "shown"} "{m["text"]}"'))
    for v in r["verified"]:
        events.setdefault(v["video_id"], []).append((v["start"], v["timecode"], f"verified: {'; '.join(v['claims'])}"))
    shown = sum(1 for v in r["videos"] if v["seconds"])
    who = f"{r['name'] or 'unnamed person'} ({', '.join(r['entity_ids'])})"
    lines = [f"{who}: {r['seconds']:.1f} s on screen in {shown} video{'s' * (shown != 1)}"]
    for video in r["videos"]:
        lines += ["", f'"{video["title"]}" video_id={video["video_id"]}: {video["seconds"]:.1f} s on screen']
        lines += [f"  {when:<24}  {what}" for _, when, what in sorted(events[video["video_id"]], key=lambda e: e[0])]
    if "strip_path" in r:
        lines += ["", f"strip: {r['strip_path'] or 'none, not on screen in these videos'}"]
    return "\n".join(lines)


# video_jockey


def _session_path(index: str, session_id: str) -> Path:
    folder = store.data_dir("work") / "sessions"
    folder.mkdir(exist_ok=True)
    return folder / f"{_checked(index)}-{session_id}.json"


def video_jockey(
    index: str,
    question: str,
    session_id: str | None = None,
    video_ids: list[str] | None = None,
    progress: Progress = None,
) -> dict[str, Any]:
    """Answers a question about the whole collection. A session keeps the earlier questions and
    answers on disk, so a follow-up passing its session_id is answered with them in view."""
    from tlv import agent

    if session_id is None:
        session_id, history = uuid.uuid4().hex[:12], []
    else:
        path = _session_path(index, session_id) if re.fullmatch(r"[0-9a-f]{12}", session_id) else None
        if path is None or not path.exists():
            raise ValueError(f'No session "{session_id}" in index "{index}": pass the session_id of an earlier answer')
        history = json.loads(path.read_text())
    with _selected(index):
        result = agent.answer(question, video_ids=video_ids, progress=progress, history=history)
    history.append({"question": question, "text": result["text"], "citations": result["citations"]})
    path = _session_path(index, session_id)
    media.write_atomically(path, lambda partial: partial.write_text(json.dumps(history, ensure_ascii=False)))
    return {**result, "session_id": session_id}


def jockey_text(r: dict[str, Any]) -> str:
    lines = [r["text"]]
    if r.get("incomplete_reason"):
        lines.append(f"(answer incomplete: {r['incomplete_reason']})")
    lines.extend(f'{c["marker"]} {c["timecode"]}  "{c["title"]}"  video_id={c["video_id"]}' for c in r["citations"])
    lines.append(f"session: {r['session_id']}")
    return "\n".join(lines)


# video_reel

REEL_ORDERS = ("found", "game", "video")


def video_reel(
    index: str,
    query: str | None = None,
    then: list[str] = (),
    music: str | None = None,
    bpm: float | None = None,
    order: str = "found",
    max_clips: int = 12,
    mute_video: bool = False,
    out: str | None = None,
    from_file: str | None = None,
    max_candidates: int = 8,
    video_ids: list[str] | None = None,
    progress: Progress = None,
) -> dict[str, Any]:
    """Cuts the confirmed clips of a find into one mp4 on the beat of `music`, each clip opening with
    a lower third, with a sources file beside it. `from_file` cuts a saved find (the JSON of `tlv
    find --json`, or an eval results file with `query` naming its run) instead of running one."""
    from tlv import reel

    if bpm is not None and music is None:
        raise ValueError("bpm is the tempo of the music: pass music too")
    if bpm is not None and not 60 <= bpm <= 200:
        raise ValueError(f"bpm must be from 60 to 200, got {bpm:g}")
    if max_clips < 1:
        raise ValueError(f"max_clips must be at least 1, got {max_clips}")
    if music is not None and not Path(music).is_file():
        raise FileNotFoundError(f"No music file at {music}")
    if from_file is not None:
        if then:
            raise ValueError("then shapes a new find, not one saved in from_file")
        found = reel.saved_find(Path(from_file), query)
    elif query is not None:
        found = video_find(index, query, then, max_candidates, cut=False, video_ids=video_ids, progress=progress)
    else:
        raise ValueError("Name what to find, or pass a saved find as from_file")
    confirmed = [o for o in found["confirmed"] if video_ids is None or o["video_id"] in video_ids][:max_clips]
    if not confirmed:
        raise ValueError(f'nothing found to cut for "{found["claim"]}"')
    with _db(index) as db:
        rows = {o["video_id"]: _video(db, o["video_id"]) for o in confirmed}
        for row in rows.values():
            _local_file(row)
        return reel.make(
            db,
            found["claim"],
            confirmed,
            rows,
            Path(music) if music else None,
            Path(out) if out else None,
            order,
            bpm,
            mute_video,
            progress,
        )


def reel_text(r: dict[str, Any]) -> str:
    tempo = f" at {r['bpm']} bpm" if r["bpm"] else ""
    lines = [
        f'reel of "{r["claim"]}": {r["path"]}',
        f"{len(r['cuts'])} clips, {format_timecode(r['duration'])}{tempo}, rendered in {r['seconds']} s",
    ]
    lines.extend(
        f'{c["reel_timecode"]}  {c["timecode"]} of "{c["title"]}"  video_id={c["video_id"]}' for c in r["cuts"]
    )
    lines.append(f"sources: {r['sources_path']}")
    return "\n".join(lines)
