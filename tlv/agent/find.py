"""search and find. search ranks moments that may match a query; find checks each of the best
by looking at it, keeps one result per play with its replays listed under it, and records every
confirmed moment as a verified observation tied to its video, time range and evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from tlv import entities, media, store
from tlv.agent import plan, retrieve, sources
from tlv.agent.connect import LONGEST_LOOK_S, Candidate, Clip, looks, rank_clips, rank_steps
from tlv.agent.inspect import Moment, Rejection, check, hear, same_play
from tlv.agent.model import Vlm
from tlv.agent.plan import Sense
from tlv.agent.replay import Play, distinct_moments, merge_replays
from tlv.agent.sources import Video
from tlv.media import Range
from tlv.timecode import format_range

Progress = Callable[[str], None] | None
# Video-time ranges to search, by video id: a search narrowed to them skips every other video and
# every hit outside them.
Within = Mapping[str, Sequence[tuple[float, float]]]

CHAIN_DEPTH = 40  # clips of each step tried in chains
CLIP_MARGIN_S = 1.0  # step times land within a second, so a cut clip keeps a second more each side
SHEET_FRAMES = 6


def observation_id(video_id: str, start: float, end: float) -> str:
    """A stable id for a confirmed observation: "obs_" and ten hex digits of its video and range
    to the tenth of a second."""
    digest = hashlib.sha1(f"{video_id}:{start:.1f}:{end:.1f}".encode()).hexdigest()
    return f"obs_{digest[:10]}"


def _r(seconds: float) -> float:
    return round(seconds, 2)


def _say(progress: Progress, message: str) -> None:
    if progress:
        progress(message)


def plain_texts(db: sqlite3.Connection, texts: list[str]) -> list[str]:
    """The texts with each @Name mention as the plain name; a mention of nobody known is an error."""
    names = sources.person_names(db) if any("@" in text for text in texts) else []
    mentions = [plan.resolve_mentions(text, names) for text in texts]
    unknown = list(dict.fromkeys(u for m in mentions for u in m.unknown))
    if unknown:
        known = f"Known: {', '.join(f'@{n}' for n in names)}" if names else "No named people yet."
        raise ValueError(f"Unknown entity {', '.join(unknown)}. {known}")
    return [m.plain for m in mentions]


def _steps(db: sqlite3.Connection, query: str, then: list[str], progress: Progress, split: bool) -> list[str]:
    """The query's steps in plain words; with `split`, a query that reads as a sequence is split
    by the model."""

    def by_model(text: str) -> list[str]:
        _say(progress, "splitting the query into steps")
        with Vlm(progress) as vlm:
            return plan.split_with_model(vlm, text)

    plain = plain_texts(db, [query, *then])
    return plan.steps(plain[0], plain[1:], by_model if split else None)


def clips_within(clips: list[Clip], within: Within) -> list[Clip]:
    """The clips overlapping one of the ranges listed for their video, in the order given."""
    return [c for c in clips if any(c.start < end and start < c.end for start, end in within.get(c.video_id, ()))]


def _videos(db: sqlite3.Connection, video_ids: list[str] | None, within: Within | None) -> list[Video]:
    """The videos to search: the named ones (every video when None) that `within` lists, if given."""
    return [v for v in sources.videos(db, video_ids) if within is None or v.video_id in within]


def _candidates(
    db: sqlite3.Connection,
    videos: list[Video],
    steps: list[str],
    senses: list[Sense],
    max_gap: float,
    limit: int,
    modalities: list[str] | None,
    within: Within | None,
) -> list[Candidate]:
    """Each step searched for in the channels of its sense (a spoken step by what is said), and an
    ordered claim also as one sentence in the channels of all its steps. With `within`, hits outside
    its ranges are dropped before the best `limit` are chosen, so the limit fills with hits inside."""
    corpus = retrieve.load(db, [v.video_id for v in videos])
    asked = [(plan.topic(step) if s == "speech" else step, [s]) for step, s in zip(steps, senses, strict=True)]
    if len(steps) > 1:
        asked.append((", then ".join(steps), senses))
    queries = [
        retrieve.Query(text, tuple(entities.resolve(db, text)), retrieve.channels(by, modalities)) for text, by in asked
    ]
    ranked = retrieve.rank(db, corpus, queries)
    if within is not None:
        ranked = [clips_within(clips, within) for clips in ranked]
    if len(steps) == 1:
        return rank_clips(ranked[0], limit)
    *per_step, whole = ranked
    return rank_steps(whole[:CHAIN_DEPTH], [clips[:CHAIN_DEPTH] for clips in per_step], max_gap, limit)


def search(
    query: str,
    then: list[str] = (),
    limit: int = 10,
    video_ids: list[str] | None = None,
    max_gap: float = 12,
    modalities: list[str] | None = None,
    within: Within | None = None,
) -> dict[str, Any]:
    """Moments that may match the query, best first, unchecked. With `then`, each hit links one
    moment per step in order, each step starting within `max_gap` seconds of the one before. With
    `within`, only hits overlapping its ranges of their video count."""
    then = list(then)
    with closing(store.connect()) as db:
        videos = _videos(db, video_ids, within)
        steps = _steps(db, query, then, None, split=False)
        titles = {v.video_id: v.title for v in videos}
        candidates = _candidates(db, videos, steps, [plan.sense(s) for s in steps], max_gap, limit, modalities, within)
    return {"query": query, "then": then, "results": [_hit(c, steps, titles) for c in candidates]}


def _hit(candidate: Candidate, steps: list[str], titles: dict[str, str]) -> dict[str, Any]:
    hit = {
        "video_id": candidate.video_id,
        "title": titles[candidate.video_id],
        "start": _r(candidate.start),
        "end": _r(candidate.end),
        "timecode": format_range(candidate.start, candidate.end),
        "kind": candidate.kind,
        "score": candidate.score,
    }
    if candidate.kind == "chain":
        hit["steps"] = [
            {
                "step": step,
                "start": _r(clip.start),
                "end": _r(clip.end),
                "timecode": format_range(clip.start, clip.end),
                "rank": clip.rank,
            }
            for step, clip in zip(steps, candidate.clips, strict=True)
        ]
    return hit


def find(
    query: str,
    then: list[str] = (),
    max_candidates: int = 8,
    pad: float = 3,
    min_confidence: float = 0.6,
    cut: bool = True,
    video_ids: list[str] | None = None,
    progress: Progress = None,
    within: Within | None = None,
) -> dict[str, Any]:
    """Checks the best candidates for a claim by looking at each, and returns the confirmed ones:
    one per play, its replays listed under it, each with its evidence. Every confirmed moment is
    saved as a verified observation. A claim made only of spoken steps is checked against the speech
    alone, with no model loaded. With `within`, only candidates overlapping its ranges of their video
    are checked, the best of them up to `max_candidates`."""
    began = time.monotonic()
    with closing(store.connect()) as db:
        videos = _videos(db, video_ids, within)
        steps = _steps(db, query, list(then), progress, split=True)
        senses = [plan.sense(step) for step in steps]
        claim = ", then ".join(steps)
        _say(progress, f"searching {len(videos)} videos for {len(steps)} steps")
        candidates = _candidates(db, videos, steps, senses, 12, max_candidates, None, within)
        stretches = looks(candidates, LONGEST_LOOK_S)[:max_candidates]
        searched = time.monotonic()
        by_id = {v.video_id: v for v in videos}
        if all(s == "speech" for s in senses):
            _say(progress, f"listening to {len(stretches)} stretches")
            outcomes = [hear(db, by_id[s.video_id], s, steps, pad) for s in stretches]
            kept, repeats = distinct_moments([o for o in outcomes if isinstance(o, Moment)])
            # A replay shows a play again, but what is said over it is said anew: a spoken moment stands alone.
            plays = [Play(m) for m in kept]
            loaded, model_seconds = searched, 0.0
        else:
            with Vlm(progress) as vlm:
                loaded = time.monotonic()
                _say(progress, f"checking {len(stretches)} stretches")
                looked = [(by_id[s.video_id], s) for s in stretches]
                outcomes = check(vlm, db, looked, claim, steps, senses, pad, min_confidence)
                kept, repeats = distinct_moments([o for o in outcomes if isinstance(o, Moment)])
                plays = merge_replays(kept, lambda a, b: same_play(vlm, a, b))
                model_seconds = vlm.seconds
        checked = time.monotonic()
        plays.sort(key=lambda p: (-p.moment.confidence, p.moment.video_id, p.moment.start))
        rejected = [
            {"video_id": o.video_id, "start": _r(o.start), "end": _r(o.end), "reason": o.reason}
            for o in outcomes
            if isinstance(o, Rejection)
        ]
        rejected += [
            {
                "video_id": m.video_id,
                "start": _r(m.start),
                "end": _r(m.end),
                "reason": f"same moment as {observation_id(of.video_id, of.start, of.end)}",
            }
            for m, of in repeats
        ]
        confirmed = [_observation(db, play, claim) for play in plays]
        _save(db, confirmed, claim)
    if cut:
        for observation, play in zip(confirmed, plays, strict=True):
            _say(progress, f'cutting {observation["timecode"]} of "{observation["title"]}"')
            observation.update(_cut(play.moment))
    seconds = {"search": _r(searched - began), "load": _r(loaded - searched), "check": _r(checked - loaded)}
    _say(
        progress,
        f"found {len(confirmed)} in {time.monotonic() - began:.0f} s: {seconds['search']:.0f} s searching, "
        f"{seconds['load']:.0f} s loading the model, {seconds['check']:.0f} s checking "
        f"({model_seconds:.0f} s of it answering)",
    )
    return {
        "claim": claim,
        "confirmed": confirmed,
        "rejected": rejected,
        "counts": {
            "candidates": len(candidates),
            "checked": len(stretches),
            "confirmed": len(confirmed),
            "rejected": len(rejected),
        },
        "seconds": seconds,
    }


def _observation(db: sqlite3.Connection, play: Play[Moment], claim: str) -> dict[str, Any]:
    m = play.moment
    video = m.video
    speech = sources.speech(db, video.video_id, m.start, m.end)
    text = sources.screen_text(db, video.video_id, m.start, m.end)
    heard = sources.sounds(db, video.video_id, m.start, m.end)
    seen = [(frame, path) for frame, path in m.shown if m.start <= frame.t <= m.end]
    named = [*entities.resolve(db, claim), *(entities.resolve(db, m.who) if m.who else [])]
    present = {p["entity_id"] for p in entities.present(db, video.video_id, m.start, m.end)}
    return {
        "id": observation_id(video.video_id, m.start, m.end),
        "video_id": video.video_id,
        "title": video.title,
        "source": video.source,
        "local_path": video.local_path,
        "start": _r(m.start),
        "end": _r(m.end),
        "timecode": format_range(m.start, m.end),
        "claim": claim,
        "who": m.who,
        "entity_id": next((e for e in named if e in present), None),
        "confidence": m.confidence,
        "evidence": {
            "summary": m.summary,
            "frames": [{"t": _r(frame.t), "path": str(path)} for frame, path in seen],
            "transcript": [{"start": _r(s.start), "end": _r(s.end), "text": s.text} for s in speech],
            "ocr": [{"t": _r(line.start), "text": line.text} for line in text],
            "sounds": [{"start": _r(s.start), "end": _r(s.end), "label": s.text} for s in heard],
        },
        "steps": [
            {"step": s.step, "start": _r(s.start), "end": _r(s.end), "timecode": format_range(s.start, s.end)}
            for s in m.steps
        ],
        "replays": [
            {
                "video_id": r.video_id,
                "start": _r(r.start),
                "end": _r(r.end),
                "timecode": format_range(r.start, r.end),
                "reason": reason,
            }
            for r, reason in play.replays
        ],
    }


def _save(db: sqlite3.Connection, confirmed: list[dict[str, Any]], claim: str) -> None:
    """One verified observation per confirmed result; finding it again replaces it."""
    rows = []
    for o in confirmed:
        evidence = o["evidence"]
        stored = {
            "frames": [f["path"] for f in evidence["frames"]],
            "transcript": " ".join(s["text"] for s in evidence["transcript"]),
            "ocr": [line["text"] for line in evidence["ocr"]],
            "sounds": [s["label"] for s in evidence["sounds"]],
            "summary": evidence["summary"],
            "who": o["who"],
            "steps": o["steps"],
            "replays": o["replays"],
        }
        rows.append(
            (
                o["id"],
                o["video_id"],
                o["start"],
                o["end"],
                o["entity_id"],
                claim,
                o["confidence"],
                json.dumps(stored, ensure_ascii=False),
                claim,
                store.now(),
            )
        )
    with store.transaction(db):
        db.executemany(
            "INSERT OR REPLACE INTO observations (obs_id, video_id, start, end, entity_id, kind, text, confidence, "
            "evidence, query, created_at) VALUES (?, ?, ?, ?, ?, 'verified', ?, ?, ?, ?, ?)",
            rows,
        )


def _cut(moment: Moment) -> dict[str, Any]:
    """The moment cut from its video with CLIP_MARGIN_S more on each side, and a contact sheet."""
    video = moment.video
    span = Range(max(0.0, moment.start - CLIP_MARGIN_S), min(video.duration, moment.end + CLIP_MARGIN_S))
    stem = f"{media.slug(video.title)}-{video.video_id[-6:]}_{span.start:.1f}-{span.end:.1f}"
    clip_path = store.data_dir("clips") / f"{stem}.mp4"
    sheet_path = store.data_dir("frames") / f"{stem}_sheet.jpg"
    try:
        media.cut(Path(video.local_path), span, clip_path)
        times = media.contact_sheet(Path(video.local_path), span, SHEET_FRAMES, sheet_path)
    except RuntimeError as error:
        return {"cut_error": str(error)}
    return {"clip_path": str(clip_path), "frame_path": str(sheet_path), "frame_times": [_r(t) for t in times]}
