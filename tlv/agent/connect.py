"""From ranked hits to candidates: one moment found twice is kept once, and the steps of an
ordered query are linked in time order inside one video. Ported from the TypeScript search.ts;
nothing here reads the store or runs a model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

# Two ranges in one video whose overlap is at least this share of their union are one moment.
SAME_MOMENT = 0.5
# One look at a candidate covers at most this many seconds: a longer stretch may hold two plays, which
# the model reads as one and cannot place (a 17-s chain over two threes came back unreadable).
LONGEST_LOOK_S = 12.0


class Ranged(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


@dataclass(frozen=True)
class Clip:
    """One hit of one query. Rank 1 is the best match."""

    video_id: str
    start: float
    end: float
    rank: int


@dataclass(frozen=True)
class Candidate:
    """A moment to inspect.

    kind is "clip" for a hit of a single-step query, "sentence" for a hit of the whole ordered
    query, and "chain" for one clip per step linked in order. score is lower-is-better: the rank
    of a clip, or the sum of step ranks of a chain. clips holds one clip per step for a chain,
    else the single hit."""

    video_id: str
    start: float
    end: float
    kind: Literal["clip", "sentence", "chain"]
    score: int
    clips: tuple[Clip, ...]


def overlap_ratio(a: Ranged, b: Ranged) -> float:
    """Overlap divided by union of two ranges."""
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 1.0
    return max(0.0, min(a.end, b.end) - max(a.start, b.start)) / union


def merge_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Keeps the better of any two candidates in one video overlapping by at least half their
    union, then returns the best `limit` by score (the earlier start on a tie)."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.score, c.start)):
        if len(kept) == limit:
            break
        if not any(k.video_id == candidate.video_id and overlap_ratio(k, candidate) >= SAME_MOMENT for k in kept):
            kept.append(candidate)
    return kept


def rank_clips(clips: list[Clip], limit: int) -> list[Candidate]:
    """Ranks the hits of a single-step query."""
    return merge_candidates([Candidate(c.video_id, c.start, c.end, "clip", c.rank, (c,)) for c in clips], limit)


@dataclass(frozen=True)
class _Link:
    clip: Clip
    cost: int
    previous: _Link | None


def chain_steps(steps: list[list[Clip]], max_gap: float) -> list[Candidate]:
    """Links one clip per step into chains inside one video. A step's clip must start at or after
    the previous step's clip start and no later than the previous end plus `max_gap`. For every
    clip of the last step, keeps the chain with the lowest rank sum."""
    if not steps:
        return []
    frontier = [_Link(clip, clip.rank, None) for clip in steps[0]]
    for step_clips in steps[1:]:
        following: list[_Link] = []
        for clip in step_clips:
            best: _Link | None = None
            for link in frontier:
                prev = link.clip
                if prev.video_id != clip.video_id or not prev.start <= clip.start <= prev.end + max_gap:
                    continue
                if best is None or link.cost < best.cost:
                    best = link
            if best:
                following.append(_Link(clip, best.cost + clip.rank, best))
        frontier = following
    chains = []
    for last in frontier:
        clips: list[Clip] = []
        link: _Link | None = last
        while link:
            clips.insert(0, link.clip)
            link = link.previous
        start, end = min(c.start for c in clips), max(c.end for c in clips)
        chains.append(Candidate(last.clip.video_id, start, end, "chain", last.cost, tuple(clips)))
    return chains


def rank_steps(sentence: list[Clip], steps: list[list[Clip]], max_gap: float, limit: int) -> list[Candidate]:
    """Ranks an ordered query: chains of the per-step hits plus the hits of the whole sentence. A
    sentence hit scores its rank once per step, so it competes with a chain whose steps all got
    that rank."""
    hits = [Candidate(c.video_id, c.start, c.end, "sentence", c.rank * len(steps), (c,)) for c in sentence]
    return merge_candidates([*chain_steps(steps, max_gap), *hits], limit)


@dataclass(frozen=True)
class Look:
    """A stretch of one video checked in one look."""

    video_id: str
    start: float
    end: float


def looks(candidates: list[Candidate], longest: float) -> list[Look]:
    """The stretches to check for the candidates, best first. Candidates in one video that overlap or
    touch are checked together while their union stays within `longest` seconds, so one moment found
    as neighbouring spans is looked at once; a longer candidate is checked in equal consecutive pieces
    no longer than `longest`."""
    merged: list[Look] = []
    for c in candidates:
        at = next((i for i, m in enumerate(merged) if _joins(m, c, longest)), None)
        if at is None:
            merged.append(Look(c.video_id, c.start, c.end))
        else:
            m = merged[at]
            merged[at] = Look(m.video_id, min(m.start, c.start), max(m.end, c.end))
    pieces = []
    for m in merged:
        count = max(1, math.ceil((m.end - m.start) / longest))
        length = (m.end - m.start) / count
        pieces += [Look(m.video_id, m.start + i * length, m.start + (i + 1) * length) for i in range(count)]
    return pieces


def _joins(look: Look, candidate: Candidate, longest: float) -> bool:
    touching = candidate.start <= look.end and look.start <= candidate.end
    union = max(look.end, candidate.end) - min(look.start, candidate.start)
    return look.video_id == candidate.video_id and touching and union <= longest
