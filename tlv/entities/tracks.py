"""Tracks: one person followed through the sampled frames of one shot, from Vision detections.

Detections arrive every 0.5 s. A detection continues a track when its box overlaps the track's
last box (IoU at least MIN_IOU), every rival overlap of either (the track with another detection,
the detection with another track) is under RIVAL times that, and the track was seen within
MAX_GAP_S. Anything else starts a new track: a broken track only costs a merge later, but a
swapped one would mix two people. Boxes are normalized x, y, w, h with a top-left origin.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Box = tuple[float, float, float, float]
Kind = Literal["face", "body"]

MIN_IOU = 0.3
RIVAL = 0.5
MAX_GAP_S = 1.0
# A face sits on a body when its center is inside the body box, in its top HEAD_SHARE.
HEAD_SHARE = 0.4
# ...in at least this share of the frames where both are seen.
LINK_SHARE = 0.6


@dataclass(frozen=True)
class Seen:
    """One detection: a face or body box at time t."""

    t: float
    box: Box
    conf: float
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Track:
    kind: Kind
    seen: tuple[Seen, ...]

    @property
    def start(self) -> float:
        return self.seen[0].t

    @property
    def end(self) -> float:
        return self.seen[-1].t

    def at(self, t: float) -> Seen | None:
        return next((s for s in self.seen if s.t == t), None)


def iou(a: Box, b: Box) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def follow(kind: Kind, seen: Sequence[Seen], cuts: Sequence[float]) -> list[Track]:
    """Links one kind of detection into tracks. `cuts` are shot start times; no track crosses one."""
    cuts = sorted(cuts)
    tracks: list[list[Seen]] = []
    live: list[int] = []
    by_time: dict[float, list[Seen]] = defaultdict(list)
    for s in seen:
        by_time[s.t].append(s)
    previous = -math.inf
    for t in sorted(by_time):
        if bisect.bisect_right(cuts, t) != bisect.bisect_right(cuts, previous):
            live = []
        previous = t
        live = [i for i in live if t - tracks[i][-1].t <= MAX_GAP_S]
        frame = by_time[t]
        overlap = {(i, j): iou(tracks[i][-1].box, s.box) for i in live for j, s in enumerate(frame)}
        per_track: dict[int, list[float]] = defaultdict(list)
        per_detection: dict[int, list[float]] = defaultdict(list)
        for (i, j), value in overlap.items():
            per_track[i].append(value)
            per_detection[j].append(value)
        continued = {
            j: i
            for (i, j), value in overlap.items()
            if value >= MIN_IOU and _unrivaled(per_track[i], value) and _unrivaled(per_detection[j], value)
        }
        for j, s in enumerate(frame):
            if j in continued:
                tracks[continued[j]].append(s)
            else:
                tracks.append([s])
                live.append(len(tracks) - 1)
    return [Track(kind, tuple(t)) for t in tracks]


def _unrivaled(values: list[float], value: float) -> bool:
    """`value` is the largest of `values`, which hold it, and every other is under RIVAL times it."""
    ranked = sorted(values, reverse=True)
    return ranked[0] == value and (len(ranked) == 1 or ranked[1] < RIVAL * value)


def sits_on(face: Box, body: Box) -> bool:
    """The face's center lies inside the body box, in its head region."""
    cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
    return body[0] <= cx <= body[0] + body[2] and body[1] <= cy <= body[1] + body[3] * HEAD_SHARE


def link(faces: Sequence[Track], bodies: Sequence[Track]) -> dict[int, int]:
    """Face track index -> body track index for faces that sit on one body, and that body under
    no other face, through most frames both tracks share."""
    shared: Counter[tuple[int, int]] = Counter()
    sitting: Counter[tuple[int, int]] = Counter()
    faces_at: dict[float, list[tuple[int, Seen]]] = defaultdict(list)
    bodies_at: dict[float, list[tuple[int, Seen]]] = defaultdict(list)
    for i, track in enumerate(faces):
        for s in track.seen:
            faces_at[s.t].append((i, s))
    for j, track in enumerate(bodies):
        for s in track.seen:
            bodies_at[s.t].append((j, s))
    for t, here in faces_at.items():
        there = bodies_at.get(t, [])
        pairs = [(i, j) for i, f in here for j, b in there if sits_on(f.box, b.box)]
        per_face = Counter(i for i, _ in pairs)
        per_body = Counter(j for _, j in pairs)
        for i, _ in here:
            for j, _ in there:
                shared[i, j] += 1
        for i, j in pairs:
            if per_face[i] == 1 and per_body[j] == 1:
                sitting[i, j] += 1
    best: dict[int, tuple[int, int]] = {}
    for (i, j), count in sitting.items():
        if count >= LINK_SHARE * shared[i, j] and count > best.get(i, (0, -1))[0]:
            best[i] = (count, j)
    return {i: j for i, (_, j) in best.items()}


def clearest(track: Track, count: int = 1) -> list[Seen]:
    """The `count` frames that show the person best: big, confident, facing the camera (faces)
    or whole inside the frame (bodies)."""

    def score(s: Seen) -> float:
        x, y, w, h = s.box
        if track.kind == "face":
            facing = 1.0 if abs(s.extra.get("yaw", 0.0)) < 0.6 else 0.5
            return h * s.conf * s.extra.get("quality", 0.0) * facing
        whole = 1.0 if x > 0.005 and y > 0.005 and x + w < 0.995 and y + h < 0.995 else 0.5
        return w * h * s.conf * whole

    return sorted(track.seen, key=score, reverse=True)[:count]
