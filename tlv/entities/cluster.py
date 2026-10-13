"""Grouping appearances into people, conservatively.

Corrections come first and override everything: "different" keeps two appearances apart, and
"same" joins two appearances, as does giving two appearances the same name, unless they were told
apart. Then evidence joins groups, strongest first: a face sitting on a body (tracks.link), a
"same" answer from the vision-language model, one jersey number on one uniform within a game, and
faces close enough in ArcFace space, at least one of them sharp, when each group's faces as a whole
agree too. A join is refused when it would put together two appearances seen at the same moment,
two bodies wearing one number in one game in uniforms that share almost no colors, two jersey
numbers, two different names, or a pair someone (or the model) said are different. A missed join
is the safe failure; a false one contaminates every later answer.

Blurred faces and backs say too little to join anyone on their own. So when a video has just been
indexed, its groups that no sharp face places become leads: each may be a person the owner named,
and the vision-language model is asked to compare it with that person (`Lead`); its "same" then
joins them like any other verdict.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tlv.entities import clues as rules
from tlv.entities.tracks import Box, Kind

FACE_SAME = 0.5  # ArcFace cosine at which two faces join without asking
FACE_GRAY = 0.3  # ...and from which the pair is worth asking the vision-language model about
FACE_CENTROID = 0.4  # the two groups' mean sharp faces must agree this well as well, so joins do not chain
# Vision's capture quality from which a face is sharp. Blurred, turned or tiny faces have ArcFace
# vectors close to one another whoever they show, so they are compared with sharp faces only. On
# broadcast video, faces below 0.3 matched different people at cosine 0.5 to 0.82, while a face of
# 0.3 or more matched no other person in its frame above 0.15, but for one pair (likely a dissolve).
FACE_QUALITY = 0.3
UNIFORM_SAME = 0.9  # two bodies with one jersey number wear one uniform when their colors agree this well
# Two bodies with one jersey number in one game whose uniform colors agree below this wear two
# uniforms, so they are two people, one per team. On the games labeled in eval/people.json, 1,205
# pairs of one player's numbered bodies never agreed below 0.195, while 20 pairs of two players
# (gold against white #23) did, down to 0.08. Telling such pairs apart before any join keeps a crop
# that shows both uniforms from chaining the two players into one person.
UNIFORM_APART = 0.15
NEIGHBORS = 15
HALF_STEP = 0.25  # half the sampling step: a track last seen at t is visible until t + HALF_STEP
BLOCK_ROWS = 1024


@dataclass(frozen=True)
class Appearance:
    id: str
    video_id: str | None
    kind: Kind
    start: float
    end: float
    box: Box
    vector: np.ndarray
    clues: tuple[dict[str, Any], ...] = ()
    body: str | None = None
    quality: float | None = None  # faces only: Vision's capture quality where embedded

    @property
    def sharp(self) -> bool:
        return self.quality is not None and self.quality >= FACE_QUALITY


@dataclass(frozen=True)
class Correction:
    kind: Literal["name", "same", "different"]
    a: str
    b: str | None = None
    value: str | None = None
    at: float = 0.0


@dataclass(frozen=True)
class Lead:
    """A group seen only in a newly indexed video that may be a person the owner named: the model is
    asked whether `candidate`, the group's clearest face or else its biggest body, shows the same
    person as `reference`, the named person's appearance of that kind. Only when it does is each of
    `bodies`, a (reference, body) pair for every body of the group, asked about too, since a face can
    sit on someone else's body in a hug."""

    kind: Kind
    reference: str
    candidate: str
    closeness: float  # ArcFace cosine to the person's nearest face, or uniform colors' to the reference body
    bodies: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Clustering:
    groups: list[list[str]]
    # Appearance pairs from different groups whose faces look alike, but not enough to join on
    # their own and with no verdict yet: the questions for the vision-language model, closest first.
    gray: list[tuple[str, str, float]]
    # Groups of the newly indexed video that may be people the owner named, closest first.
    leads: list[Lead]


@dataclass(frozen=True)
class Description:
    name: str | None
    evidence: list[dict[str, Any]]
    canonical: str
    vector: np.ndarray


def latest_names(corrections: Sequence[Correction]) -> dict[str, Correction]:
    """Each appearance's most recent name correction."""
    named: dict[str, Correction] = {}
    for c in sorted(corrections, key=lambda c: c.at):
        if c.kind == "name":
            named[c.a] = c
    return named


def cluster(
    apps: Sequence[Appearance],
    corrections: Sequence[Correction] = (),
    verdicts: dict[tuple[str, str], bool | None] | None = None,
    new_video: str | None = None,
) -> Clustering:
    """Groups appearances into people. `verdicts` maps sorted appearance id pairs to the model's
    answer (True same, False different, None unsure). With `new_video`, the video just indexed,
    also lists its leads."""
    verdicts = verdicts or {}
    index = {a.id: i for i, a in enumerate(apps)}
    groups = _Groups(apps, latest_names(corrections))
    for c in corrections:
        if c.kind == "different" and c.a in index and c.b in index:
            groups.block(index[c.a], index[c.b])
    for i, j in _forced(corrections, index, groups.corrected):
        groups.force(i, j)
    dressed = _jersey_pairs(apps)
    for i, j in _apart(apps, verdicts, index):
        groups.block(i, j)
    for similarity, i, j in dressed:
        if similarity < UNIFORM_APART:
            groups.block(i, j)
    for i, j in _links(apps, index):
        groups.join(i, j)
    for (a, b), same in sorted(verdicts.items()):
        if same and a in index and b in index:
            groups.join(index[a], index[b])
    for similarity, i, j in dressed:
        if similarity >= UNIFORM_SAME:
            groups.join(i, j)
    faces = [i for i, a in enumerate(apps) if a.kind == "face"]
    near, gray = [], []
    for i, j, similarity in _neighbors(apps, faces, [i for i in faces if apps[i].sharp]):
        (near if similarity >= FACE_SAME else gray).append((similarity, i, j))
    for similarity, i, j in sorted(near, reverse=True):
        if not groups.join(i, j, faces_agree=True):
            gray.append((similarity, i, j))
    asked = {groups.pair(index[a], index[b]) for a, b in verdicts if a in index and b in index}
    questions: dict[tuple[int, int], tuple[str, str, float]] = {}
    for similarity, i, j in sorted(gray, reverse=True):
        pair = groups.pair(i, j)
        if pair[0] != pair[1] and pair not in asked and pair not in questions and not groups.conflict(*pair):
            a, b = sorted((apps[i].id, apps[j].id))
            questions[pair] = (a, b, similarity)
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(apps)):
        members[groups.find(i)].append(i)
    leads = _leads(apps, groups, members, new_video, verdicts) if new_video else []
    return Clustering([[apps[i].id for i in group] for group in members.values()], list(questions.values()), leads)


def describe(apps: Sequence[Appearance], corrections: Sequence[Correction] = ()) -> Description:
    """Name, name evidence, canonical appearance and mean embedding of one group. A name
    correction outranks clues; clues name the group only when they agree (clues.agreed_name)."""
    named = latest_names(corrections)
    fixes = sorted((named[a.id] for a in apps if a.id in named), key=lambda c: c.at)
    agreed = rules.agreed_name(
        (a.video_id, {**c, "video_id": a.video_id, "appearance_id": a.id}) for a in apps for c in a.clues
    )
    evidence: list[dict[str, Any]] = [{"kind": "correction", "value": c.value, "appearance_id": c.a} for c in fixes]
    if agreed and (not fixes or rules.same_person(agreed[0], fixes[-1].value or "")):
        evidence += agreed[1]
    name = fixes[-1].value if fixes else agreed[0] if agreed else None
    faces = [a for a in apps if a.kind == "face"]
    pool = [a for a in faces if a.sharp] or faces or list(apps)
    canonical = max(pool, key=lambda a: (a.box[3] if faces else a.box[2] * a.box[3], a.id))
    mean = np.stack([a.vector for a in pool]).mean(axis=0)
    return Description(name, evidence, canonical.id, mean / np.linalg.norm(mean))


class _Groups:
    """Union-find over appearances that keeps, per group, what a join must not contradict."""

    def __init__(self, apps: Sequence[Appearance], named: dict[str, Correction]) -> None:
        self.apps = apps
        self.parent = list(range(len(apps)))
        self.blocked: list[set[int]] = [set() for _ in apps]
        self.jerseys: list[set[str]] = [set() for _ in apps]
        self.corrected: list[set[str]] = [set() for _ in apps]
        self.clues: list[list[tuple[str | None, dict[str, Any]]]] = [[] for _ in apps]
        # Sums of each group's sharp face vectors, and of all its face vectors.
        self.sharp: list[np.ndarray | None] = [a.vector.copy() if a.sharp else None for a in apps]
        self.seen: list[np.ndarray | None] = [a.vector.copy() if a.kind == "face" else None for a in apps]
        self._names: dict[int, frozenset[str]] = {}
        for i, a in enumerate(apps):
            if a.id in named and named[a.id].value:
                self.corrected[i].add(named[a.id].value)
            for clue in a.clues:
                if clue["kind"] == "jersey":
                    self.jerseys[i].add(clue["value"])
                else:
                    self.clues[i].append((a.video_id, clue))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def pair(self, i: int, j: int) -> tuple[int, int]:
        a, b = self.find(i), self.find(j)
        return (a, b) if a <= b else (b, a)

    def block(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.blocked[a].add(b)
            self.blocked[b].add(a)

    def names(self, root: int) -> frozenset[str]:
        """The name a group is known by: its corrections', else the one its clues agree on."""
        if root not in self._names:
            agreed = rules.agreed_name(self.clues[root])
            self._names[root] = frozenset(self.corrected[root] or ({agreed[0]} if agreed else ()))
        return self._names[root]

    def heard(self, root: int) -> frozenset[str]:
        """Every name the group's clues give, agreed or not."""
        return frozenset(clue["value"] for _, clue in self.clues[root])

    def conflict(self, a: int, b: int) -> bool:
        """Whether groups a and b cannot be one person: told apart, wearing different numbers (a
        player keeps his number from game to game), or named differently. Where only one group is
        known by a name its clues agree on, every name the other's clues give counts against it:
        one caption or one mention naming someone else is enough to stay apart. A name the owner
        gave is contradicted only by a name the other group is known by, since it may be one no
        caption or commentator uses; otherwise every mention of the person's own name would cut
        the named face from its body."""
        if b in self.blocked[a]:
            return True
        if self.jerseys[a] and self.jerseys[b] and self.jerseys[a].isdisjoint(self.jerseys[b]):
            return True
        mine, theirs = self.names(a), self.names(b)
        if not (mine or theirs):
            return False
        mine = mine or (frozenset() if self.corrected[b] else self.heard(a))
        theirs = theirs or (frozenset() if self.corrected[a] else self.heard(b))
        return bool(mine and theirs) and not any(rules.same_person(x, y) for x in mine for y in theirs)

    def face(self, root: int) -> np.ndarray | None:
        """The group's face: the sum of its sharp faces, else of its blurred ones. A blurred face so
        answers to the person's sharp faces as a whole, not to the one odd view it resembles."""
        sharp = self.sharp[root]
        return self.seen[root] if sharp is None else sharp

    def join(self, i: int, j: int, faces_agree: bool = False) -> bool:
        """Joins the groups of i and j unless that contradicts what either knows; with
        `faces_agree`, also unless their faces differ. True when they end up together."""
        a, b = self.find(i), self.find(j)
        if a == b:
            return True
        if self.conflict(a, b):
            return False
        if faces_agree and _cosine(self.face(a), self.face(b)) < FACE_CENTROID:
            return False
        self.union(a, b)
        return True

    def force(self, i: int, j: int) -> None:
        """Joins the groups of i and j, whatever the evidence says, unless they were told apart."""
        a, b = self.find(i), self.find(j)
        if b not in self.blocked[a]:
            self.union(a, b)

    def union(self, a: int, b: int) -> None:
        if a == b:
            return
        if len(self.clues[a]) < len(self.clues[b]):
            a, b = b, a
        self.parent[b] = a
        for other in self.blocked[b]:
            self.blocked[other].discard(b)
            self.blocked[other].add(a)
        self.blocked[a] |= self.blocked[b]
        self.blocked[a].discard(a)
        self.jerseys[a] |= self.jerseys[b]
        self.corrected[a] |= self.corrected[b]
        self.clues[a] += self.clues[b]
        for sums in (self.sharp, self.seen):
            if sums[b] is not None:
                sums[a] = sums[b] if sums[a] is None else sums[a] + sums[b]
        self._names.pop(a, None)
        self._names.pop(b, None)


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 1.0
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _forced(
    corrections: Sequence[Correction], index: dict[str, int], corrected: list[set[str]]
) -> Iterator[tuple[int, int]]:
    """Pairs corrections put together: "same", and appearances given the same name."""
    for c in corrections:
        if c.kind == "same" and c.a in index and c.b in index:
            yield index[c.a], index[c.b]
    by_name: dict[tuple[str, ...], int] = {}
    for i, names in enumerate(corrected):
        for name in names:
            key = tuple(rules.words(name))
            if key in by_name:
                yield by_name[key], i
            by_name.setdefault(key, i)


def _apart(
    apps: Sequence[Appearance], verdicts: dict[tuple[str, str], bool | None], index: dict[str, int]
) -> Iterator[tuple[int, int]]:
    """Pairs the evidence says are different people: seen at the same moment as the same kind, or
    said to be different by the model."""
    timeline: dict[tuple[str, Kind], list[int]] = defaultdict(list)
    for i, a in enumerate(apps):
        if a.video_id is not None:
            timeline[a.video_id, a.kind].append(i)
    for members in timeline.values():
        members.sort(key=lambda i: apps[i].start)
        live: list[int] = []
        for i in members:
            live = [j for j in live if apps[j].end >= apps[i].start]
            yield from ((j, i) for j in live)
            live.append(i)
    for (a, b), same in verdicts.items():
        if same is False and a in index and b in index:
            yield index[a], index[b]


def _links(apps: Sequence[Appearance], index: dict[str, int]) -> Iterator[tuple[int, int]]:
    for i, a in enumerate(apps):
        if a.body in index:
            yield i, index[a.body]


def _leads(
    apps: Sequence[Appearance],
    groups: _Groups,
    members: dict[int, list[int]],
    video_id: str,
    verdicts: dict[tuple[str, str], bool | None],
) -> list[Lead]:
    """The groups seen only in `video_id`, with no sharp face, that may be a person the owner named,
    closest first: groups that neither wear another number nor go by another name (_Groups.conflict),
    and whose clearest face looks at least FACE_GRAY like one of the person's faces or, showing no
    face, wear the person's number. A lead the model has already judged is not listed again."""
    named = [root for root in members if groups.corrected[root]]
    leads = []
    for root, group in members.items():
        if groups.corrected[root] or any(apps[i].video_id != video_id or apps[i].sharp for i in group):
            continue
        for person in named:
            lead = None if groups.conflict(person, root) else _lead(apps, members[person], group)
            if lead and tuple(sorted((lead.reference, lead.candidate))) not in verdicts:
                leads.append(lead)
    return sorted(leads, key=lambda lead: -lead.closeness)


def _lead(apps: Sequence[Appearance], person: list[int], group: list[int]) -> Lead | None:
    """The question that would tell whether `group` is `person`, when it may be. A face is shown
    beside the person's portrait (their biggest sharp face, else their biggest face), a body beside
    the person's biggest body wearing its number."""
    faces = [i for i in group if apps[i].kind == "face"]
    bodies = [i for i in group if apps[i].kind == "body"]
    theirs = [i for i in person if apps[i].kind == "face"]
    if faces:
        if not theirs:
            return None
        best = max(faces, key=lambda i: (apps[i].quality or 0.0, _area(apps[i]), apps[i].id))
        closeness = float(np.max(np.stack([apps[i].vector for i in theirs]) @ apps[best].vector))
        if closeness < FACE_GRAY:
            return None
        portrait = max([i for i in theirs if apps[i].sharp] or theirs, key=lambda i: (apps[i].box[3], apps[i].id))
        asked = [(_reference_body(apps, person, apps[b]), b) for b in bodies]
        pairs = tuple((apps[r].id, apps[b].id) for r, b in asked if r is not None)
        return Lead("face", apps[portrait].id, apps[best].id, closeness, pairs)
    body = max(bodies, key=lambda i: (_area(apps[i]), apps[i].id))
    reference = _reference_body(apps, person, apps[body])
    if reference is None or not _worn(apps[body]) & _worn(apps[reference]):
        return None
    return Lead("body", apps[reference].id, apps[body].id, float(apps[reference].vector @ apps[body].vector))


def _reference_body(apps: Sequence[Appearance], person: list[int], body: Appearance) -> int | None:
    """The person's biggest body wearing a number `body` wears, else any number, else any body: a
    number read off a body vouches that it is theirs."""
    theirs = [i for i in person if apps[i].kind == "body"]
    pool = [i for i in theirs if _worn(apps[i]) & _worn(body)] or [i for i in theirs if _worn(apps[i])] or theirs
    return max(pool, key=lambda i: (_area(apps[i]), apps[i].id), default=None)


def _worn(a: Appearance) -> set[str]:
    return {clue["value"] for clue in a.clues if clue["kind"] == "jersey"}


def _area(a: Appearance) -> float:
    return a.box[2] * a.box[3]


def _jersey_pairs(apps: Sequence[Appearance]) -> list[tuple[float, int, int]]:
    """Body pairs from one game with one jersey number, with how alike their uniforms are, most
    alike first."""
    wearing: dict[tuple[str | None, str], list[int]] = defaultdict(list)
    for i, a in enumerate(apps):
        if a.kind == "body" and a.video_id is not None:
            for clue in a.clues:
                if clue["kind"] == "jersey":
                    wearing[a.video_id, clue["value"]].append(i)
    pairs = []
    for members in wearing.values():
        vectors = np.stack([apps[i].vector for i in members])
        similar = vectors @ vectors.T
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pairs.append((float(similar[x, y]), members[x], members[y]))
    return sorted(pairs, reverse=True)


def _neighbors(
    apps: Sequence[Appearance], rows: Sequence[int], columns: Sequence[int]
) -> Iterator[tuple[int, int, float]]:
    """Each row member's NEIGHBORS nearest column members at FACE_GRAY or closer, each pair once."""
    if not columns:
        return
    vectors = np.stack([apps[i].vector for i in columns])
    where = {i: c for c, i in enumerate(columns)}
    seen: set[tuple[int, int]] = set()
    k = min(NEIGHBORS, len(columns))
    for top in range(0, len(rows), BLOCK_ROWS):
        block = rows[top : top + BLOCK_ROWS]
        similar = np.stack([apps[i].vector for i in block]) @ vectors.T
        for r, i in enumerate(block):
            if i in where:
                similar[r, where[i]] = -1.0
        nearest = np.argpartition(-similar, k - 1, axis=1)[:, :k]
        for r, i in enumerate(block):
            for c in nearest[r]:
                similarity = float(similar[r, c])
                pair = (min(i, columns[c]), max(i, columns[c]))
                if similarity >= FACE_GRAY and pair not in seen:
                    seen.add(pair)
                    yield pair[0], pair[1], similarity
