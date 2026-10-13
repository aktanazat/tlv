"""Clues to who a person is: jersey numbers read off body crops, and names from on-screen text,
speech and captions.

A clue is a JSON dict {"kind", "value", "t"} stored with its appearance; caption names also carry
the captioner's "source". Names only count when independent channels agree (`agreed_name`), and
jersey numbers only when several reads of one body give the same digits (`vote`).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any

from tlv.entities.tracks import Box

Clue = dict[str, Any]

NAME_KINDS = ("ocr_name", "spoken_name", "caption_name")
# The channel a name came through: two clues are independent when their channels differ, or
# when they come from different videos or at least APART_S seconds apart. A caption's name counts
# only when the captioner read it on screen or heard it; names it recognized by sight are too
# often wrong.
CHANNEL = {"ocr_name": "screen", "spoken_name": "speech"}
CAPTION_CHANNEL = {"on_screen": "screen", "speech": "speech"}
APART_S = 10.0
# A single word of a name stands for the whole name when it is this long and no other known
# name shares it ("LeBron" for "LeBron James").
MIN_WORD = 4
JERSEY = re.compile(r"\d{1,2}")
MIN_JERSEY_CONF = 0.5
# Digits centered outside this band of a body crop's width belong to someone beside the body.
CENTER_BAND = (0.25, 0.75)


def words(text: str) -> list[str]:
    """Lowercase words without accents: "Luka  Dončić!" -> ["luka", "doncic"]."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9']+", plain.lower())


def jersey(lines: Iterable[tuple[str, float, Box]]) -> tuple[str, Box] | None:
    """The number one body crop shows, with its box, from the crop's (text, confidence, box) lines:
    the single confident one- or two-digit reading centered across the crop, else None."""
    found: dict[str, Box] = {}
    for text, conf, box in lines:
        middle = box[0] + box[2] / 2
        if conf >= MIN_JERSEY_CONF and JERSEY.fullmatch(text.strip()) and CENTER_BAND[0] <= middle <= CENTER_BAND[1]:
            found.setdefault(text.strip(), box)
    return next(iter(found.items())) if len(found) == 1 else None


def vote(reads: Sequence[str | None]) -> str | None:
    """A track's jersey number from its frames' reads: one number read at least twice, with no
    other number read as often as half of that."""
    counts = Counter(read for read in reads if read is not None).most_common()
    if not counts or counts[0][1] < 2:
        return None
    if len(counts) > 1 and counts[1][1] * 2 >= counts[0][1]:
        return None
    return counts[0][0]


def said(timed: Sequence[tuple[str, float]], names: Sequence[str]) -> list[tuple[str, float]]:
    """Each mention of a known name in timed text, (word, time) pairs, with its time: the full
    name word for word, or one word that no other known name has (at least MIN_WORD long).
    Names spelled alike ("LEBRON JAMES", "LeBron James") count as one, the first given."""
    tokens = [(w, t) for text, t in timed for w in words(text)]
    unique: dict[tuple[str, ...], str] = {}
    for name in names:
        unique.setdefault(tuple(words(name)), name)
    unique.pop((), None)
    spelled = sorted(((list(parts), name) for parts, name in unique.items()), key=lambda p: -len(p[0]))
    owners: dict[str, set[str]] = defaultdict(set)
    for parts, name in spelled:
        for word in parts:
            owners[word].add(name)
    found = []
    i = 0
    while i < len(tokens):
        full = next(((p, n) for p, n in spelled if [w for w, _ in tokens[i : i + len(p)]] == p), None)
        word, t = tokens[i]
        if full:
            found.append((full[1], t))
            i += len(full[0])
            continue
        if len(word) >= MIN_WORD and len(owners.get(word, ())) == 1:
            found.append((next(iter(owners[word])), t))
        i += 1
    return found


def mentions(text: str, names: Sequence[str]) -> list[str]:
    """The known names `text` mentions, each once, in order."""
    return list(dict.fromkeys(name for name, _ in said([(text, 0.0)], names)))


def channel(clue: Clue) -> str:
    return CHANNEL.get(clue["kind"]) or CAPTION_CHANNEL[clue["source"]]


def same_person(a: str, b: str) -> bool:
    """Two names can be one person when the words of one are all in the other."""
    x, y = set(words(a)), set(words(b))
    return bool(x) and bool(y) and (x <= y or y <= x)


def independent(items: Iterable[tuple[str, str | None, float]]) -> int:
    """How many of (channel, video_id, t) are independent of each other: same-channel items from
    one video count once per APART_S stretch."""
    times: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    for chan, video, t in items:
        times[chan, video].append(t)
    count = 0
    for ts in times.values():
        ts.sort()
        count += 1 + sum(1 for a, b in pairwise(ts) if b - a >= APART_S)
    return count


def agreed_name(clues: Iterable[tuple[str | None, Clue]]) -> tuple[str, list[Clue]] | None:
    """The name a person's (video_id, clue) pairs agree on, with the clues behind it, or None.

    Partial names join the full name they belong to ("LeBron" joins "LeBron James"). A name is
    taken when at least two independent clues support it and it has at least twice the support
    of any other name.
    """
    named = [(video, clue) for video, clue in clues if clue["kind"] in NAME_KINDS and words(clue["value"])]
    full = sorted({clue["value"] for _, clue in named}, key=lambda n: -len(words(n)))
    owner: dict[str, str] = {}
    for value in full:
        homes = [n for n in full if len(words(n)) > len(words(value)) and same_person(n, value)]
        heads = {owner.get(n, n) for n in homes}
        if len(heads) <= 1:
            owner[value] = heads.pop() if heads else value
    support: dict[str, list[tuple[str | None, Clue]]] = defaultdict(list)
    for video, clue in named:
        if clue["value"] in owner:
            support[owner[clue["value"]]].append((video, clue))
    ranked = sorted(
        ((independent((channel(c), v, c["t"]) for v, c in found), name, found) for name, found in support.items()),
        key=lambda r: -r[0],
    )
    if not ranked:
        return None
    count, name, found = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0
    if count < 2 or count < 2 * runner_up:
        return None
    return name, [c for _, c in found]
