"""Follow-ups to the last find, said the way a person says them: "now only fourth quarter", "without
the replays", "same but steph". A narrowing keeps some of the clips the find confirmed and is
answered from the index in milliseconds, with no model and no new search; naming someone else
reruns the find with that person in place of the one the search named. Rules read the usual
phrasings; what they cannot read goes to the model once, and each part of its answer must name
something that exists (a game clock window, words of a video's title, a person the search names)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

from tlv.agent.clock import Window, mentioned
from tlv.agent.model import Vlm, parse_json

_LEAVE_OUT = r"(?:without|no|not|skip|drop|remove|exclude|excluding|except|minus|lose|cut|hide)"
_REPLAYS = re.compile(
    rf"\b{_LEAVE_OUT}\s+(?:the\s+|any\s+)?replays?\b|\blive\s+(?:only|plays?|footage|action)\b"
    r"|\bonly\s+(?:the\s+)?live\b",
    re.I,
)
_DROP = re.compile(rf"\b{_LEAVE_OUT}\s+(?:the\s+)?(obs_\w+(?:(?:\s*,\s*|\s+and\s+|\s+)(?:the\s+)?obs_\w+)*)", re.I)
_OBS = re.compile(r"obs_\w+")
_VIDEO_NOUN = re.compile(r"\b(?:game|games|video|videos|highlights)\b", re.I)
# "replace lebron with steph", "instead of lebron, steph", "steph instead of lebron"
_EXPLICIT = (
    re.compile(r"\b(?:replace|swap|switch)\s+(?P<old>.+?)\s+(?:with|for|to|by)\s+(?P<new>.+)", re.I),
    re.compile(r"\binstead\s+of\s+(?P<old>[^,]+),\s*(?P<new>.+)", re.I),
    re.compile(r"(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+)", re.I),
)
# Words that name no person, video or change of their own.
_FILLER = frozenset(
    {"a", "an", "the", "it", "that", "this", "them", "those", "these", "me", "in", "on", "of", "from", "at"}
    | {"now", "only", "just", "keep", "show", "give", "get", "please", "thanks", "and", "but", "with", "for"}
    | {"same", "instead", "what", "how", "about", "try", "do", "again", "redo", "repeat", "switch", "to"}
    | {"during", "find", "search", "look", "one", "ones", "clip", "clips", "play", "plays", "moment", "moments"}
    | {"result", "results", "game", "games", "video", "videos", "highlights"}
)


@dataclass(frozen=True)
class Narrow:
    """Which of the last find's clips to keep: those inside a window of game time, in some of the
    videos, not mostly replay, and not dropped by id. The empty Narrow keeps them all."""

    clock: Window | None = None
    video_ids: frozenset[str] | None = None
    no_replays: bool = False
    drop: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Swap:
    """Someone else to find: every way the last search names the old person (`old`, longest
    first) becomes `new`."""

    old: tuple[str, ...]
    new: str


@dataclass(frozen=True)
class Followup:
    """What a follow-up asks for: clips of the last find to keep, and a person to find instead,
    which takes a new find."""

    narrow: Narrow
    swap: Swap | None = None


def parse(instruction: str, steps: Sequence[str], names: Sequence[str], titles: Mapping[str, str]) -> Followup | None:
    """The follow-up the rules read in an instruction (its @mentions already plain names), given the
    steps of the last search, the names of known people and the videos' titles by id; None when
    they cannot read all of it, for the model to try."""
    text = " ".join(instruction.split())
    named_ids = frozenset(v for v in titles if re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text))
    for video_id in named_ids:
        text = text.replace(video_id, " ")
    no_replays = _REPLAYS.search(text) is not None
    text = _REPLAYS.sub(" ", text)
    drop = frozenset(obs for said in _DROP.finditer(text) for obs in _OBS.findall(said[1]))
    text = _DROP.sub(" ", text)
    window = None
    if found := mentioned(text):
        window, phrase = found
        text = re.sub(re.escape(phrase), " ", text, count=1, flags=re.I)
    words = _content(text)
    video_ids = named_ids or None
    if words and _VIDEO_NOUN.search(text) and (titled := _titled(words, titles)):
        video_ids, words = titled, []
    narrow = Narrow(window, video_ids, no_replays, drop)
    swap = _swap_said(text, words, steps, names) if words else None
    if words and swap is None:
        return None
    return Followup(narrow, swap) if swap or narrow != Narrow() else None


def _swap_said(text: str, words: list[str], steps: Sequence[str], names: Sequence[str]) -> Swap | None:
    """The person to find instead, as "replace X with Y" or "Y instead of X" say it; or, when the
    words left name exactly one known person, that person in place of the one the search names."""
    for pattern in _EXPLICIT:
        if said := pattern.search(text):
            return _swap(said["old"], said["new"], steps, names)
    person = " ".join(words)
    return _swap(None, person, steps, names) if _known(person, names) else None


def _swap(old: str | None, new: str, steps: Sequence[str], names: Sequence[str]) -> Swap | None:
    """`new` in place of `old`, read as one of the people the search names where it can be, or in
    place of the one known person the search names when `old` says no one; None when the search
    does not name the old person, or no new one is named."""
    person = " ".join(_content(new))
    said = " ".join(_content(old or ""))
    searched = _searched(steps, names)
    replaced = (_known(said, searched) or said) if said else (searched[0] if len(searched) == 1 else None)
    if not person or not replaced:
        return None
    forms = _forms(replaced, names)
    if not any(_says(step, form) for step in steps for form in forms):
        return None
    return Swap(forms, _known(person, names) or person)


def _content(text: str) -> list[str]:
    """The words of an instruction that are not filler, as written."""
    return [word for word in re.findall(r"\w(?:[\w'.-]*\w)?", text) if word.lower() not in _FILLER]


def _titled(words: Sequence[str], titles: Mapping[str, str]) -> frozenset[str]:
    """The videos whose title has every one of the words."""
    wanted = {part for word in words for part in re.findall(r"\w+", word.lower())}
    return frozenset(v for v, title in titles.items() if wanted <= set(re.findall(r"\w+", title.lower())))


def _known(person: str, names: Sequence[str]) -> str | None:
    """The one known person the words name, each word a word of the name or, from four letters on,
    the start of one ("steph" for Stephen Curry); None for no one or several."""
    said = person.lower().split()

    def names_them(name: str) -> bool:
        parts = name.lower().split()
        return all(any(part == word or (len(word) >= 4 and part.startswith(word)) for part in parts) for word in said)

    named = [name for name in names if said and names_them(name)]
    return named[0] if len(named) == 1 else None


def _forms(person: str, names: Sequence[str]) -> tuple[str, ...]:
    """The ways a search can name a person, longest first: a known person's full name and each word
    of it ("LeBron James", "LeBron", "James"); anyone else as written."""
    known = _known(person, names)
    if known is None:
        return (person,)
    return tuple(dict.fromkeys([known, *sorted(known.split(), key=len, reverse=True)]))


def _searched(steps: Sequence[str], names: Sequence[str]) -> list[str]:
    """The known people the search names: those it names in full or, when it names no one in full,
    by a word of their name. In "LeBron James dunks" the "James" is not James Harden."""
    full = [name for name in names if any(_says(step, name) for step in steps)]
    return full or [name for name in names if any(_says(step, word) for step in steps for word in name.split())]


def _says(text: str, words: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(words)}(?!\w)", text, re.I) is not None


def swapped(steps: Sequence[str], swap: Swap) -> list[str]:
    """The steps naming the new person wherever they named the old one. One pass, so a full name is
    replaced whole and a new name holding the old one is not replaced again."""
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(form)}(?!\w)" for form in swap.old), re.I)
    return [pattern.sub(lambda _: swap.new, step) for step in steps]


def narrowed(
    result: dict[str, Any],
    narrow: Narrow,
    ranges: Mapping[str, Sequence[tuple[float, float]]],
    replayed: Set[str],
) -> dict[str, Any]:
    """The find result with only the confirmed clips the narrow keeps. With a game clock window, a
    clip is inside it when it overlaps one of its video's `ranges`, as find keeps a candidate
    within them; a clip is a replay when its id is in `replayed`. Kept clips lose their replays
    when replays are left out. Rejected candidates and the other counts stay as the find left them."""
    clips = result["confirmed"]
    unknown = narrow.drop - {clip["id"] for clip in clips}
    if unknown:
        raise ValueError(f"no clip {', '.join(sorted(unknown))} in the last find")

    def kept(clip: dict[str, Any]) -> bool:
        inside = narrow.clock is None or any(
            clip["start"] < end and start < clip["end"] for start, end in ranges.get(clip["video_id"], ())
        )
        return (
            inside
            and clip["id"] not in narrow.drop
            and (narrow.video_ids is None or clip["video_id"] in narrow.video_ids)
            and not (narrow.no_replays and clip["id"] in replayed)
        )

    confirmed = [{**clip, "replays": []} if narrow.no_replays else clip for clip in clips if kept(clip)]
    return {**result, "confirmed": confirmed, "counts": {**result["counts"], "confirmed": len(confirmed)}}


def prompt(instruction: str, steps: Sequence[str], titles: Mapping[str, str]) -> str:
    return (
        "Someone searched sports videos for a moment and now asks for a change to what was found. Say what the "
        "change asks for, with null for each part it does not mention.\n"
        f"Search: {', then '.join(steps)}\n"
        f"Videos: {'; '.join(titles.values())}\n"
        f"Change: {instruction}\n"
        'Answer with JSON only: {"replace": "the words of the search naming the person to change", "with": "the '
        'person to find instead", "clock": "the game time to keep, like 4th, 2nd half, 4th 2:00-0:00 or last 2 min", '
        '"video": "words from the title of the one video to keep"}'
    )


def read_answer(
    value: object, steps: Sequence[str], names: Sequence[str], titles: Mapping[str, str]
) -> Followup | None:
    """The model's follow-up when each part it fills in names something that exists and some part
    asks for a change; None otherwise. The model writes "null" as a word for the parts a change does
    not mention. Leaving out the replays is for the rules alone: asked for it as one more part, the
    model said to leave them out for nearly every change, "make it purple" too."""
    if not isinstance(value, dict):
        return None
    clock, video, old, new = (
        None if isinstance(part, str) and part.strip().lower() == "null" else part
        for part in (value.get(key) for key in ("clock", "video", "replace", "with"))
    )
    if not all(isinstance(part, str | None) for part in (clock, video, old, new)):
        return None
    window = mentioned(clock) if clock else None
    video_ids = _titled(_content(video), titles) if video and _content(video) else None
    swap = _swap(old, new, steps, names) if new else None
    if (clock and window is None) or video_ids == frozenset() or (new and swap is None):
        return None
    narrow = Narrow(window[0] if window else None, video_ids)
    return Followup(narrow, swap) if swap or narrow != Narrow() else None


def with_model(
    vlm: Vlm, instruction: str, steps: Sequence[str], names: Sequence[str], titles: Mapping[str, str]
) -> Followup | None:
    reply = vlm.ask([], prompt(instruction, steps, titles), 160)
    return read_answer(parse_json(reply.text), steps, names, titles)
