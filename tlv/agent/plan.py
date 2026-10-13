"""From a query to the steps to look for. A step is one thing someone does ("LeBron James
scores"); an ordered query has several, found in that order inside one video. Steps come from
the caller's `then`, else from the query itself, split by the model when it reads as a sequence.
@Name mentions become the names of known people, so every step is plain text. Each step is
perceived by one sense, read from its words, which decides where it is searched for and what
checks it."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from tlv.agent.model import Vlm, parse_json

# Words that make a query read as a sequence of steps rather than one moment.
_SEQUENCE = re.compile(r"\b(then|after|afterwards|before|followed|next|once|until|later)\b", re.I)
# Connectors that keep their steps in reading order; "after", "before" and "until" may not.
_FORWARD = re.compile(r"\s*[,;]?\s*\b(?:and\s+)?(?:then|afterwards|after that|followed by|next)\b\s*", re.I)
MAX_STEPS = 4
_WORD = r"[\w]"

Sense = Literal["sight", "speech", "text", "sound"]
# A step names its sense with its words: what someone says, what is written on screen, what is heard;
# anything else is seen. The first sense named wins, in this order.
_SENSES: tuple[tuple[Sense, re.Pattern[str]], ...] = (
    (
        "speech",
        re.compile(
            r"\b(?:says?|said|saying|talk(?:s|ed|ing)?(?:\s+about)?|mention(?:s|ed|ing)?|discuss(?:es|ed|ing)?|"
            r"speak(?:s|ing)?|spoke|commentators?|announcers?|broadcasters?|narrators?)\b",
            re.I,
        ),
    ),
    ("text", re.compile(r"\b(?:on[- ]screen|graphics?|captions?|banners?|scoreboards?|score ?bugs?|text)\b", re.I)),
    ("sound", re.compile(r"\b(?:buzzers?|horns?|whistles?|sirens?|bells?|sounds?|hears?|heard|noises?)\b", re.I)),
)
# Words that ask about speech or hold a sentence together, never what is said: "when did they talk
# about the summit" listens for "summit" alone.
_FUNCTION_WORDS = frozenset(
    (
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "every",
        "each",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "someone",
        "somebody",
        "anyone",
        "anybody",
        "everyone",
        "everybody",
        "something",
        "anything",
        "who",
        "whom",
        "whose",
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "whether",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "having",
        "will",
        "would",
        "can",
        "could",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "about",
        "of",
        "on",
        "in",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "into",
        "onto",
        "over",
        "as",
        "and",
        "or",
        "but",
        "if",
        "so",
        "than",
        "find",
        "show",
        "ever",
        "time",
        "times",
        "moment",
        "moments",
        "part",
        "parts",
    )
)


def sense(step: str) -> Sense:
    """The sense that perceives a step: the first of speech, on-screen text and sound whose words it
    uses, else sight."""
    return next((name for name, named in _SENSES if named.search(step)), "sight")


def topic(step: str) -> str:
    """What a spoken step says: the step without the words that make it speech, ask about it or
    hold the sentence together."""
    said = _SENSES[0][1].sub(" ", step).split()
    return " ".join(word for word in said if word.lower().strip(".,;:!?\"'()") not in _FUNCTION_WORDS)


@dataclass(frozen=True)
class Mentions:
    """A text with each @Name replaced by the name of the person it names."""

    plain: str
    unknown: tuple[str, ...]  # mentions that name nobody, such as "@Kobe"


def resolve_mentions(text: str, names: list[str]) -> Mentions:
    """Resolves @Name mentions against the names of known people, case-insensitively, with " ",
    "_" and "-" interchangeable. The longest matching name wins. An "@" inside a word (an email
    address) or right after "<" is left alone."""
    patterns = [
        (name, re.compile(re.sub(r"\\?[ _-]", "[ _-]", re.escape(name)) + rf"(?!{_WORD})", re.I))
        for name in sorted(names, key=lambda n: -len(n))
    ]
    plain, unknown = [], []
    i = 0
    while (at := text.find("@", i)) >= 0:
        plain.append(text[i:at])
        i = at + 1
        if at > 0 and (re.match(_WORD, text[at - 1]) or text[at - 1] == "<"):
            plain.append("@")
            continue
        rest = text[i:]
        hit = next(((name, m) for name, p in patterns if (m := p.match(rest))), None)
        if hit:
            name, match = hit
            plain.append(name)
            i += match.end()
            continue
        token = re.match(r"[^\s,.;:!?)\"']*", rest).group()
        if token:
            unknown.append(f"@{token}")
        plain.append(f"@{token}")
        i += len(token)
    plain.append(text[i:])
    return Mentions("".join(plain), tuple(unknown))


def reads_as_sequence(query: str) -> bool:
    return _SEQUENCE.search(query) is not None


def split_by_connectors(query: str) -> list[str]:
    """The query cut at "then", "followed by" and similar connectors; the whole query when it has
    none, or when cutting leaves an empty step."""
    parts = [p.strip(" ,;.") for p in _FORWARD.split(query)]
    return parts if all(parts) and len(parts) <= MAX_STEPS else [query]


def split_prompt(query: str) -> str:
    return (
        "Split this search query about a video into the steps it describes, in the order they happen in the "
        "video. Each step is a short phrase saying who does what; repeat the person in every step. A query "
        "about one moment is one step.\n"
        f"Query: {query}\n"
        'Answer with JSON only: {"steps": ["first step", "next step"]}'
    )


def read_split(value: object) -> list[str] | None:
    """The model's steps when they are 1 to MAX_STEPS non-empty phrases, else None."""
    steps = value.get("steps") if isinstance(value, dict) else None
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        return None
    if not all(isinstance(s, str) and s.strip() for s in steps):
        return None
    return [s.strip() for s in steps]


def split_with_model(vlm: Vlm, query: str) -> list[str]:
    reply = vlm.ask([], split_prompt(query), 120)
    return read_split(parse_json(reply.text)) or split_by_connectors(query)


def steps(query: str, then: list[str], split: Callable[[str], list[str]] | None) -> list[str]:
    """The ordered steps of a query: the query and `then` when the caller gave steps; else the
    query split by `split` (or by its connectors) when it reads as a sequence; else the query."""
    if then:
        return [query, *then]
    if not reads_as_sequence(query):
        return [query]
    return split(query) if split else split_by_connectors(query)
