"""Finding the spans that may match a text, from the channels the index has: what the frames
look like (SigLIP), what the captions mean (a text embedding), the words in captions and on-screen
text, the words spoken (matched by how they sound), the sounds heard, and the people seen. Each
step searches the channels of its sense. Each channel ranks spans on its own, so no channel's scores
need to be comparable with another's; a span comes as high as its best rank in any channel."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from bisect import bisect_left
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from tlv import embed, entities, store
from tlv.agent.connect import Clip
from tlv.agent.plan import Sense
from tlv.agent.replay import period_ended
from tlv.agent.sources import Line

RRF_K = 60  # reciprocal-rank weight of agreement between channels, which breaks ties of best rank
DEPTH = 50  # spans each channel brings to the fusion
MODALITIES = ("visual", "audio", "transcription")
# Channels each modality searches.
CHANNELS = {
    "visual": ("frames", "captions", "screen words", "people"),
    "audio": ("sounds", "speech words"),
    "transcription": ("speech words",),
}
# Channels each sense of a step searches. Sight: measured on "lebron james scores, then lebron james
# celebrates" over three games, fusing people (presence ties at rank 1 for every span he fills), speech
# and sounds pushed two of the four true plays past the first 12 candidates; frames, captions and screen
# words alone bring all four into the first 6.
SENSE_CHANNELS: dict[Sense, tuple[str, ...]] = {
    "sight": ("frames", "captions", "screen words"),
    "text": ("screen words", "frames", "captions"),
    "speech": ("speech words",),
    "sound": ("sounds", "screen words", "captions"),
}
# A game clock read at 0.0 means a period ended, which a score bug shows only as numbers; a span showing
# it is also found by the words for that.
PERIOD_END = "end of the period quarter half game buzzer horn clock expires"
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "her",
        "his",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "then",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "while",
        "with",
    ]
)


def words(text: str) -> list[str]:
    """Lowercase word stems without stop words: plural s, then ing, ion or ed, then a final e."""
    stems = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOP:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        for suffix in ("ing", "ion", "ed"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        if len(word) > 3 and word.endswith("e"):
            word = word[:-1]
        stems.append(word)
    return stems


# Letters that sound alike share a digit (American Soundex); vowels, h, w and y have none.
_SOUNDEX = {
    letter: digit for digit, group in enumerate(("bfpv", "cgjkqsxz", "dt", "l", "mn", "r"), 1) for letter in group
}


def sound_key(word: str) -> str:
    """The American Soundex key of a word: its first letter and three digits for the consonant
    sounds after it ("Jordi" and "Jordy" are both J630). The transcriber spells names it does not
    know by ear, so spoken words are matched by key."""
    letters = [c for c in word.lower() if "a" <= c <= "z"]
    if not letters:
        return ""
    digits, previous = [], _SOUNDEX.get(letters[0])
    for letter in letters[1:]:
        digit = _SOUNDEX.get(letter)
        if digit and digit != previous:
            digits.append(str(digit))
        if letter not in "hw":
            previous = digit
    return (letters[0] + "".join(digits) + "000")[:4]


def sound_keys(text: str) -> list[str]:
    """The sound key of each word of a text that is not a stop word."""
    return [key for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOP and (key := sound_key(word))]


def channels(senses: list[Sense], modalities: list[str] | None) -> frozenset[str]:
    """The channels a query searches: those of the modalities the caller named, else those of the
    senses of its steps."""
    if modalities is not None:
        return frozenset(c for m in modalities for c in CHANNELS[m])
    return frozenset(c for s in senses for c in SENSE_CHANNELS[s])


@dataclass(frozen=True)
class Span:
    span_id: str
    video_id: str
    start: float
    end: float
    caption: str
    replay: bool | None


@dataclass
class Corpus:
    """What retrieval reads of the chosen videos, loaded once per query."""

    spans: list[Span]
    span_vectors: np.ndarray  # caption text embeddings, one row per span
    frame_vectors: np.ndarray  # SigLIP rows of the frames inside spans, grouped by span
    frame_bounds: list[tuple[int, int]]  # each span's rows in frame_vectors
    screen_words: list[Counter[str]]  # caption, captioner details and on-screen text per span
    speech_keys: list[Counter[str]]  # sound keys of the words spoken per span
    sounds: list[list[tuple[set[str], float]]]  # per span: label stems and weight (confidence x overlap)


def load(db: sqlite3.Connection, video_ids: list[str]) -> Corpus:
    spans, span_blobs = [], []
    frame_rows, bounds, screen, speech, sounds = [], [], [], [], []
    for video_id in video_ids:
        rows = db.execute(
            "SELECT span_id, start, end, caption, details, replay, embedding FROM spans WHERE video_id = ? "
            "ORDER BY start",
            (video_id,),
        ).fetchall()
        frames = db.execute(
            "SELECT t, embedding FROM frames WHERE video_id = ? AND embedding IS NOT NULL ORDER BY t", (video_id,)
        ).fetchall()
        texts = db.execute("SELECT t, text FROM ocr WHERE video_id = ? ORDER BY t, y, x", (video_id,)).fetchall()
        segments = db.execute("SELECT words FROM speech WHERE video_id = ?", (video_id,)).fetchall()
        spoken = sorted((w for row in segments for w in json.loads(row["words"])), key=lambda w: w["s"] + w["e"])
        heard = db.execute("SELECT start, end, label, conf FROM sounds WHERE video_id = ?", (video_id,)).fetchall()
        frame_times = np.array([f["t"] for f in frames])
        text_times = [t["t"] for t in texts]
        word_times = [(w["s"] + w["e"]) / 2 for w in spoken]
        for row in rows:
            start, end = row["start"], row["end"]
            replay = None if row["replay"] is None else bool(row["replay"])
            spans.append(Span(row["span_id"], video_id, start, end, row["caption"] or "", replay))
            span_blobs.append(row["embedding"])
            first, last = np.searchsorted(frame_times, [start, end])
            bounds.append((len(frame_rows), len(frame_rows) + int(last - first)))
            frame_rows.extend(f["embedding"] for f in frames[first:last])
            # A span the captioner has not reached yet has no details; it is found by its frames and words.
            details = _strings({k: v for k, v in json.loads(row["details"] or "{}").items() if k != "digest"})
            on_screen = texts[bisect_left(text_times, start) : bisect_left(text_times, end)]
            read = [t["text"] for t in on_screen] + ([PERIOD_END] if _period_ends(on_screen) else [])
            screen.append(Counter(words(" ".join([row["caption"] or "", *details, *read]))))
            said = spoken[bisect_left(word_times, start) : bisect_left(word_times, end)]
            speech.append(Counter(sound_keys(" ".join(w["w"] for w in said))))
            sounds.append(
                [
                    (set(words(h["label"].replace("_", " "))), h["conf"] * _overlap(h, start, end))
                    for h in heard
                    if h["start"] < end and h["end"] > start
                ]
            )
    has_vectors = all(span_blobs) and span_blobs
    return Corpus(
        spans,
        store.stack(span_blobs) if has_vectors else np.zeros((len(spans), 0), dtype=np.float32),
        store.stack(frame_rows),
        bounds,
        screen,
        speech,
        sounds,
    )


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _overlap(event: sqlite3.Row, start: float, end: float) -> float:
    return (min(event["end"], end) - max(event["start"], start)) / (end - start)


def _period_ends(rows: list[sqlite3.Row]) -> bool:
    """Whether a period ends in the frames of these OCR rows, given in reading order."""
    by_time: dict[float, list[str]] = {}
    for row in rows:
        by_time.setdefault(row["t"], []).append(row["text"])
    return period_ended([Line(t, t, " | ".join(parts)) for t, parts in by_time.items()])


def competition_ranks(scores: np.ndarray) -> list[tuple[int, int]]:
    """(span index, rank) of the DEPTH best positive scores; equal scores share a rank, the next
    one skipping as many ranks as shared it (1, 2, 2, 4)."""
    order = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:DEPTH]
    ranks, rank = [], 0
    for position, i in enumerate(order):
        if position == 0 or scores[i] < scores[order[position - 1]]:
            rank = position + 1
        ranks.append((i, rank))
    return ranks


def fuse(rankings: Iterable[list[tuple[int, int]]]) -> list[int]:
    """Span indices by their best rank in any channel, best first; among equal best ranks, the span
    more channels agree on (reciprocal-rank sum) first. Spans no channel ranked are left out.

    A channel's top spans are not outvoted by spans every channel ranks middling: the captions
    misname actions the frames show plainly (SigLIP put the true dunks of "lebron james dunks" at
    ranks 1 and 2, captions at 27 and 52), and every candidate is checked by looking at it anyway.
    On the eval's 21 events of sight and sound queries, 17 reach the 12 looks this way against 14
    with reciprocal-rank fusion alone."""
    best: dict[int, int] = {}
    agreement: dict[int, float] = {}
    for ranking in rankings:
        for i, rank in ranking:
            best[i] = min(best.get(i, rank), rank)
            agreement[i] = agreement.get(i, 0.0) + 1 / (RRF_K + rank)
    return sorted(best, key=lambda i: (best[i], -agreement[i], i))


def bm25(documents: list[Counter[str]], query: list[str]) -> np.ndarray:
    """Okapi BM25 (k1 1.2, b 0.75) of each document for the query stems."""
    k1, b = 1.2, 0.75
    count = len(documents)
    lengths = np.array([sum(d.values()) for d in documents], dtype=np.float64)
    average = lengths.mean() if count and lengths.mean() > 0 else 1.0
    scores = np.zeros(count)
    for term in set(query):
        frequency = np.array([d[term] for d in documents], dtype=np.float64)
        holding = int((frequency > 0).sum())
        if holding == 0:
            continue
        idf = math.log(1 + (count - holding + 0.5) / (holding + 0.5))
        scores += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * lengths / average))
    return scores


@dataclass(frozen=True)
class Query:
    """One text to rank spans for, the people it names and the channels that search for it."""

    text: str
    entity_ids: tuple[str, ...]
    channels: frozenset[str]


def rank(db: sqlite3.Connection, corpus: Corpus, queries: list[Query]) -> list[list[Clip]]:
    """Clips (spans) for each query, best first, ranked 1, 2, 3... by the fused ranks of its channels.
    An embedder is loaded only when some query searches its channel."""
    wanted = frozenset().union(*(q.channels for q in queries))
    texts = [q.text for q in queries]
    seen = "frames" in wanted and len(corpus.frame_vectors) > 0
    read = "captions" in wanted and corpus.span_vectors.size > 0
    # One load serves both embedders when both channels are searched; each load costs 3-4 s.
    if seen and read:
        looks, means = embed.queries(texts)
    else:
        looks = embed.image_queries(texts) if seen else None
        means = embed.sentences(texts, query=True) if read else None
    people = _people(db, corpus) if "people" in wanted and any(q.entity_ids for q in queries) else None
    results = []
    for i, query in enumerate(queries):
        rankings = []
        if looks is not None and "frames" in query.channels:
            rankings.append(competition_ranks(_frame_scores(corpus, looks[i])))
        if means is not None and "captions" in query.channels:
            rankings.append(competition_ranks(corpus.span_vectors @ means[i]))
        stems = words(query.text)
        if "screen words" in query.channels:
            rankings.append(competition_ranks(bm25(corpus.screen_words, stems)))
        if "speech words" in query.channels:
            rankings.append(competition_ranks(bm25(corpus.speech_keys, sound_keys(query.text))))
        if "sounds" in query.channels:
            named = set(stems)
            heard = [sum(weight for labels, weight in span if labels & named) for span in corpus.sounds]
            rankings.append(competition_ranks(np.array(heard)))
        if people is not None and "people" in query.channels and query.entity_ids:
            seen = [max((share.get(e, 0.0) for e in query.entity_ids), default=0.0) for share in people]
            rankings.append(competition_ranks(np.array(seen)))
        order = fuse(rankings)
        results.append(
            [
                Clip(corpus.spans[s].video_id, corpus.spans[s].start, corpus.spans[s].end, r)
                for r, s in enumerate(order, 1)
            ]
        )
    return results


def _frame_scores(corpus: Corpus, vector: np.ndarray) -> np.ndarray:
    """Each span's best frame match: a moment shows in some frames of a span, not all."""
    similarity = corpus.frame_vectors @ vector
    return np.array([similarity[a:b].max() if b > a else 0.0 for a, b in corpus.frame_bounds])


def _people(db: sqlite3.Connection, corpus: Corpus) -> list[dict[str, float]]:
    """For each span, the share of its time each person is seen."""
    shares = []
    for span in corpus.spans:
        length = span.end - span.start
        seen = {}
        for person in entities.present(db, span.video_id, span.start, span.end):
            inside = sum(
                min(e, span.end) - max(s, span.start) for s, e in person["times"] if e > span.start and s < span.end
            )
            seen[person["entity_id"]] = max(seen.get(person["entity_id"], 0.0), inside / length)
        shares.append(seen)
    return shares
