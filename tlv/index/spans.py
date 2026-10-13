"""Spans, the retrieval unit: a few seconds inside one shot, or a run of short shots, each with a
structured caption from the vision-language model, a text embedding for search, and a caption
observation tied to its time range and evidence.

The model sees up to FRAMES_PER_SPAN frames spread over a span, less those repeating the picture
and on-screen text of the frame shown before them, so a still slide or terminal costs it one
picture. A caption is kept across reruns while its span and everything the model saw (frames,
speech, on-screen text, prompt, model) are unchanged. Spans go to the model BATCH to a job, and a
job's captions are saved as soon as it returns, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from tlv import embed, guard, store
from tlv.index import frames, vision
from tlv.models import vlm

VERSION = 1
MIN_S = 2.0
MAX_S = 8.0
FRAMES_PER_SPAN = 4
# A picked frame showing what the frame shown before it did is left out. Two pictures match at a
# SigLIP cosine of SAME_PICTURE or more. In the everyday set a chart coming up on a slide reads
# 0.985 to the picture before it, and a talking head two seconds on mostly reads under 0.995 (the
# meeting keeps 90% of its pictures; 63% at 0.99), while a lecture slide under its moving webcam
# inset reads over it in half the picked frames.
SAME_PICTURE = 0.995
# SigLIP sees a terminal before and after a new line as one picture (cosine 1.0), so the text on
# screen must hold too: no read from one picture to the other may gain or lose NEW_WORDS words
# against the first. OCR on a still slide gains or loses one now and then, a formula read as
# "wme"; a new line, bullet or command brings two or more ("cargo build").
NEW_WORDS = 2
MAX_TOKENS = 400
# Spans per model job. The model writes a job's answers together, taking up the next span as an
# answer ends, so a span costs less model time than in a job of its own; eight keep a job under
# half a minute, the longest another process then waits for the shared model.
BATCH = 8
OCR_MIN_CONF = 0.5
OCR_LINES = 12
# Unverified: a caption is the model's first look, not a checked find.
CAPTION_CONFIDENCE = 0.5
SHOTS = ("live", "replay", "crowd", "bench", "interview", "studio", "graphic", "other")
NAME_SOURCES = ("on_screen", "speech")
JERSEY = re.compile(r"#?\s*(\d{1,2})")
WORD = re.compile(r"[a-z0-9]+")
# A word on screen, for telling whether the text changed: shorter runs are mostly OCR specks.
SCREEN_WORD = re.compile(r"[a-z0-9]{3,}")
# A string field of an answer cut off before its JSON closed.
FIELD = re.compile(r'"(caption|action|shot)"\s*:\s*"((?:[^"\\]|\\.)*)"')

# Scores and clocks stay out of captions: the model misreads them, and the ocr rows hold them
# exactly. People are named only from text or speech: in a pilot on the 76ers video, about half
# the names the model recognized by sight were wrong ("Anthony Davis" for LeBron's #23).
PROMPT = (
    "These are {count} frames from a few seconds of a video, in time order.{speech}{text}\n"
    "Reply with JSON only, in this shape:\n"
    '{{"caption": "...", "action": "...", "shot": "...", "people": [{{"description": "...", "jersey": null, '
    '"team": null, "name": null, "name_source": null}}]}}\n'
    "- caption: one or two plain sentences on what happens: who does what, and how it ends. Leave out the score "
    "and the clock. Name a person only when the name is on screen or in the speech; otherwise describe them by "
    "team, jersey number and look.\n"
    '- action: the main action in a few words, such as "makes a three-pointer", "dunks", "celebrates", '
    '"answers a reporter", or null.\n'
    '- shot: "live" for action as it happens, "replay" for earlier action shown again (slow motion, another '
    'camera angle, a replay graphic), "crowd", "bench", "interview", "studio", "graphic", or "other".\n'
    "- people: the people who matter, main subject first. jersey: the number on the jersey as digits, when "
    "readable. team: only when the team name is readable on the uniform or on screen. name: only when it is on "
    'screen ("on_screen") or in the speech ("speech"); otherwise name and name_source are null.'
)


@dataclass(frozen=True)
class Inputs:
    """What the captioner sees for one span."""

    frames: list[str]
    transcript: str
    ocr: list[str]
    sounds: list[str]


@dataclass(frozen=True)
class Span:
    """One span: its id and range, what the captioner sees, the prompt, and the digest of both."""

    span_id: str
    start: float
    end: float
    inputs: Inputs
    prompt: str
    digest: str


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None, recaption: bool = False) -> None:
    """Builds the spans, captions the ones without a current caption (all of them with
    `recaption`), then embeds every span and writes one caption observation each."""
    video_id = video["video_id"]
    shots = [
        (row["start"], row["end"])
        for row in db.execute("SELECT start, end FROM shots WHERE video_id = ? ORDER BY idx", (video_id,))
    ]
    reads = vision.text_times(video["duration"])
    spans = []
    for start, end in build(shots):
        seen = gather(db, video_id, start, end, reads)
        text = prompt(seen)
        spans.append(Span(f"{video_id}:{start:.3f}", start, end, seen, text, digest(start, end, seen, text)))
    todo = _replace(db, video_id, spans, recaption)
    if todo:
        _caption(db, todo, progress)
    captions = {
        row["span_id"]: (row["caption"], json.loads(row["details"]))
        for row in db.execute("SELECT span_id, caption, details FROM spans WHERE video_id = ?", (video_id,))
    }
    vectors = embed.sentences([document(*captions[span.span_id], span.inputs) for span in spans], progress=progress)
    now = store.now()
    with store.transaction(db):
        db.executemany(
            "UPDATE spans SET embedding = ? WHERE span_id = ?",
            [(store.to_blob(vector), span.span_id) for span, vector in zip(spans, vectors, strict=True)],
        )
        db.execute("DELETE FROM observations WHERE video_id = ? AND kind = 'caption'", (video_id,))
        db.executemany(
            "INSERT INTO observations (obs_id, video_id, start, end, kind, text, confidence, evidence, created_at)"
            " VALUES (?, ?, ?, ?, 'caption', ?, ?, ?, ?)",
            [
                (
                    f"caption:{span.span_id}",
                    video_id,
                    span.start,
                    span.end,
                    captions[span.span_id][0],
                    CAPTION_CONFIDENCE,
                    json.dumps(asdict(span.inputs)),
                    now,
                )
                for span in spans
            ],
        )


def _replace(db: sqlite3.Connection, video_id: str, spans: list[Span], recaption: bool) -> list[Span]:
    """Writes the video's spans, keeping each caption whose span and inputs are unchanged, and
    returns the spans left to caption."""
    old = {row["span_id"]: row for row in db.execute("SELECT * FROM spans WHERE video_id = ?", (video_id,))}
    rows = []
    todo = []
    for span in spans:
        kept = old.get(span.span_id)
        if recaption or not kept or kept["caption"] is None or json.loads(kept["details"])["digest"] != span.digest:
            rows.append((span.span_id, video_id, span.start, span.end, None, json.dumps({"digest": span.digest}), None))
            todo.append(span)
        else:
            rows.append(
                (span.span_id, video_id, span.start, span.end, kept["caption"], kept["details"], kept["replay"])
            )
    with store.transaction(db):
        db.execute("DELETE FROM spans WHERE video_id = ?", (video_id,))
        db.executemany(
            "INSERT INTO spans (span_id, video_id, start, end, caption, details, replay) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return todo


def build(shots: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Spans over consecutive shots. A shot of MIN_S to MAX_S seconds is one span, and a longer
    one splits into equal parts of at most MAX_S. Back-to-back shorter shots group until a group
    reaches MIN_S, and a remainder joins the run's last group. A short shot between two long ones
    stays its own span: a span never mixes a long shot with others."""
    spans: list[tuple[float, float]] = []
    run: list[tuple[float, float]] = []
    for start, end in shots:
        if end - start < MIN_S:
            run.append((start, end))
            continue
        spans += _group(run)
        run = []
        parts = math.ceil((end - start) / MAX_S)
        step = (end - start) / parts
        spans += [(start + i * step, end if i == parts - 1 else start + (i + 1) * step) for i in range(parts)]
    return spans + _group(run)


def _group(run: list[tuple[float, float]]) -> list[tuple[float, float]]:
    groups: list[tuple[float, float]] = []
    opened: float | None = None
    for start, end in run:
        opened = start if opened is None else opened
        if end - opened >= MIN_S:
            groups.append((opened, end))
            opened = None
    if opened is not None:
        if groups:
            groups[-1] = (groups[-1][0], run[-1][1])
        else:
            groups.append((opened, run[-1][1]))
    return groups


def gather(db: sqlite3.Connection, video_id: str, start: float, end: float, reads: list[float]) -> Inputs:
    """The frames, words, on-screen text and sounds inside [start, end): FRAMES_PER_SPAN frames
    spread over the span, less each repeating the frame shown before it. `reads` are the times
    the video's text is read at, vision.text_times."""
    picked = db.execute(
        "SELECT t, path, embedding FROM frames WHERE video_id = ? AND t >= ? AND t < ? ORDER BY t",
        (video_id, start, end),
    ).fetchall()
    if len(picked) > FRAMES_PER_SPAN:
        picked = [picked[(2 * i + 1) * len(picked) // (2 * FRAMES_PER_SPAN)] for i in range(FRAMES_PER_SPAN)]
    shown = picked[:1]
    for frame in picked[1:]:
        if not _repeats(db, video_id, reads, shown[-1], frame):
            shown.append(frame)
    words = [
        word["w"]
        for row in db.execute(
            "SELECT words FROM speech WHERE video_id = ? AND start < ? AND end > ? ORDER BY start",
            (video_id, end, start),
        )
        for word in json.loads(row["words"])
        if start <= (word["s"] + word["e"]) / 2 < end
    ]
    # The lines seen most often first: a score bug outlasts the passing ads around it.
    seen: dict[str, int] = {}
    for row in db.execute(
        "SELECT text FROM ocr WHERE video_id = ? AND t >= ? AND t < ? AND conf >= ? ORDER BY t",
        (video_id, start, end, OCR_MIN_CONF),
    ):
        seen[row["text"]] = seen.get(row["text"], 0) + 1
    ocr = sorted(seen, key=lambda text: -seen[text])[:OCR_LINES]
    sounds = [
        row["label"]
        for row in db.execute(
            "SELECT label, max(conf) AS conf FROM sounds WHERE video_id = ? AND start < ? AND end > ?"
            " GROUP BY label ORDER BY conf DESC",
            (video_id, end, start),
        )
    ]
    return Inputs([frame["path"] for frame in shown], " ".join(words), ocr, sounds)


def _repeats(db: sqlite3.Connection, video_id: str, reads: list[float], shown: sqlite3.Row, frame: sqlite3.Row) -> bool:
    """Whether `frame` shows what `shown`, a frame before it, did: their SigLIP vectors within
    SAME_PICTURE, and no read of the text on screen from one picture to the other gaining or losing
    NEW_WORDS words against the first."""
    if shown["embedding"] is None or frame["embedding"] is None:
        return False
    if float(store.from_blob(shown["embedding"]) @ store.from_blob(frame["embedding"])) < SAME_PICTURE:
        return False
    # A frame is the picture as it stands just before half a step past its time (ffmpeg's fps
    # filter gives each step the last frame rounding to it), and a read is the picture at its
    # time, so the reads from the last one at or before the first picture to the first one at or
    # after the second hold every change between them. A read with no text is an empty screen.
    step = frames.EVERY_S * vision.TEXT_EVERY
    after, before = shown["t"] + frames.EVERY_S / 2 - step, frame["t"] + frames.EVERY_S / 2 + step
    window = reads[bisect_right(reads, after) : bisect_left(reads, before)]
    words: dict[float, Counter[str]] = {t: Counter() for t in window}
    for row in db.execute(
        "SELECT t, text FROM ocr WHERE video_id = ? AND t > ? AND t < ? AND conf >= ?",
        (video_id, after, before, OCR_MIN_CONF),
    ):
        words.setdefault(row["t"], Counter()).update(SCREEN_WORD.findall(row["text"].lower()))
    read = [words[t] for t in sorted(words)]
    return all((later - read[0]).total() < NEW_WORDS and (read[0] - later).total() < NEW_WORDS for later in read[1:])


def prompt(seen: Inputs) -> str:
    speech = f'\nSpeech: "{seen.transcript}"' if seen.transcript else ""
    text = f"\nOn-screen text: {' | '.join(seen.ocr)}" if seen.ocr else ""
    return PROMPT.format(count=len(seen.frames), speech=speech, text=text)


def digest(start: float, end: float, seen: Inputs, text: str) -> str:
    """Names everything a caption depends on, so a changed input means a new caption."""
    key = json.dumps([start, end, seen.frames, text, frames.VERSION, vlm.MODEL, MAX_TOKENS])
    return hashlib.sha1(key.encode()).hexdigest()


def parse(answer: str) -> dict[str, Any]:
    """The captioner's answer as {"caption", "action", "shot", "people"}: the first JSON object in
    it with a caption, else the string fields of one cut off by the token limit, else the answer
    itself as the caption. An unknown shot kind reads as "other", a jersey as its digits, and a
    name is kept only with a source the prompt allows."""
    found = _first_caption_object(answer) or _fields(answer)
    caption = _string(found.get("caption")) or answer.strip()
    shot = _string(found.get("shot"))
    shot = shot.lower() if shot else "other"
    people = [_person(person) for person in found.get("people") or [] if isinstance(person, dict)]
    return {
        "caption": caption,
        "action": _string(found.get("action")),
        "shot": shot if shot in SHOTS else "other",
        "people": people,
    }


def ground(details: dict[str, Any], seen: Inputs) -> dict[str, Any]:
    """Holds the people of a parsed answer to the span's evidence: a name stays only when a word of
    it is in the on-screen text or the speech, which becomes its source (the model's claim first),
    and a team only when a word of it is on screen."""
    evidence = {"on_screen": _words(" ".join(seen.ocr)), "speech": _words(seen.transcript)}
    return {**details, "people": [_grounded(person, evidence) for person in details["people"]]}


def document(caption: str, details: dict[str, Any], seen: Inputs) -> str:
    """The text a span is searched by: its caption, who is in it, what is said and shown."""
    people = "; ".join(
        ", ".join(
            str(value)
            for value in (
                person["name"],
                person["description"],
                person["jersey"] and f"#{person['jersey']}",
                person["team"],
            )
            if value
        )
        for person in details.get("people", [])
    )
    lines = [caption, f"People: {people}" if people else "", f"Speech: {seen.transcript}" if seen.transcript else ""]
    lines.append(f"On screen: {' | '.join(seen.ocr)}" if seen.ocr else "")
    return "\n".join(line for line in lines if line)


def _caption(db: sqlite3.Connection, todo: list[Span], progress: guard.Progress | None) -> None:
    """Captions the spans with one loaded model, BATCH to a job, saving a job's captions as soon
    as it returns."""
    with guard.ModelProcess("tlv.models.vlm", progress=progress) as model:
        for first in range(0, len(todo), BATCH):
            batch = todo[first : first + BATCH]
            questions = [
                {"images": span.inputs.frames, "prompt": span.prompt, "max_tokens": MAX_TOKENS} for span in batch
            ]
            rows = []
            for span, answer in zip(batch, model.call({"batch": questions})["results"], strict=True):
                details = ground(parse(answer["text"]), span.inputs)
                replay = {"replay": 1, "live": 0}.get(details["shot"])
                rows.append((details["caption"], json.dumps({**details, "digest": span.digest}), replay, span.span_id))
            with store.transaction(db):
                db.executemany("UPDATE spans SET caption = ?, details = ?, replay = ? WHERE span_id = ?", rows)
            if progress:
                progress(f"captioned {first + len(batch)} of {len(todo)} spans")


def _first_caption_object(answer: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start in (i for i, char in enumerate(answer) if char == "{"):
        try:
            found, _ = decoder.raw_decode(answer, start)
        except json.JSONDecodeError:
            continue
        if isinstance(found, dict) and "caption" in found:
            return found
    return {}


def _fields(answer: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name, value in FIELD.findall(answer):
        found.setdefault(name, json.loads(f'"{value}"'))
    return found


def _person(found: dict[str, Any]) -> dict[str, Any]:
    jersey = JERSEY.fullmatch(str(found.get("jersey") or "").strip())
    name = _string(found.get("name"))
    source = found.get("name_source")
    named = name is not None and name not in NAME_SOURCES and source in NAME_SOURCES
    return {
        "description": _string(found.get("description")),
        "jersey": jersey.group(1) if jersey else None,
        "team": _string(found.get("team")),
        "name": name if named else None,
        "name_source": source if named else None,
    }


def _grounded(person: dict[str, Any], evidence: dict[str, set[str]]) -> dict[str, Any]:
    name, team = person["name"], person["team"]
    claims = (person["name_source"], *NAME_SOURCES)
    source = next((claim for claim in claims if name and _words(name) & evidence[claim]), None)
    return {
        **person,
        "name": name if source else None,
        "name_source": source,
        "team": team if team and _words(team) & evidence["on_screen"] else None,
    }


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _string(value: Any) -> str | None:
    """A non-empty string as itself, stripped; anything else, or "null", as None."""
    if not isinstance(value, str) or value.strip().lower() in ("", "null", "none"):
        return None
    return value.strip()
