"""answer: a question about the whole collection, answered from what the index holds and cited to
clips the model has looked at. The sources are the moments that best match the question and an
outline of each video. The model writes the answer citing them as [n]; each cited source is then
checked against its own frames, and a sentence whose sources all fail that check is left out."""

from __future__ import annotations

import math
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from tlv import entities, guard, store
from tlv.agent import plan, retrieve, sources
from tlv.agent.connect import rank_clips
from tlv.agent.find import plain_texts
from tlv.agent.inspect import clock, inspection_window, lines_text, pick_frames, stamp
from tlv.agent.model import Vlm, parse_json
from tlv.agent.sources import Video
from tlv.timecode import format_range, format_timecode

MOMENTS = 8  # best matching spans offered as sources
OUTLINE = 30  # captions offered as an outline of the videos, spread over them
HISTORY = 3  # earlier questions and answers kept in view
SOURCE_CHARS = 300  # of each source's caption, speech and on-screen text
ANSWER_TOKENS = 400
CHECK_WINDOW_S = 6.0
_MARKER = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Source:
    video: Video
    start: float
    end: float
    text: str  # what the index says happens in it


def answer(
    question: str,
    video_ids: list[str] | None = None,
    progress: guard.Progress | None = None,
    history: list[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Answers a question about the indexed videos, citing the clips that show it. `history` holds
    the earlier questions and answers of the conversation, oldest first."""
    say = progress or (lambda message: None)
    began = time.monotonic()
    with closing(store.connect()) as db:
        videos = sources.videos(db, video_ids)
        if not videos:
            raise LookupError("No videos in this index. Add one with video_add.")
        [plain] = plain_texts(db, [question])
        say(f"searching {len(videos)} videos")
        corpus = retrieve.load(db, [v.video_id for v in videos])
        question = retrieve.Query(
            plain, tuple(entities.resolve(db, plain)), retrieve.channels([plan.sense(plain)], None)
        )
        [clips] = retrieve.rank(db, corpus, [question])
        by_id = {v.video_id: v for v in videos}
        offered = [_source(db, by_id[c.video_id], c.start, c.end) for c in rank_clips(clips, MOMENTS)]
        per_video = max(1, OUTLINE // len(videos))
        for video in videos:
            spans = sources.captions(db, video.video_id, 0.0, video.duration)
            offered += [_source(db, video, s.start, s.end) for s in spans[:: max(1, math.ceil(len(spans) / per_video))]]
        with Vlm(progress) as vlm:
            say("writing the answer")
            reply = vlm.ask([], compose_prompt(plain, videos, offered, history), ANSWER_TOKENS)
            draft = reply.text.strip()
            cited = sorted({int(n) for n in _MARKER.findall(draft) if 1 <= int(n) <= len(offered)})
            held = set()
            for n in cited:
                say(f"checking source [{n}] against its frames")
                if _holds(vlm, db, offered[n - 1], _citing(draft, n)):
                    held.add(n)
    kept = keep_supported(draft, held)
    say(f"answered in {time.monotonic() - began:.0f} s, {vlm.seconds:.0f} s of it generating")
    result: dict[str, Any] = {
        "text": kept.text or "The clips checked do not show an answer.",
        "status": "completed",
        "citations": [_citation(f"[{i}]", offered[n - 1]) for i, n in enumerate(kept.order, 1)],
    }
    if reply.truncated:
        result |= {"status": "incomplete", "incomplete_reason": "max_output_tokens"}
    elif kept.left_out:
        reason = f"left out {kept.left_out} statements their clips do not show"
        result |= {"status": "incomplete", "incomplete_reason": reason}
    return result


def _source(db: sqlite3.Connection, video: Video, start: float, end: float) -> Source:
    caption = " ".join(line.text for line in sources.captions(db, video.video_id, start, end))
    said = " ".join(line.text for line in sources.speech(db, video.video_id, start, end))
    shown = " | ".join(line.text for line in sources.screen_text(db, video.video_id, start, end)[:2])
    parts = [
        caption[:SOURCE_CHARS],
        f'Said: "{said[:SOURCE_CHARS]}"' if said else "",
        f"On screen: {shown}"[:SOURCE_CHARS] if shown else "",
    ]
    return Source(video, start, end, " ".join(part for part in parts if part))


def compose_prompt(question: str, videos: list[Video], offered: list[Source], history: list[dict[str, Any]]) -> str:
    listed = "\n".join(f'- "{v.title}" ({format_timecode(v.duration)})' for v in videos)
    numbered = "\n".join(
        f'[{n}] "{s.video.title}" {clock(s.start)}-{clock(s.end)}: {s.text}' for n, s in enumerate(offered, 1)
    )
    earlier = "".join(f"Q: {h['question']}\nA: {_MARKER.sub('', h['text'])}\n" for h in history[-HISTORY:])
    return (
        f"Videos:\n{listed}\nSources, numbered:\n{numbered}\n"
        f"{f'Earlier in this conversation:{chr(10)}{earlier}' if earlier else ''}"
        f"Question: {question}\n"
        "Answer in a few sentences using only the sources. After each fact, put the number of the source that "
        "shows it in brackets, like [2]. Give times as minutes:seconds. If the sources do not answer the question, "
        "say what is missing."
    )


def _citing(draft: str, n: int) -> str:
    """The sentences of the draft that cite source n, with their markers taken out."""
    citing = [s for s in _SENTENCE.split(draft) if f"[{n}]" in s]
    return " ".join(_MARKER.sub("", s).strip() for s in citing)


def _holds(vlm: Vlm, db: sqlite3.Connection, source: Source, statement: str) -> bool:
    """Whether the model, looking at the source's frames, speech and text, finds the statement shown."""
    video = source.video
    window = inspection_window(source, 1.0, video.duration, CHECK_WINDOW_S)
    frames = pick_frames(sources.frames(db, video.video_id, window.start, window.end), window)
    if not frames:
        return False
    stamped = stamp(
        frames, store.data_dir("frames") / "answered" / f"{video.video_id}_{window.start:.1f}-{window.end:.1f}"
    )
    speech = sources.speech(db, video.video_id, window.start, window.end)
    text = sources.screen_text(db, video.video_id, window.start, window.end)
    prompt = (
        f"These {len(frames)} frames come from one video, in time order; each shows its number and its time in the "
        "top-left corner.\n"
        f"{lines_text('Speech', speech)}{lines_text('On-screen text', text)}"
        f"Statement: {statement}\n"
        "Does this clip show what the statement says about it? Parts about other clips do not count against it.\n"
        'Answer with JSON only: {"shown": true or false}'
    )
    value = parse_json(vlm.ask(stamped, prompt, 30).text)
    return isinstance(value, dict) and value.get("shown") is True


@dataclass(frozen=True)
class Kept:
    text: str
    order: list[int]  # the draft's numbers of the sources kept, in their new order
    left_out: int  # sentences left out


def keep_supported(draft: str, held: set[int]) -> Kept:
    """The draft without the sentences whose sources all failed their check, with failed or unknown
    markers taken out of the rest and the kept sources renumbered [1], [2]... by first citation."""
    sentences = _SENTENCE.split(draft)
    kept = [s for s in sentences if not (numbers := {int(n) for n in _MARKER.findall(s)}) or numbers & held]
    text = " ".join(kept)
    order = list(dict.fromkeys(int(n) for n in _MARKER.findall(text) if int(n) in held))
    renumbered = {n: i for i, n in enumerate(order, 1)}
    text = _MARKER.sub(lambda m: f"[{renumbered[int(m[1])]}]" if int(m[1]) in renumbered else "", text)
    tidy = re.sub(r" +([.,;:!?])", r"\1", re.sub(r" {2,}", " ", text)).strip()
    return Kept(tidy, order, len(sentences) - len(kept))


def _citation(marker: str, source: Source) -> dict[str, Any]:
    return {
        "marker": marker,
        "type": "clip",
        "video_id": source.video.video_id,
        "title": source.video.title,
        "start": round(source.start, 2),
        "end": round(source.end, 2),
        "timecode": format_range(source.start, source.end),
    }
