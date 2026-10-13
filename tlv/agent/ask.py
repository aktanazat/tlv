"""ask: a question about one video, or a stretch of it, answered by the vision-language model from
frames stamped with their times, the captions of the stretch, what is said in it and, in a short
stretch, the text on screen."""

from __future__ import annotations

import json
from contextlib import closing
from typing import Any

from tlv import store
from tlv.agent import sources
from tlv.agent.inspect import clock, inspection_window, lines_text, pick_frames, stamp
from tlv.agent.model import Vlm, parse_json
from tlv.agent.sources import Video, fit
from tlv.media import Range
from tlv.timecode import format_range, format_timecode

MIN_WINDOW_S = 4.0
SCREEN_TEXT_S = 60.0  # text on screen is read every second; over longer stretches the captions carry it
TEXT_CHARS = 6000  # speech and captions in one prompt, so a long video stays well inside the memory cap
ANSWER_TOKENS = 500


def ask(
    video_id: str,
    question: str,
    start: float | None = None,
    end: float | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answers a question about a video, or about start..end of it (widened to at least
    MIN_WINDOW_S). With a JSON schema the answer is a JSON value meant to match it."""
    with closing(store.connect()) as db:
        video = sources.video(db, video_id)
        windowed = start is not None or end is not None
        span = Range(0.0, video.duration)
        if windowed:
            asked = clamp(video, 0.0 if start is None else start, video.duration if end is None else end)
            span = inspection_window(asked, 0, video.duration, MIN_WINDOW_S)
        frames = pick_frames(sources.frames(db, video_id, span.start, span.end), span)
        captions = sources.captions(db, video_id, span.start, span.end)
        speech = sources.speech(db, video_id, span.start, span.end)
        text = sources.screen_text(db, video_id, span.start, span.end) if span.end - span.start <= SCREEN_TEXT_S else []
    stamped = stamp(frames, store.data_dir("frames") / "asked" / f"{video_id}_{span.start:.1f}-{span.end:.1f}")
    prompt = (
        f'These {len(frames)} frames come from the video "{video.title}" between {clock(span.start)} and '
        f"{clock(span.end)}, in time order; each shows its time (minutes:seconds) in the top-left corner.\n"
        f"{lines_text('Scenes', fit(captions, TEXT_CHARS // 2))}{lines_text('Speech', fit(speech, TEXT_CHARS // 2))}"
        f"{lines_text('On-screen text', text)}"
        f"Question: {question}\n"
        "Answer from what the frames, speech and text show, giving times as minutes:seconds. If they do not "
        "show the answer, say so."
    )
    if json_schema is not None:
        prompt += f"\nAnswer with JSON only, matching this JSON schema:\n{json.dumps(json_schema)}"
    with Vlm() as vlm:
        reply = vlm.ask(stamped, prompt, ANSWER_TOKENS)
    result: dict[str, Any] = {"video_id": video.video_id, "title": video.title}
    if windowed:
        result |= {
            "start": round(span.start, 2),
            "end": round(span.end, 2),
            "timecode": format_range(span.start, span.end),
        }
    result["truncated"] = reply.truncated
    if json_schema is None:
        return result | {"answer": reply.text.strip()}
    value = parse_json(reply.text)
    if value is None:
        cut_off = " (cut off at the token limit)" if reply.truncated else ""
        raise ValueError(f"The model returned invalid JSON{cut_off}: {reply.text[:300]}")
    return result | {"json": value}


def clamp(video: Video, start: float, end: float) -> Range:
    """The requested range inside the video; an empty range is an error."""
    clamped = Range(max(0.0, start), min(end, video.duration))
    if not clamped.end > clamped.start:
        raise ValueError(
            f'Empty range {format_range(start, end)} for "{video.title}" ({format_timecode(video.duration)} long)'
        )
    return clamped
