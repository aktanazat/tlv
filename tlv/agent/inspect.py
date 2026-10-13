"""Checking a candidate by looking at it. Frames from the index are shown to the
vision-language model at their true times, each stamped with its number, its time and the game
clock the score bug shows, together with the speech, on-screen text and sounds of the window. The
model names frames, never seconds; frame numbers become video times here, so a step lands where
its frames are. A spoken step is checked against the words spoken instead: the model cannot hear."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tlv import store
from tlv.agent import plan, retrieve, sources
from tlv.agent.connect import Ranged
from tlv.agent.model import Reply, Vlm, parse_json
from tlv.agent.plan import Sense
from tlv.agent.replay import GameClock, period_ended, read_clock
from tlv.agent.sources import Frame, Line, Video
from tlv.media import Range

FRAME_WIDTH = 448
# A check costs 4.4-5.3 s with 12 frames (1600 prompt tokens). With 8 the eval lost a buzzer and a third of a
# second of placement (recall 0.61 -> 0.57, precision 0.42 -> 0.36), so seen claims keep 12. A claim with nothing
# to see (only heard or read) needs frames only to place it, so it gets the fewest.
MIN_FRAMES, MAX_FRAMES = 8, 12
FRAME_SPACING_S = 1.0  # about one frame per second of window, within the bounds above
MIN_WINDOW_S = 6.0
VERDICT_TOKENS = 200
# Text on screen in one verdict prompt: arena ads read every second would otherwise outweigh the frames.
# A claim read on screen gets all of it: there the text is the evidence.
VERDICT_TEXT_CHARS = 800
SOUND_CHARS = 400  # sound labels in one verdict prompt, for a claim about something heard
PAIR_FRAMES = 4  # frames of each moment shown when asking whether two moments are one play
OCR_SPACING_S = 1.0  # on-screen text is read once a second
# A time a verdict names instead of a frame number, as stamped on the frames: h:mm:ss.s or m:ss.s.
_STAMPED = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")


def clock(seconds: float) -> str:
    """Seconds as m:ss.s (h:mm:ss.s from an hour on): the time stamped on frames and lines."""
    tenths = round(max(0.0, seconds) * 10)
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    head = f"{hours}:{minutes:02d}" if hours else f"{minutes}"
    return f"{head}:{rest // 10:02d}.{rest % 10}"


def inspection_window(span: Ranged, pad: float, duration: float, min_length: float) -> Range:
    """The window to inspect around a candidate: padded, clamped to the video, and widened around
    its center to at least `min_length` seconds (or the whole video if shorter)."""
    if duration <= min_length:
        return Range(0.0, duration)
    start, end = max(0.0, span.start - pad), min(duration, span.end + pad)
    if end - start < min_length:
        center = (start + end) / 2
        start, end = center - min_length / 2, center + min_length / 2
        if start < 0:
            start, end = 0.0, end - start
        if end > duration:
            start, end = start - (end - duration), duration
    return Range(start, end)


def pick_frames(frames: list[Frame], window: Range, most: int = MAX_FRAMES) -> list[Frame]:
    """About one indexed frame per second of the window, MIN_FRAMES to `most` of them, each the frame
    nearest the middle of an equal slot of the window; in time order, none twice."""
    if not frames:
        return []
    count = min(most, max(MIN_FRAMES, round((window.end - window.start) / FRAME_SPACING_S)))
    slot = (window.end - window.start) / count
    picked = {}
    for i in range(count):
        target = window.start + (i + 0.5) * slot
        nearest = min(frames, key=lambda f: abs(f.t - target))
        picked[nearest.t] = nearest
    return [picked[t] for t in sorted(picked)]


def stamp(frames: list[Frame], folder: Path, readings: list[str] | None = None) -> list[Path]:
    """Copies of the frames FRAME_WIDTH wide with "<number>  <time>" drawn in the top-left corner,
    numbered from 1, and after it the frame's reading (its game clock) when one is given."""
    folder.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=22)
    paths = []
    for number, frame in enumerate(frames, 1):
        with Image.open(frame.path) as original:
            height = round(original.height * FRAME_WIDTH / original.width)
            image = original.convert("RGB").resize((FRAME_WIDTH, height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        reading = readings[number - 1] if readings else ""
        label = "  ".join(part for part in (str(number), clock(frame.t), reading) if part)
        *_, right, bottom = draw.textbbox((6, 3), label, font=font)
        draw.rectangle((0, 0, right + 6, bottom + 4), fill="black")
        draw.text((6, 3), label, fill="yellow", font=font)
        path = folder / f"{number:02d}.jpg"
        image.save(path, quality=90)
        paths.append(path)
    return paths


def frame_span(times: list[float], window: Range, first: int, last: int) -> tuple[float, float]:
    """The video time covered by frames `first` to `last` (numbered from 1) of frames shown at
    `times`. Each frame stands for the time nearer to it than to its neighbours; the window edges
    bound the first and the last frame."""
    start = window.start if first == 1 else (times[first - 2] + times[first - 1]) / 2
    end = window.end if last == len(times) else (times[last - 1] + times[last]) / 2
    return start, end


@dataclass(frozen=True)
class StepSeen:
    """One step as the model saw it, in video time."""

    step: str
    seen: bool
    start: float
    end: float


@dataclass(frozen=True)
class Verdict:
    match: bool
    confidence: float
    who: str
    evidence: str
    replay: bool
    steps: tuple[StepSeen, ...]


def frame_number(value: object, times: list[float], window: Range) -> int | None:
    """The frame a verdict names: by its number, in digits or as text, 0 read as frame 1 (models count
    from 0 too); or by the time stamped on it, which names the frame nearest that time. None for
    anything else, such as a number past the last frame or a time outside the window."""
    if isinstance(value, str):
        text = value.strip()
        stamped = _STAMPED.fullmatch(text)
        if stamped:
            hours, minutes, seconds = stamped.groups()
            t = int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)
            # Stamps show tenths, so a stamp read back may round to just outside the window.
            if not window.start - 0.1 <= t <= window.end + 0.1:
                return None
            return 1 + min(range(len(times)), key=lambda i: abs(times[i] - t))
        value = int(text) if text.isdigit() else None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is int and 0 <= value <= len(times):
        return max(1, value)
    return None


def _named_frames(value: object, times: list[float], window: Range) -> tuple[int, int] | None:
    """The first and last frame a step names: two frames in either order, or one frame for both."""
    items = value if isinstance(value, list) else [value]
    numbers = [frame_number(item, times, window) for item in items]
    if not 1 <= len(numbers) <= 2 or None in numbers:
        return None
    return min(numbers), max(numbers)


def read_verdict(value: object, steps: list[str], times: list[float], window: Range) -> Verdict | str:
    """The model's answer as a verdict in video time, or why it cannot be read.

    Expected: {"match": bool, "confidence": number, "who": str, "evidence": str, "replay": bool,
    "steps": {"1": [first frame, last frame] or null, "2": ...}}, one entry per step seen or not.
    Steps given as a list are read in order, and frames as `frame_number` reads them."""
    if not isinstance(value, dict):
        return "unreadable verdict"
    match, confidence, seen = value.get("match"), value.get("confidence"), value.get("steps")
    if isinstance(seen, list):
        seen = {str(n): frames for n, frames in enumerate(seen, 1)}
    if not isinstance(match, bool) or not isinstance(seen, dict):
        return "unreadable verdict"
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return "unreadable verdict"
    numbers = [str(n) for n in range(1, len(seen) + 1)]
    if set(seen) != set(numbers):
        return "unreadable verdict"
    checked = []
    for i, frames in enumerate(seen[n] for n in numbers):
        step = steps[i] if i < len(steps) else ""
        if frames is None:
            checked.append(StepSeen(step, False, 0.0, 0.0))
            continue
        named = _named_frames(frames, times, window)
        if named is None:
            return f"step {i + 1} names frames {frames}, not frame numbers from 1 to {len(times)}"
        start, end = frame_span(times, window, *named)
        checked.append(StepSeen(step, True, start, end))
    who, evidence = value.get("who"), value.get("evidence")
    return Verdict(
        match,
        float(confidence),
        who if isinstance(who, str) else "",
        evidence if isinstance(evidence, str) else "",
        value.get("replay") is True,
        tuple(checked),
    )


@dataclass(frozen=True)
class CheckedStep:
    step: str
    start: float
    end: float


@dataclass(frozen=True)
class Confirmed:
    start: float
    end: float
    steps: tuple[CheckedStep, ...]


@dataclass(frozen=True)
class Rejected:
    reason: str


def judge(
    verdict: Verdict, window: Range, candidate: Ranged, step_count: int, min_confidence: float
) -> Confirmed | Rejected:
    """Confirms a candidate only when the model matched the whole claim with enough confidence and
    saw every step, each starting no earlier than the one before. The confirmed range spans the
    steps when they all fall inside the window; otherwise the candidate range stands."""
    if not verdict.match:
        return Rejected("no match")
    if not 0 <= verdict.confidence <= 1:
        return Rejected(f"confidence {verdict.confidence} outside 0..1")
    if verdict.confidence < min_confidence:
        return Rejected(f"confidence {verdict.confidence} below {min_confidence}")
    steps = verdict.steps
    if len(steps) != step_count:
        return Rejected(f"expected {step_count} steps, got {len(steps)}")
    unseen = next((i for i, s in enumerate(steps) if not s.seen), None)
    if unseen is not None:
        return Rejected(f"step {unseen + 1} not seen")
    early = next((i for i in range(1, len(steps)) if steps[i].start < steps[i - 1].start), None)
    if early is not None:
        return Rejected(f"step {early + 1} starts before step {early}")
    epsilon = 1e-6
    inside = all(window.start - epsilon <= s.start <= s.end <= window.end + epsilon for s in steps)
    start, end = min(s.start for s in steps), max(s.end for s in steps)
    if not (inside and end > start):
        start, end = candidate.start, candidate.end
    clamped = tuple(
        CheckedStep(s.step, min(window.end, max(window.start, s.start)), min(window.end, max(window.start, s.end)))
        for s in steps
    )
    return Confirmed(start, end, clamped)


def lines_text(title: str, lines: list[Line]) -> str:
    if not lines:
        return ""
    return "\n".join([f"{title}:", *(f"[{clock(line.start)}] {line.text}" for line in lines)]) + "\n"


# How the model is told to judge a step of each sense.
_HINTS: dict[Sense, str] = {
    "sight": "",
    "speech": " (spoken: judge it by the speech)",
    "text": " (written on screen: judge it by the on-screen text and the frames)",
    "sound": " (heard: judge it by the sounds and by the game clock stamped on the frames)",
}


def verdict_prompt(
    claim: str,
    steps: list[str],
    senses: list[Sense],
    frames: list[Frame],
    speech: list[Line],
    text: list[Line],
    heard: list[Line],
) -> str:
    numbered = "\n".join(f"{i}. {step}{_HINTS[s]}" for i, (step, s) in enumerate(zip(steps, senses, strict=True), 1))
    template = ", ".join(f'"{i}": [first frame number, last frame number] or null' for i in range(1, len(steps) + 1))
    return (
        f"These {len(frames)} frames come from one video, in time order. Each frame shows in its top-left corner "
        f"its number (1 to {len(frames)}), its time (minutes:seconds) and, when the score bug shows one, the game "
        "clock (quarter and time left).\n"
        f"{lines_text('Speech', speech)}{lines_text('On-screen text', text)}{lines_text('Sounds', heard)}"
        f"Claim: {claim}\nSteps, in order:\n{numbered}\n"
        "Does the whole claim happen here, every step in this order? If the person, the action or the order is "
        "uncertain, it is not a match. For each step, give the numbers of the first and last frame showing it, "
        "or null if no frame shows it.\n"
        "Answer with one line of JSON only, no other words:\n"
        '{"match": true or false, "confidence": 0 to 1, "who": "jersey number and team of who does it", '
        '"evidence": "what decides it, at most eight words", '
        f'"replay": true if the footage is a replay, "steps": {{{template}}}}}'
    )


@dataclass(frozen=True)
class Moment:
    """A confirmed candidate: where the claim happens, and what was looked at to decide it."""

    video: Video
    start: float
    end: float
    confidence: float
    who: str
    summary: str
    replay: bool  # the model or the captioner calls the footage a replay
    clock: GameClock | None  # the game clock on screen during the moment
    steps: tuple[CheckedStep, ...]
    shown: tuple[tuple[Frame, Path], ...]  # each frame shown, with its stamped copy

    @property
    def video_id(self) -> str:
        return self.video.video_id


@dataclass(frozen=True)
class Rejection:
    video_id: str
    start: float
    end: float
    reason: str


def ask_verdict(
    vlm: Vlm, images: list[Path], prompt: str, steps: list[str], times: list[float], window: Range
) -> Verdict | str:
    """The model's verdict. An answer that cannot be read is asked for once more, told what was wrong
    with it, before the candidate is given up."""
    verdict = _verdict(vlm.ask(images, prompt, VERDICT_TOKENS), steps, times, window)
    if isinstance(verdict, Verdict):
        return verdict
    again = f"{prompt}\nA previous answer could not be used: {verdict}. Answer again with JSON only, in that form."
    return _verdict(vlm.ask(images, again, VERDICT_TOKENS), steps, times, window)


def _verdict(reply: Reply, steps: list[str], times: list[float], window: Range) -> Verdict | str:
    if reply.truncated:
        return "verdict cut off at the token limit"
    return read_verdict(parse_json(reply.text), steps, times, window)


def heard_at(words: list[Line], keys: list[str]) -> tuple[float, float] | None:
    """The shortest run of spoken words holding a word that sounds like each key, or None when some
    key is not heard (or there are none)."""
    wanted = set(keys)
    hits = [(word, key) for word in words if (key := retrieve.sound_key(word.text)) in wanted]
    best: tuple[float, float] | None = None
    for i, (first, _) in enumerate(hits):
        found: set[str] = set()
        for last, key in hits[i:]:
            found.add(key)
            if found == wanted:
                if best is None or last.end - first.start < best[1] - best[0]:
                    best = (first.start, last.end)
                break
    return best


def _spoken(
    db: sqlite3.Connection, video: Video, window: Range, steps: list[str], senses: list[Sense]
) -> dict[int, tuple[float, float]] | str:
    """When each spoken step is heard in the window, by step index; or which one is not heard."""
    words = sources.spoken_words(db, video.video_id, window.start, window.end)
    spoken = {}
    for i, (step, sense) in enumerate(zip(steps, senses, strict=True)):
        if sense == "speech":
            at = heard_at(words, retrieve.sound_keys(plan.topic(step)))
            if at is None:
                return f"step {i + 1} is not heard in the speech"
            spoken[i] = at
    return spoken


def hear(db: sqlite3.Connection, video: Video, look: Ranged, steps: list[str], pad: float) -> Moment | Rejection:
    """Checks a claim made only of spoken steps against the words spoken around the stretch: each
    step's words heard, matched by how they sound, each step starting no earlier than the one before.
    No model looks, as there is nothing to see."""
    window = inspection_window(look, pad, video.duration, MIN_WINDOW_S)
    spoken = _spoken(db, video, window, steps, ["speech"] * len(steps))
    if isinstance(spoken, str):
        return Rejection(video.video_id, look.start, look.end, spoken)
    times = [spoken[i] for i in range(len(steps))]
    early = next((i for i in range(1, len(times)) if times[i][0] < times[i - 1][0]), None)
    if early is not None:
        return Rejection(video.video_id, look.start, look.end, f"step {early + 1} is heard before step {early}")
    start, end = times[0][0], max(e for _, e in times)
    said = " ".join(w.text for w in sources.spoken_words(db, video.video_id, start, end))
    return Moment(
        video,
        start,
        end,
        1.0,
        "",
        f'heard "{said}"',
        False,
        None,
        tuple(CheckedStep(step, a, b) for step, (a, b) in zip(steps, times, strict=True)),
        tuple((frame, frame.path) for frame in sources.frames(db, video.video_id, start, end)),
    )


def _game_clock(db: sqlite3.Connection, video_id: str, t: float) -> str:
    """The game clock the score bug shows at a frame, as "clock Q4 0:34.0"; empty when none is read."""
    reading = read_clock(sources.screen_text(db, video_id, t - OCR_SPACING_S / 2, t + OCR_SPACING_S / 2))
    if reading is None:
        return ""
    periods = "/".join(f"Q{p}" if p.isdigit() else p for p in sorted(reading.periods))
    return f"clock {periods} {clock(reading.low)}"


def unheard(steps: list[str], senses: list[Sense], heard: list[Line], text: list[Line]) -> int | None:
    """The first step to be heard that the index neither heard nor read in the window, by index; None
    when there is none. The model cannot hear, so such a step stands only on a sound label sharing a
    word with it, or on the game clock reaching 0.0, which the horn ending a period goes with."""
    known = [line.text for line in heard] + ([retrieve.PERIOD_END] if period_ended(text) else [])
    found = set(retrieve.words(" ".join(known)))
    pairs = enumerate(zip(steps, senses, strict=True))
    return next((i for i, (step, s) in pairs if s == "sound" and not found & set(retrieve.words(step))), None)


def check(
    vlm: Vlm,
    db: sqlite3.Connection,
    video: Video,
    look: Ranged,
    claim: str,
    steps: list[str],
    senses: list[Sense],
    pad: float,
    min_confidence: float,
) -> Moment | Rejection:
    """Shows the model the stretch's window and judges its answer. A spoken step must be heard in the
    window, and takes the time it is heard whatever the model says of it; a step to be heard must have
    been heard or read by the index there, as `unheard` says, before the model is asked."""

    def reject(reason: str) -> Rejection:
        return Rejection(video.video_id, look.start, look.end, reason)

    window = inspection_window(look, pad, video.duration, MIN_WINDOW_S)
    spoken = _spoken(db, video, window, steps, senses)
    if isinstance(spoken, str):
        return reject(spoken)
    text = sources.screen_text(db, video.video_id, window.start, window.end)
    heard = sources.sounds(db, video.video_id, window.start, window.end) if "sound" in senses else []
    silent = unheard(steps, senses, heard, text)
    if silent is not None:
        return reject(f"step {silent + 1} is not heard: no sound or game clock in the index shows it")
    most = MAX_FRAMES if "sight" in senses else MIN_FRAMES
    frames = pick_frames(sources.frames(db, video.video_id, window.start, window.end), window, most)
    if len(frames) < 2:
        return reject("the index has no frames in this window")
    stamped = stamp(frames, _folder(video, window), [_game_clock(db, video.video_id, f.t) for f in frames])
    speech = sources.speech(db, video.video_id, window.start, window.end)
    shown_text = text if "text" in senses else sources.fit(text, VERDICT_TEXT_CHARS)
    prompt = verdict_prompt(claim, steps, senses, frames, speech, shown_text, sources.fit(heard, SOUND_CHARS))
    verdict = ask_verdict(vlm, stamped, prompt, steps, [f.t for f in frames], window)
    if isinstance(verdict, str):
        return reject(verdict)
    seen = tuple(StepSeen(s.step, True, *spoken[i]) if i in spoken else s for i, s in enumerate(verdict.steps))
    judgement = judge(replace(verdict, steps=seen), window, look, len(steps), min_confidence)
    if isinstance(judgement, Rejected):
        return reject(judgement.reason)
    start, end = judgement.start, judgement.end
    return Moment(
        video,
        start,
        end,
        verdict.confidence,
        verdict.who,
        verdict.evidence,
        verdict.replay or sources.mostly_replay(db, video.video_id, start, end),
        read_clock([line for line in text if start <= line.start <= end]),
        judgement.steps,
        tuple(zip(frames, stamped, strict=True)),
    )


def _folder(video: Video, window: Range) -> Path:
    return store.data_dir("frames") / "checked" / f"{video.video_id}_{window.start:.1f}-{window.end:.1f}"


def same_play(vlm: Vlm, kept: Moment, moment: Moment) -> str | None:
    """Asks the model whether two confirmed moments show the same play, one of them a replay or
    another angle of the other. Returns its reason when it says so, else None."""
    a, b = _spread(kept), _spread(moment)
    folder = _folder(kept.video, Range(kept.start, kept.end)) / f"vs_{moment.video_id}_{moment.start:.1f}"
    stamped = stamp([*a, *b], folder)
    prompt = (
        f"Frames 1 to {len(a)} are clip A and frames {len(a) + 1} to {len(a) + len(b)} are clip B; each frame "
        "shows its number and time in the top-left corner.\n"
        "Do A and B show the same moment, the same play, with one of them a replay, slow motion or another "
        "camera angle of the other? Two different plays that look alike, such as one player scoring twice, are "
        "not the same moment.\n"
        'Answer with JSON only: {"same": true or false, "reason": "one short sentence"}'
    )
    value = parse_json(vlm.ask(stamped, prompt, 80).text)
    if not isinstance(value, dict) or value.get("same") is not True:
        return None
    reason = value.get("reason")
    return f"the model compared both: {reason}" if isinstance(reason, str) and reason else "the model compared both"


def _spread(moment: Moment) -> list[Frame]:
    """PAIR_FRAMES of the frames shown inside a moment, spread over its range."""
    inside = [frame for frame, _ in moment.shown if moment.start <= frame.t <= moment.end]
    if len(inside) <= PAIR_FRAMES:
        return inside
    return [inside[round(i * (len(inside) - 1) / (PAIR_FRAMES - 1))] for i in range(PAIR_FRAMES)]
