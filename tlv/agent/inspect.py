"""Checking a candidate by looking at it. Frames from the index are shown to the
vision-language model at their true times, each stamped with its number, its time and the game
clock the score bug shows, together with the speech, on-screen text and sounds of the window. The
model names frames, never seconds; frame numbers become video times here, so a step lands where
its frames are. A spoken step is checked against the words spoken instead: the model cannot hear.
A person the claim names whom only sight knows is pointed out in the frames the index sees them in:
the index says who someone is, the model whether the claim happens."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tlv import entities, store
from tlv.agent import plan, retrieve, sources
from tlv.agent.connect import Ranged
from tlv.agent.model import Question, Reply, Vlm, parse_json
from tlv.agent.plan import Sense
from tlv.agent.replay import GameClock, join_pieces, period_ended, read_clock
from tlv.agent.sources import Frame, Line, Video
from tlv.media import Range

# 448 px wide: 384 cut a check's prompt from 1620 to 1284 tokens (3.6 -> 3.1 s) but the eval lost
# three events (recall 0.74 -> 0.61, precision 0.45 -> 0.36). 416 (91 tokens a frame, not 112) lost two of five
# plays checked one by one.
FRAME_WIDTH = 448
# Reading the prompt is most of a check, and each frame is 112 of its tokens. Seen claims get at most 10: on the
# eval's checks that showed 9% fewer frames than 12 and took 5% less model time, finding the same 18 of 24 events
# (local-frames10-2026-10-13.json). 9 and 11 each lost a play that 10 and 12 found; with 8 the eval lost a buzzer
# and a third of a second of placement. A claim with nothing to see (only heard or read) needs frames only to
# place it, so it gets the fewest.
MIN_FRAMES, MAX_FRAMES = 8, 10
FRAME_SPACING_S = 1.0  # about one frame per second of window, within the bounds above
MIN_SEEN_FRAMES = 2  # frames a claim to be seen must cover: one frame shows a pose, not a play
MIN_WINDOW_S = 6.0
VERDICT_TOKENS = 200
# Text on screen in one verdict prompt: arena ads read every second would otherwise outweigh the frames. Cutting it
# to 400 characters, or leaving it out for a seen claim, lost plays. A claim read on screen gets all of it: there
# the text is the evidence.
VERDICT_TEXT_CHARS = 800
SOUND_CHARS = 400  # sound labels in one verdict prompt, for a claim about something heard
PAIR_FRAMES = 4  # frames of each moment shown when asking whether two moments are one play
# Answers stop at their first words when those settle them: a verdict of no match, two moments that
# are not one play. Nothing after them is read.
NO_MATCH = r'"match"\s*:\s*false'
NOT_SAME = r'"same"\s*:\s*false'
OCR_SPACING_S = 1.0  # on-screen text is read once a second
# A step read on screen that names a known person stands on the frames whose text holds the name, and is
# placed where that text is on screen: shown a window that a graphic fills only in part, the model marks the
# whole window. The text is followed this far past the step's frames to find where it starts and ends.
READ_AROUND_S = 30.0
# The words of one spoken step count as said together when they are said within this many seconds, about a
# sentence: a topic said across the edge of the window is still heard there.
SAID_TOGETHER_S = 8.0
# A question about what was said asks for the talk about it, not each word: mentions of its topic are one
# discussion while the talk runs on between them, with no pause longer than LONG_PAUSE_S, and the next comes
# within MENTIONS_WITHIN_S. In the everyday eval's meeting, mentions of one topic came 15 to 27 s apart with
# pauses of 1.9 s at most between; sports commentary naming a coach about two plays 27 s apart paused 3.2 s.
LONG_PAUSE_S = 2.5
MENTIONS_WITHIN_S = 30.0
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
    people: list[str],
) -> str:
    """The question the model is asked about a window. `people` are lines from `recognized`: who the
    index sees in which frames, for the people the claim names whom only sight knows."""
    numbered = "\n".join(f"{i}. {step}{_HINTS[s]}" for i, (step, s) in enumerate(zip(steps, senses, strict=True), 1))
    template = ", ".join(f'"{i}": [first frame number, last frame number] or null' for i in range(1, len(steps) + 1))
    known = (
        "Who is who: the index recognizes these people by how they look, which the frames alone cannot tell you.\n"
        + "".join(f"- {line}\n" for line in people)
        + "That says who someone is, not what they do: whether the claim happens, judge from the frames.\n"
        if people
        else ""
    )
    return (
        f"These {len(frames)} frames come from one video, in time order. Each frame shows in its top-left corner "
        f"its number (1 to {len(frames)}), its time (minutes:seconds) and, when the score bug shows one, the game "
        "clock (quarter and time left).\n"
        f"{lines_text('Speech', speech)}{lines_text('On-screen text', text)}{lines_text('Sounds', heard)}{known}"
        f"Claim: {claim}\nSteps, in order:\n{numbered}\n"
        "Does the whole claim happen here, every step in this order? If the person, the action or the order is "
        "uncertain, it is not a match. For each step, give the numbers of the first and last frame showing it, "
        "or null if no frame shows it.\n"
        "Answer with one line of JSON only, no other words:\n"
        '{"match": true or false, "confidence": 0 to 1, "who": "jersey number and team of who does it", '
        '"evidence": "what decides it, at most eight words", '
        f'"replay": true if the footage is a replay, "steps": {{{template}}}}}'
    )


def recognized(db: sqlite3.Connection, video_id: str, claim: str, frames: list[Frame]) -> list[str]:
    """A line for each person the claim names whom only sight knows (`entities.sight_only`) and the
    index sees in the frames: the runs of frames showing them, each with where in the frame, as
    "coco: frames 5 to 6, face at 28-57% across and 32-70% down". No word names such a person, so the
    frames alone cannot tell the model who they are; the index knows them by how they look."""
    lines = []
    for entity_id, name in entities.sight_only(db, entities.resolve(db, claim)).items():
        seen = entities.sightings(db, video_id, entity_id, frames[0].t, frames[-1].t)
        runs: list[tuple[int, int, str]] = []
        for number, frame in enumerate(frames, 1):
            here = [s for s in seen if s[0] <= frame.t <= s[1]]
            if not here:
                continue
            _, _, kind, (x, y, w, h) = min(here, key=lambda s: s[2] != "face")  # a face over the body under it
            across, down = f"{x * 100:.0f}-{(x + w) * 100:.0f}% across", f"{y * 100:.0f}-{(y + h) * 100:.0f}% down"
            where = f"{kind} at {across} and {down}"
            if runs and runs[-1][1] == number - 1 and runs[-1][2] == where:
                runs[-1] = (runs[-1][0], number, where)
            else:
                runs.append((number, number, where))
        if runs:
            shown = (f"frame {a}, {where}" if a == b else f"frames {a} to {b}, {where}" for a, b, where in runs)
            lines.append(f"{name}: {'; '.join(shown)}")
    return lines


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


def joined(earlier: Moment, later: Moment) -> Moment:
    """One moment of two pieces of it: their whole range, each step from its first start to its last end,
    every frame either was shown, the game clock read in either, and the confidence, who and evidence of
    the more confident piece (the earlier on a tie)."""
    best = later if later.confidence > earlier.confidence else earlier
    steps = tuple(
        CheckedStep(a.step, min(a.start, b.start), max(a.end, b.end))
        for a, b in zip(earlier.steps, later.steps, strict=True)
    )
    shown = {frame.t: (frame, path) for frame, path in (*earlier.shown, *later.shown)}
    clocks = [c for c in (earlier.clock, later.clock) if c]
    clock = (
        GameClock(
            frozenset().union(*(c.periods for c in clocks)), min(c.low for c in clocks), max(c.high for c in clocks)
        )
        if clocks
        else None
    )
    return replace(
        best,
        start=min(earlier.start, later.start),
        end=max(earlier.end, later.end),
        clock=clock,
        steps=steps,
        shown=tuple(shown[t] for t in sorted(shown)),
    )


@dataclass(frozen=True)
class Rejection:
    video_id: str
    start: float
    end: float
    reason: str


@dataclass(frozen=True)
class Shown:
    """A stretch made ready to ask about: the frames and prompt the model is shown, and what judging
    its answer needs."""

    video: Video
    look: Ranged
    window: Range
    frames: list[Frame]
    stamped: list[Path]  # the frames as shown, numbered and timed
    prompt: str
    text: list[Line]  # on-screen text of the window, where the game clock is read
    spoken: dict[int, tuple[float, float]]  # when each spoken step is heard, by step index


def ask_verdicts(vlm: Vlm, shown: list[Shown], steps: list[str]) -> list[Verdict | str]:
    """The model's verdicts on the stretches, asked together. An answer that cannot be read is asked
    for once more, told what was wrong with it, before its candidate is given up."""
    verdicts = _verdicts(vlm, shown, [s.prompt for s in shown], steps)
    again = [i for i, verdict in enumerate(verdicts) if isinstance(verdict, str)]
    told = "Answer again with JSON only, in that form."
    prompts = [f"{shown[i].prompt}\nA previous answer could not be used: {verdicts[i]}. {told}" for i in again]
    for i, verdict in zip(again, _verdicts(vlm, [shown[i] for i in again], prompts, steps), strict=True):
        verdicts[i] = verdict
    return verdicts


def _verdicts(vlm: Vlm, shown: list[Shown], prompts: list[str], steps: list[str]) -> list[Verdict | str]:
    questions = [
        Question(s.stamped, prompt, VERDICT_TOKENS, NO_MATCH) for s, prompt in zip(shown, prompts, strict=True)
    ]
    return [_verdict(reply, steps, s) for s, reply in zip(shown, vlm.ask_all(questions), strict=True)]


def _verdict(reply: Reply, steps: list[str], shown: Shown) -> Verdict | str:
    if reply.stopped:  # it said no match; what it would have said next cannot change that
        return Verdict(False, 0.0, "", "", False, ())
    if reply.truncated:
        return "verdict cut off at the token limit"
    return read_verdict(parse_json(reply.text), steps, [f.t for f in shown.frames], shown.window)


def heard_at(words: list[Line], said: list[str], window: Range) -> tuple[float, float] | None:
    """The shortest run of spoken words, at most SAID_TOGETHER_S long and overlapping the window,
    holding a word that sounds like each word of `said`; or None when no such run is heard (or `said`
    is empty)."""
    wanted = set(said)
    hits = [
        (word, asked)
        for word in words
        if (asked := next((w for w in said if retrieve.sounds_like(word.text, w)), None)) is not None
    ]
    best: tuple[float, float] | None = None
    for i, (first, _) in enumerate(hits):
        found: set[str] = set()
        for last, asked in hits[i:]:
            if last.end - first.start > SAID_TOGETHER_S:
                break
            found.add(asked)
            if found == wanted and last.end > window.start:
                if first.start < window.end and (best is None or last.end - first.start < best[1] - best[0]):
                    best = (first.start, last.end)
                break
    return best


def _spoken(
    db: sqlite3.Connection, video: Video, window: Range, steps: list[str], senses: list[Sense]
) -> dict[int, tuple[float, float]] | str:
    """When each spoken step is heard in the window (its words said together, perhaps across the
    window's edge), by step index; or which one is not heard."""
    near = sources.spoken_words(db, video.video_id, window.start - SAID_TOGETHER_S, window.end + SAID_TOGETHER_S)
    spoken = {}
    for i, (step, sense) in enumerate(zip(steps, senses, strict=True)):
        if sense == "speech":
            at = heard_at(near, retrieve.said_words(plan.topic(step)), window)
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


def join_discussions(db: sqlite3.Connection, moments: list[Moment]) -> list[Moment]:
    """Joins the moments a spoken claim was heard into one moment for each stretch of talk."""
    return join_pieces(moments, joined, lambda earlier, later: _one_discussion(db, earlier, later))


def _one_discussion(db: sqlite3.Connection, earlier: Moment, later: Moment) -> bool:
    """Whether `later` is heard in the same stretch of talk as `earlier`: same video, at most
    MENTIONS_WITHIN_S after it, and no pause longer than LONG_PAUSE_S in the words said between."""
    if earlier.video_id != later.video_id or later.start - earlier.end > MENTIONS_WITHIN_S:
        return False
    between = sources.spoken_words(db, earlier.video_id, earlier.end, later.start)
    edges = [earlier.end, *(t for word in between for t in (word.start, word.end)), later.start]
    return all(resume - stop <= LONG_PAUSE_S for stop, resume in zip(edges[::2], edges[1::2], strict=True))


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


def read_on_screen(lines: list[Line], words: set[str], start: float, end: float) -> tuple[float, float] | None:
    """When text holding every one of `words` is on screen around start..end: the run of frames reading
    them that reaches into start..end, from half a reading before its first frame to half a reading after
    its last; None when no frame of start..end reads them. Each line holds until the next, as
    sources.screen_text leaves out a frame repeating the one before it, and one frame missing the words
    between two that read them (OCR misreads now and then) does not end a run."""
    runs: list[tuple[float, float]] = []
    last = None  # index of the last line that read the words
    for i, line in enumerate(lines):
        if not words <= set(retrieve.words(line.text)):
            continue
        until = lines[i + 1].start if i + 1 < len(lines) else line.start + OCR_SPACING_S
        misread = last == i - 2 and line.start - lines[i - 1].start < 2 * OCR_SPACING_S
        if runs and (last == i - 1 or misread):
            runs[-1] = (runs[-1][0], until)
        else:
            runs.append((line.start, until))
        last = i
    half = OCR_SPACING_S / 2
    reaching = [(a - half, b - half) for a, b in runs if a - half <= end and b - half >= start]
    return max(reaching, key=lambda run: min(run[1], end) - max(run[0], start), default=None)


def read_steps(
    db: sqlite3.Connection, video_id: str, steps: tuple[CheckedStep, ...], senses: list[Sense]
) -> tuple[CheckedStep, ...] | str:
    """The steps, each read on screen that names a known person placed where text holding the name is on
    screen, as `read_on_screen` finds it; or why one of them is not read there."""
    people = [(name, set(retrieve.words(name))) for name in sources.person_names(db)]
    placed = []
    for i, (step, sense) in enumerate(zip(steps, senses, strict=True)):
        said = set(retrieve.words(step.step))
        named = [(name, words) for name, words in people if words and words <= said]
        if sense != "text" or not named:
            placed.append(step)
            continue
        words = set().union(*(words for _, words in named))
        lines = sources.screen_text(db, video_id, step.start - READ_AROUND_S, step.end + READ_AROUND_S)
        at = read_on_screen(lines, words, step.start, step.end)
        if at is None:
            names = " and ".join(name for name, _ in named)
            return f"step {i + 1} is not read: no on-screen text in its frames names {names}"
        placed.append(CheckedStep(step.step, *at))
    return tuple(placed)


def unseen(db: sqlite3.Connection, video_id: str, claim: str, window: Range, start: float, end: float) -> str | None:
    """A person the claim names whom the index sees in the window but not from start to end, where the
    model placed the claim: then the model gave the claim to someone else. None when there is none; one
    the index sees nowhere in the window counts for nothing, as a player far from the camera often goes
    unrecognized."""
    named = entities.resolve(db, claim)
    if not named:
        return None
    around = {p["entity_id"]: p["name"] for p in entities.present(db, video_id, window.start, window.end)}
    there = {p["entity_id"] for p in entities.present(db, video_id, start, end)}
    return next((around[e] or e for e in named if e in around and e not in there), None)


def check(
    vlm: Vlm,
    db: sqlite3.Connection,
    looks: list[tuple[Video, Ranged]],
    claim: str,
    steps: list[str],
    senses: list[Sense],
    pad: float,
    min_confidence: float,
) -> list[Moment | Rejection]:
    """Shows the model each stretch's window, all in one batch, and judges its answers. A spoken step
    must be heard in the window, and takes the time it is heard whatever the model says of it; a step
    to be heard must have been heard or read by the index there, as `unheard` says, before the model
    is asked. A step read on screen that names a known person takes the time text holding the name is
    on screen, as `read_steps` says. A claim to be seen must cover MIN_SEEN_FRAMES shown frames, and a
    person it names whom the index sees in the window must be seen where it is placed, as `unseen`
    says. The model is told where the index sees a person the claim names whom only sight knows, as
    `recognized` says; that tells it who someone is, and the claim stands only on its own verdict."""
    ready = [show(db, video, look, claim, steps, senses, pad) for video, look in looks]
    verdicts = iter(ask_verdicts(vlm, [r for r in ready if isinstance(r, Shown)], steps))
    return [
        r if isinstance(r, Rejection) else _judged(db, r, next(verdicts), claim, senses, min_confidence) for r in ready
    ]


def show(
    db: sqlite3.Connection, video: Video, look: Ranged, claim: str, steps: list[str], senses: list[Sense], pad: float
) -> Shown | Rejection:
    """The stretch's window made ready to ask about, or why the model need not be asked."""
    window = inspection_window(look, pad, video.duration, MIN_WINDOW_S)
    spoken = _spoken(db, video, window, steps, senses)
    if isinstance(spoken, str):
        return Rejection(video.video_id, look.start, look.end, spoken)
    text = sources.screen_text(db, video.video_id, window.start, window.end)
    heard = sources.sounds(db, video.video_id, window.start, window.end) if "sound" in senses else []
    silent = unheard(steps, senses, heard, text)
    if silent is not None:
        reason = f"step {silent + 1} is not heard: no sound or game clock in the index shows it"
        return Rejection(video.video_id, look.start, look.end, reason)
    most = MAX_FRAMES if "sight" in senses else MIN_FRAMES
    frames = pick_frames(sources.frames(db, video.video_id, window.start, window.end), window, most)
    if len(frames) < 2:
        return Rejection(video.video_id, look.start, look.end, "the index has no frames in this window")
    stamped = stamp(frames, _folder(video, window), [_game_clock(db, video.video_id, f.t) for f in frames])
    speech = sources.speech(db, video.video_id, window.start, window.end)
    shown_text = text if "text" in senses else sources.fit(text, VERDICT_TEXT_CHARS)
    people = recognized(db, video.video_id, claim, frames)
    prompt = verdict_prompt(claim, steps, senses, frames, speech, shown_text, sources.fit(heard, SOUND_CHARS), people)
    return Shown(video, look, window, frames, stamped, prompt, text, spoken)


def _judged(
    db: sqlite3.Connection, shown: Shown, verdict: Verdict | str, claim: str, senses: list[Sense], min_confidence: float
) -> Moment | Rejection:
    video, look = shown.video, shown.look

    def reject(reason: str) -> Rejection:
        return Rejection(video.video_id, look.start, look.end, reason)

    if isinstance(verdict, str):
        return reject(verdict)
    spoken = shown.spoken
    seen = tuple(StepSeen(s.step, True, *spoken[i]) if i in spoken else s for i, s in enumerate(verdict.steps))
    judgement = judge(replace(verdict, steps=seen), shown.window, look, len(senses), min_confidence)
    if isinstance(judgement, Rejected):
        return reject(judgement.reason)
    placed = read_steps(db, video.video_id, judgement.steps, senses)
    if isinstance(placed, str):
        return reject(placed)
    start, end = judgement.start, judgement.end
    if placed != judgement.steps:
        start, end = min(s.start for s in placed), max(s.end for s in placed)
    if "sight" in senses:
        if sum(start <= f.t <= end for f in shown.frames) < MIN_SEEN_FRAMES:
            return reject(f"the claim covers fewer than {MIN_SEEN_FRAMES} of the frames shown")
        missing = unseen(db, video.video_id, claim, shown.window, start, end)
        if missing:
            return reject(f"{missing} is seen in the window but not where the claim is placed")
    return Moment(
        video,
        start,
        end,
        verdict.confidence,
        verdict.who,
        verdict.evidence,
        verdict.replay or sources.mostly_replay(db, video.video_id, start, end),
        read_clock([line for line in shown.text if start <= line.start <= end]),
        placed,
        tuple(zip(shown.frames, shown.stamped, strict=True)),
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
    reply = vlm.ask(stamped, prompt, 80, NOT_SAME)
    value = None if reply.stopped else parse_json(reply.text)
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
