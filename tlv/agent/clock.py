"""The game clock across a whole video. OCR reads the screen once a second and `replay.read_clock`
reads the period and the time left from one frame's text; together those readings say where a
video shows a moment of game time. Two questions are answered from them, with no model: which
frames show a given time (the live play and every replay that keeps the scoreboard on screen),
and which stretches of video lie inside a window of game time, the ranges `find --clock` and
`search --clock` narrow to before anything is ranked or checked."""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass

from tlv.agent import sources
from tlv.agent.replay import CLOCK_TOLERANCE_S, read_clock
from tlv.agent.sources import Line, Video

QUARTERS = ("1", "2", "3", "4")
OVERTIMES = ("OT", "2OT", "3OT", "4OT")
# NBA scoreboards show the shot clock in tenths under five seconds, the way the game clock shows
# time under a minute, and OCR can put the shot clock nearer the period than the game clock.
SHOT_CLOCK_TENTHS_S = 5.0
# A run of readings in a window reaches past its first and last reading by up to this many seconds
# (a replay or close-up hiding the scoreboard), never past halfway to a reading outside the window.
REACH_S = 15.0

_ORDINALS = {"1st": "1", "first": "1", "2nd": "2", "second": "2", "3rd": "3", "third": "3", "4th": "4", "fourth": "4"}
_NAMES = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th"}
_COUNTS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
_PERIOD_NOUNS = ("quarter", "qtr", "period")
_TIME = r"\d{1,2}:[0-5]\d(?:\.\d)?|:[0-5]?\d(?:\.\d)?|\d{1,2}\.\d"
# A game time anywhere in a frame's text, in the shapes `read_clock` reads ("4:25", "22.7",
# ":12.3"), but not a time of day ("7:30PM ET").
_SHOWN = re.compile(r"(?<![\d:.])\d{1,2}:[0-5]\d(?:\.\d)?(?![\d:.%]|\s?[AP]M)|(?<![\d.])\d{1,2}\.\d(?![\d:.%])", re.I)
_LAST_WORDS = r"(?:last|final)\s+(?:(\d+(?:\.\d+)?|a|one|two|three|four|five|ten)\s*)?(min(?:ute)?s?|sec(?:ond)?s?)\b"
_LAST = re.compile(_LAST_WORDS)
_RANGE = re.compile(rf"(?<![\w:.])({_TIME})\s*(?:-|to)\s*({_TIME})(?![\w:.])")
_ONE_TIME = re.compile(rf"(?<![\w:.])({_TIME})(?![\w:.])")
# One word or time of a window inside a longer instruction. An ordinal naming a clip or a video
# ("the second one", "the first three") is not a period.
_TERM = (
    r"(?:q[1-4]|[1-4]q|(?:1st|2nd|3rd|4th|first|second|third|fourth)"
    r"(?!\s+(?:ones?|clips?|results?|videos?|games?|plays?|two|three|four|five|few|couple)\b)"
    rf"|\d?ot\d?|overtime|half|quarters?|qtr|period|{_LAST_WORDS}|{_TIME})"
)
_PHRASE = re.compile(rf"(?<![\w:.]){_TERM}(?:(?:\s*[,/-]\s*|\s+(?:and|to)\s+|\s+){_TERM})*(?![\w:.])")


@dataclass(frozen=True)
class Reading:
    """The scoreboard in one frame: the period ("1" to "4", "OT", "2OT"...) and the seconds left."""

    t: float
    period: str
    seconds: float


@dataclass(frozen=True)
class Window:
    """A stretch of game time: some periods, with `low` to `high` seconds left in each."""

    periods: frozenset[str]
    low: float = 0.0
    high: float = math.inf

    @classmethod
    def parse(cls, text: str) -> Window:
        """Reads "4th", "Q4", "2nd half", "OT", "3rd/4th", a time ("4th 1:12", "Q4 :42.1") or a range
        ("4th 2:00-1:00", either way round), and "last 2 min" (the 4th and any overtime). A time or
        range without a period is in every period."""
        words = " ".join(text.lower().split())
        low, high, unnamed = 0.0, math.inf, frozenset((*QUARTERS, *OVERTIMES))
        if last := _LAST.search(words):
            amount, unit = last.groups()
            low, high = 0.0, float(_COUNTS.get(amount, amount or 1)) * (60 if unit.startswith("min") else 1)
            unnamed = frozenset(("4", *OVERTIMES))
            words = words[: last.start()] + " " + words[last.end() :]
        elif span := _RANGE.search(words):
            low, high = sorted((_seconds(span[1]), _seconds(span[2])))
            words = words[: span.start()] + " " + words[span.end() :]
        elif one := _ONE_TIME.search(words):
            low = high = _seconds(one[1])
            words = words[: one.start()] + " " + words[one.end() :]
        periods = _periods(words, text)
        if not periods and (low, high) == (0.0, math.inf):
            raise ValueError(_unreadable(text))
        return cls(periods or unnamed, low, high)

    def holds(self, reading: Reading) -> bool:
        """Whether a reading is in the window, to the precision OCR reads a clock."""
        inside = self.low - CLOCK_TOLERANCE_S <= reading.seconds <= self.high + CLOCK_TOLERANCE_S
        return reading.period in self.periods and inside

    def narrowed_to(self, other: Window) -> Window:
        """The game time in both windows; none at all is an error."""
        periods, low, high = self.periods & other.periods, max(self.low, other.low), min(self.high, other.high)
        if not periods or low > high:
            raise ValueError(f"no game time is both {self} and {other}")
        return Window(periods, low, high)

    def __str__(self) -> str:
        names = [_NAMES[p] for p in QUARTERS if p in self.periods]
        if self.periods >= set(OVERTIMES):
            names.append("OT")
        else:
            names += ["1OT" if p == "OT" else p for p in OVERTIMES if p in self.periods]
        if (self.low, self.high) == (0.0, math.inf):
            return "/".join(names)
        when = clock_text(self.low) if self.low == self.high else f"{clock_text(self.high)}-{clock_text(self.low)}"
        return f"{'/'.join(names)} {when}"


def mentioned(text: str) -> tuple[Window, str] | None:
    """The window of game time named inside a longer instruction ("now only the fourth quarter"),
    with the words that name it; None when it names none."""
    for phrase in _PHRASE.finditer(text.lower()):
        try:
            return Window.parse(phrase[0]), phrase[0]
        except ValueError:
            continue
    return None


def _seconds(text: str) -> float:
    minutes, _, seconds = text.rpartition(":")
    return int(minutes or 0) * 60 + float(seconds)


def _unreadable(text: str) -> str:
    return (
        f'"{text}" is not a game clock: name a period (1st to 4th, Q1 to Q4, OT, 2nd half) with an optional time '
        'or range ("4th 1:12", "Q4 2:00-0:00"), or say "last 2 min"'
    )


def _periods(words: str, text: str) -> frozenset[str]:
    tokens = [token for token in re.split(r"[\s,/&+]+|\band\b", words) if token]
    periods: set[str] = set()
    i = 0
    while i < len(tokens):
        token, following = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""
        quarter = _ORDINALS.get(token)
        if numbered := re.fullmatch(r"q([1-4])|([1-4])q", token):
            quarter = numbered[1] or numbered[2]
        overtime = re.fullmatch(r"(\d)?ot(\d)?|overtime", token)
        if quarter in ("1", "2") and token in _ORDINALS and following == "half":
            periods |= {"1", "2"} if quarter == "1" else {"3", "4"}
            i += 2
        elif quarter:
            periods.add(quarter)
            i += 2 if following in _PERIOD_NOUNS else 1
        elif overtime and (number := overtime[1] or overtime[2]):
            periods.add("OT" if number == "1" else f"{number}OT")
            i += 1
        elif overtime:
            periods |= set(OVERTIMES)
            i += 1
        else:
            raise ValueError(_unreadable(text))
    return frozenset(periods)


def clock_text(seconds: float) -> str:
    """Time left as a scoreboard shows it: "1:12", "0:42.1"."""
    if seconds == int(seconds):
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def reading_text(reading: Reading) -> str:
    return f"{_NAMES.get(reading.period, reading.period)} {clock_text(reading.seconds)}"


def _period(name: str) -> str:
    """A period as `read_clock` names it, with every way of writing an overtime ("OT", "1OT",
    "OT2", "2OT") as "OT" for the first and "2OT" onward for the rest."""
    if name in QUARTERS:
        return name
    number = re.sub(r"\D", "", name) or "1"
    return "OT" if number == "1" else f"{number}OT"


def reading(line: Line) -> Reading | None:
    """The scoreboard in one frame's text; None when it shows none or two that disagree. A time
    under SHOT_CLOCK_TENTHS_S counts only as the one game time in the frame: beside another
    ("10:43 ... 4TH 4.8", "22.7 ... 3RD 4.1") it is the shot clock."""
    clock = read_clock([line])
    if clock is None or len(clock.periods) != 1 or clock.high - clock.low > CLOCK_TOLERANCE_S:
        return None
    if clock.low < SHOT_CLOCK_TENTHS_S and any(
        abs(_seconds(shown) - clock.low) > CLOCK_TOLERANCE_S for shown in _SHOWN.findall(line.text)
    ):
        return None
    (period,) = clock.periods
    return Reading(line.start, _period(period), clock.low)


def readings(db: sqlite3.Connection, video_id: str) -> list[Reading]:
    """Every scoreboard reading of a video, in time order."""
    lines = sources.screen_text(db, video_id, 0.0, math.inf)
    return [r for line in lines if (r := reading(line)) is not None]


@dataclass(frozen=True)
class Showing:
    """A run of frames whose scoreboard shows a time in the window, unbroken by a frame showing
    another time; `best` is its frame nearest the start of the window (the asked time itself)."""

    start: float
    end: float
    best: Reading


def showings(readings: list[Reading], window: Window) -> list[Showing]:
    """Where a video shows the window: the live play, and each replay or later view with the
    scoreboard on screen, one per run of frames."""
    runs: list[list[Reading]] = []
    held = False
    for r in readings:
        inside = window.holds(r)
        if inside and held:
            runs[-1].append(r)
        elif inside:
            runs.append([r])
        held = inside
    return [Showing(run[0].t, run[-1].t, min(run, key=lambda r: abs(r.seconds - window.high))) for run in runs]


def ranges(readings: list[Reading], window: Window, duration: float) -> list[tuple[float, float]]:
    """The stretches of video inside the window, in time order: each run of readings in it, from
    its first reading to its last, however far apart they are (a timeout, a close-up hiding the
    scoreboard), reaching past both ends halfway to the reading outside it, at most REACH_S and
    never past either end of the video."""
    spans: list[tuple[float, float]] = []
    held = False
    for i, r in enumerate(readings):
        inside = window.holds(r)
        if inside:
            after = readings[i + 1].t if i + 1 < len(readings) else math.inf
            end = min((r.t + after) / 2, r.t + REACH_S, duration)
            if held:
                spans[-1] = (spans[-1][0], end)
            else:
                before = readings[i - 1].t if i else -math.inf
                spans.append((max((before + r.t) / 2, r.t - REACH_S, 0.0), end))
        held = inside
    return spans


@dataclass(frozen=True)
class Coverage:
    """Where a window of game time is on screen across videos: the stretches of each video that
    shows some of it, and the videos whose scoreboard was never read."""

    window: Window
    ranges: dict[str, list[tuple[float, float]]]
    unread: list[str]


def coverage(db: sqlite3.Connection, videos: list[Video], window: Window) -> Coverage:
    found: dict[str, list[tuple[float, float]]] = {}
    unread = []
    for video in videos:
        read = readings(db, video.video_id)
        if not read:
            unread.append(video.video_id)
        elif spans := ranges(read, window, video.duration):
            found[video.video_id] = spans
    return Coverage(window, found, unread)
