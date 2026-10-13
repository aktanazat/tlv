"""One play, one result. Broadcasts show a play again: a replay from another camera, a
slow-motion close-up, or the same moment in another upload of the game. Confirmed moments that
show the same play are merged into the live one, and the others are listed as its replays.

Different plays must stay apart even when they look alike (two threes by one player thirteen
seconds apart), so a merge needs evidence: the same game clock on screen, or the model saying
so after comparing frames of both when one of them is a replay."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from tlv.agent.sources import Line

# A replay follows its play within this many seconds of the same video.
REPLAY_WITHIN_S = 90.0
# Two readings of one game clock this close are the same instant: OCR reads tenths at most.
CLOCK_TOLERANCE_S = 1.0
# A game clock is read next to the period it belongs to ("4TH 3:46", "6:11 3RD", "1ST :42.1"),
# which tells it from the shot clock (":10") and from other numbers on screen.
_PERIOD = re.compile(r"(?<![A-Za-z0-9])(?:([1-4])(?:ST|ND|RD|TH)|([1-4])Q|Q([1-4])|(\d?OT\d?))(?![A-Za-z0-9])", re.I)
_CLOCK = re.compile(r"(?<![\d:.])(\d{1,2}):([0-5]\d)(?:\.(\d))?(?![\d:.%])|(?<![\d.])(\d{1,2})\.(\d)(?![\d:.%])")
# Most characters between a period and its clock. OCR reads a scorebug's boxes row by row, so other
# items can come between them: "6:12 | GSW | TNT | 62 | 3RD" puts 20 characters between.
_NEAR = 24


@dataclass(frozen=True)
class GameClock:
    """The game clock as read on screen during a moment: the periods shown and the lowest and
    highest time left in the period, in seconds."""

    periods: frozenset[str]
    low: float
    high: float


class Moment(Protocol):
    @property
    def video_id(self) -> str: ...

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def confidence(self) -> float: ...

    @property
    def replay(self) -> bool: ...

    @property
    def clock(self) -> GameClock | None: ...


def distinct_moments[M: Moment](moments: list[M]) -> tuple[list[M], list[tuple[M, M]]]:
    """Drops confirmed moments that repeat one already kept: same video, overlapping by at least
    half of the shorter one (not half the union: the same moment found from two candidates moves
    by a second or two, which leaves repeats under half their union). The most confident stays,
    the earlier on a tie. Returns the kept moments, most confident first, and each repeat with the
    moment it repeats."""
    kept: list[M] = []
    repeats: list[tuple[M, M]] = []
    for moment in sorted(moments, key=lambda m: (-m.confidence, m.start)):
        of = next((k for k in kept if _repeats(moment, k)), None)
        if of is None:
            kept.append(moment)
        else:
            repeats.append((moment, of))
    return kept, repeats


def _repeats(a: Moment, b: Moment) -> bool:
    overlap = min(a.end, b.end) - max(a.start, b.start)
    return a.video_id == b.video_id and overlap >= 0.5 * min(a.end - a.start, b.end - b.start)


def _period(match: re.Match[str]) -> str:
    quarter = match.group(1) or match.group(2) or match.group(3)
    return quarter or match.group(4).upper()


def _seconds(match: re.Match[str]) -> float:
    minutes, seconds, tenths, short_seconds, short_tenths = match.groups()
    if minutes is not None:
        return int(minutes) * 60 + int(seconds) + int(tenths or 0) / 10
    return int(short_seconds) + int(short_tenths) / 10


def read_clock(lines: list[Line]) -> GameClock | None:
    """The game clock shown in on-screen text lines, or None when no line shows one."""
    periods, readings = set(), []
    for line in lines:
        clocks = list(_CLOCK.finditer(line.text))
        for period in _PERIOD.finditer(line.text):
            near = [(max(c.start() - period.end(), period.start() - c.end()), c) for c in clocks]
            near = [(gap, c) for gap, c in near if gap <= _NEAR]
            if near:
                periods.add(_period(period))
                readings.append(_seconds(min(near, key=lambda pair: pair[0])[1]))
    if not readings:
        return None
    return GameClock(frozenset(periods), min(readings), max(readings))


def period_ended(lines: list[Line]) -> bool:
    """Whether the game clock reads 0.0 in any of these lines, one line per frame: a period ended."""
    return any((reading := read_clock([line])) is not None and reading.low == 0.0 for line in lines)


def same_clock(a: GameClock, b: GameClock) -> bool:
    """Whether two moments show one instant of a game: a shared period and overlapping clock
    readings."""
    overlap = b.low <= a.high + CLOCK_TOLERANCE_S and a.low <= b.high + CLOCK_TOLERANCE_S
    return bool(a.periods & b.periods) and overlap


@dataclass(frozen=True)
class Play[M: Moment]:
    """A confirmed moment kept as a result, with the moments that show the same play again, each
    with the reason it is the same play."""

    moment: M
    replays: tuple[tuple[M, str], ...] = ()


def merge_replays[M: Moment](moments: list[M], compare: Callable[[M, M], str | None]) -> list[Play[M]]:
    """Groups moments that show one play. Live footage before replays and earlier before later:
    the first moment of a play is kept and the rest become its replays. `compare(kept, moment)`
    asks the model about a pair when the clock cannot decide; it returns why they are the same
    play, or None."""
    plays: list[Play[M]] = []
    for moment in sorted(moments, key=lambda m: (m.replay, m.start)):
        nearest = sorted(range(len(plays)), key=lambda i: _distance(plays[i].moment, moment))
        for i in nearest:
            reason = same_play(plays[i].moment, moment, compare)
            if reason:
                plays[i] = Play(plays[i].moment, (*plays[i].replays, (moment, reason)))
                break
        else:
            plays.append(Play(moment))
    return plays


def _distance(a: Moment, b: Moment) -> tuple[bool, float]:
    return a.video_id != b.video_id, abs(a.start - b.start)


def same_play[M: Moment](kept: M, moment: M, compare: Callable[[M, M], str | None]) -> str | None:
    """Why `moment` shows the play of `kept`, or None when it shows another play or nothing says.

    Game clocks decide when both moments show one. Otherwise the model compares the two, but
    only for moments of one video a replay apart where one is a replay, or shows no game clock
    while the other does (broadcasts hide the score bar during replays)."""
    if kept.clock and moment.clock:
        if not same_clock(kept.clock, moment.clock):
            return None
        if kept.video_id == moment.video_id:
            return f"same game clock ({_clock_text(moment.clock)})"
        reason = compare(kept, moment)
        return f"same game clock ({_clock_text(moment.clock)}) in another video; {reason}" if reason else None
    nearby = kept.video_id == moment.video_id and abs(moment.start - kept.end) <= REPLAY_WITHIN_S
    if nearby and (kept.replay or moment.replay or bool(kept.clock) != bool(moment.clock)):
        return compare(kept, moment)
    return None


def _clock_text(clock: GameClock) -> str:
    def text(seconds: float) -> str:
        return f"{int(seconds // 60)}:{seconds % 60:04.1f}" if seconds >= 60 else f"{seconds:.1f}"

    periods = "/".join(sorted(clock.periods))
    return f"period {periods}, {text(clock.low)}" + (f"-{text(clock.high)}" if clock.high > clock.low else "")
