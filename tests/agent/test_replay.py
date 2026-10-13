"""One play, one result: repeats dropped, pieces of one moment joined, replays merged into their play,
distinct plays kept. The distinct_moments cases are ported from test/search.test.ts."""

from dataclasses import dataclass

import pytest

from tlv.agent.replay import GameClock, distinct_moments, join_pieces, merge_replays, period_ended, read_clock
from tlv.agent.sources import Line


@dataclass(frozen=True)
class Seen:
    name: str
    video_id: str
    start: float
    end: float
    confidence: float = 0.9
    replay: bool = False
    clock: GameClock | None = None


@pytest.mark.parametrize(
    ("moments", "kept", "repeats"),
    [
        pytest.param(
            [Seen("a", "v1", 0, 10, 0.9), Seen("b", "v1", 2, 12, 0.95)],
            ["b"],
            [("a", "b")],
            id="drops the less confident of two moments in one video overlapping by half or more",
        ),
        pytest.param(
            [Seen("a", "v1", 255.5, 261, 0.95), Seen("b", "v1", 257.6, 263.6, 0.95)],
            ["a"],
            [("b", "a")],
            id="drops a repeat whose times moved two seconds, under half their union",
        ),
        pytest.param(
            [Seen("late", "v1", 1, 11, 0.9), Seen("early", "v1", 0, 10, 0.9)],
            ["early"],
            [("late", "early")],
            id="keeps the earlier of two equally confident repeats",
        ),
        pytest.param(
            [Seen("a", "v1", 0, 10, 0.9), Seen("b", "v1", 8, 18, 0.8)],
            ["a", "b"],
            [],
            id="keeps moments sharing under half the shorter and under 3 s",
        ),
        pytest.param(
            [Seen("a", "v1", 112.8, 125.0, 1.0), Seen("b", "v1", 119.0, 136.1, 0.95)],
            ["a"],
            [("b", "a")],
            id="drops a long moment sharing 3 s or more, under half the shorter",
        ),
        pytest.param(
            [Seen("a", "v1", 0, 10, 0.95), Seen("b", "v1", 5, 15, 0.9), Seen("c", "v1", 12, 22, 0.8)],
            ["a"],
            [("b", "a"), ("c", "a")],
            id="a moment repeating only a dropped repeat folds into what that one repeats",
        ),
        pytest.param(
            [Seen("a", "v1", 0, 10, 0.9), Seen("b", "v2", 0, 10, 0.8)],
            ["a", "b"],
            [],
            id="keeps the same range in different videos",
        ),
    ],
)
def test_distinct_moments(moments, kept, repeats):
    kept_moments, repeated = distinct_moments(moments)
    assert [m.name for m in kept_moments] == kept
    assert [(m.name, of.name) for m, of in repeated] == repeats


def lines(*texts):
    return [Line(float(i), float(i), text) for i, text in enumerate(texts)]


@pytest.mark.parametrize(
    ("texts", "ended"),
    [
        pytest.param(["4TH 3.3", "4TH 2.6", "4TH 2.0", "4TH 1.0", "4TH 0.0"], True, id="the game clock runs down"),
        pytest.param(["4TH 0.0"], True, id="0.0 with no reading before it"),
        pytest.param(["4TH 24.6", "4TH 23.6", "4TH 0.0"], False, id="0.0 a second after 23.6 is the shot clock"),
        pytest.param(["1ST 0.0", "1ST 12.3"], False, id="the clock reads on after 0.0"),
    ],
)
def test_a_period_ends_only_on_a_clock_running_out(texts, ended):
    assert period_ended(lines(*texts)) is ended


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        pytest.param(
            ["TAKERS 102 | 100 | ANDRE DRUMMOND | BONUS | 11 POINTS 11 REBOUNDS | 3:46 | 4TH | :10"],
            GameClock(frozenset("4"), 226, 226),
            id="clock before the period, shot clock after",
        ),
        pytest.param(
            ["LAL 105 | 100 | BONUS | :24 | 4TH | 3:42"], GameClock(frozenset("4"), 222, 222), id="shot clock first"
        ),
        pytest.param(["TAKERS 108 | 105 | BONUS | 4TH | :34.0 | :07"], GameClock(frozenset("4"), 34, 34), id="tenths"),
        pytest.param(
            ["TAKERS 110 | 105 | BONUS | :27.3 | :24 | 4TH"],
            GameClock(frozenset("4"), 27.3, 27.3),
            id="the shot clock between the clock and the period",
        ),
        pytest.param(
            ["73 | 6:12 | GSW | TNT | 62 | 3RD | LAL | 12"],
            GameClock(frozenset("3"), 372, 372),
            id="a scorebug read out of order",
        ),
        pytest.param(
            ["LOS ANGELES LA | 108 | 24 27.3 4TH | PHILADELPHIA 76ERS"],
            GameClock(frozenset("4"), 27.3, 27.3),
            id="arena board in a replay",
        ),
        pytest.param(
            ["TAKERS 108 | 105 | 4TH | :30.0", "TAKERS 108 | 105 | 4TH | :28.0", "TAKERS 110 | 105 | :27.3 | 4TH"],
            GameClock(frozenset("4"), 27.3, 30),
            id="a clock running across frames",
        ),
        pytest.param(["LAL 102 | 100 | 3:46 | :10"], None, id="no period"),
        pytest.param(["LAL 102 | 100 | 4TH | :10"], None, id="only a shot clock"),
        pytest.param(["4TH QUARTER STATS | FG 45.5% | 3PT 38.1%"], None, id="percentages"),
        pytest.param(["4TH QUARTER | NEXT GAME TUESDAY ON TNT | 7:30 PM"], None, id="a time far from the period"),
        pytest.param([], None, id="no text"),
    ],
)
def test_read_clock(texts, expected):
    assert read_clock(lines(*texts)) == expected


def clock(low, high=None, period="4"):
    return GameClock(frozenset(period), low, high if high is not None else low)


class Model:
    """Stands in for the model's pairwise check: answers `same` and records each pair asked."""

    def __init__(self, same=None):
        self.same = same
        self.asked = []

    def __call__(self, kept, moment):
        self.asked.append((kept.name, moment.name))
        return self.same


LIVE = Seen("live", "v1", 288, 295, clock=clock(27.3, 30.2))
# The baseline-camera replay of that basket: the arena board shows the same clock.
BASELINE = Seen("baseline", "v1", 296, 302, confidence=0.95, clock=clock(27.3, 29.5))
FIRST_THREE = Seen("first three", "v1", 255, 261, clock=clock(220, 224))
SECOND_THREE = Seen("second three", "v1", 269, 275, clock=clock(71, 74))
NO_CLOCK_REPLAY = Seen("no clock", "v1", 276, 280, replay=True)


def groups(plays):
    return [(p.moment.name, [m.name for m, _ in p.replays]) for p in plays]


@pytest.mark.parametrize(
    ("moments", "model", "expected", "asked"),
    [
        pytest.param(
            [BASELINE, LIVE],
            Model(),
            [("live", ["baseline"])],
            [],
            id="a replay showing the clock of a live play is merged into it, unasked",
        ),
        pytest.param(
            [FIRST_THREE, SECOND_THREE, LIVE],
            Model("same player, same shot"),
            [("first three", []), ("second three", []), ("live", [])],
            [],
            id="plays with different clocks stay apart whatever the model says",
        ),
        pytest.param(
            [SECOND_THREE, NO_CLOCK_REPLAY],
            Model("same shot from the baseline"),
            [("second three", ["no clock"])],
            [("second three", "no clock")],
            id="a replay without a clock is merged when the model says it is the same play",
        ),
        pytest.param(
            [SECOND_THREE, NO_CLOCK_REPLAY],
            Model(None),
            [("second three", []), ("no clock", [])],
            [("second three", "no clock")],
            id="a replay without a clock stays apart when the model says it is another play",
        ),
        pytest.param(
            [Seen("a", "v1", 10, 15), Seen("b", "v1", 30, 35)],
            Model("same"),
            [("a", []), ("b", [])],
            [],
            id="two live moments without clocks are never compared",
        ),
        pytest.param(
            [SECOND_THREE, Seen("much later", "v1", 400, 405, replay=True)],
            Model("same"),
            [("second three", []), ("much later", [])],
            [],
            id="a replay more than 90 s after a play is not compared with it",
        ),
        pytest.param(
            [LIVE, Seen("other game", "v2", 300, 306, clock=clock(28.0))],
            Model(None),
            [("live", []), ("other game", [])],
            [("live", "other game")],
            id="the same clock in another video is not enough without the model",
        ),
        pytest.param(
            [LIVE, Seen("other upload", "v2", 300, 306, clock=clock(28.0))],
            Model("same score and players"),
            [("live", ["other upload"])],
            [("live", "other upload")],
            id="the same clock in another video is merged when the model agrees",
        ),
    ],
)
def test_merge_replays(moments, model, expected, asked):
    assert groups(merge_replays(moments, model)) == expected
    assert model.asked == asked


def test_a_replay_shown_before_its_live_play_is_listed_under_the_live_play():
    replay = Seen("replay", "v1", 250, 256, confidence=0.99, replay=True, clock=clock(72))
    assert groups(merge_replays([replay, SECOND_THREE], Model())) == [("second three", ["replay"])]


def join(a, b):
    """Stands in for inspect.joined: one moment named after both pieces, spanning both."""
    return Seen(
        f"{a.name}+{b.name}",
        a.video_id,
        min(a.start, b.start),
        max(a.end, b.end),
        max(a.confidence, b.confidence),
        a.replay,
        a.clock or b.clock,
    )


# Clip times are those the everyday and sports runs returned (eval/results/everyday-local-2026-10-13.json,
# local-2026-10-12.json), where each pair below came back as separate clips.
@pytest.mark.parametrize(
    ("moments", "expected"),
    [
        pytest.param(
            [Seen("a", "v1", 444.5, 451.41), Seen("b", "v1", 453.28, 467.15), Seen("c", "v1", 468.25, 471.5)],
            [("a+b+c", 444.5, 471.5)],
            id="three looks at one lecture slide, 1.9 s and 1.1 s apart, are one moment",
        ),
        pytest.param(
            [Seen("b", "v1", 212.17, 223.34), Seen("a", "v1", 209.0, 213.0), Seen("c", "v1", 224.19, 227.5)],
            [("a+b+c", 209.0, 227.5)],
            id="a car chase seen in three looks, sharing 0.8 s and then 0.9 s apart, is one moment",
        ),
        pytest.param(
            [Seen("a", "v1", 159.5, 166.5), Seen("b", "v1", 166.75, 169.0)],
            [("a+b", 159.5, 169.0)],
            id="pointing at the bridge across a shot cut is one moment",
        ),
        pytest.param(
            [Seen("a", "v1", 270.0, 276.25), Seen("b", "v1", 281.75, 285.75)],
            [("a", 270.0, 276.25), ("b", 281.75, 285.75)],
            id="two baskets 5.5 s apart stay apart",
        ),
        pytest.param(
            [Seen("a", "v1", 444.5, 451.41), Seen("b", "v2", 453.28, 467.15)],
            [("a", 444.5, 451.41), ("b", 453.28, 467.15)],
            id="neighbouring times in two videos stay apart",
        ),
        pytest.param(
            [
                Seen("live", "v1", 296.25, 300.26, clock=clock(27.9)),
                Seen("replay", "v1", 298.5, 308.0, replay=True, clock=clock(27.3, 27.9)),
            ],
            [("live", 296.25, 300.26), ("replay", 298.5, 308.0)],
            id="a play and the replay overlapping it stay apart",
        ),
        pytest.param(
            [Seen("a", "v1", 100.0, 106.0, clock=clock(220, 226)), Seen("b", "v1", 107.0, 113.0, clock=clock(71, 77))],
            [("a", 100.0, 106.0), ("b", 107.0, 113.0)],
            id="a cut between two plays a second apart stays apart when the game clock jumps",
        ),
        pytest.param(
            [
                Seen("a", "v1", 100.0, 106.0, clock=clock(220, 226)),
                Seen("b", "v1", 108.0, 114.0, clock=clock(212, 218)),
            ],
            [("a+b", 100.0, 114.0)],
            id="pieces of one play 2 s apart join when the game clock runs on across the gap",
        ),
    ],
)
def test_join_pieces(moments, expected):
    joined = join_pieces(moments, join)
    assert sorted((m.name, m.start, m.end) for m in joined) == sorted(expected)
