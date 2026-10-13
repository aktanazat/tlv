"""The game clock across a video: a window of game time read from what a person types, the
scoreboard read from one frame's text (the shot clock is not the game clock), the runs of frames
showing a moment of game time, and the stretches of video inside a window. The scoreboard lines
are real OCR from the NBA highlight videos, with their non-Latin noise left out."""

import pytest

from tlv.agent.clock import OVERTIMES, Reading, Showing, Window, mentioned, ranges, reading, showings
from tlv.agent.sources import Line

FOURTH = frozenset({"4"})


@pytest.mark.parametrize(
    ("text", "window", "shown"),
    [
        ("4th", Window(FOURTH), "4th"),
        ("Q4", Window(FOURTH), "4th"),
        ("fourth quarter", Window(FOURTH), "4th"),
        ("2nd half", Window(frozenset({"3", "4"})), "3rd/4th"),
        ("first half", Window(frozenset({"1", "2"})), "1st/2nd"),
        ("3rd/4th", Window(frozenset({"3", "4"})), "3rd/4th"),
        ("OT", Window(frozenset(OVERTIMES)), "OT"),
        ("1OT", Window(frozenset({"OT"})), "1OT"),
        ("2OT", Window(frozenset({"2OT"})), "2OT"),
        ("4th 1:12", Window(FOURTH, 72, 72), "4th 1:12"),
        ("Q4 :42.1", Window(FOURTH, 42.1, 42.1), "4th 0:42.1"),
        ("4th 1:00-2:00", Window(FOURTH, 60, 120), "4th 2:00-1:00"),
        ("Q4 2:00 to 0:00", Window(FOURTH, 0, 120), "4th 2:00-0:00"),
        ("last 2 min", Window(frozenset({"4", *OVERTIMES}), 0, 120), "4th/OT 2:00-0:00"),
        ("final 30 seconds", Window(frozenset({"4", *OVERTIMES}), 0, 30), "4th/OT 0:30-0:00"),
        ("1:00", Window(frozenset({"1", "2", "3", "4", *OVERTIMES}), 60, 60), "1st/2nd/3rd/4th/OT 1:00"),
    ],
)
def test_a_typed_window_reads_as_periods_and_time_left_and_prints_back_as_itself(text, window, shown):
    assert Window.parse(text) == window
    assert str(window) == shown
    assert Window.parse(shown) == window


@pytest.mark.parametrize("text", ["purple", "5th", "the 4th", "4th 1:12 purple", ""])
def test_anything_else_is_refused_saying_how_to_name_a_game_time(text):
    with pytest.raises(ValueError, match=r"is not a game clock: name a period \(1st to 4th"):
        Window.parse(text)


@pytest.mark.parametrize(
    ("first", "then", "both"),
    [
        ("2nd half", "last 2 min", "4th 2:00-0:00"),
        ("4th", "Q4 1:12", "4th 1:12"),
        ("1st", "4th", None),
        ("4th 2:00-1:00", "4th 0:30", None),
    ],
)
def test_two_windows_narrow_to_the_game_time_in_both(first, then, both):
    if both is None:
        with pytest.raises(ValueError, match=f"no game time is both {first} and {then}"):
            Window.parse(first).narrowed_to(Window.parse(then))
    else:
        assert str(Window.parse(first).narrowed_to(Window.parse(then))) == both


@pytest.mark.parametrize(
    ("instruction", "window"),
    [
        ("now only fourth quarter", "4th"),
        ("same but steph in the 4th", "4th"),
        ("just the last 2 minutes please", "4th/OT 2:00-0:00"),
        ("only Q4 2:00 - 1:00", "4th 2:00-1:00"),
        ("drop the second one", None),
        ("the first three clips", None),
        ("only half court shots", None),
        ("same but steph", None),
    ],
)
def test_a_window_is_found_inside_an_instruction_and_an_ordinal_naming_a_clip_is_not_one(instruction, window):
    found = mentioned(instruction)
    assert (str(found[0]) if found else None) == window


@pytest.mark.parametrize(
    ("text", "shows"),
    [
        pytest.param(
            "GET THE 5 App | LAKERS.CO | 00 | 119 | 1:00 | GSW | TNT | 110: 4TH 16 | LAL | SPORTS",
            ("4", 60.0),
            id="the game clock",
        ),
        pytest.param("LAKERS 112 | 108 | BONUS | BONUS | 4TH | :03.5", ("4", 3.5), id="under 5 s, the only time"),
        pytest.param(
            "88 | 10:43 | GSW | INT | 80 | 4TH | LAL | 4.8 | SPORTS", None, id="the shot clock beside the game clock"
        ),
        pytest.param("60 | 11:04 | GSW | TNT | 47 | 3RD | LAL | 4.6 | SPORTS", None, id="the shot clock in the 3rd"),
        pytest.param("LAL 101 | 4TH | 3.2 | NEXT: SUNDAY 7:30PM ET", ("4", 3.2), id="a time of day is no game time"),
        pytest.param("OT | 2:15 | LAL", ("OT", 135.0), id="the first overtime"),
        pytest.param("OT2 | 0:45.2", ("2OT", 45.2), id="the second overtime, written after"),
        pytest.param("3RD 4:12 | 4TH 3:10", None, id="two periods"),
        pytest.param("LAKERS 112 | 108", None, id="no scoreboard"),
    ],
)
def test_a_frame_shows_the_game_clock_and_never_the_shot_clock(text, shows):
    found = reading(Line(5.0, 5.0, text))
    assert (None if found is None else (found.period, found.seconds)) == shows


def scoreboard(*shown):
    """Readings one second apart from 0 s, each (period, seconds left) or None for a frame whose
    scoreboard could not be read, which is left out as `readings` leaves it out."""
    return [Reading(float(t), *s) for t, s in enumerate(shown) if s is not None]


# The live play at 1:30 in the 4th, a frame later at 1:20, then its replay with the scoreboard on.
LIVE_AND_REPLAY = scoreboard(("4", 95), ("4", 91), ("4", 90), ("4", 89.5), ("4", 80), None, ("4", 90), ("4", 70))


def test_a_game_time_shows_once_per_run_of_frames_holding_it():
    assert showings(LIVE_AND_REPLAY, Window.parse("4th 1:30")) == [
        Showing(1.0, 3.0, Reading(2.0, "4", 90)),
        Showing(6.0, 6.0, Reading(6.0, "4", 90)),
    ]


@pytest.mark.parametrize(
    ("readings", "window", "duration", "spans"),
    [
        pytest.param(LIVE_AND_REPLAY, "4th 1:30", 60, [(0.5, 3.5), (5.0, 6.5)], id="halfway to the frames outside"),
        pytest.param(
            [Reading(10, "4", 30), Reading(100, "4", 20)], "4th 0:30", 200, [(0.0, 25.0)], id="at most REACH_S"
        ),
        pytest.param(
            [Reading(50, "4", 100), Reading(90, "4", 99), Reading(95, "3", 5)],
            "4th",
            200,
            [(35.0, 92.5)],
            id="a run however far apart its readings",
        ),
        pytest.param([Reading(2, "3", 30), Reading(4, "4", 700)], "4th", 10, [(3.0, 10.0)], id="up to the video's end"),
        pytest.param(LIVE_AND_REPLAY, "OT", 60, [], id="none in the window"),
    ],
)
def test_the_video_inside_a_window_reaches_past_its_readings_but_not_into_others(readings, window, duration, spans):
    assert ranges(readings, Window.parse(window), duration) == spans
