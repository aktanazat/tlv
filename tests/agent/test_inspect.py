"""Inspection windows, frame numbers turned into video time, and the verdict checks.
The window and judge cases are ported from test/search.test.ts, with steps in video time."""

import json
import math
from pathlib import Path

import pytest

from tlv.agent.inspect import (
    CheckedStep,
    Confirmed,
    Rejected,
    Shown,
    StepSeen,
    Verdict,
    ask_verdicts,
    heard_at,
    inspection_window,
    judge,
    read_verdict,
    unheard,
)
from tlv.agent.model import Reply
from tlv.agent.retrieve import sound_keys
from tlv.agent.sources import Frame, Line, Video
from tlv.media import Range


@pytest.mark.parametrize(
    ("span", "pad", "duration", "expected"),
    [
        pytest.param(Range(10, 20), 3, 100, Range(7, 23), id="pads both sides"),
        pytest.param(Range(1, 5), 3, 100, Range(0, 8), id="clamps padding at the video start"),
        pytest.param(Range(95, 99), 3, 100, Range(92, 100), id="clamps padding at the video end"),
        pytest.param(Range(50, 51), 0, 100, Range(47.5, 53.5), id="widens a short range around its center"),
        pytest.param(Range(0, 1), 0, 100, Range(0, 6), id="shifts a widened window that would start before 0"),
        pytest.param(Range(99, 100), 0, 100, Range(94, 100), id="shifts a widened window that would end late"),
        pytest.param(Range(1, 2), 0, 5, Range(0, 5), id="covers a video shorter than the minimum"),
        pytest.param(Range(1, 2), 0, 6, Range(0, 6), id="covers a video exactly the minimum long"),
    ],
)
def test_inspection_window(span, pad, duration, expected):
    assert inspection_window(span, pad, duration, 6) == expected


WINDOW = Range(100, 110)
CANDIDATE = Range(102, 107)
# Ten frames one second apart, a quarter second into each second of the window.
TIMES = [100.25 + i for i in range(10)]


def answer(steps, confidence=0.9):
    """The model's JSON with `steps` given in order, keyed "1", "2"... as the prompt asks."""
    numbered = {str(i): frames for i, frames in enumerate(steps, 1)}
    return {"match": True, "confidence": confidence, "who": "LeBron James", "evidence": "jersey 23", "steps": numbered}


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        pytest.param([1, 1], (100, 100.75), id="the first frame reaches back to the window start"),
        pytest.param([3, 5], (101.75, 104.75), id="a run of frames reaches halfway to each neighbour"),
        pytest.param([10, 10], (108.75, 110), id="the last frame reaches on to the window end"),
        pytest.param([1, 10], (100, 110), id="every frame covers the window"),
    ],
)
def test_a_step_covers_the_time_its_frames_stand_for(frames, expected):
    verdict = read_verdict(answer([frames]), ["dunks"], TIMES, WINDOW)
    assert (verdict.steps[0].start, verdict.steps[0].end) == pytest.approx(expected)


def test_frames_shown_unevenly_split_the_time_between_them_at_midpoints():
    verdict = read_verdict(answer([[2, 2]]), ["dunks"], [100.5, 101.0, 103.0], Range(100, 104))
    assert (verdict.steps[0].start, verdict.steps[0].end) == pytest.approx((100.75, 102.0))


def test_frame_numbers_confirm_the_moment_where_its_frames_are():
    verdict = read_verdict(answer([[3, 5], [7, 8]]), ["scores", "celebrates"], TIMES, WINDOW)
    assert judge(verdict, WINDOW, CANDIDATE, 2, 0.6) == Confirmed(
        101.75, 107.75, (CheckedStep("scores", 101.75, 104.75), CheckedStep("celebrates", 105.75, 107.75))
    )


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([[3, 11]], id="a frame past the last"),
        pytest.param([[3.5, 4]], id="a fractional frame number"),
        pytest.param([["1:59.0", "2:00.0"]], id="a time outside the window"),
        pytest.param([["start", 4]], id="a word"),
        pytest.param([[2, 3, 4]], id="three frames"),
    ],
)
def test_a_step_naming_frames_that_were_not_shown_is_unreadable(steps):
    assert isinstance(read_verdict(answer(steps), ["dunks"], TIMES, WINDOW), str)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(answer([[0, 2]]), (100, 101.75), id="frame 0 is read as the first frame"),
        pytest.param(answer([["3", "5"]]), (101.75, 104.75), id="frame numbers as text"),
        pytest.param(answer([[3.0, 5.0]]), (101.75, 104.75), id="whole numbers with a point"),
        pytest.param(answer([["1:42.2", "1:44.3"]]), (101.75, 104.75), id="stamped times name the nearest frames"),
        pytest.param(answer([[5, 3]]), (101.75, 104.75), id="frames in either order"),
        pytest.param(answer([[4]]), (102.75, 103.75), id="one frame for both ends"),
        pytest.param(answer([4]), (102.75, 103.75), id="a bare frame number"),
        pytest.param({"match": True, "confidence": 0.9, "steps": [[3, 5]]}, (101.75, 104.75), id="steps as a list"),
    ],
)
def test_a_verdict_in_a_near_miss_form_is_read_as_meant(value, expected):
    verdict = read_verdict(value, ["dunks"], TIMES, WINDOW)
    assert (verdict.steps[0].start, verdict.steps[0].end) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="no JSON"),
        pytest.param([True, 0.9], id="a list"),
        pytest.param({"confidence": 0.9, "steps": {"1": [1, 2]}}, id="no match"),
        pytest.param({"match": "yes", "confidence": 0.9, "steps": {"1": [1, 2]}}, id="match as text"),
        pytest.param({"match": True, "confidence": "high", "steps": {"1": [1, 2]}}, id="confidence as text"),
        pytest.param({"match": True, "confidence": True, "steps": {"1": [1, 2]}}, id="confidence as a boolean"),
        pytest.param({"match": True, "confidence": 0.9}, id="no steps"),
        pytest.param({"match": True, "confidence": 0.9, "steps": {"1": [1, 2], "3": [4, 5]}}, id="a step skipped"),
    ],
)
def test_an_answer_without_match_confidence_and_numbered_steps_is_unreadable(value):
    assert isinstance(read_verdict(value, ["dunks"], TIMES, WINDOW), str)


def test_a_null_step_is_a_step_not_seen():
    verdict = read_verdict(answer([[2, 3], None]), ["steals", "dunks"], TIMES, WINDOW)
    assert [s.seen for s in verdict.steps] == [True, False]


class Model:
    """Stands in for the vision model: gives its replies in turn and keeps the questions it was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.asked = []

    def ask_all(self, questions):
        self.asked += questions
        return [self.replies.pop(0) for _ in questions]


def said(text):
    return Reply(text, 0, 0, False, False)


READABLE = said(json.dumps(answer([[3, 5]])))
UNREADABLE = said("frames 3 to 5")
# What the model process returns for an answer stopped at its first words, which say no match.
STOPPED_AT_NO = Reply('{"match": false', 0, 4, False, True)
VIDEO = Video("v1", "game", "file", "/videos/game.mp4", 600.0)


def stretch(name):
    """A stretch whose prompt is `name` and whose one stamped image is named after it."""
    frames = [Frame(t, Path(f"{n}.jpg")) for n, t in enumerate(TIMES, 1)]
    return Shown(VIDEO, CANDIDATE, WINDOW, frames, [Path(f"{name}.jpg")], name, [], {})


@pytest.mark.parametrize(
    ("replies", "asked", "readable"),
    [
        pytest.param([READABLE], 1, True, id="a readable answer is taken"),
        pytest.param([UNREADABLE, READABLE], 2, True, id="an unreadable answer is asked for again"),
        pytest.param([UNREADABLE, said("still frames 3 to 5")], 2, False, id="a second unreadable answer gives up"),
    ],
)
def test_an_unreadable_verdict_is_asked_for_once_more(replies, asked, readable):
    model = Model(replies)
    (verdict,) = ask_verdicts(model, [stretch("dunk")], ["dunks"])
    assert (len(model.asked), isinstance(verdict, Verdict)) == (asked, readable)


def test_an_answer_stopped_at_no_match_is_a_no_match_verdict_without_asking_again():
    model = Model([STOPPED_AT_NO])
    (verdict,) = ask_verdicts(model, [stretch("dunk")], ["dunks"])
    assert (len(model.asked), verdict.match) == (1, False)


def test_only_unreadable_answers_of_a_batch_are_asked_again_each_with_its_own_stretch():
    model = Model([READABLE, UNREADABLE, READABLE])
    verdicts = ask_verdicts(model, [stretch("first"), stretch("second")], ["dunks"])
    assert [isinstance(v, Verdict) for v in verdicts] == [True, True]
    asked = [(q.prompt.split("\n")[0], [p.stem for p in q.images]) for q in model.asked]
    assert asked == [("first", ["first"]), ("second", ["second"]), ("second", ["second"])]


SAID = [
    Line(1.0, 1.5, "Jordi"),
    Line(8.0, 8.4, "talking"),
    Line(8.4, 8.6, "to"),
    Line(9.0, 9.5, "Fernandes"),
    Line(9.6, 10.0, "Jordy"),
]


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        pytest.param("jordi fernandez", (9.0, 10.0), id="the closest hearing of every name, spelled by ear"),
        pytest.param("fernandez", (9.0, 9.5), id="one name"),
        pytest.param("jordi embiid", None, id="a name not heard"),
        pytest.param("the", None, id="nothing to hear"),
    ],
)
def test_a_spoken_step_is_heard_where_its_words_are_said(words, expected):
    assert heard_at(SAID, sound_keys(words)) == expected


def verdict(steps=((101, 103), (105, 108)), match=True, confidence=0.9):
    seen = tuple(StepSeen(f"step {i}", True, start, end) for i, (start, end) in enumerate(steps, 1))
    return Verdict(match, confidence, "LeBron James", "jersey 23", False, seen)


@pytest.mark.parametrize(
    ("given", "reason"),
    [
        pytest.param(verdict(match=False), "no match", id="a verdict without a match"),
        pytest.param(verdict(confidence=1.5), "confidence 1.5 outside 0..1", id="a confidence above 1"),
        pytest.param(
            verdict(confidence=math.nan), "confidence nan outside 0..1", id="a confidence that is not a number"
        ),
        pytest.param(verdict(confidence=0.5), "confidence 0.5 below 0.6", id="a confidence below the minimum"),
        pytest.param(verdict(steps=((101, 103),)), "expected 2 steps, got 1", id="a different number of steps"),
        pytest.param(
            Verdict(True, 0.9, "", "", False, (StepSeen("a", True, 101, 103), StepSeen("b", False, 0, 0))),
            "step 2 not seen",
            id="an unseen step",
        ),
        pytest.param(
            verdict(steps=((105, 108), (101, 103))),
            "step 2 starts before step 1",
            id="a step that starts before the step before it",
        ),
    ],
)
def test_judge_rejects(given, reason):
    assert judge(given, WINDOW, CANDIDATE, 2, 0.6) == Rejected(reason)


def test_judge_confirms_a_confidence_equal_to_the_minimum():
    assert isinstance(judge(verdict(confidence=0.6), WINDOW, CANDIDATE, 2, 0.6), Confirmed)


def test_judge_confirms_steps_that_start_at_the_same_time():
    judgement = judge(verdict(steps=((102, 104), (102, 106))), WINDOW, CANDIDATE, 2, 0.6)
    assert (judgement.start, judgement.end) == (102, 106)


def test_judge_refines_the_range_to_the_steps_when_they_fit_the_window():
    assert judge(verdict(), WINDOW, CANDIDATE, 2, 0.6) == Confirmed(
        101, 108, (CheckedStep("step 1", 101, 103), CheckedStep("step 2", 105, 108))
    )


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(((101, 112),), id="a step ends after the window"),
        pytest.param(((99, 103),), id="a step starts before the window"),
        pytest.param(((101, 103), (106, 105)), id="a step ends before it starts"),
        pytest.param(((104, 104),), id="the steps span no time"),
    ],
)
def test_judge_keeps_the_candidate_range_when(steps):
    judgement = judge(verdict(steps=steps), WINDOW, CANDIDATE, len(steps), 0.6)
    assert (judgement.start, judgement.end) == (102, 107)


def test_judge_clamps_reported_steps_into_the_window():
    judgement = judge(verdict(steps=((99, 112),)), WINDOW, CANDIDATE, 1, 0.6)
    assert judgement.steps == (CheckedStep("step 1", 100, 110),)


@pytest.mark.parametrize(
    ("steps", "senses", "heard", "text", "expected"),
    [
        pytest.param(["the buzzer sounds"], ["sound"], [], [Line(1, 1, "GSW | 0.0 | 2ND")], None, id="clock runs out"),
        pytest.param(
            ["an air horn sounds"], ["sound"], [Line(1, 2, "air_horn")], [], None, id="a sound label names it"
        ),
        pytest.param(
            ["the buzzer sounds"], ["sound"], [Line(1, 2, "cheering")], [Line(1, 1, "GSW | 4.2 | 2ND")], 0, id="neither"
        ),
        pytest.param(["the buzzer sounds"], ["sound"], [], [Line(1, 1, "SHOT 0.0")], 0, id="a clock with no period"),
        pytest.param(["lebron dunks", "the buzzer sounds"], ["sight", "sound"], [], [], 1, id="only steps to be heard"),
    ],
)
def test_a_step_to_be_heard_needs_the_index_to_have_heard_or_read_it(steps, senses, heard, text, expected):
    assert unheard(steps, senses, heard, text) == expected
