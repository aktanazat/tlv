"""Inspection windows, frame numbers turned into video time, and the verdict checks.
The window and judge cases are ported from test/search.test.ts, with steps in video time."""

import json
import math
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tlv import store
from tlv.agent import retrieve
from tlv.agent.connect import Look
from tlv.agent.inspect import (
    CheckedStep,
    Confirmed,
    Moment,
    Rejected,
    Rejection,
    Shown,
    StepSeen,
    Verdict,
    ask_verdicts,
    check,
    hear,
    heard_at,
    inspection_window,
    joined,
    judge,
    read_on_screen,
    read_verdict,
    unheard,
)
from tlv.agent.model import Reply
from tlv.agent.replay import GameClock
from tlv.agent.retrieve import said_words
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
    Line(3.0, 3.4, "sandy,"),
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
        pytest.param("summit", None, id="a word that only shares a sound key"),
    ],
)
def test_a_spoken_step_is_heard_where_its_words_are_said(words, expected):
    assert heard_at(SAID, said_words(words), Range(0, 10)) == expected


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


GRAPHIC = "LEBRON JAMES | 27 PTS 8/11 FG 8/9 FT"
TEAM = "LOS ANGELES LAKERS | THIS SEASON 8-0 IN CLUTCH GAMES"
NAME = set(retrieve.words("LeBron James"))
# The index leaves out a frame repeating the text before it, so LeBron's graphic, on screen from 5:27.25
# to 5:48.25, is one line, and the team graphic after it another.
READ = [Line(327.25, 327.25, GRAPHIC), Line(349.25, 349.25, TEAM), Line(362.25, 362.25, "LAL 108 | 105")]


@pytest.mark.parametrize(
    ("lines", "start", "end", "expected"),
    [
        pytest.param(READ, 347.7, 359.8, (326.75, 348.75), id="a step over the graphic's end moves onto all of it"),
        pytest.param(READ, 355.6, 369.5, None, id="a step whose frames never read the name"),
        pytest.param(
            [
                Line(10.25, 10.25, GRAPHIC),
                Line(11.25, 11.25, "LEBRON | 27"),
                Line(12.25, 12.25, GRAPHIC),
                Line(13.25, 13.25, TEAM),
            ],
            12.0,
            13.0,
            (9.75, 12.75),
            id="one frame misread inside the graphic does not end it",
        ),
    ],
)
def test_a_step_read_on_screen_is_placed_where_the_name_is_on_screen(lines, start, end, expected):
    assert read_on_screen(lines, NAME, start, end) == expected


GAME = Video("v", "Game", "file", "/videos/game.mp4", 60.0)
LOOK = Look("v", 20.0, 24.0)  # shown 17.0 to 27.0, frame n at 16.5 + n


@pytest.fixture
def game(tmp_path, monkeypatch):
    """A private index holding a 60 s game with a small grey frame every second from 0.5 s."""
    monkeypatch.setenv("TLV_HOME", str(tmp_path))
    monkeypatch.delenv("TLV_INDEX", raising=False)
    with closing(store.connect()) as db:
        db.execute(
            "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
            " VALUES ('v', 'Game', 'file', '/videos/game.mp4', 60, 640, 360, 30, 0, 'ready')"
        )
        for second in range(60):
            path = tmp_path / f"{second}.jpg"
            Image.new("RGB", (64, 36), "grey").save(path)
            db.execute("INSERT INTO frames (video_id, t, path) VALUES ('v', ?, ?)", (second + 0.5, str(path)))
        db.commit()
        yield db


def seen_on(first, last):
    return said(json.dumps(answer([[first, last]])))


def test_a_claim_to_be_seen_needs_two_shown_frames(game):
    model = Model([seen_on(5, 5), seen_on(5, 6)])
    one, two = check(model, game, [(GAME, LOOK), (GAME, LOOK)], "LeBron James dunks", ["dunks"], ["sight"], 3.0, 0.6)
    assert (isinstance(one, Rejection), isinstance(two, Moment)) == (True, True)


def image_tokens(path):
    """The prompt tokens a shown frame costs the vision model: it reads the frame in 32 px squares
    (16 px patches merged two by two), each side rounded to whole squares."""
    with Image.open(path) as image:
        return round(image.width / 32) * round(image.height / 32)


def test_a_long_stretch_costs_the_model_ten_frames_of_tokens_at_most(game):
    """Reading the prompt is most of a check's time, and the frames are most of the prompt: a long
    stretch is shown in ten frames, 1120 image tokens, not the twelve (1344) it once was."""
    model = Model([seen_on(5, 6)])
    check(model, game, [(GAME, Look("v", 10.0, 40.0))], "LeBron James dunks", ["dunks"], ["sight"], 3.0, 0.6)
    (question,) = model.asked
    assert sum(image_tokens(path) for path in question.images) <= 1120


def person(db, name, appearances):
    """A named person whom the index sees in the game over each (start, end) given."""
    blob = store.to_blob(np.ones(4, dtype=np.float32))
    db.execute(
        "INSERT INTO entities (entity_id, name, canonical_appearance, embedding, updated_at) VALUES (?, ?, ?, ?, 0)",
        (f"e_{name}", name, f"{name}_0", blob),
    )
    for n, (start, end) in enumerate(appearances):
        db.execute(
            "INSERT INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path, embedding)"
            " VALUES (?, 'v', 'body', ?, ?, ?, '[0, 0, 0.1, 0.2]', 'crop.jpg', ?)",
            (f"{name}_{n}", start, end, start, blob),
        )
        db.execute("INSERT INTO entity_members (appearance_id, entity_id) VALUES (?, ?)", (f"{name}_{n}", f"e_{name}"))


@pytest.mark.parametrize(
    ("appearances", "kept"),
    [
        pytest.param([(17.5, 18.5)], False, id="seen in the window, not where the claim is placed: rejected"),
        pytest.param([(17.5, 18.5), (21.5, 22.5)], True, id="seen where the claim is placed"),
        pytest.param([(40.0, 45.0)], True, id="seen nowhere in the window, which says nothing"),
    ],
)
def test_a_named_person_seen_in_the_window_must_be_seen_where_the_claim_is_placed(game, appearances, kept):
    person(game, "LeBron James", appearances)
    (outcome,) = check(
        Model([seen_on(5, 6)]), game, [(GAME, LOOK)], "LeBron James dunks", ["dunks"], ["sight"], 3.0, 0.6
    )
    assert isinstance(outcome, Moment) is kept


def test_a_step_read_on_screen_moves_the_moment_onto_the_text_naming_the_person(game):
    person(game, "LeBron James", [])
    for t, text in [(17.5, GRAPHIC), (18.5, GRAPHIC), (19.5, GRAPHIC), (20.5, GRAPHIC), (21.5, TEAM)]:
        game.execute(
            "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES ('v', ?, ?, 0.1, 0.8, 0.5, 0.1, 1)", (t, text)
        )
    claim = "LeBron James's stat line is on screen"
    (moment,) = check(Model([seen_on(3, 6)]), game, [(GAME, LOOK)], claim, [claim], ["text"], 3.0, 0.6)
    assert (moment.start, moment.end) == (17.0, 21.0)


def speak(db, said):
    """One transcript segment of (word, start) pairs, each word half a second long."""
    words = [{"w": w, "s": s, "e": s + 0.5} for w, s in said]
    db.execute(
        "INSERT INTO speech (video_id, start, end, text, words) VALUES ('v', ?, ?, ?, ?)",
        (words[0]["s"], words[-1]["e"], " ".join(w for w, _ in said), json.dumps(words)),
    )


@pytest.mark.parametrize(
    ("step", "said", "expected"),
    [
        pytest.param(
            "when did they talk about pricing",
            [("our", 21.0), ("pricing", 21.5), ("for", 22.0), ("tiers", 22.5)],
            (21.5, 22.0),
            id="a question about what was said listens for its topic alone",
        ),
        pytest.param(
            "they mention pricing tiers",
            [("tiers", 25.0), ("pricing", 29.0)],
            (25.0, 29.5),
            id="a topic said across the edge of the window",
        ),
        pytest.param(
            "they mention pricing tiers", [("tiers", 25.0), ("pricing", 34.0)], None, id="a topic said too far apart"
        ),
    ],
)
def test_a_spoken_claim_is_heard_where_its_topic_is_said(game, step, said, expected):
    speak(game, said)
    outcome = hear(game, GAME, LOOK, [step], 3.0)
    assert (None if isinstance(outcome, Rejection) else (outcome.start, outcome.end)) == expected


def piece(start, end, confidence, summary, times, clock):
    step = CheckedStep("the golden gate bridge", start, end)
    shown = tuple((Frame(t, Path(f"{t}.jpg")), Path(f"stamped/{t}.jpg")) for t in times)
    return Moment(GAME, start, end, confidence, "", summary, False, clock, (step,), shown)


def test_two_pieces_join_into_one_moment_with_the_frames_and_clock_of_both():
    # The two looks at the vlog's bridge pointing (everyday-local-2026-10-13.json), with game clocks added.
    earlier = piece(
        159.5,
        166.5,
        0.95,
        "man points to bridge",
        [160.25, 161.75, 165.75, 167.25],
        GameClock(frozenset("4"), 220, 226),
    )
    later = piece(
        166.75, 169.0, 1.0, "points at the bridge", [165.75, 167.25, 168.25], GameClock(frozenset("4"), 212, 218)
    )
    moment = joined(earlier, later)
    assert (moment.start, moment.end, moment.confidence, moment.summary) == (159.5, 169.0, 1.0, "points at the bridge")
    assert moment.steps == (CheckedStep("the golden gate bridge", 159.5, 169.0),)
    assert [frame.t for frame, _ in moment.shown] == [160.25, 161.75, 165.75, 167.25, 168.25]
    assert moment.clock == GameClock(frozenset("4"), 212, 226)
