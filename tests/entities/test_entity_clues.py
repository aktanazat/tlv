"""Reading who a person is: a jersey number only when it is centered on the body and read again
without a rival, a name only when independent channels agree on it."""

import pytest

from tlv.entities.clues import agreed_name, jersey, said, vote

CENTER = (0.42, 0.35, 0.16, 0.12)
LOWER = (0.40, 0.50, 0.20, 0.10)
EDGE = (0.84, 0.38, 0.09, 0.06)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([("23", 1.0, CENTER)], "23"),
        ([(" 23 ", 1.0, CENTER), ("23", 0.5, LOWER)], "23"),  # the same number twice
        ([("21", 1.0, EDGE)], None),  # at the crop's edge: someone beside this body
        ([("23", 1.0, CENTER), ("7", 1.0, LOWER)], None),  # two numbers on one body
        ([("23", 0.3, CENTER)], None),  # an unsure read
        ([("108", 1.0, CENTER)], None),  # a score
        ([(":24", 1.0, CENTER)], None),  # a shot clock
        ([], None),
    ],
)
def test_jersey_takes_one_confident_number_centered_on_the_body(lines, expected):
    found = jersey(lines)
    assert (found[0] if found else None) == expected


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        (["23", "23", None], "23"),
        (["23", None, None], None),  # read once
        (["77", "72", "77"], None),  # a rival read half as often
        (["77", "77", "77", "72"], "77"),
        ([None, None, None], None),
    ],
)
def test_vote_takes_a_number_only_when_reads_repeat_without_a_rival(reads, expected):
    assert vote(reads) == expected


NAMES = ["LeBron James", "Bronny James", "LEBRON JAMES", "Al Horford"]


@pytest.mark.parametrize(
    ("timed", "expected"),
    [
        ([("LeBron James for three", 4.0)], [("LeBron James", 4.0)]),
        ([("lebron", 2.0), ("JAMES", 2.4)], [("LeBron James", 2.0)]),  # one name across timed words
        ([("Bronny!", 1.0)], [("Bronny James", 1.0)]),  # a first name no one else has
        ([("James", 1.0)], []),  # a name two people share
        ([("Al", 1.0)], []),  # too short to stand for a name
        ([("Luka Dončić", 1.0)], []),
    ],
)
def test_said_finds_known_names_in_timed_words(timed, expected):
    assert said(timed, NAMES) == expected


def clue(kind, value, t, **extra):
    return {"kind": kind, "value": value, "t": t, **extra}


@pytest.mark.parametrize(
    ("clues", "expected"),
    [
        # read on screen and heard: two channels agree
        (
            [("v1", clue("ocr_name", "LeBron James", 5.0)), ("v1", clue("spoken_name", "LeBron James", 6.0))],
            "LeBron James",
        ),
        # read twice within ten seconds: one channel, one stretch
        ([("v1", clue("ocr_name", "LeBron James", 5.0)), ("v1", clue("ocr_name", "LeBron James", 8.0))], None),
        # read twice, far apart
        (
            [("v1", clue("ocr_name", "LeBron James", 5.0)), ("v1", clue("ocr_name", "LeBron James", 30.0))],
            "LeBron James",
        ),
        # read in two videos
        (
            [("v1", clue("ocr_name", "LeBron James", 5.0)), ("v2", clue("ocr_name", "LeBron James", 5.0))],
            "LeBron James",
        ),
        # a partial name joins the full one
        ([("v1", clue("spoken_name", "LeBron", 6.0)), ("v1", clue("ocr_name", "LeBron James", 5.0))], "LeBron James"),
        # the captioner read the name on screen while the commentator said it
        (
            [
                ("v1", clue("caption_name", "LeBron James", 5.0, source="on_screen")),
                ("v1", clue("spoken_name", "LeBron James", 6.0)),
            ],
            "LeBron James",
        ),
        # two names, equally supported
        (
            [
                ("v1", clue("ocr_name", "LeBron James", 5.0)),
                ("v1", clue("spoken_name", "LeBron James", 6.0)),
                ("v1", clue("ocr_name", "Bronny James", 30.0)),
                ("v1", clue("spoken_name", "Bronny James", 31.0)),
            ],
            None,
        ),
        # four clues against one
        (
            [
                ("v1", clue("ocr_name", "LeBron James", 5.0)),
                ("v1", clue("spoken_name", "LeBron James", 6.0)),
                ("v1", clue("ocr_name", "LeBron James", 30.0)),
                ("v1", clue("spoken_name", "LeBron James", 31.0)),
                ("v1", clue("ocr_name", "Bronny James", 50.0)),
            ],
            "LeBron James",
        ),
    ],
)
def test_agreed_name_needs_two_independent_clues_and_a_clear_lead(clues, expected):
    found = agreed_name(clues)
    assert (found[0] if found else None) == expected
