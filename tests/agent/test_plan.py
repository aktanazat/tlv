"""Queries into steps. The mention cases are ported from test/search.test.ts."""

import pytest

from tlv.agent.plan import Mentions, read_split, resolve_mentions, sense, steps, topic

NAMES = ["LeBron", "LeBron James", "A.J. Green"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("@LeBron_James dunks", Mentions("LeBron James dunks", ()), id="a name written with underscores"),
        pytest.param(
            "@lebron-james dunks", Mentions("LeBron James dunks", ()), id="a name in another case, with a hyphen"
        ),
        pytest.param("@lebron james scores", Mentions("LeBron James scores", ()), id="the longest matching name"),
        pytest.param(
            "@LeBron scores", Mentions("LeBron scores", ()), id="a shorter name when the longer one does not match"
        ),
        pytest.param("@LeBron_James, then", Mentions("LeBron James, then", ()), id="a name followed by punctuation"),
        pytest.param("@a.j._green catches", Mentions("A.J. Green catches", ()), id="a name with regex characters"),
        pytest.param(
            "@AxJx Green catches",
            Mentions("@AxJx Green catches", ("@AxJx",)),
            id="no name that only matches as a pattern",
        ),
        pytest.param(
            "@LeBronJames dunks",
            Mentions("@LeBronJames dunks", ("@LeBronJames",)),
            id="no name continued by more letters",
        ),
        pytest.param("@Kobe passes", Mentions("@Kobe passes", ("@Kobe",)), id="an unknown name"),
        pytest.param("mail me@x.com", Mentions("mail me@x.com", ()), id="no email address"),
        pytest.param("<@e_lebron> dunks", Mentions("<@e_lebron> dunks", ()), id="no resolved mention"),
    ],
)
def test_resolve_mentions(text, expected):
    assert resolve_mentions(text, NAMES) == expected


@pytest.mark.parametrize(
    ("query", "then", "expected"),
    [
        pytest.param("lebron scores", ["lebron celebrates"], ["lebron scores", "lebron celebrates"], id="given steps"),
        pytest.param("lebron dunks on a fast break", [], ["lebron dunks on a fast break"], id="one moment"),
        pytest.param("lebron scores, then celebrates", [], ["lebron scores", "celebrates"], id="comma then"),
        pytest.param("lebron steals and then dunks", [], ["lebron steals", "dunks"], id="and then"),
        pytest.param("a block followed by a three", [], ["a block", "a three"], id="followed by"),
        pytest.param("lebron celebrates after he scores", [], ["lebron celebrates after he scores"], id="after"),
    ],
)
def test_steps_without_the_model(query, then, expected):
    assert steps(query, then, None) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"steps": [" lebron scores ", "lebron celebrates"]}, ["lebron scores", "lebron celebrates"]),
        pytest.param({"steps": []}, None, id="no steps"),
        pytest.param({"steps": ["a", "b", "c", "d", "e"]}, None, id="too many steps"),
        pytest.param({"steps": ["lebron scores", " "]}, None, id="an empty step"),
        pytest.param({"steps": "lebron scores"}, None, id="steps as text"),
        pytest.param(["lebron scores"], None, id="a bare list"),
    ],
)
def test_read_split(value, expected):
    assert read_split(value) == expected


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        pytest.param("the commentators talk about jordi fernandez", "speech", id="what commentators talk about"),
        pytest.param("the announcer says lebron is back", "speech", id="what someone says"),
        pytest.param("an on-screen graphic shows lebron james's stats", "text", id="a graphic on screen"),
        pytest.param("the scoreboard reads 110 to 108", "text", id="a scoreboard"),
        pytest.param("the buzzer at the end of the first half", "sound", id="a buzzer"),
        pytest.param("the referee's whistle stops play", "sound", id="a whistle"),
        pytest.param("the commentators mention the buzzer", "speech", id="speech is named before sound"),
        pytest.param("lebron james dunks", "sight", id="an action is seen"),
        pytest.param("a coach is interviewed during the game", "sight", id="an interview is seen"),
    ],
)
def test_a_step_is_perceived_by_the_sense_its_words_name(step, expected):
    assert sense(step) == expected


def test_the_topic_of_a_spoken_step_is_what_is_said():
    assert topic("the commentators talk about jordi fernandez") == "the jordi fernandez"
