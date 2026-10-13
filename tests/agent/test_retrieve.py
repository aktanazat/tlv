"""Ranking spans: word stems, rank ties, reciprocal-rank fusion and BM25."""

from collections import Counter

import numpy as np
import pytest

from tlv.agent.plan import sense, topic
from tlv.agent.retrieve import (
    Corpus,
    Query,
    Span,
    bm25,
    channels,
    competition_ranks,
    fuse,
    rank,
    sound_key,
    sound_keys,
    words,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("LeBron scores", ["lebron", "scor"], id="plural s and a final e"),
        pytest.param("he is scoring", ["scor"], id="ing and stop words"),
        pytest.param("celebrates, celebrated, celebration", ["celebrat"] * 3, id="one stem for each form"),
        pytest.param("the pass", ["pass"], id="a double s stays"),
    ],
)
def test_words_reduce_forms_of_a_word_to_one_stem(text, expected):
    assert words(text) == expected


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        pytest.param([0.2, 0.9, 0.5], [(1, 1), (2, 2), (0, 3)], id="best first"),
        pytest.param([0.5, 0.9, 0.5, 0.1], [(1, 1), (0, 2), (2, 2), (3, 4)], id="ties share a rank, the next skips"),
        pytest.param([0.0, 0.4, -0.2], [(1, 1)], id="no rank without a positive score"),
    ],
)
def test_competition_ranks(scores, expected):
    assert competition_ranks(np.array(scores)) == expected


def test_fusion_favors_a_span_ranked_well_by_several_channels():
    looks = [(7, 1), (3, 2), (5, 3)]
    words_ = [(3, 1), (5, 2)]
    assert fuse([looks, words_]) == [3, 5, 7]


def test_bm25_weighs_a_rare_word_over_a_common_one():
    documents = [Counter(["dunk", "pass"]), Counter(["lebron", "pass"]), Counter(["lebron", "pass"])]
    scores = bm25(documents, ["lebron", "dunk"])
    assert scores.argmax() == 0
    assert scores[1] == scores[2] > 0


@pytest.mark.parametrize(
    ("heard", "written"),
    [
        pytest.param("Jordy", "Jordi", id="y for i"),
        pytest.param("Fernandes", "Fernandez", id="s for z"),
        pytest.param("Edgecomb", "Edgecombe", id="a silent e"),
        pytest.param("Ashcroft", "Ascroft", id="an h between two letters of one sound"),
    ],
)
def test_names_spelled_by_ear_share_a_sound_key(heard, written):
    assert sound_key(heard) == sound_key(written)


@pytest.mark.parametrize(
    ("a", "b"),
    [pytest.param("Jordi", "Jokic", id="one first letter"), pytest.param("Fernandez", "Ferguson", id="one start")],
)
def test_names_that_sound_different_have_different_keys(a, b):
    assert sound_key(a) != sound_key(b)


def corpus(screen: list[str], spoken: list[str]) -> Corpus:
    """Five-second spans of one video with the given on-screen and spoken words, and no vectors."""
    count = len(screen)
    return Corpus(
        [Span(f"s{i}", "v", 5.0 * i, 5.0 * i + 5, "", None) for i in range(count)],
        np.zeros((count, 0), dtype=np.float32),
        np.zeros((0, 0), dtype=np.float32),
        [(0, 0)] * count,
        [Counter(words(text)) for text in screen],
        [Counter(sound_keys(text)) for text in spoken],
        [[] for _ in range(count)],
    )


CORPUS = corpus(
    screen=["jordi fernandez coach of the year", "lakers 102 sixers 99", "timeout"],
    spoken=["he takes the shot", "talking to Jordy Fernandes about it", "what a pass"],
)


@pytest.mark.parametrize(
    ("step", "found"),
    [
        pytest.param("the commentators talk about jordi fernandez", [(5, 10)], id="spoken: by the words said"),
        pytest.param("a graphic names jordi fernandez", [(0, 5)], id="on screen: by the words shown"),
    ],
)
def test_each_step_searches_the_channels_of_its_sense(step, found):
    heard_by = sense(step)
    text = topic(step) if heard_by == "speech" else step
    [clips] = rank(None, CORPUS, [Query(text, (), channels([heard_by], None))])
    assert [(c.start, c.end) for c in clips] == found
