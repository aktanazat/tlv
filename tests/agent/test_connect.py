"""Candidates from ranked hits: duplicates merged, ordered steps chained in time order.
Ported from test/search.test.ts."""

import pytest

from tlv.agent.connect import (
    Candidate,
    Clip,
    Look,
    chain_steps,
    looks,
    merge_candidates,
    overlap_ratio,
    rank_clips,
    rank_steps,
)
from tlv.media import Range


def hit(video_id, start, end, score):
    return Candidate(video_id, start, end, "clip", score, (Clip(video_id, start, end, score),))


def spans(candidates):
    return [(c.video_id, c.start, c.end, c.score) for c in candidates]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param(Range(0, 10), Range(0, 10), 1, id="identical ranges"),
        pytest.param(Range(0, 10), Range(20, 30), 0, id="disjoint ranges"),
        pytest.param(Range(0, 10), Range(5, 15), 1 / 3, id="half-shifted ranges"),
        pytest.param(Range(0, 10), Range(2, 4), 0.2, id="a range inside another"),
        pytest.param(Range(5, 5), Range(5, 5), 1, id="two equal instants"),
    ],
)
def test_overlap_ratio_is_overlap_over_union(a, b, expected):
    assert overlap_ratio(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        pytest.param(
            [hit("v1", 0, 10, 2), hit("v1", 0, 5, 1)],
            [("v1", 0, 5, 1)],
            id="drops the worse of two same-video candidates overlapping by half or more",
        ),
        pytest.param(
            [hit("v1", 0, 10, 2), hit("v1", 0, 4.9, 1)],
            [("v1", 0, 4.9, 1), ("v1", 0, 10, 2)],
            id="keeps same-video candidates overlapping by less than half",
        ),
        pytest.param(
            [hit("v1", 0, 10, 1), hit("v2", 0, 10, 2)],
            [("v1", 0, 10, 1), ("v2", 0, 10, 2)],
            id="keeps overlapping candidates from different videos",
        ),
        pytest.param(
            [hit("v1", 50, 60, 2), hit("v2", 30, 40, 1), hit("v3", 10, 20, 2)],
            [("v2", 30, 40, 1), ("v3", 10, 20, 2), ("v1", 50, 60, 2)],
            id="orders by score, then by earlier start",
        ),
    ],
)
def test_merge_candidates(candidates, expected):
    assert spans(merge_candidates(candidates, 10)) == expected


def test_merge_candidates_returns_at_most_limit_candidates_the_best_ones():
    candidates = [hit("v1", 0, 1, 3), hit("v2", 0, 1, 1), hit("v3", 0, 1, 2)]
    assert spans(merge_candidates(candidates, 2)) == [("v2", 0, 1, 1), ("v3", 0, 1, 2)]


def test_rank_clips_scores_each_clip_by_its_rank_and_merges_duplicates():
    clips = [Clip("v1", 10, 20, 2), Clip("v1", 11, 20, 1), Clip("v2", 0, 5, 3)]
    assert spans(rank_clips(clips, 10)) == [("v1", 11, 20, 1), ("v2", 0, 5, 3)]


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param(
            [[Clip("v1", 10, 15, 1)], [Clip("v1", 18, 22, 2)]],
            [("v1", 10, 22, 3)],
            id="links a next step that starts within max_gap of the previous end",
        ),
        pytest.param(
            [[Clip("v1", 10, 15, 1)], [Clip("v1", 20, 22, 1)]],
            [("v1", 10, 22, 2)],
            id="links a next step that starts exactly max_gap after the previous end",
        ),
        pytest.param(
            [[Clip("v1", 10, 15, 1)], [Clip("v1", 20.1, 22, 1)]],
            [],
            id="does not link a next step that starts later than max_gap after the previous end",
        ),
        pytest.param(
            [[Clip("v1", 10, 15, 1)], [Clip("v1", 9, 12, 1)]],
            [],
            id="does not link a next step that starts before the previous step",
        ),
        pytest.param(
            [[Clip("v1", 10, 15, 1)], [Clip("v2", 16, 18, 1)]],
            [],
            id="does not link steps across videos",
        ),
        pytest.param(
            [[Clip("v1", 10, 15, 3), Clip("v1", 11, 15, 1)], [Clip("v1", 16, 18, 1)]],
            [("v1", 11, 18, 2)],
            id="keeps the lowest rank sum for each last-step clip",
        ),
        pytest.param(
            [[Clip("v1", 10, 30, 1)], [Clip("v1", 12, 14, 1)]],
            [("v1", 10, 30, 2)],
            id="spans from the earliest start to the latest end of its steps",
        ),
        pytest.param(
            [[Clip("v1", 0, 2, 1)], [Clip("v1", 3, 4, 2)], [Clip("v1", 6, 9, 3)]],
            [("v1", 0, 9, 6)],
            id="chains three steps in order",
        ),
    ],
)
def test_chain_steps(steps, expected):
    assert spans(chain_steps(steps, 5)) == expected


def test_chain_steps_keeps_one_clip_per_step_in_step_order():
    first, second = Clip("v1", 10, 15, 1), Clip("v1", 18, 22, 2)
    [chain] = chain_steps([[first], [second]], 5)
    assert (chain.kind, chain.clips) == ("chain", (first, second))


def test_rank_steps_scores_a_sentence_hit_as_its_rank_once_per_step():
    steps = [[Clip("v1", 10, 15, 1)], [Clip("v1", 16, 20, 1)], [Clip("v1", 21, 25, 1)]]
    ranked = rank_steps([Clip("v2", 0, 5, 2)], steps, 5, 10)
    assert [(c.kind, c.video_id, c.score) for c in ranked] == [("chain", "v1", 3), ("sentence", "v2", 6)]


def test_rank_steps_merges_a_sentence_hit_into_a_better_chain_over_the_same_moment():
    steps = [[Clip("v1", 10, 15, 1)], [Clip("v1", 16, 20, 1)]]
    ranked = rank_steps([Clip("v1", 10, 20, 4)], steps, 5, 10)
    assert [(c.kind, c.start, c.end, c.score) for c in ranked] == [("chain", 10, 20, 2)]


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        pytest.param([hit("a", 10, 14, 1), hit("a", 14, 18, 2)], [Look("a", 10, 18)], id="neighbours are one look"),
        pytest.param([hit("a", 10, 14, 1), hit("a", 12, 16, 2)], [Look("a", 10, 16)], id="overlaps are one look"),
        pytest.param(
            [hit("a", 10, 14, 1), hit("b", 14, 18, 2)], [Look("a", 10, 14), Look("b", 14, 18)], id="videos stay apart"
        ),
        pytest.param(
            [hit("a", 10, 14, 1), hit("a", 15, 18, 2)], [Look("a", 10, 14), Look("a", 15, 18)], id="a gap stays"
        ),
        pytest.param(
            [hit("a", 10, 18, 1), hit("a", 18, 23, 2)],
            [Look("a", 10, 18), Look("a", 18, 23)],
            id="a union past the longest look stays apart",
        ),
        pytest.param([hit("a", 0, 17, 1)], [Look("a", 0, 8.5), Look("a", 8.5, 17)], id="a long one is cut in two"),
        pytest.param(
            [hit("a", 30, 34, 1), hit("a", 10, 14, 2), hit("a", 34, 36, 3)],
            [Look("a", 30, 36), Look("a", 10, 14)],
            id="the best candidate's look comes first",
        ),
    ],
)
def test_candidates_are_checked_in_looks_no_longer_than_the_longest(candidates, expected):
    assert looks(candidates, 12) == expected
