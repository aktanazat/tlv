"""The eval's scoring rules: the overlap threshold, one-to-one pairing that counts a replay of a
found moment as a duplicate, ignored ambiguous moments, and queries whose truth is empty."""

import pytest

from eval import score
from eval.score import Clip, Moment

V = "oyk2QtdUtKE"
EVENT = Moment(V, ((100.0, 110.0),))
LIVE_AND_REPLAY = Moment(V, ((288.0, 295.0), (297.0, 301.0)))


@pytest.mark.parametrize(
    ("clip", "moment", "expected"),
    [
        (Clip(V, 105.0, 115.0), EVENT, True),  # shares exactly half of both
        (Clip(V, 105.1, 115.1), EVENT, False),  # just under half
        (Clip(V, 108.0, 109.0), EVENT, True),  # a short clip inside the event
        (Clip(V, 109.5, 112.0), EVENT, False),  # 0.5 s of a 2.5 s clip
        (Clip(V, 90.0, 130.0), EVENT, True),  # a long clip around the event
        (Clip(V, 110.0, 120.0), EVENT, False),  # touching, nothing shared
        (Clip(V, 105.0, 105.0), EVENT, False),  # zero length
        (Clip("zHBgJqG1nJo", 100.0, 110.0), EVENT, False),  # same times, other video
        (Clip(V, 296.5, 302.0), LIVE_AND_REPLAY, True),  # matches through the replay range
    ],
)
def test_overlap_needs_half_of_the_shorter_range(clip, moment, expected):
    assert score.overlaps(clip, moment) is expected


A = Moment(V, ((100.0, 110.0),))
B = Moment(V, ((110.0, 120.0),))
FAR = Moment(V, ((300.0, 305.0),))
NEAR_A = Moment(V, ((108.0, 116.0),))


@pytest.mark.parametrize(
    ("events", "ambiguous", "clips", "labels", "precision", "recall"),
    [
        # The live play and its replay: one moment found once, the replay is a duplicate.
        ([LIVE_AND_REPLAY], [], [Clip(V, 287.0, 296.0), Clip(V, 296.5, 302.0)], ("true", "duplicate"), 0.5, 1.0),
        # A clip covering two events must leave the second clip its only match.
        ([A, B], [], [Clip(V, 100.0, 120.0), Clip(V, 100.0, 110.0)], ("true", "true"), 1.0, 1.0),
        # A clip on a moment the labels could not settle is neither right nor wrong.
        ([A], [FAR], [Clip(V, 301.0, 304.0), Clip(V, 100.0, 110.0)], ("ambiguous", "true"), 1.0, 1.0),
        # A second clip on an unsettled moment next to a found event is ignored, not a duplicate.
        ([A], [NEAR_A], [Clip(V, 100.0, 110.0), Clip(V, 106.0, 114.0)], ("true", "ambiguous"), 1.0, 1.0),
        ([A, B], [], [Clip(V, 101.0, 109.0), Clip(V, 500.0, 510.0)], ("true", "false"), 0.5, 0.5),
        # Empty truth: returning nothing is right, every returned clip is false.
        ([], [], [], (), None, None),
        ([], [], [Clip(V, 1.0, 5.0), Clip(V, 10.0, 20.0)], ("false", "false"), 0.0, None),
    ],
)
def test_tally_pairs_clips_with_events_one_to_one(events, ambiguous, clips, labels, precision, recall):
    result = score.tally(events, ambiguous, tuple(clips))
    assert result.labels == labels
    assert result.precision == precision
    assert result.recall == recall


TRUTH = {"queries": {"q": {"events": [{"video": V, "ranges": [["1:40", "1:50"]], "note": ""}], "ambiguous": []}}}


def confirmed(local_path: str, source: str) -> dict:
    clip = {"video_id": "6acd5d8eb4def3563b044971", "start": 101, "end": 109}
    return {**clip, "local_path": local_path, "source": source}


@pytest.mark.parametrize(
    ("local_path", "source", "label"),
    [
        ("/media/Youtube-oyk2QtdUtKE.mp4", "/media/Youtube-oyk2QtdUtKE.mp4", "true"),
        ("/media/game.mp4", "https://www.youtube.com/watch?v=oyk2QtdUtKE", "true"),
        ("/media/game.mp4", "/media/game.mp4", "false"),
    ],
)
def test_returned_clips_are_placed_on_their_youtube_video(local_path, source, label):
    output = {"confirmed": [confirmed(local_path, source)]}
    results = {"runs": [{"id": "q", "seconds": 40.0, "exit_code": 0, "output": output}]}
    [scored] = score.score(TRUTH, results)
    assert scored.tally.labels == (label,)


def test_a_failed_run_returns_nothing_and_misses_every_event():
    results = {"runs": [{"id": "q", "seconds": 3.0, "exit_code": 1, "output": None}]}
    [scored] = score.score(TRUTH, results)
    assert scored.failed
    assert (scored.tally.labels, scored.tally.missed) == ((), 1)
