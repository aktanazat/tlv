"""Following people through sampled frames: a track breaks rather than risk swapping two people, and
a face belongs to a body only where it clearly sits on it."""

import pytest

from tlv.entities.tracks import Seen, Track, follow, link


@pytest.mark.parametrize(
    ("frames", "cuts", "expected"),
    [
        # one person drifting right: one track
        ([(0.0, [0.10]), (0.5, [0.12]), (1.0, [0.14])], [], [[0.10, 0.12, 0.14]]),
        # a shot cut between the frames: two tracks
        ([(0.0, [0.10]), (0.5, [0.10])], [0.25], [[0.10], [0.10]]),
        # unseen for longer than a second: two tracks
        ([(0.0, [0.10]), (1.5, [0.10])], [], [[0.10], [0.10]]),
        # moved too far to overlap: two tracks
        ([(0.0, [0.10]), (0.5, [0.50])], [], [[0.10], [0.50]]),
        # neighbors who each clearly continue, though each new box also touches the other: two tracks
        ([(0.0, [0.10, 0.16]), (0.5, [0.11, 0.17])], [], [[0.10, 0.11], [0.16, 0.17]]),
        # two people crossing, each new box overlapping both old ones about alike: four tracks, no swap
        ([(0.0, [0.10, 0.14]), (0.5, [0.12, 0.13])], [], [[0.10], [0.14], [0.12], [0.13]]),
    ],
)
def test_follow_breaks_a_track_rather_than_risk_swapping_people(frames, cuts, expected):
    seen = [Seen(t, (x, 0.2, 0.1, 0.3), 0.9) for t, xs in frames for x in xs]
    tracks = follow("body", seen, cuts)
    assert [[s.box[0] for s in track.seen] for track in tracks] == expected


BODY = (0.10, 0.10, 0.20, 0.80)
NEIGHBOR = (0.15, 0.10, 0.20, 0.80)  # overlaps BODY where its head is
HEAD = (0.17, 0.12, 0.06, 0.10)  # on BODY's head, and NEIGHBOR's
AWAY = (0.60, 0.12, 0.06, 0.10)


def track(kind, boxes):
    return Track(kind, tuple(Seen(0.5 * k, box, 0.9) for k, box in enumerate(boxes)))


@pytest.mark.parametrize(
    ("face", "bodies", "expected"),
    [
        ([HEAD, HEAD, HEAD], [[BODY, BODY, BODY]], {0: 0}),
        ([HEAD, HEAD, AWAY], [[BODY, BODY, BODY]], {0: 0}),  # on the body in most frames
        ([HEAD, AWAY, AWAY], [[BODY, BODY, BODY]], {}),  # on it in one frame of three
        ([HEAD, HEAD, HEAD], [[BODY, BODY, BODY], [NEIGHBOR, NEIGHBOR, NEIGHBOR]], {}),  # on two bodies at once
    ],
)
def test_link_ties_a_face_to_a_body_only_where_it_clearly_sits(face, bodies, expected):
    assert link([track("face", face)], [track("body", boxes) for boxes in bodies]) == expected
