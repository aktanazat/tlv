"""People across videos, through the published entities API on a private database.

No model runs here: every pair of faces is either clearly alike (cosine at least 0.5) or clearly not
(under 0.3), so reclustering has nothing to ask the vision-language model.
"""

import json

import numpy as np
import pytest

from tlv import entities

UNIFORM = np.eye(27, dtype=np.float32)[0]  # one jersey's colors


def face(*weights: float) -> np.ndarray:
    """A unit ArcFace-sized vector along the first axes, so two faces' cosine is plain to read."""
    vector = np.zeros(512, dtype=np.float32)
    vector[: len(weights)] = weights
    return vector / np.linalg.norm(vector)


def jersey(number: str) -> dict:
    return {"kind": "jersey", "value": number, "t": 0.0}


def who(db, video_id: str, t: float) -> str:
    """The one person on screen at t."""
    (person,) = entities.present(db, video_id, t, t)
    return person["entity_id"]


@pytest.mark.parametrize(
    ("second_face", "correction", "together"),
    [
        ((0.8, 0.6), None, True),  # cosine 0.8: alike faces join on their own
        ((0.8, 0.6), "different", False),  # ...but never once someone says they are different people
        ((0.0, 0.0, 1.0), None, False),  # cosine 0: unlike faces stay apart
        ((0.0, 0.0, 1.0), "same", True),  # ...until someone says they are one person
    ],
)
def test_a_correction_outranks_the_faces_and_outlives_reclustering(
    db, video, appearance, second_face, correction, together
):
    appearance("app_a", video("v_a"), "face", 0.0, 1.0, face(1.0))
    appearance("app_b", video("v_b"), "face", 0.0, 1.0, face(*second_face))
    entities.recluster(db)
    if correction:
        getattr(entities, correction)(db, "app_a", "app_b")
    entities.recluster(db)
    assert (who(db, "v_a", 0.5) == who(db, "v_b", 0.5)) is together


@pytest.mark.parametrize(
    ("first_quality", "second_quality", "together"),
    [
        (0.5, 0.5, True),  # two sharp faces that match
        (0.05, 0.5, True),  # a blurred face that matches a sharp one
        (0.05, 0.05, False),  # two blurred faces: blur makes anyone look alike, so their match says nothing
        (0.25, 0.25, False),  # two soft faces, turned or small, are no better
    ],
)
def test_a_blurred_face_joins_only_a_sharp_face_it_matches(
    db, video, appearance, first_quality, second_quality, together
):
    appearance("app_a", video("v_a"), "face", 0.0, 1.0, face(1.0), quality=first_quality)
    appearance("app_b", video("v_b"), "face", 0.0, 1.0, face(0.8, 0.6), quality=second_quality)
    entities.recluster(db)
    assert (who(db, "v_a", 0.5) == who(db, "v_b", 0.5)) is together


@pytest.mark.parametrize(
    ("blurred", "together"),
    [
        ((0.6, 0.0, 0.8), True),  # alike to how the person looks
        ((0.0, 0.6, 0.8), False),  # alike only to their side view, as a blurred face of anyone can be
    ],
)
def test_a_blurred_face_joins_a_person_only_when_it_resembles_their_sharp_faces_as_a_whole(
    db, video, appearance, blurred, together
):
    # one person: front and side views on one body, and the front again in another game
    appearance("body", video("v_a"), "body", 0.0, 4.0, UNIFORM)
    appearance("front", "v_a", "face", 0.0, 1.0, face(1.0), body="body")
    appearance("side", "v_a", "face", 3.0, 4.0, face(0.0, 1.0), body="body")
    appearance("front_again", video("v_b"), "face", 0.0, 1.0, face(1.0))
    appearance("blurred", video("v_c"), "face", 0.0, 1.0, face(*blurred), quality=0.05)
    # the model has already been shown the blurred face beside the side view, and could not tell
    db.execute("INSERT INTO verdicts (a, b, same, created_at) VALUES ('blurred', 'side', NULL, 0)")
    entities.recluster(db)
    assert (who(db, "v_c", 0.5) == who(db, "v_a", 0.5)) is together


def test_a_name_outlives_reclustering_while_its_person_gains_and_loses_appearances(db, video, appearance):
    appearance("app_a1", video("v_a"), "face", 0.0, 1.0, face(1.0), height=0.3)  # the biggest face: the one named
    appearance("app_a2", video("v_b"), "face", 0.0, 1.0, face(0.8, 0.6))
    appearance("app_x", video("v_c"), "face", 0.0, 1.0, face(0.0, 0.0, 1.0))
    entities.recluster(db)
    entities.name(db, who(db, "v_a", 0.5), "Jane Doe")
    appearance("app_a3", "v_c", "face", 5.0, 6.0, face(0.8, 0.0, 0.0, 0.6))  # she turns up in another video
    entities.recluster(db)
    db.execute("DELETE FROM videos WHERE video_id = 'v_b'")  # and one of her videos goes
    entities.recluster(db)
    jane = who(db, "v_a", 0.5)
    assert entities.resolve(db, "jane doe") == [jane]
    assert who(db, "v_c", 5.5) == jane


@pytest.mark.parametrize(
    ("second_game", "second_number", "together"),
    [
        ("v_a", "23", True),  # one number in one game: one person
        ("v_a", None, True),  # no number read: the faces decide
        ("v_a", "7", False),  # two numbers in one game: two people, however alike their faces
        ("v_b", "23", True),  # the same number in another game
        ("v_b", "7", False),  # a player keeps his number from game to game
    ],
)
def test_two_jersey_numbers_keep_look_alike_faces_apart(db, video, appearance, second_game, second_number, together):
    for game in dict.fromkeys(("v_a", second_game)):
        video(game)
    appearance("body_1", "v_a", "body", 0.0, 2.0, UNIFORM, clues=[jersey("23")])
    appearance("face_1", "v_a", "face", 0.0, 2.0, face(1.0), body="body_1")
    numbers = [jersey(second_number)] if second_number else []
    appearance("body_2", second_game, "body", 10.0, 12.0, UNIFORM, clues=numbers)
    appearance("face_2", second_game, "face", 10.0, 12.0, face(0.8, 0.6), body="body_2")
    entities.recluster(db)
    assert (who(db, "v_a", 1.0) == who(db, second_game, 11.0)) is together


def introduce(db, video_id: str, person: str, ocr: bool = True, speech: bool = True) -> None:
    """Names the person shown from 0 s to 20 s in a caption of 4 s to 10 s, and unless told
    otherwise on screen at 5 s and aloud at 8 s."""
    if ocr:
        db.execute(
            "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES (?, 5.0, ?, 0.1, 0.8, 0.3, 0.05, 1.0)",
            (video_id, person.upper()),
        )
    if speech:
        words = [{"w": w, "s": 8.0 + k * 0.3, "e": 8.2 + k * 0.3} for k, w in enumerate(person.split())]
        db.execute(
            "INSERT INTO speech (video_id, start, end, text, words) VALUES (?, 8.0, 9.0, ?, ?)",
            (video_id, person, json.dumps(words)),
        )
    caption = {"people": [{"description": "a player", "jersey": None, "name": person, "name_source": "on_screen"}]}
    db.execute(
        "INSERT INTO spans (span_id, video_id, start, end, caption, details) VALUES (?, ?, 4.0, 10.0, ?, ?)",
        (f"{video_id}:4.000", video_id, f"{person} on court", json.dumps(caption)),
    )


@pytest.mark.parametrize(("second_name", "together"), [("Jane Doe", True), ("John Roe", False)])
def test_names_read_and_heard_for_look_alike_faces_keep_them_apart_when_they_disagree(
    db, video, appearance, second_name, together
):
    for video_id, person, vector in (("v_a", "Jane Doe", face(1.0)), ("v_b", second_name, face(0.8, 0.6))):
        appearance(f"face_{video_id}", video(video_id), "face", 0.0, 20.0, vector, height=0.3)
        introduce(db, video_id, person)
    entities.recluster(db)
    assert (who(db, "v_a", 1.0) == who(db, "v_b", 1.0)) is together


@pytest.mark.parametrize(
    ("second_caption", "together"),
    [
        (None, True),  # nothing said about the second face: the faces decide
        ("Jane", True),  # one caption agreeing with the name the person is known by
        ("John Roe", False),  # one caption naming someone else is enough to stay apart
    ],
)
def test_a_person_known_by_name_takes_in_no_one_whose_clues_name_someone_else(
    db, video, appearance, second_caption, together
):
    for video_id, vector in (("v_a", face(1.0)), ("v_b", face(0.8, 0.6))):
        appearance(f"face_{video_id}", video(video_id), "face", 0.0, 20.0, vector, height=0.3)
    introduce(db, "v_a", "Jane Doe")
    if second_caption:
        introduce(db, "v_b", second_caption, ocr=False, speech=False)
    entities.recluster(db)
    assert (who(db, "v_a", 1.0) == who(db, "v_b", 1.0)) is together


@pytest.fixture
def jane(db, video, appearance):
    """Jane Doe, wearing 23, and a stranger in the same game."""
    game = video("v_a")
    appearance("body_j", game, "body", 0.0, 2.0, UNIFORM, clues=[jersey("23")])
    appearance("face_j", game, "face", 0.0, 2.0, face(1.0), body="body_j")
    appearance("face_s", game, "face", 10.0, 12.0, face(0.0, 1.0))
    entities.recluster(db)
    entities.name(db, who(db, game, 1.0), "Jane Doe")
    return who(db, game, 1.0)


@pytest.mark.parametrize(
    ("query", "found"),
    [
        ("Jane Doe", True),
        ("where does jane score?", True),  # a first name no one else has stands for the whole name
        ("@Jane_Doe", True),
        ("#23", True),
        ("what is {jane} doing", True),
        ("John Roe", False),
        ("Doe Smith", False),  # a surname alone is too short to stand for her
        ("number 7", False),
        ("ent_0123456789ab", False),
        ("", False),
    ],
)
def test_resolve_points_only_at_people_it_knows(db, jane, query, found):
    assert entities.resolve(db, query.format(jane=jane)) == ([jane] if found else [])


def test_present_lists_each_person_once_with_the_stretches_they_are_seen(db, video, appearance):
    game = video("v_a")
    appearance("face_1", game, "face", 1.0, 2.0, face(1.0))
    appearance("face_2", game, "face", 2.25, 3.0, face(0.8, 0.6))  # right after face_1: one stretch
    appearance("face_3", game, "face", 6.0, 7.0, face(0.8, 0.0, 0.6))  # back later, past the window's end
    appearance("face_x", game, "face", 1.5, 1.5, face(0.0, 0.0, 0.0, 1.0))  # someone else, briefly
    entities.recluster(db)
    people = entities.present(db, game, 0.0, 6.5)
    assert [(p["start"], p["end"], p["times"]) for p in people] == [
        (0.75, 6.5, [[0.75, 3.25], [5.75, 6.5]]),
        (1.25, 1.75, [[1.25, 1.75]]),
    ]
