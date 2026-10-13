"""A confirmed moment's id names its video and range, so finding it again replaces its record; a
search narrowed to video-time ranges keeps only the hits inside them; Jev drops a play the commentary
credits to another player, and only when its key is given; a search for a person only sight knows
looks first where the index sees them."""

import json
import re
from contextlib import closing

import numpy as np
import pytest

from tlv import judge, store
from tlv.agent.connect import Clip
from tlv.agent.find import clips_within, credited, observation_id, search
from tlv.agent.inspect import Moment
from tlv.agent.replay import Play
from tlv.agent.sources import Video


def test_an_observation_id_is_obs_and_ten_hex_digits():
    assert re.fullmatch(r"obs_[0-9a-f]{10}", observation_id("Youtube-oyk2QtdUtKE", 288.2, 295.0))


@pytest.mark.parametrize(
    ("other", "same"),
    [
        pytest.param(("Youtube-oyk2QtdUtKE", 288.23, 295.04), True, id="the same range to the tenth of a second"),
        pytest.param(("Youtube-zHBgJqG1nJo", 288.2, 295.0), False, id="another video"),
        pytest.param(("Youtube-oyk2QtdUtKE", 288.5, 295.0), False, id="another start"),
        pytest.param(("Youtube-oyk2QtdUtKE", 288.2, 296.0), False, id="another end"),
    ],
)
def test_an_observation_id_changes_only_with_its_video_and_range(other, same):
    assert (observation_id(*other) == observation_id("Youtube-oyk2QtdUtKE", 288.2, 295.0)) is same


@pytest.mark.parametrize(
    ("clip", "kept"),
    [
        pytest.param(Clip("v1", 6.0, 10.0, 1), False, id="a clip ending where the range starts"),
        pytest.param(Clip("v1", 6.0, 10.5, 1), True, id="a clip reaching half a second into the range"),
        pytest.param(Clip("v1", 19.5, 24.0, 1), True, id="a clip starting half a second before the range ends"),
        pytest.param(Clip("v1", 20.0, 24.0, 1), False, id="a clip starting where the range ends"),
        pytest.param(Clip("v2", 12.0, 14.0, 1), False, id="a clip in a video with no range"),
    ],
)
def test_a_narrowed_search_keeps_only_clips_overlapping_a_range_of_their_video(clip, kept):
    assert clips_within([clip], {"v1": [(10.0, 20.0)], "v3": [(0.0, 99.0)]}) == ([clip] if kept else [])


GAME = Video("v", "Game", "file", "/videos/game.mp4", 600.0)
# Jimmy Butler's three, which the vision model gave to Curry, and Curry's own.
BUTLER = Play(Moment(GAME, 458.2, 459.2, 0.9, "Stephen Curry", "", False, None, (), ()))
CURRY = Play(Moment(GAME, 336.8, 338.8, 0.9, "Stephen Curry", "", False, None, (), ()))


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("TLV_HOME", str(tmp_path))
    monkeypatch.delenv("TLV_INDEX", raising=False)
    with closing(store.connect()) as db:
        yield db


def test_without_the_key_every_play_stays_and_jev_is_not_asked(db, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(judge, "bools", lambda question, states: pytest.fail("Jev was asked"))
    assert credited(db, [BUTLER, CURRY], "Curry makes a three", None) == ([BUTLER, CURRY], [])


def test_jev_drops_a_play_once_its_answer_reaches_the_veto(db, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(judge, "bools", lambda question, states: [judge.CREDIT_VETO, judge.CREDIT_VETO - 0.01])
    assert credited(db, [BUTLER, CURRY], "Curry makes a three", None) == ([CURRY], [BUTLER])


def test_when_jev_cannot_answer_every_play_stays(db, monkeypatch):
    def refuse(question, states):
        raise judge.JudgeError("refused")

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(judge, "bools", refuse)
    assert credited(db, [BUTLER, CURRY], "Curry makes a three", None) == ([BUTLER, CURRY], [])


def film(db, read):
    """A minute of film in 5 s scenes, two of them called scenes by their captions, and coco, a person
    the owner named, seen through most of 20-25 s and for a second of 40-45 s. With `read`, a sign at
    41.5 s reads her name while she is seen, which the index keeps as a clue of her name."""
    db.execute(
        "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
        " VALUES ('v', 'Film', 'file', '/videos/film.mp4', 60, 640, 360, 30, 0, 'ready')"
    )
    captions = {
        0.0: "a quiet street scene at dawn",
        5.0: "the same kitchen scene every morning",
        10.0: "a dog runs across the grass",
        20.0: "a girl reads by the window",
        30.0: "a car drives along the coast",
        40.0: "a girl walks down to the lake",
    }
    for start, caption in captions.items():
        db.execute(
            "INSERT INTO spans (span_id, video_id, start, end, caption) VALUES (?, 'v', ?, ?, ?)",
            (f"v:{start:.3f}", start, start + 5.0, caption),
        )
    blob = store.to_blob(np.ones(4, dtype=np.float32))
    db.execute(
        "INSERT INTO entities (entity_id, name, canonical_appearance, embedding, updated_at)"
        " VALUES ('e_coco', 'coco', 'coco_0', ?, 0)",
        (blob,),
    )
    for n, (start, end) in enumerate([(20.5, 24.5), (41.0, 42.0)]):
        clues = [{"kind": "ocr_name", "value": "coco", "t": 41.5}] if read and n == 1 else []
        db.execute(
            "INSERT INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path, embedding,"
            " clues, quality) VALUES (?, 'v', 'face', ?, ?, ?, '[0.4, 0.1, 0.1, 0.2]', 'crop.jpg', ?, ?, 0.5)",
            (f"coco_{n}", start, end, start, blob, json.dumps(clues)),
        )
        db.execute("INSERT INTO entity_members (appearance_id, entity_id) VALUES (?, 'e_coco')", (f"coco_{n}",))
    if read:
        db.execute(
            "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES ('v', 41.5, 'COCO', 0.4, 0.8, 0.2, 0.1, 1)"
        )
    db.commit()


def places(query):
    return [(hit["start"], hit["end"]) for hit in search(query)["results"]]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "every scene with coco",
            [(20.0, 25.0), (40.0, 45.0), (5.0, 10.0), (0.0, 5.0)],
            id="the scene she fills before the one she crosses, then what the words find",
        ),
        pytest.param("every scene with @coco", [(20.0, 25.0), (40.0, 45.0), (5.0, 10.0), (0.0, 5.0)], id="as @coco"),
        pytest.param(
            "coco walks down to the lake",
            [(40.0, 45.0), (20.0, 25.0)],
            id="among her scenes the words still come first",
        ),
    ],
)
def test_a_person_only_sight_knows_is_looked_for_first_where_the_index_sees_them(db, query, expected):
    """No caption, speech or on-screen text names coco, so no channel of words finds her; the scenes
    the index sees her in come first, then the rest."""
    film(db, read=False)
    assert places(query) == expected


def test_a_person_the_words_name_is_found_by_the_words_alone(db):
    """Once a sign reads coco's name where she is seen, the words find her, as they find a player the
    commentary names; where she is only seen is not moved ahead of them."""
    film(db, read=True)
    assert places("every scene with coco") == [(5.0, 10.0), (40.0, 45.0), (0.0, 5.0)]
