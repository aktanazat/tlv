"""tlv person: one person across the videos, read from the rows indexing and find stored."""

import json

import numpy as np
import pytest
from PIL import Image

from tlv import cli, store
from tlv.entities import timeline
from tlv.entities.timeline import Sighting

VECTOR = np.ones(4)
ADA = "ent_000000000ada"


def person(capsys, *argv):
    """`tlv person ... --json`, which must succeed, as its result."""
    assert cli.main(["person", *argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def say(db, video_id, start, text):
    """A transcript segment whose words are half a second apart from `start`."""
    words = [{"w": f" {w}", "s": start + i * 0.5, "e": start + i * 0.5 + 0.4} for i, w in enumerate(text.split())]
    db.execute(
        "INSERT INTO speech (video_id, start, end, text, words) VALUES (?, ?, ?, ?, ?)",
        (video_id, start, words[-1]["e"], text, json.dumps(words)),
    )


def show(db, video_id, t, text, y=0.8):
    db.execute(
        "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES (?, ?, ?, 0.1, ?, 0.3, 0.05, 1.0)",
        (video_id, t, text, y),
    )


def observe(db, obs_id, video_id, start, end, text, who, entity_id=None, kind="verified"):
    db.execute(
        "INSERT INTO observations (obs_id, video_id, start, end, entity_id, kind, text, confidence, evidence,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0.9, ?, ?)",
        (obs_id, video_id, start, end, entity_id, kind, text, json.dumps({"who": who}), store.now()),
    )


@pytest.fixture
def named(db):
    """Stores a person the entities stage found: named(entity_id, name, *appearance_ids)."""

    def add(entity_id, name, *members):
        db.execute(
            "INSERT INTO entities (entity_id, name, canonical_appearance, embedding, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (entity_id, name, members[0], store.to_blob(VECTOR), store.now()),
        )
        db.executemany("INSERT INTO entity_members VALUES (?, ?)", [(member, entity_id) for member in members])

    return add


@pytest.fixture
def ada(video, appearance, named):
    """Videos va and vb, added in that order, and Ada Lovelace: a face at 10-12 s and a body at
    13-14 s of va, and one frame of her face at 5 s of vb."""
    video("va")
    video("vb")
    appearance("a1", "va", "face", 10.0, 12.0, VECTOR, quality=0.6)
    appearance("a2", "va", "body", 13.0, 14.0, VECTOR, height=0.6)
    appearance("b1", "vb", "face", 5.0, 5.0, VECTOR, quality=0.6)
    named(ADA, "Ada Lovelace", "a1", "a2", "b1")


@pytest.fixture
def byron(appearance, named):
    """A second named person who shares Ada's last name."""
    appearance("o1", "va", "body", 50.0, 51.0, VECTOR)
    named("ent_000000000b0b", "Byron Lovelace", "o1")


def seen(start, end, video_id="v", crop="c.jpg", sharpness=0.0, size=0.5):
    return Sighting(video_id, start, end, start, crop, sharpness, size)


@pytest.mark.parametrize(
    ("second", "stretches"),
    [
        pytest.param(seen(4.0, 5.0), [(1.0, 5.0)], id="two seconds unseen"),
        pytest.param(seen(4.01, 5.0), [(1.0, 2.0), (4.01, 5.0)], id="just over two seconds"),
        pytest.param(seen(2.5, 3.0, video_id="w"), [(1.0, 2.0), (2.5, 3.0)], id="another video"),
    ],
)
def test_sightings_join_into_one_stretch_across_at_most_two_unseen_seconds(second, stretches):
    found = timeline.presences([seen(1.0, 2.0), second])
    assert [(stretch.start, stretch.end) for stretch in found] == stretches


def test_seconds_on_screen_leave_out_the_unseen_moments_inside_a_stretch():
    [stretch] = timeline.presences([seen(1.0, 2.0), seen(3.0, 3.5), seen(3.2, 4.0)])
    assert (stretch.start, stretch.end, stretch.seen) == (1.0, 4.0, 2.0)


@pytest.mark.parametrize(
    ("sightings", "crop"),
    [
        pytest.param(
            [seen(1.0, 2.0, crop="big", size=0.9), seen(2.0, 3.0, crop="sharp", sharpness=0.6, size=0.1)],
            "sharp",
            id="a sharp face over a bigger crop",
        ),
        pytest.param(
            [seen(1.0, 2.0, crop="sharp", sharpness=0.4), seen(2.0, 3.0, crop="sharper", sharpness=0.8)],
            "sharper",
            id="the sharper of two faces",
        ),
        pytest.param([seen(1.0, 2.0, crop="small"), seen(2.0, 3.0, crop="big", size=0.9)], "big", id="no sharp face"),
    ],
)
def test_a_stretch_keeps_its_sharpest_face_else_its_biggest_crop(sightings, crop):
    [stretch] = timeline.presences(sightings)
    assert stretch.crop_path == crop


def test_a_person_has_their_stretches_and_seconds_on_screen_per_video(ada, capsys):
    result = person(capsys, "@Lovelace")
    assert (result["entity_ids"], result["name"], result["seconds"]) == ([ADA], "Ada Lovelace", 4.5)
    assert [(v["video_id"], v["seconds"]) for v in result["videos"]] == [("va", 4.0), ("vb", 0.5)]
    stretches = [(p["video_id"], p["start"], p["end"]) for p in result["presences"]]
    assert stretches == [("va", 9.75, 14.25), ("vb", 4.75, 5.25)]


def test_clusters_that_share_a_name_are_one_persons_timeline(video, appearance, named, capsys):
    video("va")
    video("vb")
    appearance("a1", "va", "face", 10.0, 12.0, VECTOR)
    appearance("b1", "vb", "face", 5.0, 6.0, VECTOR)
    named(ADA, "Ada Lovelace", "a1")
    named("ent_00000000ada2", "Ada Lovelace", "b1")
    assert [p["video_id"] for p in person(capsys, "Ada Lovelace")["presences"]] == ["va", "vb"]


@pytest.mark.parametrize(
    ("kind", "text", "others", "start"),
    [
        pytest.param("screen", "LOVELACE", (), 6.0, id="the last name alone"),
        pytest.param("screen", "LOVELACE", ("Byron Lovelace",), None, id="a word another named person shares"),
        pytest.param("screen", "ADA LOVELACE", ("Byron Lovelace",), 6.0, id="the full name"),
        pytest.param("screen", "ADA", (), None, id="a word under four letters"),
        pytest.param("screen", "LOVELACES", (), None, id="part of a word"),
        pytest.param("screen", "LOVELACÉ", (), 6.0, id="accented"),
        pytest.param("speech", "and Lovelace's engine", (), 4.5, id="said, possessive"),
    ],
)
def test_the_name_counts_as_said_or_shown_by_the_rule_clues_name_people_by(
    db, video, appearance, named, capsys, kind, text, others, start
):
    video("va")
    appearance("a1", "va", "face", 1.0, 2.0, VECTOR)
    named(ADA, "Ada Lovelace", "a1")
    for i, other in enumerate(others):
        appearance(f"o{i}", "va", "body", 50.0, 51.0, VECTOR)
        named(f"ent_00000000000{i}", other, f"o{i}")
    if kind == "screen":
        show(db, "va", 6.0, text)
    else:
        say(db, "va", 4.0, text)
    mentions = person(capsys, "@Ada_Lovelace")["mentions"]
    assert [(m["kind"], m["start"]) for m in mentions] == ([] if start is None else [(kind, start)])


def test_a_name_split_over_two_lines_of_one_frame_reads_whole(db, ada, byron, capsys):
    show(db, "va", 6.0, "ADA", y=0.80)
    show(db, "va", 6.0, "LOVELACE", y=0.85)
    mentions = person(capsys, "@Ada_Lovelace")["mentions"]
    assert [(m["kind"], m["start"], m["text"]) for m in mentions] == [("screen", 6.0, "ADA | LOVELACE")]


def test_frames_showing_the_name_at_most_two_seconds_apart_are_one_showing(db, ada, capsys):
    for t in (6.0, 7.0, 9.0, 11.5):
        show(db, "va", t, "LOVELACE")
    mentions = person(capsys, "@Lovelace")["mentions"]
    assert [(m["start"], m["end"]) for m in mentions] == [(6.0, 9.0), (11.5, 11.5)]


def test_verified_moments_are_those_tied_to_or_naming_the_person_merged_where_they_overlap(db, ada, capsys):
    observe(db, "o1", "va", 20.0, 25.0, "she writes the first program", "unknown", entity_id=ADA)
    observe(db, "o2", "va", 24.0, 28.0, "a woman explains the engine", "Ada Lovelace")
    observe(db, "o3", "va", 40.0, 42.0, "ada lovelace smiles", "unknown")
    observe(db, "o4", "va", 41.0, 45.0, "babbage builds the engine", "Charles Babbage")
    observe(db, "o5", "va", 41.0, 44.0, "Ada Lovelace waves", None, kind="caption")
    verified = person(capsys, "@Lovelace")["verified"]
    assert [(m["start"], m["end"], m["claims"], m["obs_ids"]) for m in verified] == [
        (20.0, 28.0, ["she writes the first program", "a woman explains the engine"], ["o1", "o2"]),
        (40.0, 42.0, ["ada lovelace smiles"], ["o3"]),
    ]


def test_video_limits_every_part_of_the_timeline_to_those_videos(db, ada, capsys):
    for video_id in ("va", "vb"):
        show(db, video_id, 6.0, "LOVELACE")
        say(db, video_id, 4.0, "here is Lovelace")
        observe(db, f"o_{video_id}", video_id, 4.0, 7.0, "lovelace smiles", "Ada Lovelace")
    result = person(capsys, "@Lovelace", "--video", "vb")
    parts = ("videos", "presences", "mentions", "verified")
    assert {part: {item["video_id"] for item in result[part]} for part in parts} == {part: {"vb"} for part in parts}


@pytest.mark.parametrize(
    ("who", "message"),
    [
        pytest.param(
            "@Nobody",
            'No person "@Nobody" in this index. Named people: Ada Lovelace, Byron Lovelace. tlv entities lists '
            "everyone, named or not.",
            id="nobody by that name",
        ),
        pytest.param(
            "Ada Lovelace and Byron Lovelace",
            '"Ada Lovelace and Byron Lovelace" could be Ada Lovelace or Byron Lovelace: give one full name or an '
            "entity id",
            id="two people",
        ),
    ],
)
def test_a_person_the_index_cannot_pin_down_exits_1_saying_why(ada, byron, capsys, who, message):
    assert cli.main(["person", who]) == 1
    assert capsys.readouterr().err == f"tlv: {message}\n"


def test_the_strip_shows_each_stretch_by_its_clearest_crop_in_time_order(video, appearance, named, tmp_path, capsys):
    video("va")
    video("vb")
    colors = {"red": (220, 30, 30), "blue": (30, 30, 220), "green": (30, 200, 30), "yellow": (230, 220, 30)}
    for appearance_id, video_id, kind, start, end, color in [
        ("r1", "va", "face", 10.0, 10.5, "red"),
        ("b1", "va", "body", 11.0, 12.0, "blue"),
        ("g1", "va", "body", 30.0, 31.0, "green"),
        ("y1", "vb", "body", 5.0, 6.0, "yellow"),
    ]:
        appearance(appearance_id, video_id, kind, start, end, VECTOR, height=0.9, quality=0.6)
        Image.new("RGB", (60, 80), colors[color]).save(tmp_path / f"{appearance_id}.jpg")
    named(ADA, "Ada Lovelace", "r1", "b1", "g1", "y1")
    pixels = np.asarray(Image.open(person(capsys, "@Lovelace", "--strip")["strip_path"]).convert("RGB")).astype(int)
    centers = {}
    for color, rgb in colors.items():
        where = np.argwhere(np.all(np.abs(pixels - rgb) < 40, axis=-1))
        if len(where):
            centers[color] = where.mean(axis=0)
    assert set(centers) == {"red", "green", "yellow"}
    (red_y, red_x), (green_y, green_x), (yellow_y, _) = centers["red"], centers["green"], centers["yellow"]
    assert abs(red_y - green_y) < 5 and red_x < green_x
    assert yellow_y > green_y + 50
