"""Spans: how shots become spans, how a caption answer is read, what each span sees, and which
captions a rerun keeps."""

import json

import numpy as np
import pytest

from tlv import embed, guard, store
from tlv.index import frames, spans, vision


@pytest.mark.parametrize(
    ("shots", "expected"),
    [
        ([(0.0, 5.0)], [(0.0, 5.0)]),
        ([(0.0, 16.0)], [(0.0, 8.0), (8.0, 16.0)]),
        ([(0.0, 20.0)], [(0.0, 20 / 3), (20 / 3, 40 / 3), (40 / 3, 20.0)]),
        ([(0.0, 1.0), (1.0, 2.5), (2.5, 3.0), (3.0, 3.5)], [(0.0, 3.5)]),
        ([(0.0, 1.0), (1.0, 2.0), (2.0, 7.0)], [(0.0, 2.0), (2.0, 7.0)]),
        ([(0.0, 5.0), (5.0, 6.0), (6.0, 11.0)], [(0.0, 5.0), (5.0, 6.0), (6.0, 11.0)]),
        ([], []),
    ],
    ids=[
        "one shot",
        "long shot halves",
        "long shot thirds",
        "short run with leftover",
        "short run",
        "lone short",
        "none",
    ],
)
def test_spans_follow_shots(shots, expected):
    assert spans.build(shots) == pytest.approx(expected)


def person(**fields):
    return {"description": None, "jersey": None, "team": None, "name": None, "name_source": None, **fields}


FENCED = """```json
{"caption": "LeBron hits a three.", "action": "makes a three-pointer", "shot": "Live",
 "people": [{"description": "Lakers player", "jersey": "#23", "team": "Lakers"}]}
```"""

# Cut off by the token limit inside people[], as a pilot answer was.
TRUNCATED = (
    '{"caption": "A player in a white \\"23\\" jersey shoots.", "action": "shoots", "shot": "replay", "people": [{"des'
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            FENCED,
            {
                "caption": "LeBron hits a three.",
                "action": "makes a three-pointer",
                "shot": "live",
                "people": [person(description="Lakers player", jersey="23", team="Lakers")],
            },
        ),
        ('Sure: {"note": 1} {"caption": "Bench players stand.", "shot": "bench"}', {"caption": "Bench players stand."}),
        ('{"caption": "A dunk.", "shot": "close-up", "action": "null"}', {"shot": "other", "action": None}),
        (
            '{"caption": "x", "people": [{"jersey": "#23"}, {"jersey": 23}, {"jersey": " 8 "}, {"jersey": "twenty"}]}',
            {"people": [person(jersey="23"), person(jersey="23"), person(jersey="8"), person()]},
        ),
        (
            '{"caption": "x", "people": [{"name": "Austin Reaves", "name_source": "speech"},'
            ' {"name": "Joel Embiid", "name_source": "guess"}, {"name": "Paul George"},'
            ' {"name": "Anthony Davis", "name_source": "recognized"},'
            ' {"name": "on_screen", "name_source": "on_screen"}]}',
            {"people": [person(name="Austin Reaves", name_source="speech"), person(), person(), person(), person()]},
        ),
        ("A crowd cheers.", {"caption": "A crowd cheers.", "shot": "other", "people": []}),
        (
            TRUNCATED,
            {"caption": 'A player in a white "23" jersey shoots.', "action": "shoots", "shot": "replay", "people": []},
        ),
    ],
    ids=["fenced", "chatter first", "unknown shot", "jersey digits", "names need a source", "no json", "truncated"],
)
def test_caption_answers_are_read_defensively(answer, expected):
    parsed = spans.parse(answer)
    assert {key: parsed[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("ocr", "transcript", "claimed", "expected"),
    [
        ([], "James, he wants it.", person(name="LeBron James", name_source="speech"), {"LeBron James": "speech"}),
        (["LAKERS 108"], "Reaves the hockey assist.", person(name="Anthony Davis", name_source="speech"), {}),
        (
            ["LAKERS 108"],
            "LeBron again.",
            person(name="LeBron James", name_source="on_screen"),
            {"LeBron James": "speech"},
        ),
        (
            ["LEBRON JAMES 29 PTS"],
            "LeBron again.",
            person(name="LeBron James", name_source="speech"),
            {"LeBron James": "speech"},
        ),
        (
            ["ANDRE DRUMMOND 11 POINTS"],
            "",
            person(name="Andre Drummond", name_source="on_screen"),
            {"Andre Drummond": "on_screen"},
        ),
    ],
    ids=["heard", "neither read nor heard", "claimed read, only heard", "read and heard keeps the claim", "read"],
)
def test_a_name_stands_only_on_what_the_span_shows_or_says(ocr, transcript, claimed, expected):
    grounded = spans.ground({"people": [claimed]}, spans.Inputs([], transcript, ocr, []))["people"][0]
    assert ({grounded["name"]: grounded["name_source"]} if grounded["name"] else {}) == expected


@pytest.mark.parametrize(
    ("ocr", "transcript", "team", "expected"),
    [
        (["LOS ANGELES LAKERS", "4TH :27.3"], "", "Lakers", "Lakers"),
        (["LAKERS 108"], "over the Pelicans", "New Orleans Pelicans", None),
    ],
    ids=["on screen", "only said"],
)
def test_a_team_stands_only_when_it_is_on_screen(ocr, transcript, team, expected):
    grounded = spans.ground({"people": [person(team=team)]}, spans.Inputs([], transcript, ocr, []))["people"][0]
    assert grounded["team"] == expected


def add_rows(db, table, rows):
    columns = rows[0].keys()
    db.executemany(
        f"INSERT INTO {table} (video_id, {', '.join(columns)}) VALUES ('v_test', {', '.join('?' * len(columns))})",
        [tuple(row.values()) for row in rows],
    )


def ocr(t, text):
    return {"t": t, "text": text, "x": 0.1, "y": 0.9, "w": 0.1, "h": 0.05, "conf": 0.9}


def speech(start, end, *words):
    timed = [{"w": word, "s": start, "e": end} for word in words]
    return {"start": start, "end": end, "text": " ".join(words), "words": json.dumps(timed)}


@pytest.fixture
def scene(db, video):
    """Two 4 s shots with frames every 0.5 s, two words around the cut, and a score bug."""
    add_rows(db, "shots", [{"idx": 0, "start": 0.0, "end": 4.0}, {"idx": 1, "start": 4.0, "end": 8.0}])
    add_rows(db, "frames", [{"t": i * 0.5 + 0.25, "path": f"t{i}.jpg", "shot_idx": i // 8} for i in range(16)])
    add_rows(db, "speech", [speech(3.7, 4.1, "left"), speech(3.9, 4.3, "right")])
    add_rows(db, "ocr", [ocr(1.25, "LAL 102")])
    return db, video


def test_a_word_belongs_to_the_span_holding_its_middle(scene):
    db, _ = scene
    transcripts = [
        spans.gather(db, "v_test", start, end, vision.text_times(8.0)).transcript
        for start, end in [(0.0, 4.0), (4.0, 8.0)]
    ]
    assert transcripts == ["left", "right"]


@pytest.fixture
def captioner(monkeypatch):
    """Stands in for the vision-language and text-embedding models, answering a job of one question
    or a batch as the model worker does. Captions are numbered in the order they are made, "jobs"
    lists how many questions each job asked and "pictures" how many frames each question showed,
    and a test may set the shot kind the model answers and the job at which it stops."""
    answer = {"shot": "replay", "made": 0, "jobs": [], "pictures": [], "fails_at": None}

    def caption():
        answer["made"] += 1
        return {"text": json.dumps({"caption": f"caption {answer['made']}", "shot": answer["shot"]})}

    class Model:
        def __init__(self, module, progress=None):
            assert module == "tlv.models.vlm"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def call(self, job):
            questions = job.get("batch", [job])
            answer["jobs"].append(len(questions))
            answer["pictures"] += [len(question["images"]) for question in questions]
            if len(answer["jobs"]) == answer["fails_at"]:
                raise RuntimeError("the model stopped")
            return {"results": [caption() for _ in job["batch"]]} if "batch" in job else caption()

    monkeypatch.setattr(guard, "ModelProcess", Model)
    monkeypatch.setattr(embed, "sentences", lambda texts, progress=None: np.ones((len(texts), 4), dtype=np.float32))
    return answer


def captions(db):
    return {row["span_id"]: row["caption"] for row in db.execute("SELECT span_id, caption FROM spans")}


@pytest.mark.parametrize(
    ("table", "row", "recaption", "expected"),
    [
        (None, None, False, set()),
        ("ocr", ocr(5.25, "LAL 105"), False, {"v_test:4.000"}),
        ("speech", speech(1.0, 1.5, "three"), False, {"v_test:0.000"}),
        (None, None, True, {"v_test:0.000", "v_test:4.000"}),
    ],
    ids=["unchanged", "new on-screen text", "new speech", "recaption"],
)
def test_a_rerun_recaptions_only_spans_whose_inputs_changed(scene, captioner, table, row, recaption, expected):
    db, video = scene
    spans.run(db, video, None)
    before = captions(db)
    if table:
        add_rows(db, table, [row])
    spans.run(db, video, None, recaption=recaption)
    after = captions(db)
    assert {span_id for span_id in after if after[span_id] != before[span_id]} == expected


def test_each_span_has_one_caption_observation_over_its_range(scene, captioner):
    db, video = scene
    spans.run(db, video, None)
    observed = db.execute("SELECT obs_id, start, end, text FROM observations ORDER BY start").fetchall()
    assert [tuple(row) for row in observed] == [
        ("caption:v_test:0.000", 0.0, 4.0, "caption 1"),
        ("caption:v_test:4.000", 4.0, 8.0, "caption 2"),
    ]


def in_order(db):
    return [row["caption"] for row in db.execute("SELECT caption FROM spans ORDER BY start")]


def test_a_two_minute_shot_is_captioned_in_two_model_jobs(db, video, captioner):
    # The model writes a job's answers together, so a span asked in a job of its own took about 30%
    # more time in the model: fifteen spans go in jobs of eight, each answer to its span.
    add_rows(db, "shots", [{"idx": 0, "start": 0.0, "end": 120.0}])
    spans.run(db, video, None)
    assert (captioner["jobs"], in_order(db)) == ([8, 7], [f"caption {n}" for n in range(1, 16)])


def test_an_interrupted_run_keeps_the_jobs_done_and_the_rerun_captions_the_rest(db, video, captioner):
    add_rows(db, "shots", [{"idx": 0, "start": 0.0, "end": 120.0}])
    captioner["fails_at"] = 2
    with pytest.raises(RuntimeError, match="the model stopped"):
        spans.run(db, video, None)
    interrupted = in_order(db)
    captioner["fails_at"] = None
    spans.run(db, video, None)
    made = [f"caption {n}" for n in range(1, 16)]
    assert (interrupted, in_order(db)) == ([*made[:8], *[None] * 7], made)


SLIDE = ["Backpropagation", "An algorithm that trains the weights of a neural network"]
LINE = ["cargo build --release"]


def two_minutes(db, turn_per_s, screen):
    """A two-minute shot with a frame every 0.5 s, its picture turning `turn_per_s` radians a second
    in SigLIP space, and `screen(t)` the lines each text read finds."""
    db.execute("UPDATE videos SET duration = 120.0 WHERE video_id = 'v_test'")
    add_rows(db, "shots", [{"idx": 0, "start": 0.0, "end": 120.0}])
    pictures = [np.array([np.cos(turn_per_s * t), np.sin(turn_per_s * t)]) for t in frames.times(120.0)]
    add_rows(
        db,
        "frames",
        [
            {"t": t, "path": f"t{t}.jpg", "shot_idx": 0, "embedding": store.to_blob(picture)}
            for t, picture in zip(frames.times(120.0), pictures, strict=True)
        ],
    )
    add_rows(db, "ocr", [ocr(t, line) for t in vision.text_times(120.0) for line in screen(t)])
    return db.execute("SELECT * FROM videos WHERE video_id = 'v_test'").fetchone()


@pytest.mark.parametrize(
    ("turn_per_s", "screen", "pictures"),
    [
        # The guard: a still slide costs 15 pictures in two minutes (7.5 a minute of video), not 60.
        (0.0, lambda t: SLIDE, [1] * 15),
        (0.0, lambda t: SLIDE + (LINE if t > 5 else []), [2] + [1] * 14),
        # A frame is the picture a quarter second past its time, and text is read once a second, so
        # a line first read at 2.25 s may have come up after the 1.25 s frame, and one first read at
        # 8.25 s before the 7.25 s frame.
        (0.0, lambda t: SLIDE + (LINE if t > 2 else []), [2] + [1] * 14),
        (0.0, lambda t: SLIDE + (LINE if t > 8 else []), [2] + [1] * 14),
        (0.0, lambda t: LINE if t < 5 else [], [2] + [1] * 14),
        # Each picture is within 0.995 of the one 2 s before it, but not of the one 4 s before.
        (0.03, lambda t: SLIDE, [2] * 15),
        (1.0, lambda t: SLIDE, [4] * 15),
    ],
    ids=[
        "still slide",
        "a line comes up",
        "a line comes up just after a picture",
        "a line comes up just before a picture",
        "the text goes away",
        "slow pan",
        "moving picture",
    ],
)
def test_a_frame_repeating_the_picture_and_text_shown_before_it_is_left_out(
    db, video, captioner, turn_per_s, screen, pictures
):
    spans.run(db, two_minutes(db, turn_per_s, screen), None)
    assert captioner["pictures"] == pictures


@pytest.mark.parametrize(("shot", "replay"), [("replay", 1), ("live", 0), ("crowd", None)])
def test_the_shot_kind_marks_replays(scene, captioner, shot, replay):
    db, video = scene
    captioner["shot"] = shot
    spans.run(db, video, None)
    assert [row["replay"] for row in db.execute("SELECT replay FROM spans")] == [replay, replay]
