"""What prompts and evidence read from the index for a time range."""

import json
from contextlib import closing

import pytest

from tlv import store
from tlv.agent import sources
from tlv.agent.sources import Line


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("TLV_HOME", str(tmp_path))
    monkeypatch.delenv("TLV_INDEX", raising=False)
    with closing(store.connect()) as db:
        db.execute(
            "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
            " VALUES ('v', 'Game', 'https://example.com/game', '/videos/game.mp4', 60, 640, 360, 30, 0, 'ready')"
        )
        db.commit()
        yield db


def add_text(db, t, text, x, y):
    db.execute(
        "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES ('v', ?, ?, ?, ?, 0.1, 0.05, 1)", (t, text, x, y)
    )


def test_screen_text_reads_each_frame_top_to_bottom_and_skips_repeats(db):
    add_text(db, 1.25, "PHI 102", 0.1, 0.9)
    add_text(db, 1.25, "LIVE", 0.8, 0.05)
    add_text(db, 1.25, "LAL 100", 0.3, 0.9)
    add_text(db, 2.25, "PHI 102", 0.1, 0.9)
    add_text(db, 2.25, "LIVE", 0.8, 0.05)
    add_text(db, 2.25, "LAL 100", 0.3, 0.9)
    add_text(db, 3.25, "PHI 105", 0.1, 0.9)
    assert sources.screen_text(db, "v", 0, 10) == [
        Line(1.25, 1.25, "LIVE | PHI 102 | LAL 100"),
        Line(3.25, 3.25, "PHI 105"),
    ]


def test_speech_is_cut_to_the_words_spoken_inside_the_range(db):
    words = [{"w": w, "s": s, "e": s + 0.4} for w, s in [("LeBron", 1.0), ("for", 2.0), ("three", 3.0), ("yes", 6.0)]]
    db.execute(
        "INSERT INTO speech (video_id, start, end, text, words) VALUES ('v', 1.0, 6.4, 'LeBron for three yes', ?)",
        (json.dumps(words),),
    )
    assert sources.speech(db, "v", 1.8, 4.0) == [Line(2.0, 3.4, "for three")]


@pytest.mark.parametrize(
    ("replays", "expected"),
    [
        pytest.param([(8, 16)], True, id="replay over most of the range"),
        pytest.param([(10, 16)], True, id="replay over half of the range"),
        pytest.param([(5, 9), (12, 20)], True, id="replays that add up to half"),
        pytest.param([(11, 16)], False, id="replay over a third of the range"),
        pytest.param([], False, id="live footage only"),
    ],
)
def test_a_range_is_a_replay_when_the_captioner_called_half_of_it_one(db, replays, expected):
    db.execute("INSERT INTO spans (span_id, video_id, start, end, replay) VALUES ('live', 'v', 0, 30, 0)")
    for i, (start, end) in enumerate(replays):
        db.execute(
            "INSERT INTO spans (span_id, video_id, start, end, replay) VALUES (?, 'v', ?, ?, 1)", (f"r{i}", start, end)
        )
    assert sources.mostly_replay(db, "v", 7, 13) is expected


def test_text_that_fits_is_kept_whole():
    lines = [Line(i, i + 1, "x" * 10) for i in range(10)]
    assert sources.fit(lines, 100) == lines


def test_text_past_the_budget_keeps_lines_spread_over_the_whole_stretch():
    lines = [Line(i, i + 1, "x" * 100) for i in range(100)]
    kept = sources.fit(lines, 2500)
    assert sum(len(line.text) for line in kept) <= 2500
    assert kept[0].start == 0 and kept[-1].start >= 90
