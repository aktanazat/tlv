"""tlv at, --clock and tlv more through the command line, over the on-screen text of the test video:
where its scoreboard shows a game time, find and search kept to a window of game time, and
follow-ups that keep some of the last find's clips without the model or find again for someone
else. find and search are stand-ins that record what they were asked, and the model a stand-in
that answers as told, since the real ones need the models."""

import json
from pathlib import Path

import pytest

import tlv.agent
from tlv import cli, media, store
from tlv.agent import model

# The test video's scoreboard, one frame a second: the end of the 3rd, then the 4th from 1:31, with
# a replay at 7 s showing 1:30 again.
SCOREBOARD = {
    0.0: "LAL 96 | 3RD | 0:12 | GSW 95",
    1.0: "LAL 96 | 3RD | 0:11 | GSW 95",
    2.0: "LAL 98 | 4TH | 1:31 | GSW 97",
    3.0: "LAL 98 | 4TH | 1:30 | GSW 97",
    4.0: "LAL 98 | 4TH | 1:29 | GSW 97",
    5.0: "LAL 100 | 4TH | 1:20 | GSW 97",
    7.0: "REPLAY | LAL 98 | 4TH | 1:30 | GSW 97",
    8.0: "LAL 100 | 4TH | 1:05 | GSW 97",
    9.0: "LAKERS 100 | WARRIORS 97",
}


def run(capsys, *argv):
    """A tlv command with --json: its exit code, and its result or what it printed to stderr."""
    code = cli.main([*argv, "--json"])
    out, err = capsys.readouterr()
    return code, (json.loads(out) if code == 0 else err)


def show(video_id, frames):
    with store.transaction(store.connect("tlv")) as db:
        db.executemany(
            "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES (?, ?, ?, 0.1, 0.9, 0.8, 0.05, 1.0)",
            [(video_id, t, text) for t, text in frames.items()],
        )


@pytest.fixture
def scoreboard(video):
    show(video, SCOREBOARD)
    return video


def test_at_finds_the_live_play_and_the_replay_showing_a_game_time_with_a_still_and_clip_of_each(capsys, scoreboard):
    code, result = run(capsys, "at", scoreboard, "4th 1:30", "--cut")
    assert code == 0
    hits = [(hit["t"], hit["start"], hit["end"], hit["shows"]) for hit in result["hits"]]
    assert hits == [(3.0, 2.0, 4.0, "4th 1:30"), (7.0, 7.0, 7.0, "4th 1:30")]
    assert all(Path(hit["frame_path"]).stat().st_size > 0 for hit in result["hits"])
    clips = [media.probe(Path(hit["clip_path"])).duration for hit in result["hits"]]
    assert clips == [pytest.approx(6.0, abs=0.1)] * 2


@pytest.mark.parametrize(
    ("frames", "says"),
    [
        (SCOREBOARD, 'no frame of "Test Pattern" shows 2nd (the game clock is read in 8 frames)\n'),
        ({}, 'no game clock read in "Test Pattern"\n'),
    ],
)
def test_at_says_whether_the_time_is_not_shown_or_the_clock_was_never_read(capsys, video, frames, says):
    show(video, frames)
    assert cli.main(["at", video, "2nd"]) == 0
    assert capsys.readouterr().out == says


@pytest.mark.parametrize("command", ["find", "search"])
def test_find_and_search_look_only_where_the_scoreboard_shows_the_window(
    capsys, monkeypatch, scoreboard, add_video, clip_file, command
):
    add_video("tlv", "Youtube-noclock01", "No Clock", clip_file)
    within = []
    monkeypatch.setattr(tlv.agent, command, lambda query, **asked: within.append(asked["within"]) or {"query": query})
    code, result = run(capsys, command, "a dunk", "--clock", "Q4")
    assert code == 0
    assert within == [{scoreboard: [(1.5, 10.0)]}]
    assert result["clock"] == {"window": "4th", "ranges": {scoreboard: [[1.5, 10.0]]}, "unread": ["Youtube-noclock01"]}


@pytest.mark.parametrize("command", ["find", "search"])
def test_a_window_no_video_shows_is_refused_before_anything_is_searched(capsys, monkeypatch, scoreboard, command):
    monkeypatch.setattr(tlv.agent, command, lambda query, **asked: pytest.fail(f"{command} ran"))
    assert run(capsys, command, "a dunk", "--clock", "1st") == (1, "tlv: no video shows 1st on its scoreboard\n")


def clip(obs_id, video_id, start, end, replays=0):
    """A confirmed clip, with `replays` replays merged into it."""
    replay = {"video_id": video_id, "start": 7.0, "end": 8.0}
    return {"id": obs_id, "video_id": video_id, "start": start, "end": end, "replays": [replay] * replays}


class NoModel:
    def __init__(self, progress=None):
        raise AssertionError("the model was started")


class Answering:
    """The model, answering whatever it is asked with `text`."""

    def __init__(self, text):
        self.text = text

    def __call__(self, progress=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def ask(self, images, prompt, max_tokens):
        return model.Reply(self.text, 0, 0, truncated=False, stopped=False)


@pytest.fixture
def lebron_find(capsys, monkeypatch, scoreboard):
    """`tlv find "LeBron James dunks" --no-cut` run on the scoreboard video, which the captioner calls a
    replay from 6 to 9 s, with LeBron James and Stephen Curry known. Returns what each find was asked."""
    video = scoreboard
    with store.transaction(store.connect("tlv")) as db:
        db.execute("INSERT INTO spans (span_id, video_id, start, end, replay) VALUES ('s1', ?, 6.0, 9.0, 1)", (video,))
        db.executemany(
            "INSERT INTO entities (entity_id, name, canonical_appearance, embedding, updated_at)"
            " VALUES (?, ?, 'a1', x'00', 0)",
            [("ent_lebron", "LeBron James"), ("ent_curry", "Stephen Curry")],
        )
    clips = {
        "LeBron James dunks": [
            clip("obs_3rd", video, 0.0, 1.0),
            clip("obs_live", video, 2.0, 4.5, replays=1),
            clip("obs_replay", video, 6.5, 8.0),
        ],
        "Stephen Curry dunks": [clip("obs_curry", video, 2.5, 4.0), clip("obs_curry_replay", video, 6.5, 8.5)],
    }
    asked = []

    def find(query, **kwargs):
        asked.append((query, kwargs["within"], kwargs["cut"]))
        confirmed = clips[query]
        counts = {"candidates": 5, "checked": 5, "confirmed": len(confirmed), "rejected": 5 - len(confirmed)}
        return {"claim": query, "confirmed": confirmed, "rejected": [], "counts": counts, "seconds": 1.0}

    monkeypatch.setattr(tlv.agent, "find", find)
    assert run(capsys, "find", "LeBron James dunks", "--no-cut")[0] == 0
    return asked


def test_follow_ups_narrow_the_last_find_without_the_model_and_a_new_name_finds_again(
    capsys, monkeypatch, scoreboard, lebron_find
):
    monkeypatch.setattr(model, "Vlm", NoModel)
    seen = []
    for instruction in ["now only fourth quarter", "without the replays", "same but steph"]:
        code, result = run(capsys, "more", instruction)
        assert code == 0, result
        followup = result["followup"]
        clips = [(c["id"], len(c["replays"])) for c in result["confirmed"]]
        seen.append((clips, followup["clock"], followup["no_replays"], followup["read_by"], followup["reran"]))
    assert seen == [
        ([("obs_live", 1), ("obs_replay", 0)], "4th", False, "rules", False),
        ([("obs_live", 0)], "4th", True, "rules", False),
        ([("obs_curry", 0)], "4th", True, "rules", True),
    ]
    assert lebron_find == [
        ("LeBron James dunks", None, False),
        ("Stephen Curry dunks", {scoreboard: [(1.5, 10.0)]}, False),
    ]


@pytest.mark.parametrize(
    ("answer", "kept"),
    [('{"clock": "3rd quarter"}', ["obs_3rd"]), ('{"clock": "purple"}', None)],
)
def test_the_model_reads_what_the_rules_cannot_and_counts_only_a_reading_that_names_a_game_time(
    capsys, monkeypatch, lebron_find, answer, kept
):
    monkeypatch.setattr(model, "Vlm", Answering(answer))
    code, result = run(capsys, "more", "the period before the last one")
    if kept is None:
        assert code == 1
        assert result.startswith('tlv: could not tell what to change from "the period before the last one": say')
    else:
        assert (code, [c["id"] for c in result["confirmed"]], result["followup"]["read_by"]) == (0, kept, "model")


def test_a_follow_up_with_no_find_before_it_says_to_find_first(capsys, home):
    assert run(capsys, "more", "without the replays") == (
        1,
        'tlv: no find to follow up in index "tlv": run tlv find first\n',
    )
