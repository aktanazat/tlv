"""tlv reel: a find's confirmed moments cut to a song's beat, with where each clip comes from."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tlv import cli, reel, store


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _clicks(path: Path, bpm: float, seconds: float, offset: float = 0.0) -> Path:
    """A click track: a 10 ms, 1 kHz tick on every beat from `offset` seconds in."""
    tick = f"0.9*sin(2*PI*1000*t)*lt(mod(t-{offset},{60 / bpm}),0.01)"
    _ffmpeg("-f", "lavfi", "-i", f"aevalsrc='{tick}':s=16000:d={seconds}", str(path))
    return path


def _found(obs_id: str, video_id: str, start: float, end: float) -> dict:
    """A confirmed result as `tlv find --json` prints it."""
    return {
        "id": obs_id,
        "video_id": video_id,
        "title": "Test Pattern",
        "start": start,
        "end": end,
        "claim": "The pattern scrolls",
        "who": None,
        "entity_id": None,
        "confidence": 0.9,
        "evidence": {
            "summary": "",
            "transcript": [],
            "sounds": [],
            "ocr": [],
            "frames": [],
        },
    }


def _saved(path: Path, *found: dict) -> Path:
    path.write_text(json.dumps({"claim": "The pattern scrolls", "confirmed": list(found)}))
    return path


def _two_moments(path: Path, video_id: str) -> Path:
    return _saved(path, _found("obs_a", video_id, 2.0, 3.0), _found("obs_b", video_id, 6.0, 7.5))


def _stream(path: Path, kind: str, field: str) -> str:
    command = ["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", f"stream={field}", "-of", "csv=p=0"]
    return subprocess.run([*command, str(path)], capture_output=True, text=True, check=True).stdout.strip()


def _ticks(path: Path) -> np.ndarray:
    """When each tick in the sound of a video starts, in seconds."""
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    pcm = np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, np.int16)
    loud = np.flatnonzero(np.abs(pcm) > 0.2 * 32768) / 16000
    return loud[np.concatenate([[True], np.diff(loud) > 0.1])]


def _moment(obs_id: str, start: float, end: float, video_id: str = "v1", claim: str = "Makes a three") -> reel.Moment:
    return reel.Moment(obs_id, video_id, start, end, 0.9, claim, "", None)


def _video(video_id: str = "v1", shots: tuple[float, ...] = ()) -> reel.Video:
    return reel.Video(video_id, "Game", f"https://example.com/{video_id}", Path("game.mp4"), 600.0, shots, True)


@pytest.mark.parametrize(("bpm", "offset"), [(120, 0.0), (95, 0.3)])
def test_beats_reads_the_tempo_and_first_beat_of_a_click_track(tmp_path, home, bpm, offset):
    grid = reel.beats(_clicks(tmp_path / "clicks.wav", bpm, 20, offset))
    assert (grid.bpm, grid.first) == (pytest.approx(bpm, abs=0.1), pytest.approx(offset, abs=0.01))


def test_beats_refuses_a_song_without_a_steady_beat(tmp_path, home):
    noise = tmp_path / "noise.wav"
    _ffmpeg("-f", "lavfi", "-i", "anoisesrc=d=20:c=white:r=16000:seed=1", str(noise))
    with pytest.raises(ValueError, match="no steady beat"):
        reel.beats(noise)


def test_plan_moves_clip_edges_onto_camera_cuts_and_cuts_on_beats():
    """A clip runs a second either side of its moment. An edge within half a second of a camera
    cut moves onto it; the clip then lasts whole beats, losing its part beat when it ends on a
    camera cut, and the next clip starts where it ends."""
    grid = reel.Beats(bpm=120.0, first=0.0, duration=60.0)
    cuts = reel.plan(
        [_moment("a", 4.2, 6.6), _moment("b", 12.0, 13.1)], {"v1": _video(shots=(3.0, 7.9))}, grid, "found"
    )
    assert [t for c in cuts for t in (c.start, c.end, c.at)] == pytest.approx([3.0, 7.5, 0.0, 11.0, 14.0, 4.5])


def test_plan_in_video_order_keeps_videos_in_found_order_each_in_time_order():
    moments = [
        _moment("b2", 50, 51, "vb"),
        _moment("a2", 40, 41, "va"),
        _moment("b1", 10, 11, "vb"),
        _moment("a1", 5, 6, "va"),
    ]
    cuts = reel.plan(moments, {"va": _video("va"), "vb": _video("vb")}, None, "video")
    assert [c.moment.obs_id for c in cuts] == ["b1", "b2", "a1", "a2"]


def test_a_long_lower_third_stays_clear_of_the_score_bug(tmp_path):
    """Broadcasts show the score bottom right, from about x = 940 of 1280. The lower third starts
    at x = 48 and ends before it however long the claim and the title are."""
    claim = "Stephen Curry pulls up from well beyond the arc over two defenders and makes a three pointer"
    title = "WARRIORS at LAKERS | FULL GAME HIGHLIGHTS | April 3, 2025 | NBA Regular Season 2024-25"
    reel.lower_third(_moment("a", 1, 2, claim=claim), title, tmp_path / "third.png")
    assert reel.MARGIN + Image.open(tmp_path / "third.png").width < 940


def test_reel_in_game_order_follows_the_scoreboard(tmp_path, video, capsys):
    """Clips go by period and game clock, overtime after the 4th and clips without a scoreboard
    last. Each takes the scoreboard the index read nearest its middle in the frames its clip shows,
    a second either side of the moment: the clip can open on the play before a camera cut (4.5 s),
    and a stat line can cover the scoreboard while the moment plays (the 1st, read at 5.75 s)."""
    scoreboard = {
        2.75: "LAL 110 GSW 110 OT 2:00",
        4.5: "4TH 9:30",
        5.0: "4TH 2:07",
        5.75: "1ST 10:00",
        8.25: "4TH 9:00",
    }
    with store.transaction(store.connect("tlv")) as db:
        db.executemany(
            "INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES (?, ?, ?, 0.7, 0.9, 0.2, 0.05, 1.0)",
            [(video, t, text) for t, text in scoreboard.items()],
        )
    saved = _saved(
        tmp_path / "find.json",
        _found("unread", video, 1.0, 1.5),
        _found("ot", video, 2.5, 3.0),
        _found("4th-late", video, 4.5, 5.5),
        _found("1st", video, 6.5, 7.0),
        _found("4th-early", video, 8.0, 8.5),
    )
    out = tmp_path / "reel.mp4"
    assert cli.main(["reel", "--from", str(saved), "--order", "game", "--out", str(out), "--json"]) == 0
    cuts = json.loads(capsys.readouterr().out)["cuts"]
    assert [(c["obs_id"], c["game_clock"]) for c in cuts] == [
        ("1st", "1st 10:00"),
        ("4th-early", "4th 9:00"),
        ("4th-late", "4th 2:07"),
        ("ot", "OT 2:00"),
        ("unread", None),
    ]


def test_reel_of_a_saved_eval_run_lasts_whole_beats_frame_for_frame(tmp_path, video, capsys):
    """Each clip lasts whole beats of the song (0.5 s at 120 bpm), so every cut lands on a beat,
    and the file holds exactly those frames at 30 fps, with sound to the end."""
    output = json.loads(_two_moments(tmp_path / "find.json", video).read_text())
    runs = [
        {"id": "scrolls", "query": "the pattern scrolls", "then": [], "output": output},
        {"id": "other", "query": "something else", "then": [], "output": {"claim": "Something else", "confirmed": []}},
    ]
    saved = tmp_path / "runs.json"
    saved.write_text(json.dumps({"runs": runs}))
    song, out = _clicks(tmp_path / "clicks.wav", 120, 20), tmp_path / "reel.mp4"
    assert cli.main(["reel", "scrolls", "--from", str(saved), "--music", str(song), "--out", str(out), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [(c["start"], c["end"], c["reel_start"]) for c in result["cuts"]] == [(1.0, 4.0, 0.0), (5.0, 8.5, 3.0)]
    assert (_stream(out, "v", "nb_frames"), float(_stream(out, "a", "duration"))) == (
        "195",
        pytest.approx(6.5, abs=0.05),
    )


def test_a_song_shorter_than_the_reel_loops_on_its_beat(tmp_path, video):
    """The song lasts 5.3 s and the reel 6.5 s: the song loops whole bars from its first beat, so
    its ticks stay half a second apart through the loop and carry on to the end."""
    song, out = _clicks(tmp_path / "clicks.wav", 120, 5.3), tmp_path / "reel.mp4"
    saved = _two_moments(tmp_path / "find.json", video)
    assert cli.main(["reel", "--from", str(saved), "--music", str(song), "--mute-video", "--out", str(out)]) == 0
    ticks = _ticks(out)
    assert (np.diff(ticks).tolist(), ticks[-1] > 5.9) == (pytest.approx([0.5] * (len(ticks) - 1), abs=0.02), True)


def test_reel_lists_where_each_clip_comes_from_beside_it(tmp_path, video):
    out = tmp_path / "reel.mp4"
    assert cli.main(["reel", "--from", str(_two_moments(tmp_path / "find.json", video)), "--out", str(out)]) == 0
    listed = json.loads(out.with_suffix(".sources.json").read_text())
    source = "https://example.com/Youtube-pattern01.mp4"
    assert [(c["reel_timecode"], c["source"], c["timecode"], c["obs_id"]) for c in listed] == [
        ("00:00:00.0 -> 00:00:03.0", source, "00:00:01.0 -> 00:00:04.0", "obs_a"),
        ("00:00:03.0 -> 00:00:06.5", source, "00:00:05.0 -> 00:00:08.5", "obs_b"),
    ]
    assert f"{source} at 00:00:05.0 -> 00:00:08.5" in out.with_suffix(".sources.md").read_text()


@pytest.mark.parametrize(
    ("found", "args", "message"),
    [
        ([], [], 'nothing found to cut for "The pattern scrolls"'),
        (None, ["--bpm", "120"], "bpm is the tempo of the music: pass music too"),
    ],
)
def test_reel_refuses_what_it_cannot_cut(tmp_path, video, capsys, found, args, message):
    saved = _two_moments(tmp_path / "find.json", video) if found is None else _saved(tmp_path / "find.json", *found)
    assert cli.main(["reel", "--from", str(saved), "--out", str(tmp_path / "reel.mp4"), *args]) == 1
    assert capsys.readouterr().err == f"tlv: {message}\n"
