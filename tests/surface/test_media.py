"""list, cut and frames work from the store and the local copy: the right database, ranges
clamped to the video, files named like the TypeScript tool's, and plain errors for a bad range
or a missing file."""

import json
from pathlib import Path

import pytest
from PIL import Image

from tlv import cli, media


def run(capsys, *argv):
    code = cli.main([*argv, "--json"])
    out, err = capsys.readouterr()
    return code, (json.loads(out) if code == 0 else err)


def test_cut_clamps_to_the_video_and_samples_the_sheet_across_the_clip(capsys, video):
    code, result = run(capsys, "cut", video, "--from", "8", "--to", "0:20", "--frame", "sheet")
    assert code == 0
    assert (result["start"], result["end"]) == (8.0, 10.0)
    assert result["clip_path"].endswith("/clips/test-pattern-tern01_8.0-10.0.mp4")
    assert media.probe(Path(result["clip_path"])).duration == pytest.approx(2.0, abs=0.1)
    assert result["frame_times"] == [8.17, 8.5, 8.83, 9.17, 9.5, 9.83]


@pytest.mark.parametrize(
    ("frame", "frames"),
    [("mid", {"frame_path": "test-pattern-tern01_1.0-3.0_mid.jpg", "frame_times": [2.0]}), ("none", {})],
)
def test_cut_returns_the_middle_frame_or_no_frame(capsys, video, frame, frames):
    code, result = run(capsys, "cut", video, "--from", "1", "--to", "3", "--frame", frame)
    assert code == 0
    returned = {key: result[key] for key in ("frame_path", "frame_times") if key in result}
    if "frame_path" in returned:
        returned["frame_path"] = Path(returned["frame_path"]).name
    assert returned == frames


def test_frames_tiles_count_frames_into_one_sheet(capsys, video):
    code, result = run(capsys, "frames", video, "--from", "2", "--to", "6", "--count", "4")
    assert code == 0
    assert result["times"] == [2.5, 3.5, 4.5, 5.5]
    with Image.open(result["sheet_path"]) as sheet:
        assert sheet.width <= media.SHEET_MAX_WIDTH
        assert sheet.width > sheet.height  # 2 x 2 tiles of a 4:3 frame


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["frames", "{video}", "--from", "1", "--to", "2", "--count", "13"],
            "count must be a whole number from 1 to 12",
        ),
        (["cut", "{video}", "--from", "12", "--to", "15"], "Empty range 00:00:12.0 -> 00:00:15.0"),
        (["cut", "nope", "--from", "1", "--to", "2"], 'No video "nope" in this index'),
        (["timeline", "nope", "--from", "1", "--to", "2"], 'No video "nope" in this index'),
    ],
)
def test_bad_requests_fail_with_status_1_and_a_reason(capsys, video, argv, message):
    code, err = run(capsys, *[arg.format(video=video) for arg in argv])
    assert code == 1
    assert message in err


def test_a_missing_local_copy_says_how_to_restore_it(capsys, add_video, tmp_path):
    add_video("tlv", "gone-video", "Gone", tmp_path / "deleted.mp4")
    code, err = run(capsys, "cut", "gone-video", "--from", "1", "--to", "2")
    assert code == 1
    assert 'tlv add "https://example.com/gone-video.mp4"' in err
    code, result = run(capsys, "list")
    assert [v["local_exists"] for v in result["videos"]] == [False]


def test_index_selects_its_own_database(capsys, add_video, clip_file):
    add_video("tlv", "default-one", "Default", clip_file)
    add_video("practice", "practice-one", "Practice", clip_file)
    listed = {
        index: [v["video_id"] for v in run(capsys, "list", "--index", index)[1]["videos"]]
        for index in ("tlv", "practice", "empty")
    }
    assert listed == {"tlv": ["default-one"], "practice": ["practice-one"], "empty": []}
