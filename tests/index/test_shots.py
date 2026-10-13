"""Shot boundaries: a cut where the scene score reaches the threshold, one cut per burst, and
shots that tile the whole video."""

import subprocess

import pytest

from tlv.index import shots


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([(3.0, 9.9), (6.0, 2.0)], [(0.0, 10.0)]),
        ([(4.0, 10.0)], [(0.0, 4.0), (4.0, 10.0)]),
        ([(4.0, 12.0), (4.05, 40.0), (4.1, 15.0)], [(0.0, 4.05), (4.05, 10.0)]),
        ([(4.0, 20.0), (4.6, 20.0)], [(0.0, 4.0), (4.0, 4.6), (4.6, 10.0)]),
        ([(0.2, 50.0), (5.0, 30.0)], [(0.0, 5.0), (5.0, 10.0)]),
        ([(5.0, 30.0), (9.7, 50.0)], [(0.0, 5.0), (5.0, 10.0)]),
    ],
    ids=["below threshold", "at threshold", "burst keeps strongest", "cuts 0.6 s apart", "cut at start", "cut at end"],
)
def test_shots_cut_at_strong_scores_and_tile_the_video(scores, expected):
    assert shots.shots(scores, 10.0) == expected


def test_a_hard_cut_in_a_real_video_splits_it_in_two(tmp_path):
    """Guards the parsing of ffmpeg's scene-score printout, which zero cuts would silently pass."""
    path = tmp_path / "cut.mp4"
    lavfi = "color=c=red:s=320x240:r=25:d=2[a];color=c=blue:s=320x240:r=25:d=2[b];[a][b]concat=n=2:v=1:a=0"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            lavfi,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    found = shots.shots(shots.scores(str(path)), 4.0)
    assert [(round(start, 1), round(end, 1)) for start, end in found] == [(0.0, 2.0), (2.0, 4.0)]
