"""The stretch a question is about."""

import pytest

from tlv.agent.ask import clamp
from tlv.agent.sources import Video
from tlv.media import Range

VIDEO = Video("v", "Game", "https://example.com/game", "/videos/game.mp4", 60.0)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        pytest.param(10, 20, Range(10, 20), id="inside the video"),
        pytest.param(-5, 90, Range(0, 60), id="cut to the video"),
    ],
)
def test_a_range_is_cut_to_the_video(start, end, expected):
    assert clamp(VIDEO, start, end) == expected


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(20, 10, id="ending before it starts"),
        pytest.param(60, 90, id="starting at the end"),
        pytest.param(70, 90, id="past the end"),
    ],
)
def test_an_empty_range_is_an_error(start, end):
    with pytest.raises(ValueError, match="Empty range"):
        clamp(VIDEO, start, end)
