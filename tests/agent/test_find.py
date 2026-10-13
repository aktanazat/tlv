"""A confirmed moment's id names its video and range, so finding it again replaces its record; a
search narrowed to video-time ranges keeps only the hits inside them."""

import re

import pytest

from tlv.agent.connect import Clip
from tlv.agent.find import clips_within, observation_id


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
