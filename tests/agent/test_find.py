"""A confirmed moment's id names its video and range, so finding it again replaces its record."""

import re

import pytest

from tlv.agent.find import observation_id


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
