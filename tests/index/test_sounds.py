"""Sound events: classifier windows of one label that overlap or touch become one event."""

import pytest

from tlv.index.sounds import Event, merge


def window(start, end, **labels):
    return {"start": start, "end": end, "labels": [{"label": label, "conf": conf} for label, conf in labels.items()]}


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        (
            [window(0.0, 1.0, cheering=0.4), window(0.5, 1.5, cheering=0.9), window(1.0, 2.0, cheering=0.5)],
            [Event(0.0, 2.0, "cheering", 0.9)],
        ),
        ([window(0.0, 1.0, music=0.6), window(1.0, 2.0, music=0.3)], [Event(0.0, 2.0, "music", 0.6)]),
        (
            [window(0.0, 1.0, music=0.6), window(1.5, 2.5, music=0.3)],
            [Event(0.0, 1.0, "music", 0.6), Event(1.5, 2.5, "music", 0.3)],
        ),
        (
            [window(0.0, 1.0, speech=0.8, cheering=0.4), window(0.5, 1.5, speech=0.7)],
            [Event(0.0, 1.0, "cheering", 0.4), Event(0.0, 1.5, "speech", 0.8)],
        ),
    ],
    ids=["overlapping", "touching", "apart", "two labels"],
)
def test_windows_of_one_label_merge_into_events(windows, expected):
    assert merge(windows) == expected
