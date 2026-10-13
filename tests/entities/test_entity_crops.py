"""What crops measure: a face warped so its landmarks land on ArcFace's template, and jersey colors
that match one uniform and tell it from others."""

import cv2
import numpy as np
import pytest

from tlv.entities.cluster import UNIFORM_SAME
from tlv.entities.crops import TEMPLATE, align, colors


def test_align_puts_each_landmark_on_its_template_point():
    # white dots where a tilted, enlarged, shifted face's landmarks are, on a black 640x360 frame
    angle = np.deg2rad(12)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = (TEMPLATE - 56) @ rotation.T * 1.7 + [300, 180]
    frame = np.zeros((360, 640, 3), np.uint8)
    for x, y in points:
        cv2.circle(frame, (round(x), round(y)), 2, (255, 255, 255), -1)
    face = align(frame, points / [640, 360])
    assert min(int(face[round(y), round(x)].min()) for x, y in TEMPLATE) > 128


WHITE, GOLD, BLACK, RED, BLUE = (235, 235, 235), (39, 185, 253), (25, 25, 25), (40, 40, 200), (200, 80, 30)  # BGR
WARM_WHITE = (161, 195, 224)  # a white jersey under warm arena light, as measured: saturation 0.28, hue 32 degrees


def shirt(color, seed):
    """A 120x80 crop of a jersey in one color with a purple number on it, and camera noise."""
    image = np.full((120, 80, 3), color, np.uint8)
    cv2.putText(image, "23", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (130, 40, 85), 4)
    noise = np.random.default_rng(seed).normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        (WHITE, WHITE, True),  # one uniform in two frames
        (WHITE, WARM_WHITE, True),  # ...also under warmer light
        (WHITE, GOLD, False),
        (WARM_WHITE, GOLD, False),  # a warm-lit white jersey is still not a gold one
        (WHITE, BLACK, False),
        (RED, BLUE, False),  # two strong colors, told apart by hue alone
    ],
)
def test_colors_match_one_uniform_and_tell_it_from_others(first, second, same):
    assert (float(colors(shirt(first, 1)) @ colors(shirt(second, 2))) >= UNIFORM_SAME) is same
