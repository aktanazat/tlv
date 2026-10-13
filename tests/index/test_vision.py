"""On-screen text read by Apple Vision comes back in top-left normalized image coordinates, the
frame every consumer of the ocr table crops and draws in."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from tlv.index import vision

WIDTH, HEIGHT = 1280, 720


@pytest.mark.usefixtures("db")
def test_text_boxes_are_normalized_from_the_top_left(tmp_path):
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=64)
    expected = {}
    for text, corner in [("LAKERS 102", (64, 36)), ("PHILA 100", (832, 576))]:
        draw.text(corner, text, font=font, fill="black")
        left, top, right, bottom = draw.textbbox(corner, text, font=font)
        expected[text] = (left / WIDTH, top / HEIGHT, (right - left) / WIDTH, (bottom - top) / HEIGHT)
    path = tmp_path / "score.jpg"
    image.save(path)
    [lines] = vision.read_text([path])
    assert {line.text: line.box for line in lines} == {
        text: pytest.approx(box, abs=0.02) for text, box in expected.items()
    }
