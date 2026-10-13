"""On-screen text read by Apple Vision comes back in top-left normalized image coordinates, the
frame every consumer of the ocr table crops and draws in, and a video is read from its file alone,
on the frames' half-second grid, whatever its codec."""

import subprocess

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


# VP9 is what YouTube and browsers' screen recorders often hand over; Apple's decoders refuse it.
@pytest.mark.parametrize("codec", ["libx264", "libvpx-vp9"])
def test_a_video_is_read_on_its_own_with_text_once_a_second(db, video, tmp_path, codec):
    still = tmp_path / "score.png"
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    ImageDraw.Draw(image).text((64, 36), "LAKERS 102", font=ImageFont.load_default(size=64), fill="black")
    image.save(still)
    encode = ["-t", str(video["duration"]), "-r", "30", "-c:v", codec, "-pix_fmt", "yuv420p", video["local_path"]]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-loop", "1", "-i", str(still), *encode], check=True)
    vision.run(db, video, None)
    read = [(row["t"], row["text"]) for row in db.execute("SELECT t, text FROM ocr ORDER BY t")]
    assert read == [(second + 0.25, "LAKERS 102") for second in range(int(video["duration"]))]
