"""Pictures of people cut from full-resolution video frames: a display crop per appearance (also
what the vision-language model sees), faces aligned for ArcFace, the body crops jersey numbers are
read from, and the colors of a jersey. Boxes are normalized x, y, w, h with a top-left origin."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from tlv.entities.tracks import Box

# ArcFace's five-point template for a 112x112 face: the eyes, the nose tip, the mouth corners.
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)
FACE_SIDE = 112
# Jersey colors: a pixel this unsaturated or dark counts as black, gray or white; the rest by hue.
# White under warm arena light reads up to about 0.3 saturation, close to a gold jersey's hue.
PLAIN_SATURATION, DARK_VALUE = 0.35, 0.2
HUES = 12


def frames(path: Path, times: Sequence[float]) -> Iterator[tuple[float, np.ndarray]]:
    """The decoded frame nearest each of `times` (seconds), in time order, as BGR pixels."""
    wanted = sorted(set(times))
    capture = cv2.VideoCapture(str(path))
    half = 0.5 / (capture.get(cv2.CAP_PROP_FPS) or 30.0)
    k = 0
    try:
        while k < len(wanted) and capture.grab():
            now = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if wanted[k] > now + half:
                continue
            _, frame = capture.retrieve()
            while k < len(wanted) and wanted[k] <= now + half:
                yield wanted[k], frame
                k += 1
    finally:
        capture.release()


def cut(frame: np.ndarray, box: Box, scale: float = 1.0, square: bool = False) -> np.ndarray:
    """The part of `frame` inside `box` grown `scale` times around its center, within the frame."""
    height, width = frame.shape[:2]
    x, y, w, h = box
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    half_w, half_h = w * width * scale / 2, h * height * scale / 2
    if square:
        half_w = half_h = max(half_w, half_h)
    x0, y0 = max(0, round(cx - half_w)), max(0, round(cy - half_h))
    x1, y1 = min(width, round(cx + half_w)), min(height, round(cy + half_h))
    return frame[y0:y1, x0:x1]


def align(frame: np.ndarray, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """The face with these five normalized landmarks warped onto ArcFace's template, 112x112."""
    height, width = frame.shape[:2]
    points = np.asarray(landmarks, dtype=np.float64) * [width, height]
    return cv2.warpAffine(frame, _similarity(points, TEMPLATE), (FACE_SIDE, FACE_SIDE), borderValue=(0, 0, 0))


def save(image: np.ndarray, path: Path, longest: int) -> None:
    """Writes `image`, shrunk so its longer side is at most `longest` pixels."""
    scale = longest / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def colors(image: np.ndarray) -> np.ndarray:
    """A unit-length color histogram of BGR pixels, for telling uniforms apart by dot product: three
    brightness bins for plain pixels, then HUES hues each dim or bright for the rest."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int64)
    hue, saturation, value = hsv[:, 0], hsv[:, 1] / 255, hsv[:, 2] / 255
    plain = (saturation < PLAIN_SATURATION) | (value < DARK_VALUE)
    colored = 3 + hue * HUES // 180 * 2 + (value >= 0.5)
    bins = np.where(plain, np.minimum(value * 3, 2).astype(np.int64), colored)
    counts = np.bincount(bins, minlength=3 + HUES * 2).astype(np.float32)
    return counts / np.linalg.norm(counts)


def _similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The least-squares rotation, uniform scale and shift taking `source` points onto `target`
    (Umeyama), as a 2x3 matrix."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mean_s, target - mean_t
    u, sigma, vt = np.linalg.svd(t.T @ s / len(source))
    d = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ d @ vt
    scale = np.trace(np.diag(sigma) @ d) / s.var(axis=0).sum()
    return np.hstack([scale * rotation, (mean_t - scale * rotation @ mean_s)[:, None]])
