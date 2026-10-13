"""The transcript: Whisper segments with word times, from the video's audio track.

The audio is transcribed in pieces of one to two minutes, one model job each, so a search waits
at most for one piece rather than a whole video. A seam sits in the quietest half second of a
piece's second minute, a pause when there is one, and the two pieces share OVERLAP_S around it.
The seam falls on a word both pieces heard at the same time: the earlier piece gives the words up
to it, the later one the words after, so no word is lost, doubled or cut at the edge of a piece.
Whisper drops words near the very edge of its audio even when silence parts them from the edge
(a cut in a 0.5 s pause of a meeting lost "if you want to"), which is why pieces overlap even
where they could be cut in a pause.
"""

from __future__ import annotations

import json
import re
import sqlite3
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from tlv import guard, media, store

VERSION = 1
# A piece is MIN_S to MAX_S seconds of audio.
MIN_S = 60.0
MAX_S = 120.0
# Seconds two pieces share around a seam: each side of it gets half, so the words there are
# heard well inside both pieces.
OVERLAP_S = 6.0
# Loudness is read in FRAME_S frames, and a seam goes in the quietest GAP_S stretch.
FRAME_S = 0.02
GAP_S = 0.5
# Two pieces heard a seam word alike when their times for it are this close.
AGREE_S = 1.0
WORD = re.compile(r"\W+")
# The (segment, word) position before a piece's first word.
NONE = (-1, -1)


@dataclass(frozen=True)
class Piece:
    start: float
    end: float


def run(db: sqlite3.Connection, video: sqlite3.Row, progress: guard.Progress | None) -> None:
    video_id = video["video_id"]
    audio = audio_file(video)
    segments: list[dict[str, Any]] = []
    if audio:
        segments = transcribe(audio, progress)
    rows = [
        (video_id, segment["start"], segment["end"], segment["text"], json.dumps(segment["words"]))
        for segment in segments
        if segment["text"]
    ]
    with store.transaction(db):
        db.execute("DELETE FROM speech WHERE video_id = ?", (video_id,))
        db.executemany("INSERT INTO speech (video_id, start, end, text, words) VALUES (?, ?, ?, ?, ?)", rows)


def audio_file(video: sqlite3.Row) -> Path | None:
    """The video's audio as a 16 kHz mono WAV under work/<video_id>/, extracted once and shared
    by speech and sounds, which run at the same time; None when the video has no sound track, as
    screen recordings often do."""
    source = Path(video["local_path"])
    if not media.has_audio(source):
        return None
    path = store.data_dir("work") / video["video_id"] / "audio.wav"
    with store.lock(f"audio-{video['video_id']}"):
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            media.audio(source, path)
    return path


def transcribe(audio: Path, progress: guard.Progress | None) -> list[dict[str, Any]]:
    """The WAV's segments in its own time, one model job per piece. Every piece after the first
    keeps the language Whisper heard in the first, as one job over the whole file did. A piece is
    not prompted with the text before it: Whisper already decodes each 30 s window without the one
    before, so one misheard stretch cannot echo on, and pieces stay independent of each other."""
    pieces = plan(audio)
    joined: list[dict[str, Any]] = []
    language = None
    with guard.ModelProcess("tlv.models.whisper", progress=progress) as whisper:
        for i, piece in enumerate(pieces):
            job = {"audio": str(audio), "start": piece.start, "end": piece.end, "language": language}
            result = whisper.call(job)
            language = language or result["language"]
            joined = seam(joined, result["segments"], piece.start, pieces[i - 1].end) if i else result["segments"]
            if progress and len(pieces) > 1:
                progress(f"transcribed {i + 1} of {len(pieces)} pieces")
    return joined


def plan(audio: Path) -> list[Piece]:
    """The pieces the WAV is transcribed in. Each ends OVERLAP_S / 2 past the middle of the
    quietest GAP_S between MIN_S and MAX_S into it, and the next starts OVERLAP_S / 2 before."""
    power, total = _power(audio)
    gap = round(GAP_S / FRAME_S)
    pieces = []
    start = 0.0
    while total - start > MAX_S:
        first = round((start + MIN_S) / FRAME_S)
        last = round((start + MAX_S - OVERLAP_S / 2) / FRAME_S)
        level = np.convolve(power[first:last], np.ones(gap) / gap, mode="valid")
        cut = (first + int(np.argmin(level)) + gap / 2) * FRAME_S
        pieces.append(Piece(start, cut + OVERLAP_S / 2))
        start = cut - OVERLAP_S / 2
    return [*pieces, Piece(start, total)]


def seam(before: list[dict[str, Any]], after: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """Joins the segments of two pieces that share [start, end]. The seam is the word both heard
    alike (the same word, within AGREE_S) nearest the middle of the overlap: `before` gives the
    words up to it, `after` the words past it. When no word agrees, the middle of the overlap
    parts them by time."""
    middle = (start + end) / 2
    left = [(at, word) for at, word in _words(before) if _middle(word) >= start]
    right = [(at, word) for at, word in _words(after) if _middle(word) < end]
    matcher = SequenceMatcher(None, [_key(w) for _, w in left], [_key(w) for _, w in right], autojunk=False)
    agreed = [
        (i + k, j + k)
        for i, j, size in matcher.get_matching_blocks()
        for k in range(size)
        if _key(left[i + k][1]) and abs(_middle(left[i + k][1]) - _middle(right[j + k][1])) <= AGREE_S
    ]
    if agreed:
        i, j = min(agreed, key=lambda pair: abs(_middle(left[pair[0]][1]) - middle))
        (through_before, word), (through_after, _) = left[i], right[j]
        seam_t = word["e"]
    else:
        seam_t = middle
        through_before = max((at for at, word in _words(before) if _middle(word) < middle), default=NONE)
        through_after = max((at for at, word in _words(after) if _middle(word) < middle), default=NONE)
    return _trim(before, through_before, seam_t, earlier=True) + _trim(after, through_after, seam_t, earlier=False)


def _trim(
    segments: list[dict[str, Any]], through: tuple[int, int], seam_t: float, earlier: bool
) -> list[dict[str, Any]]:
    """The earlier piece's segments up to the word at `through` (segment, word), or the later
    piece's past it, each cut down to the words kept. A segment without words stays when its
    middle is before `seam_t` in the earlier piece, or not before it in the later one."""
    out = []
    for si, segment in enumerate(segments):
        words = segment["words"]
        if not words:
            if ((segment["start"] + segment["end"]) / 2 < seam_t) == earlier:
                out.append(segment)
            continue
        kept = [word for wi, word in enumerate(words) if ((si, wi) <= through) == earlier]
        if len(kept) == len(words):
            out.append(segment)
        elif kept:
            out.append(
                {
                    "start": segment["start"] if kept[0] is words[0] else kept[0]["s"],
                    "end": segment["end"] if kept[-1] is words[-1] else kept[-1]["e"],
                    "text": " ".join(word["w"] for word in kept),
                    "words": kept,
                }
            )
    return out


def _words(segments: list[dict[str, Any]]) -> list[tuple[tuple[int, int], dict[str, Any]]]:
    return [((si, wi), word) for si, segment in enumerate(segments) for wi, word in enumerate(segment["words"])]


def _middle(word: dict[str, Any]) -> float:
    return (word["s"] + word["e"]) / 2


def _key(word: dict[str, Any]) -> str:
    return WORD.sub("", word["w"].lower())


def _power(audio: Path) -> tuple[np.ndarray, float]:
    """Mean power of each FRAME_S frame of a 16-bit mono WAV, read a few minutes at a time, and
    the WAV's length in seconds."""
    with wave.open(str(audio)) as wav:
        rate = wav.getframerate()
        frame = round(rate * FRAME_S)
        blocks = [np.zeros(0)]
        while chunk := wav.readframes(frame * 10_000):
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768
            count = len(pcm) // frame
            blocks.append((pcm[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
        return np.concatenate(blocks), wav.getnframes() / rate
