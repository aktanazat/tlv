"""Times in seconds, written and read as timecodes."""

from __future__ import annotations

import re

_TIME = re.compile(r"^(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d+)?)$")


def format_timecode(seconds: float) -> str:
    """Seconds as HH:MM:SS.s, rounded to the nearest tenth."""
    tenths = round(max(0.0, seconds) * 10)
    hours, rest = divmod(tenths, 36000)
    minutes, rest = divmod(rest, 600)
    return f"{hours:02d}:{minutes:02d}:{rest // 10:02d}.{rest % 10}"


def format_range(start: float, end: float) -> str:
    """A range as "00:01:23.4 -> 00:01:31.0"."""
    return f"{format_timecode(start)} -> {format_timecode(end)}"


def parse_time(text: str) -> float:
    """Parses "83.4", "1:23.4" or "00:01:23.4" into seconds."""
    match = _TIME.match(text.strip())
    if not match:
        raise ValueError(f'"{text}" is not a time: use seconds (83.4) or a timecode (1:23.4, 00:01:23.4)')
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)
