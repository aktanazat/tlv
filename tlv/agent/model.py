"""The vision-language model as the agent uses it: one guarded process held for a whole find,
ask or answer, asked short questions that it answers in compact JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tlv import guard

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    generation_tokens: int
    seconds: float
    truncated: bool


def parse_json(text: str) -> Any:
    """The JSON value in a model reply, code fences and chatter around it ignored; None when
    there is none."""
    body = _FENCE.sub("", text.strip())
    start = min((i for i in (body.find("{"), body.find("[")) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError:
        return None
    return value


class Vlm:
    """with Vlm(progress) as vlm: vlm.ask(images, prompt, max_tokens)"""

    def __init__(self, progress: guard.Progress | None = None) -> None:
        self._process = guard.ModelProcess("tlv.models.vlm", progress=progress)
        self.seconds = 0.0  # generation time of every reply so far

    def __enter__(self) -> Vlm:
        self._process.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._process.__exit__(*exc)

    @property
    def peak_bytes(self) -> int:
        return self._process.peak

    def ask(self, images: list[Path], prompt: str, max_tokens: int) -> Reply:
        out = self._process.call({"images": [str(p) for p in images], "prompt": prompt, "max_tokens": max_tokens})
        self.seconds += out["seconds"]
        return Reply(
            out["text"],
            out["prompt_tokens"],
            out["generation_tokens"],
            out["seconds"],
            out["generation_tokens"] >= max_tokens,
        )
