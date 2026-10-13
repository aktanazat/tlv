"""The vision-language model as the agent uses it: one guarded process held for a whole find,
ask or answer, asked short questions that it answers in compact JSON. Questions asked together
are answered in one job, which writes their answers side by side (tlv.models.vlm)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tlv import guard

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True)
class Question:
    """What the model is shown and asked. Its answer ends at `max_tokens`, or as soon as its text
    matches `stop`, a pattern for first words that already settle the answer."""

    images: list[Path]
    prompt: str
    max_tokens: int
    stop: str | None = None


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    generation_tokens: int
    truncated: bool  # cut off at the token limit
    stopped: bool  # ended where its text matched the question's stop pattern


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
    """with Vlm(progress) as vlm: vlm.ask(images, prompt, max_tokens), or vlm.ask_all(questions)"""

    def __init__(self, progress: guard.Progress | None = None) -> None:
        self._process = guard.ModelProcess("tlv.models.vlm", progress=progress)
        self.seconds = 0.0  # time spent answering so far

    def __enter__(self) -> Vlm:
        self._process.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._process.__exit__(*exc)

    def ask(self, images: list[Path], prompt: str, max_tokens: int, stop: str | None = None) -> Reply:
        return self.ask_all([Question(images, prompt, max_tokens, stop)])[0]

    def ask_all(self, questions: list[Question]) -> list[Reply]:
        """The replies to the questions, in order, answered together in one job."""
        if not questions:
            return []
        out = self._process.call(
            {
                "batch": [
                    {
                        "images": [str(p) for p in q.images],
                        "prompt": q.prompt,
                        "max_tokens": q.max_tokens,
                        "stop": q.stop,
                    }
                    for q in questions
                ]
            }
        )
        self.seconds += out["seconds"]
        return [
            Reply(
                r["text"],
                r["prompt_tokens"],
                r["generation_tokens"],
                not r["stopped"] and r["generation_tokens"] >= q.max_tokens,
                r["stopped"],
            )
            for q, r in zip(questions, out["results"], strict=True)
        ]
