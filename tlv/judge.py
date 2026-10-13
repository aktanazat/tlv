"""Jev, TypeSafe's judge model, asked yes-or-no questions about text the index already holds:
claims, transcript lines, captions, text read off frames. That text goes to TypeSafe's cloud API and
each question costs a fraction of a cent; video and frames never leave the Mac. Without
TYPESAFE_API_KEY in the environment (`mem-secret run` or the MCP entry sets it) every call raises
JudgeError and the caller keeps its own checks, so tlv stays local unless the key is given.

A batch of states comes back in about half a second, where the vision model takes about four seconds
per look. On the eval's 112 checked stretches Jev still cannot stand in for the vision model: as the
checker it finds 14 events where the vision model finds 17, and no threshold screens out false
stretches without a true one. It does settle one question the vision model gets wrong: CREDIT."""

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
# Requests in flight at once: the API answers one state per request, so a search's clips go out together.
PARALLEL = 12
TIMEOUT_S = 30.0


class JudgeError(RuntimeError):
    """Jev did not answer: no key in the environment, or the API refused or could not be reached."""


@dataclass(frozen=True)
class Question:
    """A yes-or-no question about each state: what to judge, and what a yes and a no mean."""

    instructions: str
    yes: str
    no: str


# Asked of a clip the vision model confirmed, with the state {"claim": the claim, "clip":
# "m:ss.s-m:ss.s", "speech": the commentary lines around it, each "[m:ss.s] words"}. The vision model
# confirmed Curry for Jimmy Butler's and Brandin Podziemski's threes though its prompt held the call
# naming each shooter. On the eval's warm run Jev answers 0.95 and 0.97 for those two clips and at
# most 0.55 for the other 40 confirmed clips.
CREDIT = Question(
    instructions="`claim` names a basketball player and what he does, and a video model says it happens in "
    "`clip`, a range of the video's time. `speech` is the broadcast commentary around it, each line stamped "
    "with its time. Does the commentary say that a different player made that action at that time?",
    yes="Near `clip`, the commentary names another player as the one who made the claimed action.",
    no="The commentary names the claimed player for it, names nobody for it, or talks about something else.",
)
# A clip whose CREDIT answer reaches this is someone else's play. It sits well above the 0.55 of the
# most doubtful clip kept, a true one, so a small drift in Jev's answers does not cost a true clip.
CREDIT_VETO = 0.7


def bools(question: Question, states: list[dict[str, Any]]) -> list[float]:
    """The probability of a yes for each state, in the order given. Raises JudgeError."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JudgeError("Jev needs TYPESAFE_API_KEY in the environment")
    asked = {
        "type": "noul",
        "instructions": question.instructions,
        "criteria": {"true": question.yes, "false": question.no},
    }
    with ThreadPoolExecutor(PARALLEL) as pool:
        return list(pool.map(lambda state: _ask(key, asked, state), states))


def _ask(key: str, question: dict[str, Any], state: dict[str, Any]) -> float:
    request = urllib.request.Request(
        URL,
        data=json.dumps({"state": state, "model": MODEL, "questions": {"q": question}}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "tlv"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.load(response)["answers"]["q"]["noul"]
    except urllib.error.HTTPError as error:
        reason = error.read().decode(errors="replace")[:300]
        raise JudgeError(f"Jev refused the question: HTTP {error.code} {reason}") from error
    except OSError as error:
        raise JudgeError(f"Jev could not be reached: {error}") from error
