"""Measure Jev's credit veto (tlv.judge.CREDIT) on an eval run: ask it about every confirmed clip,
drop the clips whose commentary credits the claimed action to a different player, and write the run
back without them for score.py. Sends each clip's claim and commentary, as text, to TypeSafe's cloud
API, so it needs TYPESAFE_API_KEY:

    mem-secret run TYPESAFE_API_KEY -- uv run python eval/jev_bench.py eval/results/local-warm-2026-10-12.json \\
        --out /tmp/vetoed.json
    uv run python eval/score.py /tmp/vetoed.json
"""

import argparse
import json
import sys
import time
from contextlib import closing
from pathlib import Path

from tlv import judge, store
from tlv.agent import sources
from tlv.agent.inspect import clock

# Commentary read around a clip: about the stretch the vision model read it in (3 s of padding, 6 s
# at least), since a call often starts before the frames the model names.
SPEECH_PAD_S = 4.0


def credit_state(db, claim: str, clip: dict) -> dict:
    """What Jev reads for one confirmed clip: the claim, where it was confirmed, the commentary around it."""
    lines = sources.speech(db, clip["video_id"], clip["start"] - SPEECH_PAD_S, clip["end"] + SPEECH_PAD_S)
    return {
        "claim": claim,
        "clip": f"{clock(clip['start'])}-{clock(clip['end'])}",
        "speech": [f"[{clock(line.start)}] {line.text}" for line in lines],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    asked = [(run, clip) for run in results["runs"] for clip in run["output"]["confirmed"]]
    with closing(store.connect()) as db:
        states = [credit_state(db, run["output"]["claim"], clip) for run, clip in asked]
    began = time.monotonic()
    answers = judge.bools(judge.CREDIT, states)
    seconds = time.monotonic() - began
    for (run, clip), credit in zip(asked, answers, strict=True):
        vetoed = credit >= judge.CREDIT_VETO
        print(f"{run['id']:24} {clip['timecode']}  credit {credit:.2f}{'  vetoed' if vetoed else ''}")
        if vetoed:
            run["output"]["confirmed"].remove(clip)
            run["output"]["rejected"].append(
                {
                    "video_id": clip["video_id"],
                    "start": clip["start"],
                    "end": clip["end"],
                    "reason": "credited to another player",
                }
            )
    vetoed = sum(credit >= judge.CREDIT_VETO for credit in answers)
    print(f"asked about {len(asked)} confirmed clips in {seconds:.2f} s; vetoed {vetoed}", file=sys.stderr)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
