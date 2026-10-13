"""Runs `tlv find` over the eval queries and saves its raw outputs and timings.

    uv run python eval/run.py                  # every query
    uv run python eval/run.py curry-three      # only the named queries
    uv run python eval/run.py --out FILE       # into FILE instead of today's file
    uv run python eval/run.py --set everyday   # the set in eval/everyday/, searched in tlv's everyday index

Results go to eval/results/local-<date>.json (eval/results/NAME-local-<date>.json for --set NAME),
rewritten after each query so an interrupted run keeps what finished; running named queries again
replaces just their records in the file.
eval/results/twelvelabs-2026-10-12.json is the yardstick: the same queries and flags run once
through the Twelve Labs client this tool replaced, before that client was removed.
"""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

from tlv.media import write_atomically

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
FLAGS = ["--max-candidates", "12", "--no-cut", "--json"]


def command(query: dict, index: str | None) -> list[str]:
    then = [arg for step in query["then"] for arg in ("--then", step)]
    named = ["--index", index] if index else []
    return ["uv", "run", "tlv", *named, "find", query["query"], *then, *FLAGS]


def run(query: dict, index: str | None) -> dict:
    """One `find` call: its parsed JSON output (None when it failed), exit code, stderr and wall time."""
    argv = command(query, index)
    started = time.monotonic()
    done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    return {
        "id": query["id"],
        "query": query["query"],
        "then": query["then"],
        "argv": argv,
        "seconds": round(time.monotonic() - started, 1),
        "exit_code": done.returncode,
        "output": json.loads(done.stdout) if done.returncode == 0 else None,
        "stderr": done.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="query ids to run (default: all)")
    parser.add_argument("--out", type=Path, help="results file (default: eval/results/local-<date>.json)")
    parser.add_argument("--set", metavar="NAME", help="the queries in eval/NAME/, searched in tlv's NAME index")
    args = parser.parse_args()

    queries = json.loads(((EVAL / args.set if args.set else EVAL) / "queries.json").read_text())
    unknown = set(args.ids) - {query["id"] for query in queries}
    if unknown:
        parser.error(f"unknown query ids: {', '.join(sorted(unknown))}")
    chosen = [query for query in queries if not args.ids or query["id"] in args.ids]

    prefix = f"{args.set}-" if args.set else ""
    path = args.out or EVAL / "results" / f"{prefix}local-{datetime.date.today().isoformat()}.json"
    path.parent.mkdir(exist_ok=True)
    saved = json.loads(path.read_text())["runs"] if path.exists() else []
    runs = {record["id"]: record for record in saved}
    for query in chosen:
        print(f"local: {query['id']}", flush=True)
        runs[query["id"]] = run(query, args.set)
        record = runs[query["id"]]
        print(f"  exit {record['exit_code']} in {record['seconds']} s", flush=True)
        ordered = [runs[q["id"]] for q in queries if q["id"] in runs]
        results = {"system": "local", "flags": FLAGS, "runs": ordered}
        write_atomically(path, lambda partial, results=results: partial.write_text(json.dumps(results, indent=1)))
    print(path)


if __name__ == "__main__":
    main()
