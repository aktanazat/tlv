"""Runs one search system's `find` over the eval queries and saves its raw outputs and timings.

    uv run python eval/run.py twelvelabs            # every query
    uv run python eval/run.py local curry-three     # only the named queries

Results go to eval/results/<system>-<date>.json, rewritten after each query so an interrupted run
keeps what finished; running named queries again replaces just their records in today's file.
`twelvelabs` is the old TypeScript tool kept as a yardstick in its own worktree, with the API key
injected by mem-secret; `local` is this repository's `tlv`.
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
YARDSTICK = Path.home() / "Developer/tlv-yardstick"
FLAGS = ["--max-candidates", "12", "--no-cut", "--json"]
SYSTEMS = {
    "twelvelabs": (
        [
            str(Path.home() / ".local/bin/mem-secret"),
            *("run", "TWELVELABS_API_KEY", "--"),
            str(Path.home() / ".bun/bin/bun"),
            str(YARDSTICK / "src/cli.ts"),
            "find",
        ],
        YARDSTICK,
    ),
    "local": (["uv", "run", "tlv", "find"], REPO),
}


def command(system: str, query: dict) -> list[str]:
    then = [arg for step in query["then"] for arg in ("--then", step)]
    return [*SYSTEMS[system][0], query["query"], *then, *FLAGS]


def run(system: str, query: dict) -> dict:
    """One `find` call: its parsed JSON output (None when it failed), exit code, stderr and wall time."""
    argv = command(system, query)
    started = time.monotonic()
    done = subprocess.run(argv, cwd=SYSTEMS[system][1], capture_output=True, text=True)
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
    parser.add_argument("system", choices=SYSTEMS)
    parser.add_argument("ids", nargs="*", help="query ids to run (default: all)")
    args = parser.parse_args()

    queries = json.loads((EVAL / "queries.json").read_text())
    unknown = set(args.ids) - {query["id"] for query in queries}
    if unknown:
        parser.error(f"unknown query ids: {', '.join(sorted(unknown))}")
    chosen = [query for query in queries if not args.ids or query["id"] in args.ids]

    path = EVAL / "results" / f"{args.system}-{datetime.date.today().isoformat()}.json"
    path.parent.mkdir(exist_ok=True)
    saved = json.loads(path.read_text())["runs"] if path.exists() else []
    runs = {record["id"]: record for record in saved}
    for query in chosen:
        print(f"{args.system}: {query['id']}", flush=True)
        runs[query["id"]] = run(args.system, query)
        record = runs[query["id"]]
        print(f"  exit {record['exit_code']} in {record['seconds']} s", flush=True)
        ordered = [runs[q["id"]] for q in queries if q["id"] in runs]
        results = {"system": args.system, "flags": FLAGS, "runs": ordered}
        write_atomically(path, lambda partial, results=results: partial.write_text(json.dumps(results, indent=1)))
    print(path)


if __name__ == "__main__":
    main()
