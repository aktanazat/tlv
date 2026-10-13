"""Scores saved `find` runs against the hand-labeled truth in eval/truth.json.

    uv run python eval/score.py eval/results/twelvelabs-2026-10-12.json
    uv run python eval/score.py --set everyday eval/results/everyday-local-<date>.json  # eval/everyday/truth.json

A returned clip matches a labeled event when it is on the event's video and overlaps one of the
event's ranges (the live play or any replay of the same moment) by at least half of the shorter
of the two. Clips and events are paired one to one, as many pairs as possible. A clip left
unpaired is "ambiguous" (ignored) when it matches a moment the labels could not settle, a
"duplicate" when it matches an event that another clip already took (a replay or a second cut of
the same moment), and "false" otherwise.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tlv.timecode import format_range, parse_time

EVAL = Path(__file__).resolve().parent
YOUTUBE_ID = re.compile(r"(?:Youtube-|[?&]v=|youtu\.be/)([\w-]{11})")
LABELS = ("true", "duplicate", "false", "ambiguous")


@dataclass(frozen=True)
class Clip:
    video: str
    start: float
    end: float


@dataclass(frozen=True)
class Moment:
    """A labeled event, or an ambiguous moment: one video, one or more ranges showing it."""

    video: str
    ranges: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Tally:
    """How a query's returned clips fared: one label per clip, in rank order."""

    events: int
    labels: tuple[str, ...]

    def count(self, label: str) -> int:
        return self.labels.count(label)

    @property
    def missed(self) -> int:
        return self.events - self.count("true")

    @property
    def precision(self) -> float | None:
        """True clips over judged clips (ambiguous ones are not judged); None with nothing judged."""
        judged = len(self.labels) - self.count("ambiguous")
        return self.count("true") / judged if judged else None

    @property
    def recall(self) -> float | None:
        """Events found over events labeled; None when the truth is empty."""
        return self.count("true") / self.events if self.events else None


@dataclass(frozen=True)
class Scored:
    id: str
    seconds: float
    failed: bool
    clips: tuple[Clip, ...]
    tally: Tally


def overlaps(clip: Clip, moment: Moment) -> bool:
    """True when the clip shares at least half of the shorter of itself and one of the moment's ranges."""
    if clip.video != moment.video:
        return False
    for start, end in moment.ranges:
        shared = min(clip.end, end) - max(clip.start, start)
        if shared > 0 and shared >= 0.5 * min(clip.end - clip.start, end - start):
            return True
    return False


def pairs(events: list[Moment], clips: tuple[Clip, ...]) -> dict[int, int]:
    """A maximum one-to-one matching of clips to the events they overlap: event index by clip index.

    Augmenting paths, clips in rank order, so a clip covering two events never costs a later clip
    its only match."""
    owner: dict[int, int] = {}

    def claim(c: int, seen: set[int]) -> bool:
        for e, event in enumerate(events):
            if e not in seen and overlaps(clips[c], event):
                seen.add(e)
                if e not in owner or claim(owner[e], seen):
                    owner[e] = c
                    return True
        return False

    for c in range(len(clips)):
        claim(c, set())
    return {c: e for e, c in owner.items()}


def tally(events: list[Moment], ambiguous: list[Moment], clips: tuple[Clip, ...]) -> Tally:
    paired = pairs(events, clips)

    def label(c: int, clip: Clip) -> str:
        if c in paired:
            return "true"
        if any(overlaps(clip, moment) for moment in ambiguous):
            return "ambiguous"
        if any(overlaps(clip, event) for event in events):
            return "duplicate"
        return "false"

    return Tally(len(events), tuple(label(c, clip) for c, clip in enumerate(clips)))


def moment(labeled: dict) -> Moment:
    ranges = tuple((parse_time(start), parse_time(end)) for start, end in labeled["ranges"])
    return Moment(labeled["video"], ranges)


def youtube_id(confirmed: dict) -> str:
    """The YouTube id in a clip's local file name or source URL; the system's own id when neither has one."""
    found = YOUTUBE_ID.search(f"{confirmed['local_path']} {confirmed['source']}")
    return found.group(1) if found else confirmed["video_id"]


def returned(output: dict | None) -> tuple[Clip, ...]:
    """The confirmed clips of a `find` output in rank order; none when the run failed."""
    if output is None:
        return ()
    return tuple(Clip(youtube_id(c), c["start"], c["end"]) for c in output["confirmed"])


def score(truth: dict, results: dict) -> list[Scored]:
    scored = []
    for run in results["runs"]:
        labeled = truth["queries"][run["id"]]
        clips = returned(run["output"])
        events = [moment(event) for event in labeled["events"]]
        ambiguous = [moment(event) for event in labeled["ambiguous"]]
        failed = run["exit_code"] != 0
        scored.append(Scored(run["id"], run["seconds"], failed, clips, tally(events, ambiguous, clips)))
    return scored


def ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def table(scored: list[Scored]) -> str:
    """A Markdown table with one row per query and a total row, then each clip's label."""
    lines = [
        "| query | events | clips | true | duplicate | false | ambiguous | missed | precision | recall | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scored:
        name = f"{s.id} (failed)" if s.failed else s.id
        counts = " | ".join(str(s.tally.count(label)) for label in LABELS)
        lines.append(
            f"| {name} | {s.tally.events} | {len(s.clips)} | {counts} | {s.tally.missed}"
            f" | {ratio(s.tally.precision)} | {ratio(s.tally.recall)} | {s.seconds:.0f} |"
        )
    total = Tally(sum(s.tally.events for s in scored), tuple(label for s in scored for label in s.tally.labels))
    counts = " | ".join(str(total.count(label)) for label in LABELS)
    seconds = sum(s.seconds for s in scored)
    lines.append(
        f"| all | {total.events} | {len(total.labels)} | {counts} | {total.missed}"
        f" | {ratio(total.precision)} | {ratio(total.recall)} | {seconds:.0f} |"
    )
    lines.append("")
    lines.append(f"mean seconds per query: {seconds / len(scored):.0f}")
    lines.append("")
    for s in scored:
        for clip, label in zip(s.clips, s.tally.labels, strict=True):
            lines.append(f"- {s.id}: {clip.video} {format_range(clip.start, clip.end)} {label}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="*", type=Path, help="results files saved by eval/run.py")
    parser.add_argument("--set", metavar="NAME", help="score against eval/NAME/truth.json instead of eval/truth.json")
    args = parser.parse_args()

    truth = json.loads(((EVAL / args.set if args.set else EVAL) / "truth.json").read_text())
    for path in args.results:
        results = json.loads(path.read_text())
        print(f"## {results['system']} ({path.name})\n")
        print(table(score(truth, results)))
        print()


if __name__ == "__main__":
    main()
