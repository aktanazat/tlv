"""Scores how the index groups people against the hand-checked labels in eval/people.json.

    uv run python eval/people.py                      # the live index, read-only
    uv run python eval/people.py --db ~/.local/share/tlv/peopletest.db
    uv run python eval/people.py --db ... --name coco --person "Rui Hachimura"
    uv run python eval/people.py --prove /private/var/tmp/akyl/People/proofhome

Each label says which person one appearance (a face or body track) shows. A label finds its
appearance by id, else by the same video and kind, a clearest frame within MATCH_S seconds and a
box overlapping by MATCH_IOU, so the labels outlive a re-index that moves track ids.

Per person: how many groups (entities) hold their labels, how many sit in the main one, whether one
group holds them in every video they appear in, and cross-video recall: of the pairs of their
labels from two different videos, the share in one group. Then every group holding two labeled
people (a false merge) and precision: of the labeled pairs that share a group, the share showing
one person. Unlabeled appearances count for neither.

With --name and --person: whether the group(s) so named hold exactly that person's labels in every
video, and which groups `tlv.entities.resolve` returns for "@NAME" and the bare name.

With --prove HOME: the correction proof, on a copy of the index made in HOME (which becomes
$TLV_HOME; the index itself is only read). The owner's corrections go through tlv.entities, which
regroups without the model, and so do the extra regroups, so no model process starts. Rui
Hachimura's main group is named "coco", then the other groups holding his labels: the name must
hold exactly his labels in every video, as the same group with the same members, through two more
regroups. Then the light-haired man's face, which indexing tied to LeBron's body, is told apart from
LeBron's group: it must leave with none of LeBron's labels and not his name, while LeBron's group
keeps every label it held, through two more regroups. Exits 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

EVAL = Path(__file__).resolve().parent
LIVE = Path.home() / ".local/share/tlv/tlv.db"
MATCH_S = 0.5
MATCH_IOU = 0.3
PEOPLE_HEADER = [
    "person",
    "videos",
    "labels",
    "groups",
    "main group",
    "in main group",
    "one group in every video",
    "cross-video pairs together",
]
PROOF_INDEX = "people-proof"
NICKNAME, NAMED = "coco", "Rui Hachimura"
STRAY, HOST = "Man with light hair", "LeBron James"


@dataclass(frozen=True)
class Label:
    appearance_id: str
    video: str
    kind: str
    t: float
    box: tuple[float, float, float, float]
    person: str


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"no index at {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def load(path: Path) -> list[Label]:
    data = json.loads(path.read_text())
    return [
        Label(x["appearance_id"], x["video_id"], x["kind"], x["t"], tuple(x["box"]), x["person"])
        for x in data["labels"]
    ]


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    inter = max(w, 0.0) * max(h, 0.0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def locate(db: sqlite3.Connection, labels: list[Label]) -> tuple[dict[Label, str], Counter[str]]:
    """Each label's appearance id in this index, and how many were found by id, by place, or not."""
    rows = db.execute("SELECT appearance_id, video_id, kind, best_t, box FROM appearances").fetchall()
    ids = {row["appearance_id"] for row in rows}
    found: dict[Label, str] = {}
    how: Counter[str] = Counter()
    for label in labels:
        if label.appearance_id in ids:
            found[label] = label.appearance_id
            how["by id"] += 1
            continue
        near = [
            (iou(label.box, tuple(json.loads(row["box"]))), row["appearance_id"])
            for row in rows
            if row["video_id"] == label.video and row["kind"] == label.kind and abs(row["best_t"] - label.t) <= MATCH_S
        ]
        best = max(near, default=(0.0, ""))
        if best[0] >= MATCH_IOU:
            found[label] = best[1]
            how["by place"] += 1
        else:
            how["missing"] += 1
    return found, how


def table(header: list[str], rows: list[list[object]]) -> str:
    cells = [header] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[k]) for row in cells) for k in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def percent(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"


class Grouping:
    """The index's groups, seen through the labels."""

    def __init__(self, db: sqlite3.Connection, labels: list[Label], short: dict[str, str]) -> None:
        self.found, self.how = locate(db, labels)
        member = {row[0]: row[1] for row in db.execute("SELECT appearance_id, entity_id FROM entity_members")}
        self.names = {row[0]: row[1] for row in db.execute("SELECT entity_id, name FROM entities")}
        self.short = short
        self.entity = {label: member.get(app) for label, app in self.found.items()}
        self.people: dict[str, list[Label]] = defaultdict(list)
        for label in labels:
            if self.entity.get(label):
                self.people[label.person].append(label)

    def title(self, entity: str) -> str:
        name = self.names.get(entity)
        return f'{entity} "{name}"' if name else entity

    def videos(self, person: str) -> list[str]:
        return sorted({label.video for label in self.people[person]}, key=self.short.get)

    def groups(self, person: str) -> Counter[str]:
        """The groups holding a person's labels, by label count."""
        return Counter(self.entity[label] for label in self.people[person])

    def main(self, person: str) -> str:
        """The group holding most of a person's labels."""
        return min(self.groups(person).items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def held(self, entity: str) -> Counter[str]:
        """The labeled people in a group, by label count."""
        return Counter(label.person for label, e in self.entity.items() if e == entity)

    def called(self, name: str) -> list[str]:
        wanted = " ".join(name.lower().split())
        return [e for e, n in self.names.items() if n and " ".join(n.lower().split()) == wanted]

    def per_person(self) -> tuple[list[list[object]], int, int, int, int]:
        rows, joined, pairs, whole, several = [], 0, 0, 0, 0
        for person, mine in sorted(self.people.items(), key=lambda kv: (-len(self.videos(kv[0])), -len(kv[1]))):
            groups = self.groups(person)
            main = self.main(person)
            held = groups[main]
            videos = self.videos(person)
            spans = any(
                all(any(self.entity[x] == entity and x.video == v for x in mine) for v in videos) for entity in groups
            )
            cross = [(a, b) for a, b in combinations(mine, 2) if a.video != b.video]
            together = sum(self.entity[a] == self.entity[b] for a, b in cross)
            if len(videos) > 1:
                joined, pairs, several = joined + together, pairs + len(cross), several + 1
                whole += spans
            rows.append(
                [
                    person,
                    " ".join(self.short[v] for v in videos),
                    len(mine),
                    len(groups),
                    self.title(main),
                    percent(held, len(mine)),
                    ("yes" if spans else "no") if len(videos) > 1 else "-",
                    percent(together, len(cross)),
                ]
            )
        return rows, joined, pairs, whole, several

    def breakdown(self, person: str) -> list[str]:
        mine = self.people[person]
        videos = self.videos(person)
        total = Counter(label.video for label in mine)
        lines = []
        groups = self.groups(person)
        for entity, _ in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0])):
            here = Counter(label.video for label in mine if self.entity[label] == entity)
            counts = "  ".join(f"{self.short[v]} {here[v]}/{total[v]}" for v in videos if here[v])
            lines.append(f"    {self.title(entity)}: {counts}")
        return lines

    def mixed(self) -> tuple[list[list[object]], int, int]:
        by_entity: dict[str, list[Label]] = defaultdict(list)
        for label, entity in self.entity.items():
            if entity:
                by_entity[entity].append(label)
        rows, same, pairs = [], 0, 0
        for entity, labels in sorted(by_entity.items()):
            for a, b in combinations(labels, 2):
                pairs += 1
                same += a.person == b.person
            people = Counter(label.person for label in labels)
            if len(people) > 1:
                for person, count in people.most_common():
                    shown = sorted((x for x in labels if x.person == person), key=lambda x: (self.short[x.video], x.t))
                    ids = " ".join(f"{x.appearance_id[4:10]}@{self.short[x.video]}:{x.t:g}" for x in shown[:8])
                    more = f" +{len(shown) - 8}" if len(shown) > 8 else ""
                    rows.append([self.title(entity), person, count, ids + more])
        return rows, same, pairs

    def named(self, name: str, person: str) -> bool:
        """Prints what the groups called `name` hold; True when they hold exactly `person`'s labels."""
        groups = self.called(name)
        if not groups:
            print(f'no group is named "{name}"')
            return False
        print(f'groups named "{name}": {", ".join(groups)}')
        mine = self.people[person]
        held_anywhere = {x.video for x, e in self.entity.items() if e in groups}
        videos = sorted({x.video for x in mine} | held_anywhere, key=self.short.get)
        rows, exact = [], len(groups) == 1
        for v in videos:
            in_video = [x for x in mine if x.video == v]
            held = [x for x in in_video if self.entity[x] in groups]
            others = Counter(
                x.person for x, e in self.entity.items() if e in groups and x.video == v and x.person != person
            )
            exact = exact and len(held) == len(in_video) and not others
            rows.append(
                [
                    self.short[v],
                    f"{len(held)} of {len(in_video)}",
                    ", ".join(f"{p} {n}" for p, n in others.most_common()) or "none",
                ]
            )
        print(table(["video", f"{person}'s labels in it", "other people's labels in it"], rows))
        return exact


def resolved(db: sqlite3.Connection, name: str) -> None:
    from tlv.entities import resolve  # reads the index only; imported here so scoring needs no tlv

    for query in (f"@{name} dunking", name):
        print(f"resolve({query!r}) -> {resolve(db, query)}")


def prove(source: Path, home: Path, labels: list[Label], short: dict[str, str]) -> bool:
    """The correction proof the module docstring describes; True when every check holds."""
    home.mkdir(parents=True, exist_ok=True)
    os.environ["TLV_HOME"] = str(home)
    os.environ.pop("TLV_INDEX", None)
    from tlv import entities, store  # imported once TLV_HOME points at the copy

    copy = home / f"{PROOF_INDEX}.db"
    for path in (copy, Path(f"{copy}-wal"), Path(f"{copy}-shm")):
        path.unlink(missing_ok=True)
    original, duplicate = connect(source), sqlite3.connect(copy)
    original.backup(duplicate)
    original.close()
    duplicate.close()
    db = store.connect(PROOF_INDEX)
    checks: list[tuple[str, bool]] = []

    def look() -> Grouping:
        return Grouping(db, labels, short)

    def regroup_twice() -> None:
        for _ in range(2):
            entities._regroup(db, lambda _: None, ask=False)  # recluster() would ask the model about look-alikes

    def nicknamed(seen: Grouping) -> tuple[list[str], set[str]]:
        """The groups called NICKNAME and their members."""
        groups = seen.called(NICKNAME)
        marks = ",".join("?" * len(groups))
        rows = db.execute(f"SELECT appearance_id FROM entity_members WHERE entity_id IN ({marks})", groups)
        return groups, {row[0] for row in rows}

    def show(seen: Grouping) -> None:
        print(table(PEOPLE_HEADER, [row for row in seen.per_person()[0] if row[0] in (HOST, STRAY)]))
        mixed, same, together = seen.mixed()
        print(
            f"precision: {percent(same, together)}; groups holding more than one labeled person:"
            f" {len({row[0] for row in mixed})}"
        )

    seen = look()
    main = seen.main(NAMED)
    before = seen.groups(NAMED)[main]
    print(f"== 1. name {NAMED}'s main group {main} {NICKNAME!r} ({before} of his {len(seen.people[NAMED])} labels)")
    entities.name(db, main, NICKNAME)
    seen = look()
    seen.named(NICKNAME, NAMED)
    groups = seen.called(NICKNAME)
    kept = sum(seen.groups(NAMED)[e] for e in groups)
    alone = all(set(seen.held(e)) == {NAMED} for e in groups)
    checks.append(
        (
            f"1. {NICKNAME!r} keeps the {before} labels of the group so named and takes in no one else",
            kept >= before and alone,
        )
    )

    pieces = [e for e in seen.groups(NAMED) if e not in groups and set(seen.held(e)) == {NAMED}]
    print(f"\n== 2. name the other groups holding his labels {NICKNAME!r} too: {', '.join(pieces) or 'none'}")
    for piece in pieces:
        entities.name(db, piece, NICKNAME)
    seen = look()
    checks.append((f"2. {NICKNAME!r} holds exactly {NAMED}'s labels in every video", seen.named(NICKNAME, NAMED)))
    settled = nicknamed(seen)

    print("\n== 3. regroup twice more")
    regroup_twice()
    seen = look()
    exact = seen.named(NICKNAME, NAMED)
    resolved(db, NICKNAME)
    checks.append(("3. still exactly, as the same group with the same members", exact and nicknamed(seen) == settled))

    host = seen.main(HOST)
    held = seen.groups(HOST)[host]
    (stray,) = seen.people[STRAY]
    print(f"\n== 4. tell {STRAY}'s appearance {seen.found[stray]} apart from {HOST}'s main group {host}; before:")
    show(seen)
    entities.different(db, seen.found[stray], host)
    for step in ("4", "5"):
        if step == "4":
            print("after:")
        else:
            print("\n== 5. regroup twice more")
            regroup_twice()
        seen = look()
        show(seen)
        group, host_name = seen.entity[stray], seen.names.get(host)
        own = seen.names.get(group)
        checks.append(
            (
                f"{step}. {STRAY} holds none of {HOST}'s labels and not the name {host_name!r}",
                set(seen.held(group)) == {STRAY} and (own is None or own != host_name),
            )
        )
        checks.append(
            (
                f"{step}. {HOST}'s group keeps all {held} of their labels it held",
                seen.main(HOST) == host and seen.groups(HOST)[host] >= held,
            )
        )
        checks.append((f"{step}. {NICKNAME!r} unchanged", nicknamed(seen) == settled))
    db.close()

    print("\n== checks")
    for what, ok in checks:
        print(f"{'yes' if ok else 'no '}  {what}")
    return all(ok for _, ok in checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=LIVE, help="index to score, opened read-only")
    parser.add_argument("--labels", type=Path, default=EVAL / "people.json")
    parser.add_argument("--name", help="check the group so named...")
    parser.add_argument("--person", help="...against this labeled person")
    parser.add_argument("--prove", type=Path, metavar="HOME", help="run the correction proof on a copy made in HOME")
    args = parser.parse_args()
    if bool(args.name) != bool(args.person):
        parser.error("--name and --person go together")
    if args.prove and args.prove.expanduser().resolve() == LIVE.parent.resolve():
        parser.error("--prove needs a scratch home, not the live one")
    labels = load(args.labels)
    short = json.loads(args.labels.read_text())["videos"]
    if args.prove:
        raise SystemExit(0 if prove(args.db.expanduser(), args.prove.expanduser(), labels, short) else 1)
    db = connect(args.db.expanduser())
    seen = Grouping(db, labels, short)
    print(f"index {args.db}")
    print(f"labels {len(labels)}: " + ", ".join(f"{n} {how}" for how, n in sorted(seen.how.items())))
    print()

    rows, joined, pairs, whole, several = seen.per_person()
    print("people")
    print(table(PEOPLE_HEADER, rows))
    print()
    print(
        f"cross-video recall: {percent(joined, pairs)} of label pairs; one group in every video for {whole} of"
        f" {several} people seen in more than one video"
    )
    print()
    for person in [p for p in seen.people if len(seen.videos(p)) > 1]:
        print(f"  {person}")
        print("\n".join(seen.breakdown(person)))
    print()

    mixed, same, together = seen.mixed()
    print(f"precision: {percent(same, together)} of labeled pairs sharing a group show one person")
    if mixed:
        print(f"false merges: {len({row[0] for row in mixed})} groups hold more than one labeled person")
        print(table(["group", "person", "labels", "labels (id@video:seconds)"], mixed))
    else:
        print("false merges: none")

    if args.name:
        print()
        exact = seen.named(args.name, args.person)
        resolved(db, args.name)
        answer = "yes" if exact else "no"
        print(f'"{args.name}" names exactly {args.person}\'s labeled appearances in every video: {answer}')


if __name__ == "__main__":
    main()
