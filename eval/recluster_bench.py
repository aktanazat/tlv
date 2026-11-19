"""Times regrouping people when a video is added, on a synthetic collection, with no model.

    uv run python eval/recluster_bench.py --videos 40 --out /tmp/groups.json

Builds a private $TLV_HOME of `--videos` synthetic basketball videos (two teams of ten from a
league of 80 named players; per 6-second shot four players seen as a face on a numbered body; on-
screen text, speech and captions naming them; the owner's names on a few), adding each video the
way index_video does (its appearances, then recluster with its id), and prints the seconds each
recluster took. The vision-language model is stood in by "unsure" (no answers), so only the
grouping is timed. `--out` writes every appearance's group (as sorted member lists), each
entity's name and each caption observation's person, to compare two runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FIRST = [
    "Ada",
    "Ben",
    "Cal",
    "Dev",
    "Eli",
    "Fay",
    "Gus",
    "Hal",
    "Ivy",
    "Jon",
    "Kai",
    "Lev",
    "Max",
    "Ned",
    "Oli",
    "Pia",
    "Quin",
    "Ray",
    "Sol",
    "Tia",
]
LAST = ["Amos", "Brook", "Cole", "Drake"]
PLAYERS = 80
TEAMS = 8
SHOT_S = 6.0
SHOTS = 30
FACE_DIM, COLOR_DIM = 512, 27


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def build(home: Path, videos: int, seed: int) -> tuple[list[float], dict]:
    os.environ["TLV_HOME"] = str(home)
    os.environ.pop("TLV_INDEX", None)
    from tlv import entities, store

    entities._ask = lambda db, pairs, say: {}
    entities._ask_leads = lambda db, leads, say: {}
    rng = np.random.default_rng(seed)
    names = [f"{FIRST[i % len(FIRST)]} {LAST[i // len(FIRST)]}" for i in range(PLAYERS)]
    faces = [_unit(rng.normal(size=FACE_DIM)) for _ in range(PLAYERS)]
    uniforms = [_unit(rng.random(COLOR_DIM) ** 4) for _ in range(TEAMS)]
    team = [i % TEAMS for i in range(PLAYERS)]
    number = [str(10 + i // TEAMS) for i in range(PLAYERS)]
    db = store.connect()
    seconds = []
    for v in range(videos):
        video_id = f"v_{v:04d}"
        db.execute(
            "INSERT INTO videos (video_id, title, source, local_path, duration, width, height, fps, added_at, status)"
            " VALUES (?, ?, ?, ?, ?, 1280, 720, 30, ?, 'ready')",
            (video_id, video_id, video_id, str(home / f"{video_id}.mp4"), SHOT_S * SHOTS, store.now()),
        )
        home_team, away = rng.choice(TEAMS, size=2, replace=False)
        roster = [p for p in range(PLAYERS) if team[p] in (home_team, away)]
        apps, ocr, speech, spans, obs = [], [], [], [], []
        for s in range(SHOTS):
            start, end = s * SHOT_S, (s + 1) * SHOT_S - 0.5
            cast = rng.choice(roster, size=4, replace=False)
            for k, p in enumerate(cast):
                body_id = f"app_{video_id}_{s}_{k}_b"
                color = _unit(uniforms[team[p]] + rng.normal(scale=0.02, size=COLOR_DIM).clip(0))
                jersey = [{"kind": "jersey", "value": number[p], "t": start}] if rng.random() < 0.6 else []
                height = 0.6 if k == 0 else 0.3
                apps.append((body_id, "body", start, end, [0.1 * k, 0.2, 0.15, height], color, jersey, None, None))
                sharp = rng.random() < 0.6
                noise = 0.7 if sharp else 1.6
                face = _unit(faces[p] + rng.normal(scale=noise / np.sqrt(FACE_DIM), size=FACE_DIM))
                fh = 0.15 if k == 0 else 0.05
                quality = 0.5 if sharp else 0.15
                apps.append(
                    (
                        f"app_{video_id}_{s}_{k}_f",
                        "face",
                        start,
                        end,
                        [0.1 * k, 0.1, fh / 2, fh],
                        face,
                        [],
                        body_id,
                        quality,
                    )
                )
            lead = names[cast[0]]
            ocr.append((video_id, start + 1.0, f"{lead.upper()} 12 PTS"))
            ocr.append((video_id, start + 3.0, "LAKERS 88 RAPTORS 84"))
            line = f"and {lead.split()[0]} drives to the basket what a play by {lead}"
            words = [{"w": w, "s": start + 0.3 * i, "e": start + 0.3 * i + 0.25} for i, w in enumerate(line.split())]
            speech.append((video_id, start, start + 3.0, line, json.dumps(words)))
            people = [{"name": lead, "name_source": "on_screen", "jersey": number[cast[0]]}]
            span_id = f"sp_{video_id}_{s}"
            spans.append((span_id, video_id, start, end, f"{lead} drives", json.dumps({"people": people})))
            obs.append((f"caption:{span_id}", video_id, start, end, f"{lead} drives", store.now()))
        with store.transaction(db):
            db.executemany(
                "INSERT INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path,"
                " embedding, clues, body, quality) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        i,
                        video_id,
                        kind,
                        a,
                        b,
                        a,
                        json.dumps(box),
                        str(home / f"{i}.jpg"),
                        store.to_blob(vec),
                        json.dumps(clues),
                        body,
                        q,
                    )
                    for i, kind, a, b, box, vec, clues, body, q in apps
                ],
            )
            db.executemany("INSERT INTO ocr (video_id, t, text, x, y, w, h, conf) VALUES (?, ?, ?, 0, 0, 1, 1, 1)", ocr)
            db.executemany("INSERT INTO speech (video_id, start, end, text, words) VALUES (?, ?, ?, ?, ?)", speech)
            db.executemany(
                "INSERT INTO spans (span_id, video_id, start, end, caption, details) VALUES (?, ?, ?, ?, ?, ?)", spans
            )
            db.executemany(
                "INSERT INTO observations (obs_id, video_id, start, end, kind, text, confidence, evidence, created_at)"
                " VALUES (?, ?, ?, ?, 'caption', ?, 1, '{}', ?)",
                obs,
            )
        if v == 0:
            for k in range(3):
                face_id = f"app_{video_id}_{k}_0_f"
                db.execute(
                    "INSERT INTO corrections (kind, a, value, created_at) VALUES ('name', ?, ?, ?)",
                    (face_id, f"Owner Pick {k}", store.now()),
                )
        began = time.perf_counter()
        entities.recluster(db, video_id=video_id)
        seconds.append(time.perf_counter() - began)
        print(
            f"video {v + 1:3d}: {len([a for a in apps]):4d} appearances added, recluster {seconds[-1]:.3f} s",
            file=sys.stderr,
        )
    return seconds, snapshot(db)


def snapshot(db) -> dict:
    groups: dict[str, list[str]] = {}
    for appearance_id, entity_id in db.execute("SELECT appearance_id, entity_id FROM entity_members"):
        groups.setdefault(entity_id, []).append(appearance_id)
    names = {e: n for e, n in db.execute("SELECT entity_id, name FROM entities")}
    canon = {e: c for e, c in db.execute("SELECT entity_id, canonical_appearance FROM entities")}
    caption = {o: e for o, e in db.execute("SELECT obs_id, entity_id FROM observations WHERE kind = 'caption'")}
    clues = {a: c for a, c in db.execute("SELECT appearance_id, clues FROM appearances")}
    return {
        "groups": sorted(
            [{"members": sorted(m), "name": names[e], "canonical": canon[e]} for e, m in groups.items()],
            key=lambda g: g["members"][0],
        ),
        "entity_ids": sorted(groups),
        "captions": caption,
        "clues": clues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as home:
        seconds, groups = build(Path(home), args.videos, args.seed)
    for n in sorted({1, 10, 20, 40, 80, args.videos}):
        if n <= len(seconds):
            print(f"recluster adding video {n}: {seconds[n - 1]:.3f} s")
    print(f"total {sum(seconds):.2f} s; {len(groups['groups'])} people")
    if args.out:
        args.out.write_text(json.dumps(groups, indent=0, sort_keys=True))


if __name__ == "__main__":
    main()
