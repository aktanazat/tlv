"""People across the collection.

`index_video` turns one video's Vision detections into appearances: one person followed through one
shot (tracks.py), as a face embedded with ArcFace, or a body known by the jersey number read off it
or by the face on it. `recluster` groups every appearance in the collection into entities
(cluster.py), first asking the vision-language model about a bounded number of look-alike faces
and, after a video is indexed, whether its people that no sharp face places are people the owner
named; it names each entity from corrections or agreeing clues (clues.py), and points each caption
observation at the person it is about. Corrections are stored against appearances, so they survive
reclustering and re-indexing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tlv import guard, store
from tlv.entities import clues as rules
from tlv.entities import crops
from tlv.entities.cluster import FACE_QUALITY, HALF_STEP, Appearance, Correction, Lead, cluster, describe
from tlv.entities.tracks import MIN_IOU, Box, Kind, Seen, Track, clearest, follow, iou, link
from tlv.index import vision

VERSION = 1

MIN_FACE_PX = 40  # a face shorter than this in every frame carries too little for ArcFace
MIN_BODY_H = 0.2  # a body shorter than this share of the frame is too small to read a number off
FACE_FRAMES = 3  # aligned faces averaged into one face appearance's vector
JERSEY_FRAMES = 3  # body crops read per body appearance for a jersey number
NEAR_S = 0.07  # a body seen in fewer frames is also read this long before and after its clearest one
READ_PX = 320  # body crops shorter than this are enlarged before reading
FACE_CROP, FACE_ROOM = 224, 1.6  # longest side of a saved face crop, and the room around the face
BODY_CROP = 384  # longest side of a saved body crop
AROUND_NUMBER = (3.0, 2.0)  # a jersey's colors are taken this many digit widths and heights around its number
MIDDLE = (0.3, 0.2, 0.4, 0.4)  # ...or across this part of a body crop when no number was read
FACE_BATCH = 64
# A name on screen or in speech is about the one person shown big then: a face or a body at least
# this share of the frame height.
BIG_FACE, BIG_BODY = 0.12, 0.5
MAX_QUESTIONS = 80  # look-alike face pairs the vision-language model is asked about per recluster
PER_QUESTION = 4  # candidates shown beside one reference face
# Questions about leads (cluster.Lead) asked per minute of a newly indexed video. In one batch, a
# pair of face crops took the model 0.47 s and a pair of body crops 0.63 s, so this keeps the check
# under 10 s a minute; an ESPN clip of 2.7 minutes used 29 of its 41, a 3-minute reel 2 of its 46.
LEADS_PER_MINUTE = 15
SAME_PERSON = {
    "face": "Image 1 and image 2 each show a person's face. Are they the same person? Judge the face itself: its"
    " shape, eyes, nose, mouth, skin, hair and beard. Ignore clothes, pose, lighting and background. Answer with"
    " one word: same, different or unsure.",
    "body": "Image 1 and image 2 each show a person at a basketball game. Are they the same person? Judge what tells"
    " people apart: the uniform and its colors, the number and the name on the jersey, skin, hair and build. A"
    " player, a referee, a coach and a fan are different people. Answer with one word: same, different or unsure.",
}
MIN_LISTED = 3  # unnamed people seen fewer times are left out of `entities()`
JERSEY_QUERY = re.compile(r"(?:#\s*|\bnumber\s+|\bno\.?\s*|\bjersey\s+)(\d{1,2})\b", re.IGNORECASE)
ENTITY_ID = re.compile(r"\bent_[0-9a-f]{12}\b")


def _quiet(_: str) -> None:
    return None


# Indexing one video.


@dataclass(frozen=True)
class _Found:
    """An appearance measured from a video, before it is stored."""

    id: str
    kind: Kind
    track: Track
    best: Seen
    crop: Path
    vector: np.ndarray
    clues: tuple[dict[str, Any], ...]
    body: str | None
    quality: float | None


def index_video(db: sqlite3.Connection, video_id: str, progress: guard.Progress | None = None) -> None:
    """Finds the people in one video, replacing its earlier appearances, then reclusters."""
    say = progress or _quiet
    video = db.execute("SELECT local_path, height FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if video is None:
        raise ValueError(f"No video {video_id}")
    faces, bodies = _tracks(db, video_id, video["height"])
    say(f"following {len(faces)} faces and {len(bodies)} bodies")
    work = store.data_dir("work") / "entities" / video_id
    shutil.rmtree(work, ignore_errors=True)
    try:
        found = _measure(Path(video["local_path"]), video_id, video["height"], faces, bodies, work, say)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with store.lock("entities", lambda: say("waiting for another entities run")):
        _replace(db, video_id, found)
    recluster(db, progress, video_id)


def _tracks(db: sqlite3.Connection, video_id: str, height: int) -> tuple[list[Track], list[Track]]:
    """The video's face and body tracks big enough to recognize; faces need landmarks to align."""
    cuts = [row["start"] for row in db.execute("SELECT start FROM shots WHERE video_id = ?", (video_id,))]
    seen: dict[str, list[Seen]] = {"face": [], "body": []}
    for row in db.execute(
        "SELECT t, kind, x, y, w, h, conf, extra FROM detections WHERE video_id = ? ORDER BY t", (video_id,)
    ):
        box = (row["x"], row["y"], row["w"], row["h"])
        seen[row["kind"]].append(Seen(row["t"], box, row["conf"], json.loads(row["extra"] or "{}")))
    faces = [track for track in follow("face", seen["face"], cuts) if any(_alignable(s, height) for s in track.seen)]
    bodies = [track for track in follow("body", seen["body"], cuts) if max(s.box[3] for s in track.seen) >= MIN_BODY_H]
    return faces, bodies


def _measure(
    path: Path,
    video_id: str,
    height: int,
    faces: list[Track],
    bodies: list[Track],
    work: Path,
    say: guard.Progress,
) -> list[_Found]:
    """Crops, embeds and reads every track, decoding the video once. A body is kept only when it says
    who it is: by the jersey number read off it, or by the face on it."""
    folder = store.data_dir("crops") / video_id
    face_ids = [_track_id(video_id, track) for track in faces]
    body_ids = [_track_id(video_id, track) for track in bodies]
    face_picks = [_face_picks(track, height) for track in faces]
    body_picks = [clearest(track, JERSEY_FRAMES) for track in bodies]
    read_at = [_read_frames(picks) for picks in body_picks]
    aligned = [[work / f"{i}-{k}.png" for k in range(len(p))] for i, p in zip(face_ids, face_picks, strict=True)]
    reads = [[work / f"{i}-read{k}.png" for k in range(len(r))] for i, r in zip(body_ids, read_at, strict=True)]
    cutters: dict[float, list[Callable[[np.ndarray], None]]] = defaultdict(list)
    for face_id, picks, outs in zip(face_ids, face_picks, aligned, strict=True):
        cutters[picks[0].t].append(partial(_face_crop, picks[0].box, folder / f"{face_id}.jpg"))
        for s, out in zip(picks, outs, strict=True):
            cutters[s.t].append(partial(_aligned, s.extra["landmarks"], out))
    for moments, outs in zip(read_at, reads, strict=True):
        for (t, box), out in zip(moments, outs, strict=True):
            cutters[t].append(partial(_read_crop, box, out))
    for t, frame in crops.frames(path, list(cutters)):
        for cutter in cutters[t]:
            cutter(frame)
    say(f"cropped people from {len(cutters)} frames")
    # A read just after a body's clearest frame can fall past the video's last frame, so it is never cut.
    reads = [[p for p in paths if p.exists()] for paths in reads]
    face_vectors = _mean_rows(_face_vectors([p for ps in aligned for p in ps], say), [len(ps) for ps in aligned])
    jerseys = _jerseys(reads)
    links = link(faces, bodies)
    linked = set(links.values())
    found = []
    for k, (i, track, picks, paths, (number, colors)) in enumerate(
        zip(body_ids, bodies, body_picks, reads, jerseys, strict=True)
    ):
        if number is None and k not in linked:
            continue
        crops.save(cv2.imread(str(paths[0])), folder / f"{i}.jpg", BODY_CROP)
        found.append(
            _Found(i, "body", track, picks[0], folder / f"{i}.jpg", colors, _jersey_clue(number, picks[0]), None, None)
        )
    for k, (i, track, picks, vector) in enumerate(zip(face_ids, faces, face_picks, face_vectors, strict=True)):
        body = body_ids[links[k]] if k in links else None
        found.append(
            _Found(i, "face", track, picks[0], folder / f"{i}.jpg", vector, (), body, _quality(picks[0].extra))
        )
    return found


def _alignable(s: Seen, height: int) -> bool:
    """A face detection ArcFace can use: landmarks to align it by, and enough pixels."""
    return bool(s.extra.get("landmarks")) and s.box[3] * height >= MIN_FACE_PX


def _face_picks(track: Track, height: int) -> list[Seen]:
    """The frames a face is embedded from: its clearest usable ones, only the sharp ones when it
    has any, so blur does not drag a sharp face's vector toward everyone else's."""
    usable = [s for s in clearest(track, len(track.seen)) if _alignable(s, height)]
    return ([s for s in usable if _quality(s.extra) >= FACE_QUALITY] or usable)[:FACE_FRAMES]


def _quality(extra: dict[str, Any]) -> float:
    """Vision's capture quality of a face detection; none given counts as unusable."""
    return extra.get("quality", 0.0)


def _read_frames(picks: list[Seen]) -> list[tuple[float, Box]]:
    """Where a body's jersey is read: its clearest frames, made up to JERSEY_FRAMES with moments
    just before and after the clearest one, so a body seen once is still read more than once."""
    best = picks[0]
    return ([(s.t, s.box) for s in picks] + [(best.t - NEAR_S, best.box), (best.t + NEAR_S, best.box)])[:JERSEY_FRAMES]


def _track_id(video_id: str, track: Track) -> str:
    first = track.seen[0]
    return _hashed_id(f"{video_id}:{track.kind}:{first.t:.2f}:{first.box}".encode())


def _hashed_id(key: bytes) -> str:
    return "app_" + hashlib.sha1(key).hexdigest()[:12]


def _face_crop(box: Box, out: Path, frame: np.ndarray) -> None:
    crops.save(crops.cut(frame, box, FACE_ROOM, square=True), out, FACE_CROP)


def _aligned(landmarks: list[list[float]], out: Path, frame: np.ndarray) -> None:
    crops.save(crops.align(frame, landmarks), out, crops.FACE_SIDE)


def _read_crop(box: Box, out: Path, frame: np.ndarray) -> None:
    image = crops.cut(frame, box)
    if image.shape[0] < READ_PX:
        scale = READ_PX / image.shape[0]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crops.save(image, out, max(image.shape[:2]))


def _face_vectors(paths: list[Path], say: guard.Progress) -> np.ndarray:
    """ArcFace vectors (unit rows) of aligned faces, one model load for all of them."""
    if not paths:
        return np.zeros((0, 512), dtype=np.float32)
    parts = []
    with guard.ModelProcess("tlv.models.faces", progress=say) as model:
        for start in range(0, len(paths), FACE_BATCH):
            reply = model.call({"images": [str(p) for p in paths[start : start + FACE_BATCH]]})
            parts.append(np.frombuffer(base64.b64decode(reply["vectors"]), dtype=np.float16).reshape(reply["shape"]))
    return np.concatenate(parts).astype(np.float32)


def _mean_rows(rows: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    """Unit means of consecutive runs of `counts` rows."""
    means, start = [], 0
    for count in counts:
        mean = rows[start : start + count].mean(axis=0)
        means.append(mean / np.linalg.norm(mean))
        start += count
    return means


def _jerseys(reads: list[list[Path]]) -> list[tuple[str | None, np.ndarray]]:
    """Each body's jersey number, voted over its reads, and the colors of its jersey."""
    lines = iter(vision.read_text([p for paths in reads for p in paths]))
    jerseys = []
    for paths in reads:
        seen = [rules.jersey((line.text, line.conf, line.box) for line in next(lines)) for _ in paths]
        number = rules.vote([read[0] if read else None for read in seen])
        jerseys.append((number, _uniform(paths, seen, number)))
    return jerseys


def _uniform(paths: list[Path], seen: list[tuple[str, Box] | None], number: str | None) -> np.ndarray:
    """The colors of a body's jersey: around its number where that was read biggest, else across
    the middle of its clearest crop."""
    shown = [(path, read[1]) for path, read in zip(paths, seen, strict=True) if read and read[0] == number]
    if not shown:
        return crops.colors(crops.cut(cv2.imread(str(paths[0])), MIDDLE))
    path, (x, y, w, h) = max(shown, key=lambda item: item[1][2] * item[1][3])
    wide, tall = AROUND_NUMBER
    around = (x + w / 2 - w * wide / 2, y + h / 2 - h * tall / 2, w * wide, h * tall)
    return crops.colors(crops.cut(cv2.imread(str(path)), around))


def _jersey_clue(number: str | None, best: Seen) -> tuple[dict[str, Any], ...]:
    return ({"kind": "jersey", "value": number, "t": best.t},) if number else ()


def _replace(db: sqlite3.Connection, video_id: str, found: list[_Found]) -> None:
    """Swaps a video's appearances for `found`, moving corrections from appearances that are gone
    to the new appearance of the same person, and dropping the crops of the ones that are gone."""
    old = db.execute(
        "SELECT appearance_id, kind, best_t, box, crop_path FROM appearances WHERE video_id = ?", (video_id,)
    ).fetchall()
    fresh = {f.id for f in found}
    gone = [row for row in old if row["appearance_id"] not in fresh]
    with store.transaction(db):
        for row in gone:
            heir = _heir(row, found)
            if heir:
                db.execute("UPDATE corrections SET a = ? WHERE a = ?", (heir, row["appearance_id"]))
                db.execute("UPDATE corrections SET b = ? WHERE b = ?", (heir, row["appearance_id"]))
        db.executemany("DELETE FROM appearances WHERE appearance_id = ?", [(row["appearance_id"],) for row in gone])
        db.executemany(
            "INSERT INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path, embedding,"
            " clues, body, quality) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (appearance_id) DO UPDATE"
            " SET start = excluded.start, end = excluded.end, best_t = excluded.best_t, box = excluded.box,"
            " crop_path = excluded.crop_path, embedding = excluded.embedding, clues = excluded.clues,"
            " body = excluded.body, quality = excluded.quality",
            [
                (
                    f.id,
                    video_id,
                    f.kind,
                    f.track.start,
                    f.track.end,
                    f.best.t,
                    json.dumps(f.best.box),
                    str(f.crop),
                    store.to_blob(f.vector),
                    json.dumps(f.clues),
                    f.body,
                    f.quality,
                )
                for f in found
            ],
        )
    for row in gone:
        Path(row["crop_path"]).unlink(missing_ok=True)


def _heir(old: sqlite3.Row, found: list[_Found]) -> str | None:
    """The new appearance of the same kind whose box at the old one's clearest frame overlaps it
    most, if by at least MIN_IOU."""
    box = tuple(json.loads(old["box"]))
    best, overlap = None, MIN_IOU
    for f in found:
        seen = f.track.at(old["best_t"]) if f.kind == old["kind"] else None
        if seen and iou(seen.box, box) >= overlap:
            best, overlap = f.id, iou(seen.box, box)
    return best


# Grouping the collection.


def recluster(db: sqlite3.Connection, progress: guard.Progress | None = None, video_id: str | None = None) -> None:
    """Regroups every appearance into entities, honoring every correction; the vision-language
    model is first asked about up to MAX_QUESTIONS look-alike face pairs it has not judged, then,
    with `video_id`, the video just indexed, about its leads: up to LEADS_PER_MINUTE questions per
    minute of it."""
    _regroup(db, progress or _quiet, ask=True, new_video=video_id)


def _regroup(db: sqlite3.Connection, say: guard.Progress, ask: bool, new_video: str | None = None) -> None:
    with store.lock("entities", lambda: say("waiting for another entities run")):
        _group(db, say, ask, new_video)


def _group(db: sqlite3.Connection, say: guard.Progress, ask: bool, new_video: str | None = None) -> None:
    """`_regroup` for a caller holding the entities lock."""
    _refresh_names(db)
    apps, fixes, verdicts = _appearances(db), _corrections(db), _verdicts(db)
    grouped = cluster(apps, fixes, verdicts, new_video)
    if ask and grouped.gray:
        answers = _ask(db, [(a, b) for a, b, _ in grouped.gray[:MAX_QUESTIONS]], say)
        if answers:
            verdicts |= answers
            grouped = cluster(apps, fixes, verdicts, new_video)
    if ask and new_video and grouped.leads:
        (minutes,) = db.execute("SELECT duration / 60.0 FROM videos WHERE video_id = ?", (new_video,)).fetchone()
        answers = _ask_leads(db, _within(grouped.leads, math.ceil(LEADS_PER_MINUTE * minutes)), say)
        if answers:
            grouped = cluster(apps, fixes, verdicts | answers)
    _save(db, apps, fixes, grouped.groups)
    _link_observations(db)
    say(f"{len(grouped.groups)} people")


@contextmanager
def leaving(db: sqlite3.Connection, video_id: str, waiting: Callable[[], None] | None = None) -> Iterator[None]:
    """Takes a video's people out of the collection while the caller deletes the video, all in one
    transaction under the entities lock. Before the caller's block, each correction given through
    one of the video's appearances moves to the appearance its person is best known by among those
    left (cluster.describe's canonical one), so a person also seen in other videos keeps the names,
    joins and splits the owner gave them; a correction about someone seen only in this video goes.
    Corrections on the appearances left stay as they are: a moved name goes only to an appearance
    the owner has not named. The block deletes the video, and with it its appearances; everyone left
    is then regrouped without asking the model, so people seen only in this video are gone."""
    with store.lock("entities", waiting), store.transaction(db):
        _hand_down(db, video_id)
        yield
        _group(db, _quiet, ask=False)


def _hand_down(db: sqlite3.Connection, video_id: str) -> None:
    apps = _appearances(db)
    by_id = {a.id: a for a in apps}
    gone = {a.id for a in apps if a.video_id == video_id}
    entity_of = {row[0]: row[1] for row in db.execute("SELECT appearance_id, entity_id FROM entity_members")}
    members: dict[str, list[str]] = defaultdict(list)
    for appearance_id, entity_id in entity_of.items():
        if appearance_id not in gone:
            members[entity_id].append(appearance_id)
    heirs: dict[str, str | None] = {}
    for appearance_id in gone:
        left = members.get(entity_of.get(appearance_id, ""), [])
        heirs[appearance_id] = describe([by_id[i] for i in left]).canonical if left else None
    named = {row[0] for row in db.execute("SELECT a FROM corrections WHERE kind = 'name'") if row[0] not in gone}
    moved, dropped = [], []
    for row in db.execute("SELECT id, kind, a, b FROM corrections ORDER BY created_at DESC, id DESC").fetchall():
        if row["a"] not in gone and row["b"] not in gone:
            continue
        a = heirs[row["a"]] if row["a"] in gone else row["a"]
        b = heirs[row["b"]] if row["b"] in gone else row["b"]
        if row["kind"] == "name" and a is not None and a not in named:
            named.add(a)
            moved.append((a, b, row["id"]))
        elif row["kind"] != "name" and a is not None and b is not None and a != b:
            moved.append((a, b, row["id"]))
        else:
            dropped.append((row["id"],))
    db.executemany("UPDATE corrections SET a = ?, b = ? WHERE id = ?", moved)
    db.executemany("DELETE FROM corrections WHERE id = ?", dropped)


def _appearances(db: sqlite3.Connection) -> list[Appearance]:
    return [
        Appearance(
            row["appearance_id"],
            row["video_id"],
            row["kind"],
            row["start"],
            row["end"],
            tuple(json.loads(row["box"])),
            store.from_blob(row["embedding"]),
            tuple(json.loads(row["clues"])),
            row["body"],
            row["quality"],
        )
        for row in db.execute(
            "SELECT appearance_id, video_id, kind, start, end, box, embedding, clues, body, quality FROM appearances"
            " ORDER BY appearance_id"
        )
    ]


def _corrections(db: sqlite3.Connection) -> list[Correction]:
    return [
        Correction(row["kind"], row["a"], row["b"], row["value"], row["created_at"])
        for row in db.execute("SELECT kind, a, b, value, created_at FROM corrections ORDER BY id")
    ]


def _verdicts(db: sqlite3.Connection) -> dict[tuple[str, str], bool | None]:
    return {
        (row["a"], row["b"]): None if row["same"] is None else bool(row["same"])
        for row in db.execute("SELECT a, b, same FROM verdicts")
    }


def _save(db: sqlite3.Connection, apps: list[Appearance], fixes: list[Correction], groups: list[list[str]]) -> None:
    """Writes the groups as entities. A group keeps the id of the earlier entity it shares the most
    appearances with, so ids stay put while membership shifts."""
    by_id = {a.id: a for a in apps}
    before = {row[0]: row[1] for row in db.execute("SELECT appearance_id, entity_id FROM entity_members")}
    taken: set[str] = set()
    entities, members = [], []
    for group in sorted(groups, key=lambda g: (-len(g), min(g))):
        about = describe([by_id[i] for i in group], fixes)
        votes = Counter(before[i] for i in group if i in before)
        kept = next((e for e, _ in votes.most_common() if e not in taken), None)
        entity_id = kept or "ent_" + hashlib.sha1(",".join(sorted(group)).encode()).hexdigest()[:12]
        taken.add(entity_id)
        vector = store.to_blob(about.vector)
        entities.append((entity_id, about.name, json.dumps(about.evidence), about.canonical, vector, store.now()))
        members += [(i, entity_id) for i in group]
    with store.transaction(db):
        db.execute("DELETE FROM entities")
        db.executemany(
            "INSERT INTO entities (entity_id, name, name_evidence, canonical_appearance, embedding, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            entities,
        )
        db.executemany("INSERT INTO entity_members (appearance_id, entity_id) VALUES (?, ?)", members)


# Asking the vision-language model about look-alike faces, and about leads.


def _ask(
    db: sqlite3.Connection, pairs: list[tuple[str, str]], say: guard.Progress
) -> dict[tuple[str, str], bool | None]:
    """The model's verdict on each pair, stored so no pair is asked twice. When the model cannot
    run, the pairs stay unasked, and so apart."""
    crop = _crops(db)
    questions = _questions(pairs)
    answers: dict[tuple[str, str], bool | None] = {}
    try:
        with guard.ModelProcess("tlv.models.vlm", progress=say) as model:
            for k, (reference, others) in enumerate(questions, 1):
                reply = model.call(
                    {
                        "images": [crop[reference], *(crop[o] for o in others)],
                        "prompt": _prompt(len(others)),
                        "max_tokens": 16 + 12 * len(others),
                    }
                )
                for other, same in zip(others, _verdict_list(reply["text"], len(others)), strict=True):
                    answers[min(reference, other), max(reference, other)] = same
                say(f"compared look-alike faces {k} of {len(questions)}")
    except (guard.ModelError, guard.HeadroomError) as error:
        say(f"left look-alike faces apart: {error}")
    _remember(db, answers)
    return answers


def _ask_leads(db: sqlite3.Connection, leads: list[Lead], say: guard.Progress) -> dict[tuple[str, str], bool | None]:
    """The model's verdict on each lead, then on the bodies of each lead it says is the named
    person, stored as `_ask` stores its own. A body it says is someone else is also stored as a
    different person from every face sitting on it, so the face joins the named person alone."""
    crop = _crops(db)
    answers: dict[tuple[str, str], bool | None] = {}
    bodies: list[tuple[str, str, str]] = []
    try:
        with guard.ModelProcess("tlv.models.vlm", progress=say) as model:
            answers |= _judge(model, crop, [(lead.kind, lead.reference, lead.candidate) for lead in leads])
            found = [lead for lead in leads if answers[_pair(lead.reference, lead.candidate)]]
            bodies = [("body", reference, body) for lead in found for reference, body in lead.bodies]
            if bodies:
                answers |= _judge(model, crop, bodies)
            say(f"compared {len(leads)} people in the new video with the people you named")
    except (guard.ModelError, guard.HeadroomError) as error:
        say(f"left the new video's unknown people apart: {error}")
    for _, reference, body in bodies:
        if answers.get(_pair(reference, body)) is False:
            sitting = db.execute("SELECT appearance_id FROM appearances WHERE body = ?", (body,)).fetchall()
            answers |= {_pair(face, body): False for (face,) in sitting}
    _remember(db, answers)
    return answers


def _within(leads: list[Lead], budget: int) -> list[Lead]:
    """The closest leads whose questions, each lead's own and its bodies', fit in `budget`."""
    taken = []
    for lead in leads:
        budget -= 1 + len(lead.bodies)
        if budget < 0:
            break
        taken.append(lead)
    return taken


def _judge(
    model: guard.ModelProcess, crop: dict[str, str], pairs: list[tuple[str, str, str]]
) -> dict[tuple[str, str], bool | None]:
    """The model's same/different/unsure on each (kind, reference, candidate), asked as one batch."""
    jobs = [{"images": [crop[r], crop[c]], "prompt": SAME_PERSON[kind], "max_tokens": 4} for kind, r, c in pairs]
    reply = model.call({"batch": jobs})
    return {_pair(r, c): _word(result["text"]) for (_, r, c), result in zip(pairs, reply["results"], strict=True)}


def _word(text: str) -> bool | None:
    """The model's one-word answer; anything else is unsure."""
    words = re.findall(r"[a-z]+", text.lower())
    return {"same": True, "different": False}.get(words[0] if words else "")


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _crops(db: sqlite3.Connection) -> dict[str, str]:
    return {row[0]: row[1] for row in db.execute("SELECT appearance_id, crop_path FROM appearances")}


def _remember(db: sqlite3.Connection, answers: dict[tuple[str, str], bool | None]) -> None:
    with store.transaction(db):
        db.executemany(
            "INSERT OR REPLACE INTO verdicts (a, b, same, created_at) VALUES (?, ?, ?, ?)",
            [(a, b, None if same is None else int(same), store.now()) for (a, b), same in answers.items()],
        )


def _questions(pairs: list[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    """Pairs gathered into questions: one reference face shown with up to PER_QUESTION others,
    the face in the most pairs first."""
    left = list(pairs)
    questions = []
    while left:
        counts = Counter(x for pair in left for x in pair)
        reference = max(counts, key=lambda x: (counts[x], x))
        mine = [pair for pair in left if reference in pair][:PER_QUESTION]
        questions.append((reference, [b if a == reference else a for a, b in mine]))
        left = [pair for pair in left if pair not in mine]
    return questions


def _prompt(count: int) -> str:
    numbers = ", ".join(str(k) for k in range(2, count + 2))
    example = ", ".join(f'"{k}": "different"' for k in range(2, count + 2))
    return (
        f"Image 1 shows a person's face. Images {numbers} each show a face too. For each of them, decide whether it "
        "is the same person as image 1. Judge the face itself: its shape, eyes, nose, mouth, skin, hair and beard. "
        "Ignore clothes, pose, lighting and background. Say unsure when you cannot tell. "
        f'Answer with JSON only, in this form: {{{example}}}, each value "same", "different" or "unsure".'
    )


def _verdict_list(text: str, count: int) -> list[bool | None]:
    """The model's same/different/unsure for each candidate; anything unreadable is unsure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        reply = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        reply = {}
    if not isinstance(reply, dict):
        reply = {}
    meaning = {"same": True, "different": False}
    return [meaning.get(str(reply.get(str(k), "")).strip().lower()) for k in range(2, count + 2)]


# Name clues: a known name on screen, in speech or in a caption, tied to the one person shown big.


def _refresh_names(db: sqlite3.Connection) -> None:
    """Recomputes every appearance's name clues; jersey clues stay."""
    names = _vocabulary(db)
    found = _name_clues(db, names) if names else {}
    changes = []
    for row in db.execute("SELECT appearance_id, clues FROM appearances"):
        before = json.loads(row["clues"])
        after = [c for c in before if c["kind"] == "jersey"] + found.get(row["appearance_id"], [])
        if after != before:
            changes.append((json.dumps(after), row["appearance_id"]))
    if changes:
        with store.transaction(db):
            db.executemany("UPDATE appearances SET clues = ? WHERE appearance_id = ?", changes)


def _vocabulary(db: sqlite3.Connection) -> list[str]:
    """Every name a caption read or heard for a person, or a correction gave an appearance."""
    names = [row["value"] for row in db.execute("SELECT value FROM corrections WHERE kind = 'name'")]
    for row in db.execute("SELECT details FROM spans WHERE details IS NOT NULL"):
        names += [said for person in json.loads(row["details"]).get("people") or [] if (said := _caption_name(person))]
    return list(dict.fromkeys(name.strip() for name in names if rules.words(name)))


def _caption_name(person: dict[str, Any]) -> str | None:
    """The name a caption gives a person, when the captioner read it on screen or heard it."""
    said = (person.get("name") or "").strip()
    return said if rules.words(said) and person.get("name_source") in rules.CAPTION_CHANNEL else None


@dataclass(frozen=True)
class _Shown:
    """An appearance as seen from its video: when, how big, whose, wearing what."""

    id: str
    kind: str
    start: float
    end: float
    height: float
    person: str  # a face linked to a body is that body's person
    jersey: str | None


def _name_clues(db: sqlite3.Connection, names: list[str]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video_id, shown in _cast(db).items():
        lead = _lead(shown)
        for row in db.execute("SELECT t, text FROM ocr WHERE video_id = ?", (video_id,)):
            for name in rules.mentions(row["text"], names):
                for app in lead(row["t"]):
                    found[app].append({"kind": "ocr_name", "value": name, "t": row["t"]})
        for row in db.execute("SELECT words FROM speech WHERE video_id = ?", (video_id,)):
            timed = [(w["w"], (w["s"] + w["e"]) / 2) for w in json.loads(row["words"])]
            for name, t in rules.said(timed, names):
                for app in lead(t):
                    found[app].append({"kind": "spoken_name", "value": name, "t": round(t, 2)})
        for row in db.execute(
            "SELECT start, end, details FROM spans WHERE video_id = ? AND details IS NOT NULL", (video_id,)
        ):
            people = json.loads(row["details"]).get("people") or []
            for person in people:
                said = _caption_name(person)
                if not said:
                    continue
                clue = {
                    "kind": "caption_name",
                    "value": said,
                    "t": round((row["start"] + row["end"]) / 2, 2),
                    "source": person["name_source"],
                }
                for app in _captioned(shown, lead, row["start"], row["end"], person, len(people)):
                    found[app].append(clue)
    return found


def _cast(db: sqlite3.Connection) -> dict[str, list[_Shown]]:
    """Each video's appearances, with whose they are and the jersey number they wear."""
    cast: dict[str, list[_Shown]] = defaultdict(list)
    for row in db.execute(
        "SELECT appearance_id, video_id, kind, start, end, box, clues, body FROM appearances WHERE video_id IS NOT NULL"
    ):
        jersey = next((c["value"] for c in json.loads(row["clues"]) if c["kind"] == "jersey"), None)
        height = json.loads(row["box"])[3]
        person = row["body"] or row["appearance_id"]
        cast[row["video_id"]].append(
            _Shown(row["appearance_id"], row["kind"], row["start"], row["end"], height, person, jersey)
        )
    return cast


def _lead(shown: list[_Shown]) -> Callable[[float], list[str]]:
    """For a time, the appearances of the one person shown big then; none when nobody, or more than
    one person, is."""
    seconds: dict[int, list[_Shown]] = defaultdict(list)
    for s in shown:
        for second in range(int(s.start - HALF_STEP), int(s.end + HALF_STEP) + 1):
            seconds[second].append(s)

    def at(t: float) -> list[str]:
        here = [s for s in seconds.get(int(t), ()) if s.start - HALF_STEP <= t <= s.end + HALF_STEP]
        big = {s.person for s in here if s.height >= (BIG_FACE if s.kind == "face" else BIG_BODY)}
        return [s.id for s in here if s.person in big] if len(big) == 1 else []

    return at


def _captioned(
    shown: list[_Shown],
    lead: Callable[[float], list[str]],
    start: float,
    end: float,
    person: dict[str, Any],
    count: int,
) -> list[str]:
    """The appearances a caption's person is: the one person in the span wearing their jersey
    number, else, when the caption names only one person, the one shown big mid-span."""
    if person.get("jersey"):
        during = [s for s in shown if s.start <= end and s.end >= start]
        wearing = {s.person for s in during if s.jersey == str(person["jersey"])}
        return [s.id for s in during if s.person in wearing] if len(wearing) == 1 else []
    return lead((start + end) / 2) if count == 1 else []


def _link_observations(db: sqlite3.Connection) -> None:
    """Points each caption observation at the person its caption is mainly about: the one present
    person that its name or jersey number picks out, else nobody."""
    captions = db.execute(
        "SELECT o.obs_id, o.video_id, o.start, o.end, o.entity_id, s.details FROM observations o"
        " JOIN spans s ON 'caption:' || s.span_id = o.obs_id WHERE o.kind = 'caption'"
    ).fetchall()
    if not captions:
        return
    names = {row["entity_id"]: row["name"] for row in db.execute("SELECT entity_id, name FROM entities")}
    shown: dict[str, list[tuple[float, float, str, str | None]]] = defaultdict(list)
    for row in db.execute(
        "SELECT a.video_id, a.start, a.end, a.clues, m.entity_id FROM appearances a"
        " JOIN entity_members m USING (appearance_id) WHERE a.video_id IS NOT NULL"
    ):
        jersey = next((c["value"] for c in json.loads(row["clues"]) if c["kind"] == "jersey"), None)
        shown[row["video_id"]].append((row["start"] - HALF_STEP, row["end"] + HALF_STEP, row["entity_id"], jersey))
    changes = []
    for row in captions:
        people = json.loads(row["details"] or "{}").get("people") or []
        here = [s for s in shown[row["video_id"]] if s[0] <= row["end"] and s[1] >= row["start"]]
        entity_id = _subject(people[0], here, names) if people else None
        if entity_id != row["entity_id"]:
            changes.append((entity_id, row["obs_id"]))
    if changes:
        with store.transaction(db):
            db.executemany("UPDATE observations SET entity_id = ? WHERE obs_id = ?", changes)


def _subject(
    person: dict[str, Any], here: list[tuple[float, float, str, str | None]], names: dict[str, str | None]
) -> str | None:
    """The one present entity a caption's person is, by name, else by jersey number."""
    present = {entity_id for _, _, entity_id, _ in here}
    said = _caption_name(person)
    if said:
        named = {e for e in present if names.get(e) and rules.same_person(said, names[e] or "")}
        if named:
            return named.pop() if len(named) == 1 else None
    if person.get("jersey"):
        wearing = {entity_id for _, _, entity_id, jersey in here if jersey == str(person["jersey"])}
        return wearing.pop() if len(wearing) == 1 else None
    return None


# Questions about people.

_LISTING = """
SELECT e.entity_id, e.name, e.name_evidence, c.crop_path, COUNT(*) AS appearances,
       COUNT(DISTINCT a.video_id) AS videos, GROUP_CONCAT(DISTINCT a.video_id) AS video_ids
FROM entities e
JOIN appearances c ON c.appearance_id = e.canonical_appearance
JOIN entity_members m ON m.entity_id = e.entity_id
JOIN appearances a ON a.appearance_id = m.appearance_id
"""


def entities(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every named person, and every unnamed one seen at least MIN_LISTED times, most seen first."""
    rows = db.execute(
        _LISTING + " GROUP BY e.entity_id HAVING e.name IS NOT NULL OR COUNT(*) >= ?"
        " ORDER BY e.name IS NULL, appearances DESC, e.entity_id",
        (MIN_LISTED,),
    )
    return [_row(row) for row in rows]


def _row(row: sqlite3.Row) -> dict[str, Any]:
    evidence = json.loads(row["name_evidence"])
    named_by = None if row["name"] is None else "correction" if evidence[0]["kind"] == "correction" else "clues"
    return {
        "entity_id": row["entity_id"],
        "name": row["name"],
        "named_by": named_by,
        "appearances": row["appearances"],
        "videos": row["videos"],
        "video_ids": sorted(row["video_ids"].split(",")) if row["video_ids"] else [],
        "crop_path": row["crop_path"],
    }


def _rows_of(db: sqlite3.Connection, appearance_ids: Iterable[str]) -> list[dict[str, Any]]:
    """The entity row holding each appearance."""
    rows = []
    for appearance_id in appearance_ids:
        (entity_id,) = db.execute(
            "SELECT entity_id FROM entity_members WHERE appearance_id = ?", (appearance_id,)
        ).fetchone()
        rows.append(_row(db.execute(_LISTING + " WHERE e.entity_id = ? GROUP BY e.entity_id", (entity_id,)).fetchone()))
    return rows


def resolve(db: sqlite3.Connection, text: str) -> list[str]:
    """The entity ids a query points to, by a person's name ("LeBron", "@LeBron_James"), by an
    entity id, or failing those by a jersey number ("#23"); empty when unsure."""
    sizes = {row[0]: row[1] for row in db.execute("SELECT entity_id, COUNT(*) FROM entity_members GROUP BY entity_id")}
    named: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in db.execute("SELECT entity_id, name FROM entities WHERE name IS NOT NULL"):
        named[tuple(rules.words(row["name"]))].append(row["entity_id"])
    found = [e for e in ENTITY_ID.findall(text) if e in sizes]
    for spelled in rules.mentions(text, [" ".join(key) for key in named]):
        found += named[tuple(rules.words(spelled))]
    if not found:
        found = [e for number in JERSEY_QUERY.findall(text) for e in _wearing(db, number)]
    return sorted(set(found), key=lambda e: (-sizes.get(e, 0), e))


def _wearing(db: sqlite3.Connection, number: str) -> list[str]:
    """The people seen wearing `number`: the named ones when their names agree, else the only one."""
    people: dict[str, str | None] = {}
    for row in db.execute(
        "SELECT m.entity_id, e.name, a.clues FROM appearances a JOIN entity_members m USING (appearance_id)"
        " JOIN entities e USING (entity_id) WHERE a.kind = 'body'"
    ):
        if any(c["kind"] == "jersey" and c["value"] == number for c in json.loads(row["clues"])):
            people[row["entity_id"]] = row["name"]
    named = [n for n in people.values() if n]
    if named:
        agree = all(rules.same_person(a, b) for a in named for b in named)
        return [e for e, n in people.items() if n] if agree else []
    return list(people) if len(people) == 1 else []


def present(db: sqlite3.Connection, video_id: str, start: float, end: float) -> list[dict[str, Any]]:
    """Who is on screen in a video between start and end (seconds), with the stretches each is
    seen, earliest first."""
    rows = db.execute(
        "SELECT m.entity_id, e.name, a.start, a.end FROM appearances a"
        " JOIN entity_members m USING (appearance_id) JOIN entities e USING (entity_id)"
        " WHERE a.video_id = ? AND a.end + ? >= ? AND a.start - ? <= ? ORDER BY a.start",
        (video_id, HALF_STEP, start, HALF_STEP, end),
    )
    stretches: dict[str, list[list[float]]] = defaultdict(list)
    names: dict[str, str | None] = {}
    for row in rows:
        names[row["entity_id"]] = row["name"]
        s, e = max(start, row["start"] - HALF_STEP), min(end, row["end"] + HALF_STEP)
        times = stretches[row["entity_id"]]
        if times and s <= times[-1][1]:
            times[-1][1] = max(times[-1][1], e)
        else:
            times.append([s, e])
    found = [
        {"entity_id": e, "name": names[e], "start": times[0][0], "end": times[-1][1], "times": times}
        for e, times in stretches.items()
    ]
    return sorted(found, key=lambda p: (p["start"], p["entity_id"]))


# Corrections.


def name(db: sqlite3.Connection, target: str, name: str) -> list[dict[str, Any]]:
    """Names a person. An entity id names everyone in it, which keeps them together as that person
    whatever later evidence says, until one of them is told apart; an appearance id says who that
    one appearance shows. Names are stored against appearances, so they survive reclustering."""
    value = _clean_name(name)
    appearance_id = _appearance_of(db, target)
    named = _side(db, target)
    now = store.now()
    with store.transaction(db):
        db.executemany("DELETE FROM corrections WHERE kind = 'name' AND a = ?", [(a,) for a in named])
        db.executemany(
            "INSERT INTO corrections (kind, a, value, created_at) VALUES ('name', ?, ?, ?)",
            [(a, value, now) for a in named],
        )
    _regroup(db, _quiet, ask=False)
    return _rows_of(db, [appearance_id])


def same(db: sqlite3.Connection, a: str, b: str) -> list[dict[str, Any]]:
    """Says two entities or appearances are one person; returns the merged person."""
    x, y = _appearance_of(db, a), _appearance_of(db, b)
    if x == y:
        raise ValueError(f"{a} and {b} are already one person")
    _correct(db, "same", _side(db, a), _side(db, b), [(x, y)])
    return _rows_of(db, [x])


def different(db: sqlite3.Connection, a: str, b: str) -> list[dict[str, Any]]:
    """Says two entities or appearances are different people; returns both. One appearance told
    apart from a person is told apart from everyone in them and loses the name it shares with
    them, so what tied it there (a face laid on the wrong body, say) cannot take part of them along."""
    x, y = _appearance_of(db, a), _appearance_of(db, b)
    left, right = _side(db, a), _side(db, b)
    if len(left) > 1 and len(right) > 1:  # two people, told apart through their canonical appearances
        pairs = [(x, y)] if x != y else []
    else:
        pairs = [(p, q) for p in left for q in right if p != q]
    if not pairs:
        raise ValueError(f"{a} and {b} are one appearance; name two of its appearances to split a person")
    unnamed = [
        lone[0]
        for lone, person, others in ((left, b, right), (right, a, left))
        if len(lone) == 1 and len(others) > 1 and _named_as(db, lone[0], person)
    ]
    _correct(db, "different", left, right, pairs, unnamed)
    return _rows_of(db, [x, y])


def _side(db: sqlite3.Connection, target: str) -> list[str]:
    """The appearances a correction about `target` speaks of: everyone in an entity, or the one given."""
    members = [row[0] for row in db.execute("SELECT appearance_id FROM entity_members WHERE entity_id = ?", (target,))]
    return members or [_appearance_of(db, target)]


def _named_as(db: sqlite3.Connection, appearance_id: str, entity_id: str) -> bool:
    """Whether a name correction gives the appearance the name of the entity's person."""
    person = db.execute("SELECT name FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()
    given = db.execute(
        "SELECT value FROM corrections WHERE kind = 'name' AND a = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (appearance_id,),
    ).fetchone()
    return bool(person and person[0] and given and rules.same_person(given[0], person[0]))


def _correct(
    db: sqlite3.Connection,
    kind: str,
    left: Sequence[str],
    right: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    unnamed: Sequence[str] = (),
) -> None:
    """Records same/different for `pairs`, replacing every earlier answer between the two sides,
    drops the name corrections of `unnamed`, and regroups."""
    left_set, right_set = set(left), set(right)
    now = store.now()
    with store.transaction(db):
        earlier = [
            (row["id"],)
            for row in db.execute("SELECT id, a, b FROM corrections WHERE kind IN ('same', 'different')")
            if (row["a"] in left_set and row["b"] in right_set) or (row["a"] in right_set and row["b"] in left_set)
        ]
        db.executemany("DELETE FROM corrections WHERE id = ?", earlier)
        db.executemany("DELETE FROM corrections WHERE kind = 'name' AND a = ?", [(a,) for a in unnamed])
        db.executemany(
            "INSERT INTO corrections (kind, a, b, created_at) VALUES (?, ?, ?, ?)",
            [(kind, p, q, now) for p, q in pairs],
        )
    _regroup(db, _quiet, ask=False)


def _appearance_of(db: sqlite3.Connection, target: str) -> str:
    """An appearance id as given, or an entity's canonical appearance."""
    row = db.execute("SELECT canonical_appearance FROM entities WHERE entity_id = ?", (target,)).fetchone()
    if row:
        return row[0]
    if db.execute("SELECT 1 FROM appearances WHERE appearance_id = ?", (target,)).fetchone():
        return target
    raise ValueError(f"No person or appearance {target}")


def _clean_name(name: str) -> str:
    value = name.strip().removeprefix("@").strip()
    if not rules.words(value):
        raise ValueError(f"{name!r} is not a name")
    return value


def add_reference(
    db: sqlite3.Connection,
    name: str,
    images: Sequence[str | Path] = (),
    frames: Sequence[tuple[str, float]] = (),
) -> dict[str, Any]:
    """Teaches who a person is from photos of their face (local image files) and from moments in
    indexed videos where they are the one shown biggest, each (video_id, seconds)."""
    value = _clean_name(name)
    if not images and not frames:
        raise ValueError("Give at least one reference image or (video_id, seconds) frame")
    known = {tuple(rules.words(row[0])) for row in db.execute("SELECT name FROM entities WHERE name IS NOT NULL")}
    targets = [_shown_at(db, video_id, float(t)) for video_id, t in frames]
    if images:
        targets += _photographed(db, [(str(p), Path(p).expanduser()) for p in images], None, 0.0)
    now = store.now()
    with store.transaction(db):
        db.executemany(
            "INSERT INTO corrections (kind, a, value, created_at) VALUES ('name', ?, ?, ?)",
            [(t, value, now) for t in targets],
        )
    _regroup(db, _quiet, ask=False)
    (row,) = _rows_of(db, targets[:1])
    return {
        "entity_id": row["entity_id"],
        "name": value,
        "added_images": len(images) + len(frames),
        "extended": tuple(rules.words(value)) in known,
        "appearances": row["appearances"],
        "videos": row["videos"],
    }


def _shown_at(db: sqlite3.Connection, video_id: str, t: float) -> str:
    """The appearance of the person shown biggest at t (faces before bodies), taken from a still
    of that moment when nobody was tracked then."""
    here = db.execute(
        "SELECT appearance_id, kind, box FROM appearances WHERE video_id = ? AND start - ? <= ? AND end + ? >= ?",
        (video_id, HALF_STEP, t, HALF_STEP, t),
    ).fetchall()
    pool = [row for row in here if row["kind"] == "face"] or here
    if pool:
        return max(pool, key=lambda row: json.loads(row["box"])[3])["appearance_id"]
    video = db.execute("SELECT local_path FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if video is None:
        raise ValueError(f"No video {video_id}")
    still = store.data_dir("work") / "entities" / f"{video_id}-{t:.2f}.png"
    for _, frame in crops.frames(Path(video["local_path"]), [t]):
        crops.save(frame, still, max(frame.shape[:2]))
    if not still.exists():
        raise ValueError(f"{video_id} has no frame at {t:.2f} s")
    try:
        (appearance_id,) = _photographed(db, [(f"{video_id} at {t:.2f} s", still)], video_id, t)
    finally:
        still.unlink()
    return appearance_id


def _photographed(
    db: sqlite3.Connection, pictures: list[tuple[str, Path]], video_id: str | None, t: float
) -> list[str]:
    """Stores the biggest clear face in each (label, image file) as an appearance; returns the ids.
    A photo's appearance has no video; a still's belongs to its video at time t."""
    missing = [label for label, path in pictures if not path.is_file()]
    if missing:
        raise ValueError(f"No image at {', '.join(missing)}")
    images = [cv2.imread(str(path)) for _, path in pictures]
    unreadable = [label for (label, _), image in zip(pictures, images, strict=True) if image is None]
    if unreadable:
        raise ValueError(f"Not an image: {', '.join(unreadable)}")
    faces = []
    for (label, _), found in zip(pictures, vision.detect([path for _, path in pictures]), strict=True):
        clear = [d for d in found if d.kind == "face" and d.extra.get("landmarks")]
        if not clear:
            raise ValueError(f"No clear face in {label}")
        faces.append(max(clear, key=lambda d: d.box[3]))
    folder = store.data_dir("crops") / (video_id or "references")
    work = store.data_dir("work") / "entities"
    ids = []
    for (_, path), image, face in zip(pictures, images, faces, strict=True):
        key = path.read_bytes() if video_id is None else f"{video_id}:face:{t:.2f}:{tuple(face.box)}".encode()
        ids.append(_hashed_id(key))
        _face_crop(tuple(face.box), folder / f"{ids[-1]}.jpg", image)
        _aligned(face.extra["landmarks"], work / f"{ids[-1]}.png", image)
    aligned = [work / f"{i}.png" for i in ids]
    try:
        vectors = _face_vectors(aligned, _quiet)
    finally:
        for path in aligned:
            path.unlink(missing_ok=True)
    with store.transaction(db):
        db.executemany(
            "INSERT OR REPLACE INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path,"
            " embedding, quality) VALUES (?, ?, 'face', ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    i,
                    video_id,
                    t,
                    t,
                    t,
                    json.dumps(list(face.box)),
                    str(folder / f"{i}.jpg"),
                    store.to_blob(vector),
                    _quality(face.extra),
                )
                for i, face, vector in zip(ids, faces, vectors, strict=True)
            ],
        )
    return ids
