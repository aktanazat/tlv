"""Removing a video: no row and no file of it remain, another video and its people stay as they
were, a file another index uses stays, a follow-up never returns it, and the same file added again
indexes anew. Each stage is replaced by a stand-in that writes the rows and files its real stage
writes, since the real stages need the models."""

import json
import subprocess

import numpy as np
import pytest

from tlv import agent, core, entities, store
from tlv.index import frames, pipeline, shots, sounds, spans, speech, vision

# Face vectors: one per person, so faces of one person join and different people stay apart.
FACES = {name: np.eye(512, dtype=np.float32)[i] for i, name in enumerate(("alice", "quinn", "rae"))}
CAST = {"one": ("alice", "quinn"), "two": ("alice", "rae")}  # who each video shows, by title
# The tables with a video_id column.
VIDEO_TABLES = ("videos", "stages", "shots", "frames", "speech", "ocr", "detections", "sounds", "spans")
VIDEO_TABLES += ("appearances", "observations")


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _shots(db, video, progress):
    db.execute("INSERT INTO shots (video_id, idx, start, end) VALUES (?, 0, 0, 1)", (video["video_id"],))


def _frames(db, video, progress):
    path = _write(store.data_dir("frames") / video["video_id"] / "000000.jpg")
    db.execute(
        "INSERT INTO frames (video_id, t, path, shot_idx, embedding) VALUES (?, 0, ?, 0, ?)",
        (video["video_id"], path, store.to_blob(np.ones(4))),
    )


def _speech(db, video, progress):
    _write(store.data_dir("work") / video["video_id"] / "audio.wav")
    words = json.dumps([{"w": "hello", "s": 0.1, "e": 0.4}])
    db.execute("INSERT INTO speech VALUES (?, 0.1, 0.4, 'hello', ?)", (video["video_id"], words))


def _vision(db, video, progress):
    _write(store.data_dir("work") / video["video_id"] / "h264.mp4")
    db.execute("INSERT INTO ocr VALUES (?, 0, 'SCORE', 0, 0, 0.1, 0.1, 0.9)", (video["video_id"],))
    db.execute(
        "INSERT INTO detections (video_id, t, kind, x, y, w, h, conf) VALUES (?, 0, 'face', 0.4, 0.1, 0.1, 0.2, 0.9)",
        (video["video_id"],),
    )


def _sounds(db, video, progress):
    db.execute("INSERT INTO sounds VALUES (?, 0, 1, 'Speech', 0.8)", (video["video_id"],))


def _spans(db, video, progress, recaption=False):
    video_id = video["video_id"]
    db.execute(
        "INSERT INTO spans (span_id, video_id, start, end, caption, details, embedding)"
        " VALUES (?, ?, 0, 1, ?, '{}', ?)",
        (f"span_{video_id}", video_id, f"a scene of {video['title']}", store.to_blob(np.ones(4))),
    )
    db.execute(
        "INSERT INTO observations (obs_id, video_id, start, end, kind, text, confidence, evidence, created_at)"
        " VALUES (?, ?, 0, 1, 'caption', 'a scene', 1, '{}', ?)",
        (f"caption:span_{video_id}", video_id, store.now()),
    )


def _people(db, video_id, progress):
    """Stores a face appearance and its crop per person the video shows, then reclusters."""
    (title,) = db.execute("SELECT title FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    for person in CAST[title]:
        appearance_id = f"app_{title}_{person}"
        db.execute(
            "INSERT INTO appearances (appearance_id, video_id, kind, start, end, best_t, box, crop_path, embedding,"
            " quality) VALUES (?, ?, 'face', 0, 1, 0, '[0.4, 0.1, 0.1, 0.2]', ?, ?, 0.5)",
            (
                appearance_id,
                video_id,
                _write(store.data_dir("crops") / video_id / f"{appearance_id}.jpg"),
                store.to_blob(FACES[person]),
            ),
        )
    entities.recluster(db, progress)


@pytest.fixture
def stand_ins(monkeypatch):
    for module, run in ((shots, _shots), (frames, _frames), (speech, _speech), (vision, _vision)):
        monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(sounds, "run", _sounds)
    monkeypatch.setattr(spans, "run", _spans)
    monkeypatch.setattr(entities, "index_video", _people)


@pytest.fixture
def files(tmp_path):
    """Two short videos of different content, named by their titles."""
    made = {}
    for seconds, title in enumerate(CAST, 1):
        made[title] = tmp_path / f"{title}.mp4"
        pattern = f"testsrc=duration={seconds}:size=160x120:rate=10"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", pattern, str(made[title])], check=True)
    return made


@pytest.fixture
def ids(db, stand_ins, files):
    """Both videos added and indexed into the "tlv" index, a clip cut from each, and the people
    named: Alice through her appearance in the video removed, Quinn (seen only there) and Rae
    (seen only in the other)."""
    added = {title: pipeline.add(str(path))["video_id"] for title, path in files.items()}
    for video_id in added.values():
        core.video_cut("tlv", video_id, 0, 0.5)
    for appearance_id, name in (("app_one_alice", "Alice"), ("app_one_quinn", "Quinn"), ("app_two_rae", "Rae")):
        entities.name(db, appearance_id, name)
    return added


def _rows_naming(db, needles):
    """(table, needle) for each row of every table holding one of `needles` in a value."""
    found = []
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
        for row in db.execute(f"SELECT * FROM {table}"):
            text = " ".join(map(str, row))
            found += [(table, needle) for needle in needles if needle in text]
    return found


def _files_of(video_id):
    """Every file under $TLV_HOME whose path names the video, but the lock files."""
    home = store.home()
    return {
        path: path.read_bytes()
        for path in home.rglob("*")
        if path.is_file() and video_id[-6:] in str(path.relative_to(home)) and path.parent.name != "locks"
    }


def test_no_row_and_no_file_of_a_removed_video_remain(db, ids):
    gone = ids["one"]
    evidence = {"summary": "a scene", "replays": [{"video_id": gone, "start": 0, "end": 1}]}
    db.execute(
        "INSERT INTO observations (obs_id, video_id, start, end, kind, text, confidence, evidence, created_at)"
        " VALUES ('obs_found', ?, 0, 1, 'verified', 'a scene', 1, ?, ?)",
        (ids["two"], json.dumps(evidence), store.now()),
    )
    assert _files_of(gone), "the setup made no files to remove"
    core.video_remove("tlv", gone)
    assert (_rows_naming(db, [gone, "app_one_alice", "app_one_quinn"]), sorted(_files_of(gone))) == ([], [])


def test_another_videos_rows_files_and_corrections_are_untouched(db, ids):
    kept = ids["two"]

    def rows():
        return {
            table: sorted(map(tuple, db.execute(f"SELECT * FROM {table} WHERE video_id = ?", (kept,))), key=repr)
            for table in VIDEO_TABLES
        }

    def corrections():
        return set(map(tuple, db.execute("SELECT * FROM corrections WHERE a IN ('app_two_alice', 'app_two_rae')")))

    before = (rows(), _files_of(kept), corrections(), entities.present(db, kept, 0, 2))
    core.video_remove("tlv", ids["one"])
    after = (rows(), _files_of(kept), corrections() & before[2], entities.present(db, kept, 0, 2))
    assert after == before


def test_people_seen_elsewhere_keep_their_names_and_people_seen_only_there_go(db, ids):
    core.video_remove("tlv", ids["one"])
    listed = sorted((e["name"], e["appearances"], e["video_ids"]) for e in entities.entities(db))
    assert listed == [("Alice", 1, [ids["two"]]), ("Rae", 1, [ids["two"]])]


def test_files_another_index_uses_stay(db, ids, files):
    shared = ids["one"]
    core.video_add("other", str(files["one"]))
    before = _files_of(shared)
    core.video_remove("tlv", shared)
    status = core.video_status("other", shared)
    assert (sorted(before.keys() - _files_of(shared).keys()), status["status"]) == ([], "ready")


def test_a_follow_up_never_returns_a_removed_video(ids, monkeypatch):
    clip = {"id": "obs_1", "video_id": ids["one"], "start": 0, "end": 1, "replays": []}
    found = {"claim": "a scene", "confirmed": [clip], "rejected": [], "counts": {"confirmed": 1, "candidates": 1}}
    monkeypatch.setattr(agent, "find", lambda *args, **kwargs: found)
    core.video_find("tlv", "a scene")
    core.video_remove("tlv", ids["one"])
    with pytest.raises(LookupError, match="no find to follow up"):
        core.video_more("tlv", "without the replays")


def test_the_same_file_added_again_is_indexed_anew(db, ids, files):
    removed = ids["one"]
    counts = pipeline.status(removed)["counts"]
    core.video_remove("tlv", removed)
    again = pipeline.add(str(files["one"]))
    status = pipeline.status(removed)
    assert (again["already_indexed"], status["status"], status["counts"]) == (False, "ready", counts)
