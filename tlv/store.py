"""The local store: one SQLite database plus media folders under $TLV_HOME (~/.local/share/tlv).

Every fact the engine knows lives here, tied to a video and a time range in seconds from the
start of that video. Each table block below names the module that writes it; other modules
only read it.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np

Sub = Literal["media", "clips", "frames", "crops", "locks", "models", "work"]

SCHEMA = """
-- tlv.index.pipeline: one row per added video.
CREATE TABLE IF NOT EXISTS videos (
  video_id   TEXT PRIMARY KEY,
  title      TEXT NOT NULL,
  source     TEXT NOT NULL,              -- the URL or file path it was added from
  local_path TEXT NOT NULL,              -- the copy every cut, frame and model reads
  duration   REAL NOT NULL,
  width      INTEGER NOT NULL,
  height     INTEGER NOT NULL,
  fps        REAL NOT NULL,
  added_at   REAL NOT NULL,              -- unix seconds
  status     TEXT NOT NULL CHECK (status IN ('queued', 'indexing', 'ready', 'failed')),
  error      TEXT
);

-- tlv.index.pipeline: a stage is redone only when its version changes.
CREATE TABLE IF NOT EXISTS stages (
  video_id TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  stage    TEXT NOT NULL,
  version  INTEGER NOT NULL,
  seconds  REAL NOT NULL,                -- wall time the stage took
  done_at  REAL NOT NULL,
  PRIMARY KEY (video_id, stage)
);

-- tlv.index.shots: camera cuts.
CREATE TABLE IF NOT EXISTS shots (
  video_id TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  idx      INTEGER NOT NULL,
  start    REAL NOT NULL,
  end      REAL NOT NULL,
  PRIMARY KEY (video_id, idx)
);

-- tlv.index.frames: sampled frames on disk and their image embeddings (float16, unit length).
CREATE TABLE IF NOT EXISTS frames (
  video_id  TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  t         REAL NOT NULL,
  path      TEXT NOT NULL,
  shot_idx  INTEGER,
  embedding BLOB,
  PRIMARY KEY (video_id, t)
);

-- tlv.index.speech: transcript segments with word times.
CREATE TABLE IF NOT EXISTS speech (
  video_id TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  start    REAL NOT NULL,
  end      REAL NOT NULL,
  text     TEXT NOT NULL,
  words    TEXT NOT NULL                 -- JSON [{"w": str, "s": float, "e": float}]
);
CREATE INDEX IF NOT EXISTS speech_by_time ON speech (video_id, start);

-- tlv.index.vision: on-screen text per sampled frame; box is normalized x, y, w, h (top-left origin).
CREATE TABLE IF NOT EXISTS ocr (
  video_id TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  t        REAL NOT NULL,
  text     TEXT NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  conf     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ocr_by_time ON ocr (video_id, t);

-- tlv.index.vision: faces and bodies found per sampled frame, same box convention as ocr.
CREATE TABLE IF NOT EXISTS detections (
  det_id   INTEGER PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  t        REAL NOT NULL,
  kind     TEXT NOT NULL CHECK (kind IN ('face', 'body')),
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  conf     REAL NOT NULL,
  extra    TEXT                          -- JSON, e.g. face landmarks or roll/yaw
);
CREATE INDEX IF NOT EXISTS detections_by_time ON detections (video_id, t);

-- tlv.index.sounds: audio events.
CREATE TABLE IF NOT EXISTS sounds (
  video_id TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  start    REAL NOT NULL,
  end      REAL NOT NULL,
  label    TEXT NOT NULL,
  conf     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sounds_by_time ON sounds (video_id, start);

-- tlv.index.spans: the retrieval unit, a few seconds inside one shot or a run of short shots.
-- caption and details come from the vision-language model; embedding is a text embedding of
-- caption + speech + on-screen text, for search.
CREATE TABLE IF NOT EXISTS spans (
  span_id   TEXT PRIMARY KEY,
  video_id  TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  start     REAL NOT NULL,
  end       REAL NOT NULL,
  caption   TEXT,
  details   TEXT,                        -- JSON from the captioner
  replay    INTEGER,                     -- 1 replay, 0 live, NULL unknown
  embedding BLOB
);
CREATE INDEX IF NOT EXISTS spans_by_time ON spans (video_id, start);

-- tlv.entities: one appearance is one person seen continuously inside one shot, as a face or a
-- body. A face sitting on a body through the same frames is that body's person: `body` links them.
-- A reference image taught with add_reference has no video (video_id NULL, times 0).
CREATE TABLE IF NOT EXISTS appearances (
  appearance_id TEXT PRIMARY KEY,
  video_id      TEXT REFERENCES videos ON DELETE CASCADE,
  kind          TEXT NOT NULL CHECK (kind IN ('face', 'body')),
  start         REAL NOT NULL,
  end           REAL NOT NULL,
  best_t        REAL NOT NULL,           -- the clearest frame
  box           TEXT NOT NULL,           -- JSON [x, y, w, h] at best_t
  crop_path     TEXT NOT NULL,
  embedding     BLOB NOT NULL,           -- ArcFace for a face; a body's jersey colors (entities.crops.colors)
  clues         TEXT NOT NULL DEFAULT '[]', -- JSON [{"kind", "value", "t"}]; jersey|spoken_name|ocr_name|caption_name
  body          TEXT REFERENCES appearances ON DELETE SET NULL CHECK (body IS NULL OR kind = 'face'),
  quality       REAL CHECK ((quality IS NULL) = (kind = 'body')) -- a face's Vision capture quality where embedded
);
CREATE INDEX IF NOT EXISTS appearances_by_video ON appearances (video_id, start);

-- tlv.entities: one person across the collection.
CREATE TABLE IF NOT EXISTS entities (
  entity_id            TEXT PRIMARY KEY,
  name                 TEXT,
  name_evidence        TEXT NOT NULL DEFAULT '[]', -- JSON: the correction, or the agreeing clues, that gave the name
  canonical_appearance TEXT NOT NULL,
  embedding            BLOB NOT NULL,
  updated_at           REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS entity_members (
  appearance_id TEXT PRIMARY KEY REFERENCES appearances ON DELETE CASCADE,
  entity_id     TEXT NOT NULL REFERENCES entities ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS members_by_entity ON entity_members (entity_id);

-- tlv.entities: a person's corrections, keyed by appearances so they survive re-clustering.
CREATE TABLE IF NOT EXISTS corrections (
  id         INTEGER PRIMARY KEY,
  kind       TEXT NOT NULL CHECK (kind IN ('name', 'same', 'different')),
  a          TEXT NOT NULL,              -- appearance id
  b          TEXT,                       -- appearance id, for same/different
  value      TEXT,                       -- the name, for name
  created_at REAL NOT NULL
);

-- tlv.entities: the vision-language model's answer to "same person?" for two appearances (a < b),
-- kept so re-clustering never asks twice.
CREATE TABLE IF NOT EXISTS verdicts (
  a          TEXT NOT NULL REFERENCES appearances ON DELETE CASCADE,
  b          TEXT NOT NULL REFERENCES appearances ON DELETE CASCADE,
  same       INTEGER,                    -- 1 same, 0 different, NULL unsure
  created_at REAL NOT NULL,
  PRIMARY KEY (a, b)
);

-- tlv.agent and tlv.index.spans: a fact tied to a source video and a time range, with paths back
-- to the evidence. kind 'caption' comes from indexing; 'verified' from a checked find.
CREATE TABLE IF NOT EXISTS observations (
  obs_id     TEXT PRIMARY KEY,
  video_id   TEXT NOT NULL REFERENCES videos ON DELETE CASCADE,
  start      REAL NOT NULL,
  end        REAL NOT NULL,
  entity_id  TEXT,
  kind       TEXT NOT NULL CHECK (kind IN ('caption', 'verified')),
  text       TEXT NOT NULL,
  confidence REAL NOT NULL,
  -- evidence: JSON {"frames": [path], "transcript": str, "ocr": [str], "sounds": [str]};
  -- verified rows add "summary", "who", "steps", "replays"
  evidence   TEXT NOT NULL,
  query      TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS observations_by_video ON observations (video_id, start);
CREATE INDEX IF NOT EXISTS observations_by_entity ON observations (entity_id);
"""


def home() -> Path:
    return Path(os.environ.get("TLV_HOME") or Path.home() / ".local/share/tlv")


def data_dir(sub: Sub) -> Path:
    """A folder under $TLV_HOME, created on first use."""
    path = home() / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect(index: str | None = None) -> sqlite3.Connection:
    """Opens the index's database ($TLV_HOME/<index>.db; the index defaults to $TLV_INDEX, else "tlv"),
    creating the schema on first use. Safe across processes (WAL)."""
    path = home() / f"{index or os.environ.get('TLV_INDEX') or 'tlv'}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 60000")
    db.executescript(SCHEMA)
    return db


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolled back on any error."""
    db.execute("BEGIN IMMEDIATE")
    try:
        yield db
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float16).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def stack(blobs: list[bytes]) -> np.ndarray:
    """Rows of float32 from equal-length float16 blobs; an empty list gives shape (0, 0)."""
    if not blobs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(len(blobs), -1).astype(np.float32)


@contextmanager
def lock(name: str, waiting: Callable[[], None] | None = None) -> Iterator[None]:
    """Holds $TLV_HOME/locks/<name> across processes. The kernel drops it if the holder dies,
    so a crashed run never leaves it stuck. `waiting` is called once if another process has it."""
    path = data_dir("locks") / name
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if waiting:
                waiting()
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def now() -> float:
    return time.time()
