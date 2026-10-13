"""Embeddings for search, each returned as float32 rows of unit length.

SigLIP 2 puts frames (`images`) and the text that looks for them (`image_queries`) in one
space. A small text embedder (`sentences`) covers captions, speech and on-screen text. Each
call loads its model once under the guard and feeds it in batches. A search wants query
vectors from both models, and `queries` makes them in one child that holds both.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tlv import guard

IMAGE_BATCH = 16
TEXT_BATCH = 32
PROGRESS_BATCHES = 10
QUERY_INSTRUCTION = (
    "Given a search query about a video, retrieve the moments whose description, speech or on-screen text match it"
)


def images(paths: Sequence[str | Path], progress: guard.Progress | None = None) -> np.ndarray:
    """SigLIP 2 vectors of image files, one row per path."""
    [vectors] = _embed("tlv.models.siglip", [("images", [str(path) for path in paths])], IMAGE_BATCH, progress)
    return vectors


def image_queries(texts: Sequence[str]) -> np.ndarray:
    """SigLIP 2 vectors of texts, comparable with `images` rows by dot product."""
    [vectors] = _embed("tlv.models.siglip", [("texts", list(texts))], TEXT_BATCH)
    return vectors


def sentences(texts: Sequence[str], query: bool = False, progress: guard.Progress | None = None) -> np.ndarray:
    """Text vectors for captions, speech and on-screen text; `query=True` prefixes the
    embedder's retrieval instruction, which queries need and documents must not get."""
    items = _instructed(texts) if query else list(texts)
    # Texts of similar length share a batch, so little of each batch is padding.
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    [rows] = _embed("tlv.models.textembed", [("texts", [items[i] for i in order])], TEXT_BATCH, progress)
    ordered = np.empty_like(rows)
    ordered[order] = rows
    return ordered


def queries(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """`image_queries(texts)` and `sentences(texts, query=True)` from one guarded child that holds
    both models, so a search starts one model process instead of two."""
    looks, means = _embed(
        "tlv.models.queries", [("image_queries", list(texts)), ("sentences", _instructed(texts))], TEXT_BATCH
    )
    return looks, means


def _instructed(texts: Sequence[str]) -> list[str]:
    return [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" for text in texts]


def _embed(
    module: str, jobs: list[tuple[str, list[Any]]], batch: int, progress: guard.Progress | None = None
) -> list[np.ndarray]:
    """Rows for each (job key, items) in turn, all from one guarded child of `module`."""
    if not any(items for _, items in jobs):
        return [np.zeros((0, 0), dtype=np.float32) for _ in jobs]
    with guard.ModelProcess(module, progress=progress) as model:
        return [_rows(model, key, items, batch, progress) for key, items in jobs]


def _rows(
    model: guard.ModelProcess, key: str, items: list[Any], batch: int, progress: guard.Progress | None
) -> np.ndarray:
    parts = []
    for n, start in enumerate(range(0, len(items), batch), 1):
        result = model.call({key: items[start : start + batch]})
        vectors = np.frombuffer(base64.b64decode(result["vectors"]), dtype=np.float16)
        parts.append(vectors.reshape(result["shape"]))
        done = min(start + batch, len(items))
        if progress and len(items) > batch and (n % PROGRESS_BATCHES == 0 or done == len(items)):
            progress(f"embedded {done} of {len(items)}")
    return np.concatenate(parts).astype(np.float32)
