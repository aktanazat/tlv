"""The child side of tlv.guard.ModelProcess: JSON jobs in on stdin, one JSON line out per job.

Libraries print to stdout freely, so the protocol keeps a private copy of stdout and points
file descriptor 1 at stderr before anything is imported.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def serve(load: Callable[[], Handler]) -> None:
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    limit = int(os.environ.get("TLV_MEMORY_LIMIT_BYTES", "0"))
    if limit:
        import mlx.core as mx

        mx.set_memory_limit(limit)
        mx.set_cache_limit(limit // 8)
    handle = load()
    out.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        job = json.loads(line)
        try:
            result = handle(job)
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        out.write(json.dumps(result) + "\n")
