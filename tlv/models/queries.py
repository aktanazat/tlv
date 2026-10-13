"""Search-query worker: SigLIP 2's text side and the text embedder in one process, run under
tlv.guard.ModelProcess. A search needs vectors from both, and one child pays for the process
start, the imports and the memory check once instead of twice.

Job:    {"image_queries": [str, ...]} or {"sentences": [str, ...]}
Result: {"vectors": base64 of float16 rows, unit length, "shape": [n, d]}, made by the
        tlv.models.siglip and tlv.models.textembed handlers, so the rows are theirs.
"""

from __future__ import annotations

from typing import Any

from tlv.models import siglip, textembed
from tlv.models.protocol import Handler, serve


def load() -> Handler:
    image_queries, sentences = siglip.load(), textembed.load()

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        if "image_queries" in job:
            return image_queries({"texts": job["image_queries"]})
        return sentences({"texts": job["sentences"]})

    return handle


if __name__ == "__main__":
    serve(load)
