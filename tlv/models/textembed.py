"""Text embedding worker (Qwen3-Embedding on mlx-embeddings), run under tlv.guard.ModelProcess.

Job:    {"texts": [str, ...]}
Result: {"vectors": base64 of float16 rows, unit length, "shape": [n, d]}

The caller adds the query instruction to queries; documents go in as they are. The tokenizer
appends the end-of-text token the model pools on. TLV_TEXT_EMBED_MODEL picks another checkpoint.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from tlv.models.protocol import Handler, serve

MODEL = os.environ.get("TLV_TEXT_EMBED_MODEL", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
MAX_TOKENS = 512


def load() -> Handler:
    import mlx.core as mx
    import numpy as np
    from mlx_embeddings.utils import load as load_model

    model, tokenizer = load_model(MODEL)

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer(job["texts"], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="np")
        output = model(mx.array(tokens["input_ids"]), attention_mask=mx.array(tokens["attention_mask"]))
        # The model normalizes in bfloat16; normalizing again in float32 makes the rows unit length.
        features = output.text_embeds.astype(mx.float32)
        features = features / mx.linalg.norm(features, axis=-1, keepdims=True)
        rows = np.array(features).astype(np.float16)
        mx.clear_cache()
        return {"vectors": base64.b64encode(rows.tobytes()).decode(), "shape": list(rows.shape)}

    return handle


if __name__ == "__main__":
    serve(load)
