"""SigLIP 2 image-text embedding worker (mlx-embeddings), run under tlv.guard.ModelProcess.

Job:    {"images": [image path, ...]} or {"texts": [str, ...]}
Result: {"vectors": base64 of float16 rows, unit length, "shape": [n, d]}

Images and texts land in one space, so a text query ranks frames. Texts are lowercased and
padded to 64 tokens, as SigLIP 2 was trained. The weights load lazily, so a text-only run never
reads the vision tower. TLV_SIGLIP_MODEL picks another MLX checkpoint.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from tlv.models.protocol import Handler, serve

MODEL = os.environ.get("TLV_SIGLIP_MODEL", "mlx-community/siglip2-so400m-patch16-384")
TEXT_TOKENS = 64


def load() -> Handler:
    import mlx.core as mx
    import numpy as np
    from mlx_embeddings.utils import load as load_model
    from PIL import Image

    model, processor = load_model(MODEL, lazy=True)

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        if "images" in job:
            pictures = [Image.open(path).convert("RGB") for path in job["images"]]
            pixels = processor.image_processor(pictures, return_tensors="np")["pixel_values"]
            features = model.get_image_features(mx.array(pixels))
        else:
            texts = [text.lower() for text in job["texts"]]
            tokens = processor.tokenizer(
                texts, padding="max_length", max_length=TEXT_TOKENS, truncation=True, return_tensors="np"
            )
            features = model.get_text_features(mx.array(tokens["input_ids"]))
        features = features.astype(mx.float32)
        features = features / mx.linalg.norm(features, axis=-1, keepdims=True)
        rows = np.array(features).astype(np.float16)
        mx.clear_cache()
        return {"vectors": base64.b64encode(rows.tobytes()).decode(), "shape": list(rows.shape)}

    return handle


if __name__ == "__main__":
    serve(load)
