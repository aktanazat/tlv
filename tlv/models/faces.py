"""Face embedding worker (ArcFace on onnxruntime), run under tlv.guard.ModelProcess.

Job:    {"images": [path of an aligned 112x112 face, ...]}
Result: {"vectors": base64 of float16 rows, unit length, "shape": [n, 512]}

Faces arrive already aligned to the ArcFace five-point template (tlv.entities aligns them from
Vision landmarks). The checkpoint is InsightFace's buffalo_l recognition model, w600k_r50: a
ResNet-50 trained with ArcFace loss on WebFace600K, pinned to a Hugging Face mirror revision whose
file matches the other public mirrors byte for byte. InsightFace releases these weights for
non-commercial research use.
"""

from __future__ import annotations

import base64
from typing import Any

from tlv.models.protocol import Handler, serve

REPO = "public-data/insightface"
FILE = "models/buffalo_l/w600k_r50.onnx"
REVISION = "33c1063c49c785b7652d3fd529f86fa4f149392b"


def load() -> Handler:
    import numpy as np
    import onnxruntime
    from huggingface_hub import hf_hub_download
    from PIL import Image

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3  # the checkpoint declares batch 1 on its output; batches still work
    session = onnxruntime.InferenceSession(
        hf_hub_download(REPO, FILE, revision=REVISION),
        options,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    feed = session.get_inputs()[0].name

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        faces = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) for path in job["images"]])
        vectors = session.run(None, {feed: ((faces - 127.5) / 127.5).transpose(0, 3, 1, 2)})[0]
        rows = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)
        return {"vectors": base64.b64encode(rows.tobytes()).decode(), "shape": list(rows.shape)}

    return handle


if __name__ == "__main__":
    serve(load)
