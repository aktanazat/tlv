"""Vision-language model worker (Qwen3-VL on MLX), run under tlv.guard.ModelProcess.

Job:    {"images": [jpeg path, ...], "prompt": str, "max_tokens": int}
Result: {"text": str, "prompt_tokens": int, "generation_tokens": int, "seconds": float}

Images are shown in the order given; callers scale them down first, since every image costs
prompt tokens. TLV_VLM_MODEL picks another MLX checkpoint.
"""

from __future__ import annotations

import os
import time
from typing import Any

from tlv.models.protocol import Handler, serve

MODEL = os.environ.get("TLV_VLM_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit")


def load() -> Handler:
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm import load as load_model
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load_model(MODEL)

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        images: list[str] = job.get("images", [])
        prompt = apply_chat_template(processor, model.config, job["prompt"], num_images=len(images))
        began = time.monotonic()
        result = generate(
            model,
            processor,
            prompt,
            image=images or None,
            max_tokens=int(job.get("max_tokens", 512)),
            temperature=0.0,
            verbose=False,
        )
        mx.clear_cache()
        return {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "seconds": round(time.monotonic() - began, 2),
        }

    return handle


if __name__ == "__main__":
    serve(load)
