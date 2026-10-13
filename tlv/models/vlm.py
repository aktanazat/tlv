"""Vision-language model worker (Qwen3-VL on MLX), run under tlv.guard.ModelProcess.

Job:    {"images": [jpeg path, ...], "prompt": str, "max_tokens": int}
Result: {"text": str, "prompt_tokens": int, "generation_tokens": int, "seconds": float}

Batch:  {"batch": [{"images": [...], "prompt": str, "max_tokens": int, "stop": regex (optional)}, ...]}
Result: {"results": [{"text", "prompt_tokens", "generation_tokens", "stopped": bool}, ...], "seconds": float}

Images are shown in the order given; callers scale them down first, since every image costs
prompt tokens. TLV_VLM_MODEL picks another MLX checkpoint.

A batch answers its questions together. Each prompt is read alone, as a single job reads it, but
the answers being written advance one token each per step, so writing them all costs little more
than writing the longest, and the next question's images load while the model reads the current
one. An answer ends at its token limit, at the end of its text, or as soon as its text matches its
stop pattern ("stopped"), for a caller that needs only the first words of some answers. A question
is taken up only while the prompts and token limits of the answers being written stay within
TOKENS_AT_ONCE (one is always taken up alone), which bounds the memory their caches hold.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from tlv.models.protocol import Handler, serve

MODEL = os.environ.get("TLV_VLM_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit")
# Two verdict checks of about 2200 tokens each held 5.5 GB at their peak, the most measured under the
# guard's 6 GB cap; eight of them went over it.
TOKENS_AT_ONCE = 4400


def load() -> Handler:
    from concurrent.futures import Future, ThreadPoolExecutor

    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm import load as load_model
    from mlx_vlm.generate import BatchGenerator
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

    model, processor = load_model(MODEL)
    tokenizer = processor.tokenizer
    special_tokens = should_add_special_tokens(model.config.model_type, processor)
    image_token = getattr(model.config, "image_token_index", None)
    loader = ThreadPoolExecutor(1)  # loads a batch's next question while the model works on one

    def prompt(question: dict[str, Any]) -> str:
        images = question.get("images", [])
        return apply_chat_template(processor, model.config, question["prompt"], num_images=len(images))

    def one(job: dict[str, Any]) -> dict[str, Any]:
        images: list[str] = job.get("images", [])
        began = time.monotonic()
        result = generate(
            model,
            processor,
            prompt(job),
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

    def inputs(question: dict[str, Any]) -> dict[str, Any]:
        """The question's tokens and images as the model takes them (CPU work only)."""
        return prepare_inputs(
            processor,
            images=question.get("images") or None,
            prompts=prompt(question),
            image_token_index=image_token,
            add_special_tokens=special_tokens,
        )

    def batch(questions: list[dict[str, Any]]) -> dict[str, Any]:
        began = time.monotonic()
        stops = [re.compile(q["stop"]) if q.get("stop") else None for q in questions]
        limits = [int(q.get("max_tokens", 512)) for q in questions]
        tokens: list[list[int]] = [[] for _ in questions]
        prompt_tokens = [0] * len(questions)
        stopped = [False] * len(questions)
        loading: dict[int, Future[dict[str, Any]]] = {}
        writing: dict[int, int] = {}  # generator uid -> question index, for answers not yet ended

        def load_ahead(i: int) -> None:
            if i < len(questions):
                loading[i] = loader.submit(inputs, questions[i])

        def fits(i: int, prompt_length: int) -> bool:
            held = sum(prompt_tokens[j] + limits[j] for j in writing.values())
            return not writing or held + prompt_length + limits[i] <= TOKENS_AT_ONCE

        load_ahead(0)
        load_ahead(1)
        added = 0
        gen = BatchGenerator(
            model.language_model,
            processor,
            completion_batch_size=len(questions),  # TOKENS_AT_ONCE bounds the answers written at once
            prefill_batch_size=1,
            compute_logprobs=False,
        )
        try:
            while added < len(questions) or gen.has_work:
                ready = loading[added].result() if added < len(questions) and not gen.has_pending_prompts else None
                if ready is not None and fits(added, ready["input_ids"].shape[1]):
                    del loading[added]
                    load_ahead(added + 2)
                    ids = ready["input_ids"]
                    rest = {k: v for k, v in ready.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
                    embedded = model.get_input_embeddings(
                        ids, ready.get("pixel_values"), mask=ready.get("attention_mask"), **rest
                    )
                    kwargs = {**rest, **{k: v for k, v in embedded.to_dict().items() if v is not None}}
                    (uid,) = gen.insert([ids[0].tolist()], [limits[added]], prompt_kwargs=[kwargs])
                    writing[uid] = added
                    prompt_tokens[added] = ids.shape[1]
                    added += 1
                _, responses = gen.next()
                for response in responses:
                    i = writing[response.uid]
                    if response.finish_reason != "stop":
                        tokens[i].append(response.token)
                    if response.finish_reason is None and stops[i] and stops[i].search(tokenizer.decode(tokens[i])):
                        gen.remove(response.uid)
                        stopped[i] = True
                    if response.finish_reason is not None or stopped[i]:
                        del writing[response.uid]
        finally:
            gen.close()
            mx.clear_cache()
        return {
            "results": [
                {
                    "text": tokenizer.decode(row),
                    "prompt_tokens": prompt_tokens[i],
                    "generation_tokens": len(row),
                    "stopped": stopped[i],
                }
                for i, row in enumerate(tokens)
            ],
            "seconds": round(time.monotonic() - began, 2),
        }

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        return batch(job["batch"]) if "batch" in job else one(job)

    return handle


if __name__ == "__main__":
    serve(load)
