"""Speech-to-text worker with word times (Whisper large-v3-turbo on MLX), run under tlv.guard.ModelProcess.

Job:    {"audio": path of a 16 kHz mono WAV}
Result: {"language": str, "segments": [{"start", "end", "text", "words": [{"w", "s", "e"}]}]}

Each 30 s window is decoded without the previous window's text as its prompt, so one misheard
stretch cannot echo through the rest of a video, and a long silence followed by a likely
hallucination is skipped. TLV_WHISPER_MODEL picks another MLX checkpoint.
"""

from __future__ import annotations

import os
from typing import Any

from tlv.models.protocol import Handler, serve

MODEL = os.environ.get("TLV_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
# Seconds of silence before a suspected hallucination that are skipped rather than transcribed.
HALLUCINATION_SILENCE_S = 2.0


def load() -> Handler:
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    # transcribe() decodes in float16 and loads through this holder, so it reuses this copy.
    ModelHolder.get_model(MODEL, mx.float16)

    def handle(job: dict[str, Any]) -> dict[str, Any]:
        result = mlx_whisper.transcribe(
            job["audio"],
            path_or_hf_repo=MODEL,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=HALLUCINATION_SILENCE_S,
        )
        mx.clear_cache()
        segments = [
            {
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "text": segment["text"].strip(),
                "words": [
                    {"w": word["word"].strip(), "s": round(word["start"], 3), "e": round(word["end"], 3)}
                    for word in segment.get("words", [])
                ],
            }
            for segment in result["segments"]
        ]
        return {"language": result["language"], "segments": segments}

    return handle


if __name__ == "__main__":
    serve(load)
