"""Real RTX 5090 Qwen3-TTS megakernel decoder bridge."""

from __future__ import annotations

from typing import AsyncIterator, Optional

import numpy as np

from .tts_engine import MegakernelTTSEngine, TTSConfig


class MegakernelDecoder:
    """Streaming TTS decoder backed by the adapted CUDA megakernel.

    This path is intended for the rented RTX 5090 environment. Local
    development should use ``MEGAKERNEL_TTS_MODE=fake`` so imports and server
    tests do not compile CUDA or download model weights.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        *,
        vocoder_path: Optional[str] = None,
        device: str = "cuda",
        chunk_frames: int = 10,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 2048,
    ):
        self.config = TTSConfig(
            model_path=model_name,
            vocoder_path=vocoder_path or model_name,
            chunk_frames=chunk_frames,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        self.engine = MegakernelTTSEngine(config=self.config, device=device)

    def initialize(self) -> None:
        self.engine.initialize()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        async for chunk, _sample_rate in self.stream_audio(text):
            yield chunk

    async def stream_audio(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        async for audio, sample_rate in self.engine.synthesize_streaming(
            text,
            chunk_frames=self.config.chunk_frames,
        ):
            yield _float32_to_pcm16(audio), sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self.engine.synthesize(text)

    def step(self, *_args, **_kwargs):
        raise NotImplementedError(
            "MegakernelDecoder is a TTS streaming bridge. Use stream_audio() "
            "or synthesize(); token-level step() is only available on the fake "
            "local decoder for scaffold smoke tests."
        )


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()
