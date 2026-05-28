"""Cheap local stand-in for the RTX 5090 Qwen3-TTS megakernel path.

The fake decoder intentionally does not download a HuggingFace model or touch
CUDA. It keeps the same streaming surface as the real decoder so server,
benchmark, and Pipecat wiring can be exercised before renting GPU time.
"""

from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Iterator

import numpy as np
import torch


class FakeMegakernelDecoder:
    """Deterministic fake decoder for local integration tests."""

    def __init__(
        self,
        model_name: str = "fake-qwen3-tts",
        *,
        sample_rate: int = 24000,
        chunk_ms: int = 80,
        realtime: bool = False,
        vocab_size: int = 3072,
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.realtime = realtime
        self.vocab_size = vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token-like compatibility shim for the early assignment smoke tests."""
        input_ids = input_ids.to(self.device)
        last_token = input_ids[:, -1]
        return (last_token + 1) % self.vocab_size

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM16 audio bytes, matching the real decoder's byte stream."""
        async for chunk, _sample_rate in self.stream_audio(text):
            yield chunk

    async def stream_audio(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        """Yield fake speech-like PCM16 chunks."""
        chunk_duration = self.chunk_ms / 1000.0
        for chunk in self.iter_audio_chunks(text):
            if self.realtime:
                await asyncio.sleep(chunk_duration)
            yield chunk, self.sample_rate
            await asyncio.sleep(0)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return a full fake waveform as float32 audio."""
        chunks = [
            np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
            for chunk in self.iter_audio_chunks(text)
        ]
        if not chunks:
            return np.array([], dtype=np.float32), self.sample_rate
        return np.concatenate(chunks), self.sample_rate

    def iter_audio_chunks(self, text: str) -> Iterator[bytes]:
        """Generate deterministic PCM16 chunks for a text prompt."""
        duration_s = self._duration_for_text(text)
        total_samples = max(1, int(duration_s * self.sample_rate))
        chunk_samples = max(1, int(self.sample_rate * self.chunk_ms / 1000))
        frequency = self._frequency_for_text(text)

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            yield self._tone_chunk(start, end, total_samples, frequency)

    def _duration_for_text(self, text: str) -> float:
        word_count = max(len(text.split()), 1)
        return min(max(word_count / 2.5, 0.40), 8.0)

    def _frequency_for_text(self, text: str) -> float:
        checksum = sum(ord(char) for char in text)
        return 180.0 + float(checksum % 220)

    def _tone_chunk(
        self,
        start: int,
        end: int,
        total_samples: int,
        frequency: float,
    ) -> bytes:
        samples = np.arange(start, end, dtype=np.float32)
        t = samples / float(self.sample_rate)
        wave = 0.20 * np.sin(2.0 * math.pi * frequency * t)
        wave += 0.04 * np.sin(2.0 * math.pi * frequency * 2.0 * t)

        fade_samples = max(1, int(0.010 * self.sample_rate))
        if start < fade_samples:
            fade = samples / float(fade_samples)
            wave *= np.clip(fade, 0.0, 1.0)
        if end > total_samples - fade_samples:
            remaining = (total_samples - samples) / float(fade_samples)
            wave *= np.clip(remaining, 0.0, 1.0)

        pcm16 = np.clip(wave * 32767.0, -32768, 32767).astype("<i2")
        return pcm16.tobytes()
