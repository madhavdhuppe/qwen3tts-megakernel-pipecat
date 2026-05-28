"""Optional HuggingFace reference path for parity checks.

This path is not the assignment's optimized deliverable. It exists to compare
server and Pipecat behavior against a stock Qwen3-TTS style implementation
before switching to the RTX 5090 megakernel.
"""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np
import torch


class HFReferenceDecoder:
    """Lazy stock-model TTS wrapper with the same surface as Decoder."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        *,
        device: str | None = None,
        chunk_ms: int = 80,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_ms = chunk_ms
        self.sample_rate = 24000
        self._loaded = False

    def initialize(self) -> None:
        """Load the reference model lazily.

        The public Qwen3-TTS APIs have changed across Transformers releases, so
        this method keeps failures explicit and isolated from the fake path.
        """
        if self._loaded:
            return

        try:
            from transformers import AutoProcessor, Qwen3TTSForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "HF reference mode requires a Transformers build with "
                "Qwen3TTSForConditionalGeneration. Use fake mode locally or "
                "real mode on the 5090 box."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        async for chunk, _sample_rate in self.stream_audio(text):
            yield chunk

    async def stream_audio(self, text: str) -> AsyncIterator[tuple[bytes, int]]:
        audio, sample_rate = self.synthesize(text)
        chunk_samples = max(1, int(sample_rate * self.chunk_ms / 1000))
        pcm = _float32_to_pcm16(audio)
        bytes_per_sample = 2
        chunk_bytes = chunk_samples * bytes_per_sample
        for start in range(0, len(pcm), chunk_bytes):
            yield pcm[start:start + chunk_bytes], sample_rate

    @torch.no_grad()
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.initialize()
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)

        if isinstance(output, dict):
            audio = output.get("audio") or output.get("waveform")
        else:
            audio = output

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio, self.sample_rate


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()
