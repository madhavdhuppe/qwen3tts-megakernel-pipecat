"""Round-trip validation for the megakernel TTS adapter.

This script mimics the benchmark's end-to-end chain without requiring a live
GPU-backed model load. It uses a deterministic synthetic decoder by default so
that the pipeline shape, WAV packaging, and streaming chunks can be validated.
If a real decoder is available, the same harness can be pointed at it.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np

from pipecat_service.tts_service import MegakernelTTSService
from server.routes import _wav_from_pcm16


@dataclass
class SyntheticDecoder:
    sample_rate: int = 24000
    seconds: float = 1.0

    async def stream_audio(self, text: str) -> AsyncGenerator[tuple[bytes, int], None]:
        duration = max(self.seconds, 0.25)
        samples = int(duration * self.sample_rate)
        t = np.arange(samples, dtype=np.float32) / self.sample_rate
        base = np.sin(2 * math.pi * 220.0 * t)
        envelope = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        waveform = (base * envelope).astype(np.float32)
        clipped = np.clip(waveform, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2").tobytes()
        # Emit two chunks to exercise streaming behavior.
        chunk_size = max(samples // 2, 1)
        first = pcm[: chunk_size * 2]
        second = pcm[chunk_size * 2 :]
        yield first, self.sample_rate
        if second:
            yield second, self.sample_rate


async def _collect_audio(service: MegakernelTTSService, text: str) -> tuple[bytes, int]:
    pcm = bytearray()
    sample_rate = 24000
    async for audio, sr in service.decoder.stream_audio(text):
        pcm.extend(audio)
        sample_rate = sr
    return bytes(pcm), sample_rate


async def run_round_trip(text: str, output_path: Path, use_synthetic: bool = True) -> dict:
    if use_synthetic:
        decoder = SyntheticDecoder()
        service = MegakernelTTSService(decoder=decoder)
    else:
        service = MegakernelTTSService(mode="real")

    transcript = f"STT: {text}"
    llm_response = f"LLM: {transcript}"

    audio_bytes, sample_rate = await _collect_audio(service, llm_response)
    wav_bytes = _wav_from_pcm16(audio_bytes, sample_rate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)

    with wave.open(str(output_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

    return {
        "transcript": transcript,
        "llm_response": llm_response,
        "sample_rate": sample_rate,
        "wav_frames": frames,
        "wav_rate": frame_rate,
        "output_path": str(output_path),
        "audio_bytes": len(audio_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the megakernel TTS round trip.")
    parser.add_argument(
        "--text",
        default="A quick validation of the streaming TTS chain.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="output/roundtrip_validation.wav",
        help="Destination WAV path.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use the real megakernel decoder when available instead of the synthetic fallback.",
    )
    args = parser.parse_args()

    asyncio.run(run_round_trip(args.text, Path(args.output), use_synthetic=not args.real))

    print(f"Round-trip validation written to {args.output}")


if __name__ == "__main__":
    main()
