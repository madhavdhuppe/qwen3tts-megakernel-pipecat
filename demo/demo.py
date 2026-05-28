"""Demo entry point for RTX 5090 megakernel TTS streaming."""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat_service.tts_service import MegakernelTTSService


async def run_demo(args):
    service = MegakernelTTSService(
        mode=args.mode,
        model_path=args.model,
    )

    pcm = bytearray()
    sample_rate = 24000
    first_chunk = True
    async for audio, sample_rate in service.decoder.stream_audio(args.text):
        if first_chunk:
            print("first_chunk_bytes", len(audio))
            first_chunk = False
        pcm.extend(audio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(pcm))

    print(f"mode={service.mode}")
    print(f"samples={len(pcm) // 2}")
    print(f"saved={output_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel demo")
    parser.add_argument("--mode", default="real", help="hf or real")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--text", default="Hello from the Qwen3-TTS megakernel.")
    parser.add_argument("--output", default="output/qwen3tts_megakernel_demo.wav")
    args = parser.parse_args()

    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
