"""Benchmark HF-reference or real RTX 5090 Qwen3-TTS streaming."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat_service.tts_service import MegakernelTTSService


async def run_once(args) -> dict:
    service = MegakernelTTSService(
        mode=args.mode,
        model_path=args.model,
        chunk_ms=args.chunk_ms,
        chunk_frames=args.chunk_frames,
        realtime=args.realtime,
    )

    started = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    sample_rate = 24000
    chunks = 0

    async for audio, sr in service.decoder.stream_audio(args.text):
        now = time.perf_counter()
        if first_chunk_at is None:
            first_chunk_at = now
        chunks += 1
        total_bytes += len(audio)
        sample_rate = sr

    ended = time.perf_counter()
    elapsed_s = ended - started
    audio_s = (total_bytes / 2) / sample_rate if sample_rate else 0.0
    return {
        "mode": service.mode,
        "ttfc_ms": ((first_chunk_at or ended) - started) * 1000.0,
        "elapsed_s": elapsed_s,
        "audio_s": audio_s,
        "rtf": elapsed_s / audio_s if audio_s else float("inf"),
        "chunks": chunks,
        "bytes": total_bytes,
    }


async def main_async(args):
    results = []
    for i in range(args.runs):
        result = await run_once(args)
        results.append(result)
        print(
            f"run={i + 1} mode={result['mode']} "
            f"ttfc_ms={result['ttfc_ms']:.2f} "
            f"rtf={result['rtf']:.3f} "
            f"chunks={result['chunks']} "
            f"audio_s={result['audio_s']:.2f}"
        )

    ttfc = [r["ttfc_ms"] for r in results]
    rtf = [r["rtf"] for r in results]
    print("-" * 56)
    print(f"avg_ttfc_ms={statistics.mean(ttfc):.2f}")
    print(f"avg_rtf={statistics.mean(rtf):.3f}")
    print("target_ttfc_ms=<60 assignment / <90 reference")
    print("target_rtf=<0.1 assignment / <0.3 reference")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel benchmark")
    parser.add_argument("--mode", default=None, help="hf or real")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument(
        "--text",
        default=(
            "The quick brown fox jumps over the lazy dog. "
            "This is a streaming text to speech benchmark."
        ),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
