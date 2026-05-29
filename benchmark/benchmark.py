"""Benchmark streaming TTS on RTX 5090 (TTFC, RTF)."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat_service.tts_service import MegakernelTTSService


def _decoder_position(service: MegakernelTTSService) -> int | None:
    engine = getattr(getattr(service, "decoder", None), "engine", None)
    if engine is None or not getattr(engine, "_initialized", False):
        return None
    metrics = engine.get_metrics()
    return int(metrics.get("position", 0))


async def run_once(args, service: MegakernelTTSService) -> dict[str, Any]:
    started = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    sample_rate = 24000
    chunks = 0
    position_before = _decoder_position(service)

    async for audio, sr in service.decoder.stream_audio(args.text):
        now = time.perf_counter()
        if first_chunk_at is None:
            first_chunk_at = now
        chunks += 1
        total_bytes += len(audio)
        sample_rate = sr

    ended = time.perf_counter()
    position_after = _decoder_position(service)
    talker_steps = position_after
    if position_before is not None and position_after is not None and position_after > position_before:
        talker_steps = position_after - position_before

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
        "talker_steps": talker_steps,
        "talker_steps_per_s": (talker_steps / elapsed_s) if talker_steps and elapsed_s else 0.0,
    }


async def main_async(args):
    service = MegakernelTTSService(
        mode=args.mode or "real",
        model_path=args.model,
        chunk_frames=args.chunk_frames,
        device=args.device,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    init_started = time.perf_counter()
    service.decoder.initialize()
    init_ms = (time.perf_counter() - init_started) * 1000.0
    print(f"cold_init_ms={init_ms:.2f}")

    results = []
    for i in range(args.runs):
        result = await run_once(args, service)
        results.append(result)
        print(
            f"run={i + 1} mode={result['mode']} "
            f"warm_ttfc_ms={result['ttfc_ms']:.2f} "
            f"rtf={result['rtf']:.3f} "
            f"chunks={result['chunks']} "
            f"audio_s={result['audio_s']:.2f} "
            f"talker_steps={result['talker_steps']} "
            f"talker_steps_per_s={result['talker_steps_per_s']:.1f}"
        )

    ttfc = [r["ttfc_ms"] for r in results]
    rtf = [r["rtf"] for r in results]
    steps_per_s = [r["talker_steps_per_s"] for r in results if r["talker_steps_per_s"]]
    print("-" * 56)
    print(f"cold_init_ms={init_ms:.2f}")
    print(f"avg_warm_ttfc_ms={statistics.mean(ttfc):.2f}")
    print(f"avg_rtf={statistics.mean(rtf):.3f}")
    if steps_per_s:
        print(f"avg_talker_steps_per_s={statistics.mean(steps_per_s):.1f}")
    print("target_ttfc_ms=<60 assignment / <90 reference")
    print("target_rtf=<0.1 assignment / <0.3 reference")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel benchmark")
    parser.add_argument(
        "--mode",
        choices=["real", "megakernel", "cuda", "gpu", "hf", "reference", "hf_reference"],
        default="real",
        help="Decoder mode",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--text",
        default=(
            "The quick brown fox jumps over the lazy dog. "
            "This is a streaming text to speech benchmark."
        ),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--no-sample", action="store_true", help="Disable stochastic TTS sampling")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.chunk_frames < 1:
        parser.error("--chunk-frames must be >= 1")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
