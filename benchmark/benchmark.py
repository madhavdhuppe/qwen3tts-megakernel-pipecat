"""Benchmark streaming TTS: TTFC, RTF, chunk count (RTX 5090 real mode)."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat_service.tts_service import MegakernelTTSService


async def run_once(args) -> dict:
    service = MegakernelTTSService(
        mode=args.mode or "real",
        model_path=args.model,
        chunk_frames=args.chunk_frames,
    )

    if args.warmup_engine and hasattr(service.decoder, "initialize"):
        service.decoder.initialize()

    started = time.perf_counter()
    first_chunk_at = None
    total_bytes = 0
    sample_rate = 24000
    chunks = 0
    chunk_gaps_ms: list[float] = []
    prev_chunk_at = None

    async for audio, sr in service.decoder.stream_audio(args.text):
        now = time.perf_counter()
        if first_chunk_at is None:
            first_chunk_at = now
        if prev_chunk_at is not None:
            chunk_gaps_ms.append((now - prev_chunk_at) * 1000.0)
        prev_chunk_at = now
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
        "avg_inter_chunk_ms": statistics.mean(chunk_gaps_ms) if chunk_gaps_ms else 0.0,
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
            f"audio_s={result['audio_s']:.2f} "
            f"avg_inter_chunk_ms={result['avg_inter_chunk_ms']:.2f}"
        )

    ttfc = [r["ttfc_ms"] for r in results]
    rtf = [r["rtf"] for r in results]
    summary = {
        "runs": results,
        "avg_ttfc_ms": statistics.mean(ttfc),
        "avg_rtf": statistics.mean(rtf),
        "targets": {"ttfc_ms": 90, "rtf": 0.3},
    }
    print("-" * 56)
    print(f"avg_ttfc_ms={summary['avg_ttfc_ms']:.2f} (target < 90 ms)")
    print(f"avg_rtf={summary['avg_rtf']:.3f} (target < 0.3)")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel streaming benchmark")
    parser.add_argument("--mode", default="real", help="real or hf")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument(
        "--text",
        default=(
            "The quick brown fox jumps over the lazy dog. "
            "This is a streaming text to speech benchmark."
        ),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--warmup-engine", action="store_true", help="Initialize engine before timed runs")
    parser.add_argument("--json", default=None, help="Write summary JSON path")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
