"""Benchmark talker megakernel decode throughput (tokens/sec) on RTX 5090."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megakernel_adapter.model_tts import CODEC_BOS, TTSDecoder, load_tts_weights


def _bench_talker_steps(
    talker: TTSDecoder,
    *,
    decode_steps: int,
    warmup: int,
    runs: int,
) -> list[float]:
    """Return tok/s for each measured run (talker step() only, post-prefill)."""
    embed = talker.embed_weight[CODEC_BOS].clone()
    results: list[float] = []

    def _one_run() -> float:
        talker.reset()
        for _ in range(8):
            talker.step_with_embed(embed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            talker.step_with_embed(embed)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return decode_steps / elapsed if elapsed > 0 else 0.0

    for _ in range(warmup):
        _one_run()

    for _ in range(runs):
        results.append(_one_run())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Talker megakernel decode tok/s")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required. Run on RTX 5090.")

    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("Loading weights and JIT-compiling megakernel (first run is slow)...")
    weights = load_tts_weights(args.model, verbose=True)
    talker = TTSDecoder(weights=weights, verbose=False)

    rates = _bench_talker_steps(
        talker,
        decode_steps=args.decode_steps,
        warmup=args.warmup,
        runs=args.runs,
    )

    for i, rate in enumerate(rates, start=1):
        print(f"run={i} talker_decode_tok_s={rate:.1f}")

    print("-" * 56)
    print(f"avg_talker_decode_tok_s={statistics.mean(rates):.1f}")
    print(f"median_talker_decode_tok_s={statistics.median(rates):.1f}")
    print("note=Measures talker.step_with_embed() only (28-layer megakernel + 3072 vocab LM head).")


if __name__ == "__main__":
    main()
