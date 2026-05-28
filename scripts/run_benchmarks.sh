#!/usr/bin/env bash
# Run all performance benchmarks on RTX 5090 (after venv + model download).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

export MEGAKERNEL_TTS_MODE=real

echo "== GPU check =="
python3 scripts/verify_5090_env.py

echo "== Talker decode tok/s =="
python3 benchmark/benchmark_decode.py --runs 5 --decode-steps 100 | tee output/benchmark_decode.txt

echo "== Streaming TTS TTFC/RTF =="
python3 benchmark/benchmark.py --mode real --runs 5 --warmup-engine \
  --json output/benchmark_tts.json | tee output/benchmark_tts.txt

echo "Done. Copy numbers into docs/PERFORMANCE.md"
