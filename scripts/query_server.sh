#!/usr/bin/env bash
# Quick smoke tests against a running uvicorn server.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
OUT_DIR="${2:-output}"
mkdir -p "$OUT_DIR"

echo "== health =="
curl -sS "$BASE_URL/health" | tee "$OUT_DIR/health.json"
echo

echo "== tts/wav =="
curl -sS -X POST "$BASE_URL/tts/wav" \
  -H "content-type: application/json" \
  -d '{"text":"Hello from the megakernel server.","mode":"real"}' \
  --output "$OUT_DIR/server_smoke.wav"
echo "saved $OUT_DIR/server_smoke.wav"

echo "== tts/stream (first 64KiB) =="
curl -sS -N -X POST "$BASE_URL/tts/stream" \
  -H "content-type: application/json" \
  -d '{"text":"Streaming test.","mode":"real"}' \
  | head -c 65536 > "$OUT_DIR/server_smoke.pcm"
echo "saved $OUT_DIR/server_smoke.pcm (truncated)"
