# Qwen3-TTS Megakernel × Pipecat

RTX 5090 megakernel-backed **Qwen3-TTS talker decode** with streaming audio into a Pipecat voice pipeline.

Based on [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) and adapted for `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

## Architecture

```text
Text prompt
  → tokenizer + text projection + codec/tag prefill (PyTorch)
  → talker autoregressive loop (CUDA megakernel, 28 layers, bf16)
       each frame: code predictor (5-layer megakernel) + embedding sum
  → speech tokenizer / vocoder (Qwen3-TTS)
  → PCM chunks → FastAPI or Pipecat TTSAudioRawFrame (streaming, not buffered)
```

| Component | Implementation |
|---|---|
| Talker decoder (0.6B) | Adapted megakernel (`megakernel_adapter/csrc/kernel.cu`) |
| Code predictor | Same megakernel, 5 layers, per-frame |
| Vocoder | `qwen-tts` speech tokenizer decode |
| Server | FastAPI (`server/app.py`) |
| Voice agent | Pipecat (`demo/pipecat_voice_agent.py`) |

See [docs/model_comparison.md](docs/model_comparison.md) for Qwen3-0.6B vs TTS talker deltas.

## Kernel modifications (vs upstream qwen_megakernel)

| Change | Why |
|---|---|
| `LDG_VOCAB_SIZE=3072` | TTS codec vocab (was 151936 text tokens) |
| `LDG_LM_NUM_BLOCKS=16` | Smaller LM head grid for 3072 rows |
| `rope_theta=1_000_000` in host RoPE tables | Matches TTS talker config |
| Untied `talker.codec_head` weights | LM head ≠ embedding table |
| **Embedding sentinel** `token_id < 0` | Skip embed lookup; use `hidden_buffer` for precomputed text+codec sums |

Sentinel in kernel:

```cuda
const __nv_bfloat16 *embed_row =
    (input_token_id >= 0) ? embed_weight + input_token_id * HIDDEN_SIZE
                          : hidden_buffer;
```

Build flags: [megakernel_adapter/build_tts.py](megakernel_adapter/build_tts.py) (`-arch=sm_120a`).

## Requirements

- **GPU:** NVIDIA RTX 5090 (Blackwell, `sm_120`)
- **CUDA:** 12.8+
- **Python:** 3.10 or 3.11
- **Disk:** ~40 GB (model + JIT cache)
- **RAM:** 32 GB+

## Build (single RTX 5090)

```bash
git clone --recurse-submodules <your-repo-url>
cd qwen3tts-megakernel-pipecat

python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel

# Install CUDA 12.8 PyTorch for Blackwell if the image does not include it.
pip install -r requirements.txt

python scripts/verify_5090_env.py
```

First real run JIT-compiles the extension (~1 min) and may download `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

Detailed rental steps: [docs/vast_ai_5090_runbook.md](docs/vast_ai_5090_runbook.md).

## Run

### Demo WAV

```bash
export MEGAKERNEL_TTS_MODE=real
python demo/demo.py --mode real --text "Hello from the megakernel." \
  --output output/demo.wav
```

### HTTP server

```bash
export MEGAKERNEL_TTS_MODE=real
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Query (another terminal):

```bash
bash scripts/query_server.sh
# or
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/tts/wav \
  -H "content-type: application/json" \
  -d '{"text":"Hello","mode":"real"}' -o output/out.wav
```

Streaming PCM:

```bash
curl -N -X POST http://127.0.0.1:8000/tts/stream \
  -H "content-type: application/json" \
  -d '{"text":"Streaming test","mode":"real"}' -o output/stream.pcm
```

### Pipecat voice agent (deliverable demo)

End-to-end: **mic → STT → LLM → megakernel TTS → speaker** (audio frames streamed as generated).

```bash
export MEGAKERNEL_TTS_MODE=real
export MEGAKERNEL_TTS_USE_PIPECAT=1
export OPENAI_API_KEY=...
export DEEPGRAM_API_KEY=...   # or: export PIPECAT_STT=openai

python demo/pipecat_voice_agent.py
```

Record screen + microphone for the submission demo. Confirm you hear audio before the full LLM reply finishes printing (streaming TTS).

Programmatic TTS only:

```python
from pipecat_service.tts_service import MegakernelTTSService

tts = MegakernelTTSService(mode="real")
async for frame in tts.run_tts("Hello there"):
    ...
```

## Performance benchmarks

On the 5090 box:

```bash
bash scripts/run_benchmarks.sh
```

Or individually:

```bash
python benchmark/benchmark_decode.py --runs 5    # talker decode tok/s
python benchmark/benchmark.py --mode real --runs 5 --json output/benchmark_tts.json
```

Copy results into [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Metric | Deliverable target | Command |
|---|---|---|
| Talker decode tok/s | (report; ~1k tok/s is reference blog number for Qwen3-0.6B kernel) | `benchmark_decode.py` |
| TTFC | < 90 ms | `benchmark.py` |
| RTF | < 0.3 | `benchmark.py` |
| E2E voice latency | report honestly | Pipecat demo + stopwatch / metrics |

**Note:** Full TTS TTFC includes prefill, first codec frame, code predictor, and first vocoder chunk—not talker tok/s alone.

## Repo layout

| Path | Purpose |
|---|---|
| `megakernel_adapter/` | TTS weights, engine, CUDA extension |
| `server/` | FastAPI streaming API |
| `pipecat_service/` | Pipecat `TTSService` wrapper |
| `demo/` | CLI demo + Pipecat voice agent |
| `benchmark/` | Decode + streaming benchmarks |
| `third_party/qwen_megakernel/` | Upstream reference (submodule) |

## Known limitations

- Requires RTX 5090-class GPU (`sm_120`); will not run on older architectures without re-tuning.
- Vocoder load depends on `qwen-tts` + Transformers compatibility; see `MegakernelTTSEngine._load_vocoder`.
- Pipecat E2E demo needs cloud STT/LLM API keys unless you swap providers.
- EOS detection is heuristic; very long inputs may need `max_new_tokens` tuning.

## Deliverables checklist

- [x] Working repo + build instructions (this README + runbook)
- [x] Architecture + kernel modification docs
- [x] Pipecat demo entrypoint (`demo/pipecat_voice_agent.py`)
- [ ] **Performance numbers** — run on 5090, fill [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- [ ] **Demo recording** — attach `docs/demo.mp4` or link in PERFORMANCE.md
