# Qwen3-TTS Megakernel × Pipecat

RTX 5090 megakernel-backed **Qwen3-TTS talker decode** with streaming audio for FastAPI and Pipecat.

## Requirements

- NVIDIA RTX 5090 (Blackwell, `sm_120`)
- CUDA 12.8+, Python 3.10 or 3.11
- ~40 GB disk for model weights

## Build

```bash
python3 -m venv venv && source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
python scripts/verify_5090_env.py
```

See [docs/architecture.md](docs/architecture.md) for the system architecture and
voice-agent execution flow.
See [docs/implementation_mental_model.md](docs/implementation_mental_model.md)
for the decode/runtime boundary used by this implementation.
See [docs/folder_structure.md](docs/folder_structure.md) for the repo ownership
map.
See [docs/options.md](docs/options.md) for all CLI flags, HTTP fields, and
environment variables.
See [docs/vast_ai_5090_runbook.md](docs/vast_ai_5090_runbook.md) for Vast.ai
RTX 5090 setup.

## Run

```bash
export MEGAKERNEL_TTS_MODE=real

# CLI demo → WAV
python demo/demo.py --mode real --text "Hello" --output output/demo.wav

# HTTP server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Benchmark
python benchmark/benchmark.py --mode real --runs 5
```

Query server:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/tts/wav \
  -H "content-type: application/json" \
  -d '{"text":"Hello","mode":"real"}' -o output/out.wav
```

## Pipecat

```python
from pipecat_service.tts_service import MegakernelTTSService

tts = MegakernelTTSService(mode="real")
```

`MegakernelTTSService` auto-detects Pipecat when it is installed. Set
`MEGAKERNEL_TTS_USE_PIPECAT=0` only when you need the lightweight local shim.

Optional HF reference: `MEGAKERNEL_TTS_MODE=hf`

## Full voice-agent demo

Run the microphone-style Pipecat demo (STT → LLM → megakernel TTS → audio):

```bash
export DEEPGRAM_API_KEY=your-key
export OPENAI_API_KEY=your-key
python demo/live_voice_agent.py --port 8765
```


## Kernel changes

- `LDG_VOCAB_SIZE=3072` (codec vocab)
- Untied `codec_head` LM weights
- Embedding sentinel (`token_id < 0`) for precomputed inputs

Details: [docs/model_comparison.md](docs/model_comparison.md)

## Benchmarking

The benchmark keeps one initialized TTS service alive and reports cold startup
separately from warm request latency:

```bash
python benchmark/benchmark.py --mode real --runs 5 --chunk-frames 10
```

Reported metrics:

- `cold_init_ms`: model load, CUDA extension build, weight upload, and warmup.
- `warm_ttfc_ms`: initialized request start to first PCM chunk.
- `rtf`: generation wall time divided by emitted audio duration.
- `talker_steps_per_s`: approximate talker decode steps per second.
- `chunks`: number of streamed audio chunks.

Previous cold-path measurement:

```text
run=5 mode=real ttfc_ms=5376.00 rtf=0.615 chunks=18 audio_s=13.60
avg_ttfc_ms=6316.11
avg_rtf=0.684
target_ttfc_ms=<60 assignment / <90 reference
target_rtf=<0.1 assignment / <0.3 reference
```
