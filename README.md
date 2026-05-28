# Qwen3-TTS Megakernel Pipecat Adapter

This repository targets RTX 5090 megakernel-backed Qwen3-TTS streaming, with an
optional HF reference mode for comparison.

## RTX 5090 Run

Use this on a CUDA 12.8+ Blackwell machine with PyTorch CUDA support:

```bash
export MEGAKERNEL_TTS_MODE=real
python demo/demo.py --mode real --output output/qwen3tts_megakernel.wav
python benchmark/benchmark.py --mode real --runs 5
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Streaming smoke test:

```bash
curl -N -X POST http://127.0.0.1:8000/tts/stream \
  -H "content-type: application/json" \
  -d '{"text":"hello from megakernel mode","mode":"real"}' \
  --output output/qwen3tts_stream.pcm
```

The real path adapts Qwen3-TTS by compiling the megakernel with
`LDG_VOCAB_SIZE=3072`, using the TTS talker weights, and adding the
embedding-sentinel flow required for precomputed text+codec embeddings.

## HF Reference Mode

```bash
export MEGAKERNEL_TTS_MODE=hf
python demo/demo.py --mode hf --output output/qwen3tts_hf.wav
python benchmark/benchmark.py --mode hf --runs 3
```

## Pipecat Service

```python
from pipecat_service.tts_service import MegakernelTTSService

tts = MegakernelTTSService(mode="real")
```

`run_tts()` streams start/audio/stop frames and does not buffer the full
utterance before sending audio. Set `MEGAKERNEL_TTS_USE_PIPECAT=1` when running
inside a real Pipecat pipeline.
