# Qwen3-TTS Megakernel Pipecat Adapter

This repo keeps the assignment split into two modes:

- `fake` mode is the default. It runs locally with deterministic PCM audio and no model downloads, CUDA build, or GPU spend.
- `real` mode uses the adapted Qwen3-TTS megakernel path copied from the reference implementation and is intended for the rented RTX 5090 box.

## Local Fake Run

```bash
python demo/test_adapter.py
python demo/test_fake_decoder.py
python demo/test_fake_audio.py
python demo/demo.py --mode fake --output output/fake_qwen3tts.wav
python benchmark/benchmark.py --mode fake --runs 3
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Streaming smoke test:

```bash
curl -N -X POST http://127.0.0.1:8000/tts/stream \
  -H "content-type: application/json" \
  -d '{"text":"hello from fake local mode","mode":"fake"}' \
  --output output/fake_qwen3tts.pcm
```

## RTX 5090 Real Run

Use this only on a CUDA 12.8+ Blackwell machine with PyTorch CUDA support:

```bash
export MEGAKERNEL_TTS_MODE=real
python demo/demo.py --mode real --output output/qwen3tts_megakernel.wav
python benchmark/benchmark.py --mode real --runs 5
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The real path adapts Qwen3-TTS by compiling the megakernel with `LDG_VOCAB_SIZE=3072`, using the TTS talker weights, and adding the embedding-sentinel flow required for precomputed text+codec embeddings.

## Pipecat Service

```python
from pipecat_service.tts_service import MegakernelTTSService

tts = MegakernelTTSService(mode="fake")
# On 5090:
# tts = MegakernelTTSService(mode="real")
```

`run_tts()` streams start/audio/stop frames and does not buffer the full utterance before sending audio. Local tests use lightweight frame classes to avoid Pipecat import-time downloads; set `MEGAKERNEL_TTS_USE_PIPECAT=1` when running inside a real Pipecat pipeline.
