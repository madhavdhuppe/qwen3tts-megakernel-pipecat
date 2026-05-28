# Performance Results

Fill this file after running benchmarks on an **RTX 5090** (CUDA 12.8+, `sm_120`).

## Environment

| Field | Value |
|---|---|
| GPU | RTX 5090 |
| CUDA | |
| PyTorch | |
| Driver | |
| Model | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| Date | |

## Talker decode (megakernel)

Command:

```bash
python benchmark/benchmark_decode.py --runs 5 --decode-steps 100
```

| Metric | Target (reference) | Measured |
|---|---:|---:|
| Talker decode tok/s | ~1000 (Qwen3-0.6B blog baseline on same kernel family) | |

## Streaming TTS (audio out)

Command:

```bash
python benchmark/benchmark.py --mode real --runs 5 --json output/benchmark_tts.json
```

| Metric | Target (deliverable) | Measured |
|---|---:|---:|
| TTFC (ms) | < 90 | |
| RTF | < 0.3 | |
| Audio duration (s) | | |
| Chunks streamed | | |

## End-to-end voice agent (optional)

Command: `python demo/pipecat_voice_agent.py` (see README). Record round-trip with stopwatch or Pipecat metrics.

| Metric | Measured |
|---|---:|
| Speech-in → first audio chunk (ms) | |
| Speech-in → playback complete (ms) | |

## Bottlenecks (honest notes)

- Prefill + first vocoder chunk dominate TTFC.
- Code predictor runs per codec frame (15 extra LM heads per frame).
- Vocoder decode adds latency per chunk.

## Demo recording

- File: `docs/demo.mp4` (or link)
- Shows: speak → STT → LLM reply → megakernel TTS → speaker output, streaming (no full-utterance buffer).
