# Folder Structure

This layout follows the implementation boundary in `implementation_mental_model.md`.

```text
.
|-- benchmark/
|   `-- benchmark.py              # Warm-service TTFC, RTF, chunk, and step metrics
|-- demo/
|   |-- demo.py                   # Text-to-WAV smoke demo
|   |-- index.html                # Browser client for the WebSocket voice agent
|   `-- live_voice_agent.py       # Pipecat STT -> LLM -> TTS -> audio pipeline
|-- docs/
|   |-- architecture.md           # End-to-end diagrams and voice-agent topology
|   |-- folder_structure.md       # Ownership map for this repo
|   |-- implementation_mental_model.md
|   |-- model_comparison.md       # Source kernel vs Qwen3-TTS talker differences
|   |-- options.md                # CLI, HTTP, and environment configuration
|   `-- vast_ai_5090_runbook.md   # RTX 5090 rental/server setup notes
|-- megakernel_adapter/
|   |-- csrc/                     # Adapted CUDA kernel and torch bindings
|   |-- build_tts.py              # sm_120a JIT build flags for the TTS kernel
|   |-- hf_reference.py           # Optional stock/reference decoder path
|   |-- megakernel_decoder.py     # PCM streaming facade around the TTS engine
|   |-- model_tts.py              # Weight loading, talker decoder, code predictor
|   `-- tts_engine.py             # Text -> codec frames -> vocoder -> audio
|-- pipecat_service/
|   `-- tts_service.py            # Pipecat-compatible TTSService wrapper
|-- server/
|   |-- app.py                    # FastAPI app factory surface
|   |-- routes.py                 # Streaming and WAV endpoints
|   `-- schemas.py                # HTTP request/response schemas
`-- third_party/
    `-- qwen_megakernel/          # Vendored upstream source for comparison
```

## Ownership Rules

| Area | Owns | Does not own |
| --- | --- | --- |
| `megakernel_adapter/csrc` | CUDA decode kernels, torch op bindings, compile-time constants. | Pipecat frames, HTTP request parsing, model download policy. |
| `megakernel_adapter/model_tts.py` | TTS weight mapping, RoPE tables, KV caches, talker/code predictor wrappers. | Server lifetimes or user-facing APIs. |
| `megakernel_adapter/tts_engine.py` | Qwen3-TTS sequencing and streaming audio generation. | Network transport or Pipecat-specific classes. |
| `pipecat_service` | Converting decoder byte streams into Pipecat TTS frames. | CUDA/model internals. |
| `server` | HTTP streaming and WAV response surfaces. | End-to-end voice agent orchestration. |
| `demo` | Human-runnable demos and recording taps. | Core decode logic. |
| `benchmark` | Measurement harness and reporting. | Product/runtime behavior. |

## Change Placement Guide

- Kernel shape, vocab, sentinel, or launch flag changes go in `megakernel_adapter/csrc`
  and `megakernel_adapter/build_tts.py`.
- New Qwen3-TTS weight names or architecture constants go in
  `megakernel_adapter/model_tts.py`.
- Changes to text formatting, codec frame generation, vocoder loading, or chunking go
  in `megakernel_adapter/tts_engine.py`.
- Pipecat compatibility changes go in `pipecat_service/tts_service.py`.
- HTTP parameters and request caching go in `server/schemas.py` and
  `server/routes.py`.
- Benchmark methodology changes go in `benchmark/benchmark.py` and should be
  reflected in the README performance section.
