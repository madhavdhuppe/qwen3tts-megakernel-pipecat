# Options Reference

This page lists the supported configuration surfaces for the Qwen3-TTS
megakernel adapter.

## Environment Variables

| Variable | Default | Used by | Description |
| --- | --- | --- | --- |
| `MEGAKERNEL_TTS_MODE` | `real` | All TTS service entrypoints | Decoder mode. Supported values: `real`, `megakernel`, `cuda`, `gpu`, `hf`, `reference`, `hf_reference`. |
| `MEGAKERNEL_TTS_USE_PIPECAT` | auto | `MegakernelTTSService` | Set `0` to force the lightweight local shim. By default, the service uses real Pipecat classes when Pipecat is installed. |
| `DEEPGRAM_API_KEY` | required | Voice agent | API key for `DeepgramSTTService`. |
| `OPENAI_API_KEY` | required | Voice agent | API key for `OpenAILLMService`. |

## Voice Agent

Run:

```bash
python demo/live_voice_agent.py --host 0.0.0.0 --port 8006
```

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | WebSocket bind host. |
| `--port` | `8765` | WebSocket bind port. |
| `--model` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS model path or Hugging Face repo. |
| `--device` | `cuda` | Torch device for the megakernel decoder. |
| `--chunk-frames` | `10` | Decoder chunk size in codec frames. |
| `--llm-model` | `gpt-4o-mini` | OpenAI chat model used by the voice agent. |
| `--audio-in-sample-rate` | `16000` | Browser microphone sample rate sent through Pipecat. |
| `--audio-out-sample-rate` | `24000` | Assistant audio playback sample rate. |
| `--no-sample` | off | Disable stochastic TTS sampling. |
| `--temperature` | `0.9` | TTS sampling temperature. |
| `--top-k` | `50` | TTS top-k sampling cutoff. |
| `--max-new-tokens` | `2048` | Maximum TTS tokens to generate. |
| `--record-dir` | `output/voice_agent_recordings` | Directory for live user and assistant WAV recordings. |

The voice agent writes two WAV files per run:

```text
YYYYMMDDTHHMMSSZ_user_mic.wav
YYYYMMDDTHHMMSSZ_assistant_tts.wav
```

## Browser Client

File: `demo/index.html`

| Setting | Default | Description |
| --- | --- | --- |
| WebSocket URL | `ws://localhost:8006` | Target server URL. Use this with an SSH tunnel from a Mac. |
| Recorder sample rate | `16000` | Browser microphone sample rate passed to the Pipecat WebSocket transport. |
| Player sample rate | `24000` | Browser playback sample rate passed to the Pipecat WebSocket transport. |
| Camera | disabled | The demo is audio-only. |
| Microphone | enabled after click | The browser requests mic permission when `Connect Mic` is clicked. |

## HTTP API

Start:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Returns status and active default mode. |
| `POST` | `/tts/stream` | Streams PCM16 audio chunks. |
| `POST` | `/stream` | Alias for `/tts/stream`. |
| `GET` | `/stream?text=...` | Simple streaming endpoint using default request settings. |
| `POST` | `/tts/wav` | Returns a complete WAV file. |

`POST` request body:

| Field | Default | Description |
| --- | --- | --- |
| `text` | required | Text to synthesize. |
| `mode` | `MEGAKERNEL_TTS_MODE` or `real` | Decoder mode override. |
| `model_path` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS model path or repo. |
| `vocoder_path` | `null` | Optional vocoder path for real megakernel mode. |
| `device` | `cuda` | Torch device for real megakernel mode. |
| `sample_rate` | `24000` | Output sample rate metadata. |
| `chunk_frames` | `10` | Real megakernel streaming chunk size. |
| `chunk_ms` | `80` | HF reference streaming chunk size. |
| `do_sample` | `true` | Enable stochastic TTS sampling. |
| `temperature` | `0.9` | TTS sampling temperature. |
| `top_k` | `50` | TTS top-k sampling cutoff. |
| `max_new_tokens` | `2048` | Maximum TTS tokens to generate. |
| `realtime` | `false` | Reserved for reference/local compatibility. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/tts/wav \
  -H "content-type: application/json" \
  -d '{
    "text": "Hello from Qwen3 TTS",
    "mode": "real",
    "chunk_frames": 10,
    "temperature": 0.9,
    "top_k": 50
  }' \
  -o output/out.wav
```

## WAV Demo

Run:

```bash
python demo/demo.py --mode real --text "Hello" --output output/demo.wav
```

| Option | Default | Description |
| --- | --- | --- |
| `--mode` | `real` | Decoder mode. |
| `--model` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS model path or repo. |
| `--text` | demo sentence | Text to synthesize. |
| `--output` | `output/qwen3tts_megakernel_demo.wav` | Destination WAV path. |

## Benchmark

Run:

```bash
python benchmark/benchmark.py --mode real --runs 5 --chunk-frames 10
```

| Option | Default | Description |
| --- | --- | --- |
| `--mode` | `real` | Decoder mode. |
| `--model` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Qwen3-TTS model path or repo. |
| `--text` | benchmark sentence | Text used for each run. |
| `--runs` | `3` | Number of benchmark repetitions. Must be at least 1. |
| `--chunk-frames` | `10` | Decoder chunk size in codec frames. Must be at least 1. |

## Round-Trip Validator

Run:

```bash
python scripts/validate_roundtrip.py --text "Hello" --output output/roundtrip_validation.wav
```

| Option | Default | Description |
| --- | --- | --- |
| `--text` | validation sentence | Text to synthesize. |
| `--output` | `output/roundtrip_validation.wav` | Destination WAV path. |
| `--real` | off | Use the real megakernel decoder instead of the synthetic decoder. |
