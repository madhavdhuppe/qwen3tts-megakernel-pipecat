# Implementation Mental Model

This repo is organized around one narrow acceleration boundary:

```text
text -> Qwen3-TTS talker decode -> codec frames -> vocoder audio -> Pipecat frames
```

The RTX 5090 megakernel is used for the Qwen3-TTS talker decoder. It is not a
replacement for the whole TTS stack. Text conditioning, codebook completion,
codec-to-waveform decoding, HTTP streaming, and Pipecat frame delivery stay as
separate layers.

## Runtime Pipeline

```text
input text
  -> tokenizer and Qwen3-TTS text projection
  -> prefill embeddings for the talker LM
  -> megakernel talker step
       output: codebook-0 token + post-norm hidden state
  -> code predictor
       output: codebooks 1-15
  -> codec embedding sum + next trailing text embedding
  -> next megakernel talker step
  -> codec frame stream
  -> speech tokenizer/vocoder
  -> PCM16 audio chunks
  -> HTTP bytes or Pipecat TTSAudioRawFrame
```

The main implementation lives in `megakernel_adapter/tts_engine.py`.
`MegakernelTTSEngine.synthesize_streaming()` yields audio as soon as enough codec
frames have accumulated. The first chunk uses one codec frame to make TTFC
visible; later chunks use the configured `chunk_frames`.

## Kernel Boundary

The adapted megakernel keeps the original Qwen3-0.6B decode shape where the
TTS talker matches it:

| Field | Qwen3-TTS talker value |
| --- | ---: |
| layers | 28 |
| hidden size | 1024 |
| intermediate size | 3072 |
| query heads | 16 |
| KV heads | 8 |
| head dim | 128 |
| dtype | bfloat16 |

The TTS-specific differences are:

| Difference | Why it matters |
| --- | --- |
| `LDG_VOCAB_SIZE=3072` | The talker predicts codec tokens, not text tokens. |
| `talker.codec_head.weight` | Qwen3-TTS uses an untied codec LM head. |
| `ROPE_THETA=1000000` | The talker config differs from the source Qwen3 kernel. |
| `token_id < 0` sentinel | Lets Python pass a precomputed fused embedding instead of a token ID. |

`TTSDecoder.step()` is the normal codec-token path. `TTSDecoder.step_with_embed()`
is the TTS path used for prefill and recurrent fused codec/text embeddings.

## Streaming Boundary

Streaming is validated at three layers:

1. `MegakernelTTSEngine.synthesize_streaming()` yields repeated `(audio, sr)`
   chunks instead of waiting for the full utterance.
2. `MegakernelDecoder.stream_audio()` converts those chunks to PCM16 bytes.
3. `MegakernelTTSService.run_tts()` yields Pipecat `TTSAudioRawFrame` objects
   as chunks arrive.

The `/tts/wav` endpoint is intentionally buffered because WAV files need a full
header. Use `/tts/stream` or the Pipecat service for streaming validation.

## Benchmark Model

Treat these as separate numbers:

| Metric | Meaning |
| --- | --- |
| `cold_init_ms` | Model download/cache load, CUDA extension build, weight load, warmup. |
| `warm_ttfc_ms` | Initialized service request start to first PCM chunk. |
| `rtf` | Total generation wall time divided by emitted audio duration. |
| `talker_steps_per_s` | Approximate talker decoder steps per second for the run. |
| `chunks` | Number of audio chunks emitted before completion. |

The assignment targets should be compared to warm request metrics, while cold
initialization should be reported honestly as deployment startup cost.

## Bottleneck Expectations

If talker decode is fast but TTFC is still high, inspect:

- Speech tokenizer/vocoder first-call latency.
- Chunk size and first-frame decode behavior.
- Code predictor sampling and per-group LM heads.
- Accidental service reinitialization per request.
- CUDA synchronization from `.item()` or CPU-side control flow in hot loops.

The benchmark and HTTP server now reuse initialized services to avoid measuring
cold startup as per-request latency.
