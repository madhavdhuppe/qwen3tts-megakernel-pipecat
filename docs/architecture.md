# Architecture

This project wraps Qwen3-TTS with a CUDA megakernel decoder and exposes it through
three surfaces:

- A FastAPI TTS endpoint for request/response WAV generation.
- A Pipecat `MegakernelTTSService` for streaming TTS frames.
- A WebSocket voice-agent demo that chains browser audio, STT, LLM, TTS, playback,
  and WAV recording.

For the complete configuration matrix, see [options.md](options.md).

## Components

```mermaid
flowchart LR
    browser["Browser demo\nMic + playback"]
    ws["Pipecat WebSocket transport\nProtobuf frames"]
    stt["Deepgram STT"]
    llm["OpenAI LLM"]
    tts["MegakernelTTSService"]
    decoder["MegakernelDecoder\nCUDA / RTX 5090"]
    model["Qwen3-TTS weights\nQwen/Qwen3-TTS-12Hz-0.6B-Base"]
    recorder["WAV recorders\nuser_mic + assistant_tts"]

    browser <--> ws
    ws --> recorder
    ws --> stt
    stt --> llm
    llm --> tts
    tts --> decoder
    model --> decoder
    decoder --> tts
    tts --> recorder
    tts --> ws
    ws --> browser
```

## Voice-Agent Pipeline

`demo/live_voice_agent.py` builds one Pipecat pipeline. Audio enters through the
WebSocket input transport as browser microphone frames. The pipeline records the
raw user audio, transcribes it, feeds the transcript into the LLM context, streams
the LLM response into Qwen3-TTS, records the generated assistant audio, then sends
that audio back to the browser.

```mermaid
flowchart TD
    input["transport.input()"]
    userTap["UserMicRecorder\nwrites *_user_mic.wav"]
    stt["DeepgramSTTService"]
    userAgg["LLM user aggregator\nupdates context"]
    llm["OpenAILLMService"]
    tts["MegakernelTTSService"]
    assistantTap["AssistantTTSRecorder\nwrites *_assistant_tts.wav"]
    output["transport.output()"]
    assistantAgg["LLM assistant aggregator\nstores assistant text"]

    input --> userTap
    userTap --> stt
    stt --> userAgg
    userAgg --> llm
    llm --> tts
    tts --> assistantTap
    assistantTap --> output
    output --> assistantAgg
```

## Execution Flow

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Browser as Browser client
    participant SSH as SSH tunnel
    participant Server as GPU voice agent
    participant STT as Deepgram STT
    participant LLM as OpenAI LLM
    participant TTS as Qwen3 Megakernel TTS
    participant Files as WAV recordings

    User->>Browser: Click Connect Mic
    Browser->>SSH: Open ws://localhost:8006
    SSH->>Server: Forward WebSocket frames
    Server->>Browser: Send greeting audio after client connect
    User->>Browser: Speak
    Browser->>Server: Stream microphone audio frames
    Server->>Files: Append user mic PCM to *_user_mic.wav
    Server->>STT: Stream user audio
    STT->>Server: Transcript frames
    Server->>LLM: Send updated conversation context
    LLM->>Server: Stream response text
    Server->>TTS: Stream text to MegakernelTTSService
    TTS->>Server: PCM audio chunks
    Server->>Files: Append TTS PCM to *_assistant_tts.wav
    Server->>Browser: Stream assistant audio frames
    Browser->>User: Play assistant voice
    User->>Browser: Disconnect
    Browser->>Server: Close WebSocket
    Server->>Files: Close WAV files
```

## Startup Path

1. `main()` validates `DEEPGRAM_API_KEY` and `OPENAI_API_KEY`.
2. `_configure_gpu()` requires CUDA, selects device `0`, and sets
   `MEGAKERNEL_TTS_MODE=real` by default.
3. `MegakernelTTSService` initializes `MegakernelDecoder` with the Qwen3-TTS model.
4. `tts.decoder.initialize()` loads model weights and warms up the CUDA path.
5. `WebsocketServerTransport` starts with protobuf serialization and browser audio
   input/output enabled.
6. `PipelineRunner` runs the Pipecat task until the browser disconnects.

## Recording Behavior

The voice-agent demo records live sessions on the GPU server. By default, files
are written under:

```text
output/voice_agent_recordings/
```

Each run uses a UTC timestamp prefix:

```text
YYYYMMDDTHHMMSSZ_user_mic.wav
YYYYMMDDTHHMMSSZ_assistant_tts.wav
```

Use `--record-dir` to override the destination:

```bash
python demo/live_voice_agent.py --port 8006 --host 0.0.0.0 \
  --record-dir output/my_demo_recordings
```

## Local Demo Topology

When the agent runs on a remote GPU box and the browser runs on a Mac, the common
setup is:

```mermaid
flowchart LR
    macBrowser["Mac browser\nhttp://localhost:8080/demo/"]
    tunnel["SSH tunnel\n-L 8006:localhost:8006"]
    gpuAgent["GPU box\nvoice agent on :8006"]
    recordings["GPU filesystem\noutput/voice_agent_recordings"]

    macBrowser <--> tunnel
    tunnel <--> gpuAgent
    gpuAgent --> recordings
```

Run the tunnel from the Mac:

```bash
ssh -p 51848 root@79.117.32.66 -L 8006:localhost:8006
```

Then open the local browser client at `http://localhost:8080/demo/` and connect
to `ws://localhost:8006`.
