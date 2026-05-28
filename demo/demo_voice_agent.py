#!/usr/bin/env python3
"""Full Pipecat voice agent pipeline: STT → LLM → Megakernel TTS → Audio Output.

This mirrors the upstream reference voice-agent demo while using the local
megakernel-backed TTS service.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import wave
from datetime import datetime
from pathlib import Path
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


class WavAudioRecorder:
    """Writes Pipecat AudioRawFrame chunks to a single PCM16 WAV file."""

    def __init__(self, path: Path):
        self.path = path
        self._wav = None
        self._sample_rate = None
        self._num_channels = None

    def write_frame(self, frame) -> None:
        sample_rate = int(getattr(frame, "sample_rate"))
        num_channels = int(getattr(frame, "num_channels", 1))

        if self._wav is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._sample_rate = sample_rate
            self._num_channels = num_channels
            self._wav = wave.open(str(self.path), "wb")
            self._wav.setnchannels(num_channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)
            logger.info("Recording audio to %s", self.path)
        elif sample_rate != self._sample_rate or num_channels != self._num_channels:
            logger.warning(
                "Skipping audio frame with changed format: %s Hz/%s ch for %s",
                sample_rate,
                num_channels,
                self.path,
            )
            return

        self._wav.writeframes(getattr(frame, "audio"))

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None
            logger.info("Saved recording to %s", self.path)


def _configure_gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the RTX 5090 megakernel demo.")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("GPU detected: %s", gpu_name)
    if "RTX 5090" not in gpu_name:
        logger.warning(
            "Expected an RTX 5090 for real megakernel mode, but detected %s. "
            "Performance and compatibility may differ.",
            gpu_name,
        )
    os.environ.setdefault("MEGAKERNEL_TTS_MODE", "real")
    return gpu_name


def _load_websocket_transport_classes():
    try:
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    except ImportError:
        from pipecat.transports.network.fastapi_websocket import (  # type: ignore
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )
    return FastAPIWebsocketParams, FastAPIWebsocketTransport


async def run_voice_pipeline(args):
    """Run the full STT → LLM → TTS voice agent pipeline."""
    os.environ["MEGAKERNEL_TTS_USE_PIPECAT"] = "1"

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService

    _configure_gpu()

    from pipecat_service.tts_service import MegakernelTTSService

    class AudioRecordingProcessor(FrameProcessor):
        def __init__(self, recorder: WavAudioRecorder, **kwargs):
            super().__init__(**kwargs)
            self._recorder = recorder

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if (
                direction == FrameDirection.DOWNSTREAM
                and hasattr(frame, "audio")
                and hasattr(frame, "sample_rate")
            ):
                self._recorder.write_frame(frame)
            await self.push_frame(frame, direction)

        async def cleanup(self):
            self._recorder.close()
            await super().cleanup()

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    tts = MegakernelTTSService(
        model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode="real",
        device="cuda",
        chunk_frames=10,
    )
    tts.decoder.initialize()
    logger.info("Megakernel decoder initialized on GPU.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant powered by a custom CUDA megakernel "
                "TTS engine. Keep answers concise and conversational."
            ),
        }
    ]

    context = LLMContext(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    record_dir = Path(args.record_dir)
    user_audio_recorder = WavAudioRecorder(record_dir / f"{session_id}_user_mic.wav")
    assistant_audio_recorder = WavAudioRecorder(record_dir / f"{session_id}_assistant_tts.wav")
    user_audio_tap = AudioRecordingProcessor(user_audio_recorder, name="UserMicRecorder")
    assistant_audio_tap = AudioRecordingProcessor(
        assistant_audio_recorder,
        name="AssistantTTSRecorder",
    )

    if args.transport == "websocket":
        transport = WebsocketServerTransport(
        host=args.host,
        port=args.port,
        params=WebsocketServerParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            serializer=ProtobufFrameSerializer(),
        ),
    )
    elif args.transport == "daily":
        from pipecat.transports.daily.transport import DailyParams, DailyTransport

        transport = DailyTransport(
            room_url=os.getenv("DAILY_ROOM_URL", ""),
            token=os.getenv("DAILY_TOKEN", ""),
            bot_name="Megakernel TTS Bot",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
            ),
        )
    else:
        raise ValueError(f"Unknown transport: {args.transport}")

    pipeline = Pipeline(
        [
            transport.input(),
            user_audio_tap,
            stt,
            user_aggregator,
            llm,
            tts,
            assistant_audio_tap,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting conversation")
        messages.append(
            {
                "role": "system",
                "content": "Greet the user briefly and tell them you are ready to chat.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        user_audio_recorder.close()
        assistant_audio_recorder.close()
        await task.cancel()

    runner = PipelineRunner()
    logger.info(f"Voice agent running ({args.transport} transport)")
    await runner.run(task)


async def run_text_only_pipeline(args):
    """Text-only mode: type text, hear synthesized speech via Pipecat frames."""
    import numpy as np
    import soundfile as sf

    from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

    from pipecat_service.tts_service import MegakernelTTSService

    _configure_gpu()

    tts = MegakernelTTSService(
        model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode="real",
        device="cuda",
        chunk_frames=10,
    )
    tts.decoder.initialize()
    logger.info("Megakernel decoder initialized on GPU.")

    print("=" * 60)
    print("MEGAKERNEL TTS — TEXT-ONLY PIPECAT MODE")
    print("=" * 60)
    print("Type text and press Enter to synthesize speech.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            break

        print(f"Synthesizing: '{text}'")
        audio_chunks = []
        chunk_count = 0

        async for frame in tts.run_tts(text, context_id="text-mode"):
            if isinstance(frame, TTSStartedFrame):
                print("  [TTS Started]")
            elif isinstance(frame, TTSAudioRawFrame):
                chunk_count += 1
                audio_chunks.append(np.frombuffer(frame.audio, dtype=np.int16))
                print(f"  Chunk {chunk_count}: {len(frame.audio)} bytes")
            elif isinstance(frame, TTSStoppedFrame):
                print("  [TTS Stopped]")

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            output_path = "output/voice_agent_output_text.wav"
            sf.write(output_path, full_audio, 24000)
            duration = len(full_audio) / 24000
            print(f"  Saved: {output_path} ({duration:.2f}s)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Pipecat voice agent with megakernel TTS")
    parser.add_argument(
        "--transport",
        choices=["websocket", "daily"],
        default="websocket",
        help="Transport type (default: websocket)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument(
        "--record-dir",
        default="output/voice_agent_recordings",
        help="Directory for live voice-agent WAV recordings",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text-only mode (no STT, type text → hear TTS)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.text_only:
        asyncio.run(run_text_only_pipeline(args))
        return

    missing = []
    if not os.getenv("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Set them before running the voice pipeline:")
        print("  export DEEPGRAM_API_KEY=your-key")
        print("  export OPENAI_API_KEY=your-key")
        print("\nOr use --text-only mode to test TTS without external services.")
        sys.exit(1)

    asyncio.run(run_voice_pipeline(args))


if __name__ == "__main__":
    main()
