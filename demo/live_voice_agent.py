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
        model=args.llm_model,
    )
    tts = MegakernelTTSService(
        model_path=args.model,
        mode="real",
        device=args.device,
        chunk_frames=args.chunk_frames,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
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

    transport = WebsocketServerTransport(
        host=args.host,
        port=args.port,
        params=WebsocketServerParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            audio_in_sample_rate=args.audio_in_sample_rate,
            audio_out_sample_rate=args.audio_out_sample_rate,
            serializer=ProtobufFrameSerializer(),
        ),
    )

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
    logger.info("Voice agent running (websocket transport)")
    await runner.run(task)


def main():
    parser = argparse.ArgumentParser(description="Pipecat voice agent with megakernel TTS")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Qwen3-TTS model path or Hugging Face repo",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for the megakernel decoder")
    parser.add_argument("--chunk-frames", type=int, default=10, help="Decoder chunk size in codec frames")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI chat model for the voice agent")
    parser.add_argument(
        "--audio-in-sample-rate",
        type=int,
        default=16000,
        help="Browser microphone sample rate",
    )
    parser.add_argument(
        "--audio-out-sample-rate",
        type=int,
        default=24000,
        help="Assistant playback sample rate",
    )
    parser.add_argument("--no-sample", action="store_true", help="Disable stochastic TTS sampling")
    parser.add_argument("--temperature", type=float, default=0.9, help="TTS sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="TTS top-k sampling cutoff")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum TTS tokens to generate")
    parser.add_argument(
        "--record-dir",
        default="output/voice_agent_recordings",
        help="Directory for live voice-agent WAV recordings",
    )
    args = parser.parse_args()

    if args.port < 1:
        parser.error("--port must be >= 1")
    if args.chunk_frames < 1:
        parser.error("--chunk-frames must be >= 1")
    if args.audio_in_sample_rate < 1:
        parser.error("--audio-in-sample-rate must be >= 1")
    if args.audio_out_sample_rate < 1:
        parser.error("--audio-out-sample-rate must be >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")

    logging.basicConfig(level=logging.INFO)

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
        sys.exit(1)

    asyncio.run(run_voice_pipeline(args))


if __name__ == "__main__":
    main()
