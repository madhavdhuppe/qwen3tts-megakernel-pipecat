#!/usr/bin/env python3
"""Pipecat voice agent: STT → LLM → Megakernel TTS → local audio out.

Requires API keys (typical local demo):
  export OPENAI_API_KEY=...
  export DEEPGRAM_API_KEY=...   # or set PIPECAT_STT=openai

Run on RTX 5090:
  export MEGAKERNEL_TTS_MODE=real
  export MEGAKERNEL_TTS_USE_PIPECAT=1
  python demo/pipecat_voice_agent.py

Record the session for the deliverable demo (screen + mic).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.version_info < (3, 10):
    raise SystemExit("Pipecat voice agent requires Python 3.10+ (use 3.10 or 3.11 on the 5090 box).")

os.environ.setdefault("MEGAKERNEL_TTS_MODE", "real")
os.environ.setdefault("MEGAKERNEL_TTS_USE_PIPECAT", "1")


async def main() -> None:
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except Exception as exc:
        raise SystemExit(
            "Pipecat import failed. Use Python 3.10+, then: pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc

    from pipecat_service.tts_service import MegakernelTTSService

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Set OPENAI_API_KEY for STT+LLM (or extend this script for other providers).")

    stt_backend = os.getenv("PIPECAT_STT", "deepgram").lower()
    if stt_backend == "openai":
        from pipecat.services.openai.stt import OpenAISTTService

        stt = OpenAISTTService(api_key=openai_key)
    else:
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_key:
            raise SystemExit("Set DEEPGRAM_API_KEY or PIPECAT_STT=openai")
        from pipecat.services.deepgram.stt import DeepgramSTTService

        stt = DeepgramSTTService(api_key=deepgram_key)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    llm = OpenAILLMService(api_key=openai_key, model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    tts = MegakernelTTSService(mode=os.getenv("MEGAKERNEL_TTS_MODE", "real"))

    messages = [{"role": "system", "content": "You are a concise voice assistant."}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    print("Pipecat voice agent running. Speak into the microphone. Ctrl+C to stop.")
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
