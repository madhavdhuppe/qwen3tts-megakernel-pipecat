import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat_service.tts_service import MegakernelTTSService


async def main():
    tts = MegakernelTTSService(mode="fake")
    audio_frames = 0
    total_bytes = 0

    async for frame in tts.run_tts("hello from the fake local path", context_id="demo"):
        audio = getattr(frame, "audio", None)
        if audio is None:
            print(type(frame).__name__)
            continue

        audio_frames += 1
        total_bytes += len(audio)
        print(type(frame).__name__, len(audio))

    print(f"audio_frames={audio_frames} total_bytes={total_bytes}")


asyncio.run(main())
