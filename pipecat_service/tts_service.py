"""Pipecat TTS service for RTX 5090 megakernel and HF reference modes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from megakernel_adapter.hf_reference import HFReferenceDecoder
from megakernel_adapter.megakernel_decoder import MegakernelDecoder

logger = logging.getLogger(__name__)

_PIPECAT_FLAG = os.getenv("MEGAKERNEL_TTS_USE_PIPECAT")
_USE_REAL_PIPECAT = _PIPECAT_FLAG is None or _PIPECAT_FLAG.strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass
class LocalTTSAudioRawFrame:
    audio: bytes
    sample_rate: int
    num_channels: int = 1
    context_id: Optional[str] = None


@dataclass
class LocalTTSStartedFrame:
    context_id: Optional[str] = None


@dataclass
class LocalTTSStoppedFrame:
    context_id: Optional[str] = None


@dataclass
class LocalErrorFrame:
    error: str


class LocalTTSService:
    def __init__(self, *args, **kwargs):
        pass

    async def start_ttfb_metrics(self):
        pass

    async def stop_ttfb_metrics(self):
        pass

    async def start_tts_usage_metrics(self, text: str):
        pass


Frame = object
TTSAudioRawFrame = LocalTTSAudioRawFrame
TTSStartedFrame = LocalTTSStartedFrame
TTSStoppedFrame = LocalTTSStoppedFrame
ErrorFrame = LocalErrorFrame
TTSService = LocalTTSService

if _USE_REAL_PIPECAT:
    try:
        from pipecat.frames.frames import (  # type: ignore
            ErrorFrame,
            Frame,
            TTSAudioRawFrame,
            TTSStartedFrame,
            TTSStoppedFrame,
        )
        from pipecat.services.tts_service import TTSService  # type: ignore
    except ImportError:
        try:
            from pipecat.frames.frames import AudioRawFrame as TTSAudioRawFrame  # type: ignore
            from pipecat.frames.frames import ErrorFrame, Frame  # type: ignore

            TTSStartedFrame = None
            TTSStoppedFrame = None

            from pipecat.services.ai_services import TTSService  # type: ignore
        except Exception:
            logger.debug("Pipecat is unavailable; using local TTS service shim.", exc_info=True)
    except Exception:
        logger.debug("Pipecat is unavailable; using local TTS service shim.", exc_info=True)


def _mode() -> str:
    explicit = os.getenv("MEGAKERNEL_TTS_MODE")
    if explicit:
        return explicit.strip().lower()
    return "real"


def _build_decoder(mode: str, model_path: str, *, sample_rate: int = 24000, **kwargs):
    if mode in {"hf", "reference", "hf_reference"}:
        kwargs.pop("chunk_frames", None)
        return HFReferenceDecoder(model_path, **kwargs)
    if mode in {"real", "megakernel", "cuda", "gpu"}:
        kwargs.pop("chunk_ms", None)
        kwargs.pop("realtime", None)
        return MegakernelDecoder(model_path, **kwargs)
    raise ValueError(f"Unsupported MEGAKERNEL_TTS_MODE={mode!r}")


def _frame(frame_cls: Any, **kwargs):
    if frame_cls is None:
        return None
    try:
        return frame_cls(**kwargs)
    except TypeError:
        kwargs.pop("context_id", None)
        return frame_cls(**kwargs)


class MegakernelTTSService(TTSService):
    """Pipecat-compatible streaming TTS service (default: real megakernel on RTX 5090)."""

    def __init__(
        self,
        *,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode: Optional[str] = None,
        sample_rate: int = 24000,
        num_channels: int = 1,
        decoder: Any = None,
        **decoder_kwargs,
    ):
        try:
            super().__init__(sample_rate=sample_rate)
        except TypeError:
            super().__init__()

        self.mode = (mode or _mode()).lower()
        self._sample_rate = sample_rate
        self.num_channels = num_channels
        self.decoder = decoder or _build_decoder(
            self.mode,
            model_path,
            sample_rate=sample_rate,
            **decoder_kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(
        self,
        text: str,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[Frame, None]:
        logger.debug("%s generating TTS in %s mode", self.__class__.__name__, self.mode)
        ttfb_stopped = False

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            started = _frame(TTSStartedFrame, context_id=context_id)
            if started is not None:
                yield started

            async for audio, sample_rate in self.decoder.stream_audio(text):
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True
                yield _frame(
                    TTSAudioRawFrame,
                    audio=audio,
                    sample_rate=sample_rate,
                    num_channels=self.num_channels,
                    context_id=context_id,
                )
        except Exception as exc:
            logger.exception("Megakernel TTS exception")
            yield _frame(ErrorFrame, error=f"Megakernel TTS error: {exc}")
        finally:
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            stopped = _frame(TTSStoppedFrame, context_id=context_id)
            if stopped is not None:
                yield stopped
