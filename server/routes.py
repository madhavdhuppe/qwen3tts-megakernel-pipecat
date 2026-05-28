"""FastAPI routes for megakernel and HF-reference TTS streaming."""

from __future__ import annotations

import io
import os
import wave
from typing import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import Response, StreamingResponse

from pipecat_service.tts_service import MegakernelTTSService
from server.schemas import HealthResponse, TTSRequest

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    mode = os.getenv("MEGAKERNEL_TTS_MODE", "real")
    return HealthResponse(
        status="ok",
        mode=mode,
    )


@router.post("/tts/stream")
async def tts_stream(request: TTSRequest) -> StreamingResponse:
    service = _service_from_request(request)
    return StreamingResponse(
        _pcm_stream(service, request.text),
        media_type=f"audio/L16; rate={request.sample_rate}; channels=1",
    )


@router.post("/stream")
async def stream_alias(request: TTSRequest) -> StreamingResponse:
    return await tts_stream(request)


@router.get("/stream")
async def stream_get(
    text: str = Query("Hello from the Qwen3-TTS megakernel path."),
) -> StreamingResponse:
    request = TTSRequest(text=text)
    return await tts_stream(request)


@router.post("/tts/wav")
async def tts_wav(request: TTSRequest) -> Response:
    service = _service_from_request(request)
    pcm = bytearray()
    sample_rate = request.sample_rate

    async for audio, sr in service.decoder.stream_audio(request.text):
        pcm.extend(audio)
        sample_rate = sr

    wav_bytes = _wav_from_pcm16(bytes(pcm), sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


def _service_from_request(request: TTSRequest) -> MegakernelTTSService:
    return MegakernelTTSService(
        model_path=request.model_path,
        mode=request.mode,
        sample_rate=request.sample_rate,
        chunk_ms=request.chunk_ms,
        realtime=request.realtime,
    )


async def _pcm_stream(
    service: MegakernelTTSService,
    text: str,
) -> AsyncIterator[bytes]:
    async for audio, _sample_rate in service.decoder.stream_audio(text):
        yield audio


def _wav_from_pcm16(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()
