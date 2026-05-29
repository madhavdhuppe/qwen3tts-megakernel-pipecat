"""Pydantic request and response schemas for the local TTS server."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: Optional[str] = Field(
        default=None,
        description="Override decoder mode: hf or real.",
    )
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    vocoder_path: Optional[str] = None
    device: str = "cuda"
    sample_rate: int = 24000
    chunk_frames: int = 10
    chunk_ms: int = 80
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 2048
    realtime: bool = False
