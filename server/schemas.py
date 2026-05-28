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
        description="Override decoder mode: fake, hf, or real.",
    )
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    sample_rate: int = 24000
    chunk_ms: int = 80
    realtime: bool = False
