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
        description=(
            "Override decoder mode: real, megakernel, cuda, gpu, hf, "
            "reference, or hf_reference."
        ),
    )
    model_path: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    vocoder_path: Optional[str] = None
    device: str = "cuda"
    sample_rate: int = Field(default=24000, gt=0)
    chunk_frames: int = Field(default=10, gt=0)
    chunk_ms: int = Field(default=80, gt=0)
    do_sample: bool = True
    temperature: float = Field(default=0.9, gt=0)
    top_k: int = Field(default=50, ge=0)
    max_new_tokens: int = Field(default=2048, gt=0)
    realtime: bool = False
