"""Minimal Pipecat pipeline assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .tts_service import MegakernelTTSService


@dataclass
class LocalPipeline:
    processors: Sequence[Any]


def build_tts_only_pipeline(**tts_kwargs):
    """Build the smallest pipeline surface needed for local verification."""
    tts = MegakernelTTSService(**tts_kwargs)
    try:
        from pipecat.pipeline.pipeline import Pipeline

        return Pipeline([tts])
    except ImportError:
        return LocalPipeline([tts])
