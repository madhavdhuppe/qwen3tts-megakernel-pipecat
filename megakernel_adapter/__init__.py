"""Decoder selection for HF reference and RTX 5090 megakernel runs."""

from __future__ import annotations

import os


def _selected_mode() -> str:
    explicit = os.getenv("MEGAKERNEL_TTS_MODE")
    if explicit:
        return explicit.strip().lower()
    return "real"


MODE = _selected_mode()

if MODE in {"hf", "reference", "hf_reference"}:
    from .hf_reference import HFReferenceDecoder as Decoder
elif MODE in {"real", "megakernel", "cuda", "gpu"}:
    from .megakernel_decoder import MegakernelDecoder as Decoder
else:
    raise ValueError(
        f"Unsupported MEGAKERNEL_TTS_MODE={MODE!r}. Use hf or real."
    )


__all__ = ["Decoder", "MODE"]
