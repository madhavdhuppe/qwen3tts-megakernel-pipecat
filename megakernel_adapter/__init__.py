"""Decoder selection for local fake, HF reference, and RTX 5090 megakernel runs."""

from __future__ import annotations

import os


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "fake"}


def _selected_mode() -> str:
    explicit = os.getenv("MEGAKERNEL_TTS_MODE")
    if explicit:
        return explicit.strip().lower()
    return "fake" if _is_truthy(os.getenv("MEGAKERNEL_USE_FAKE", "1")) else "real"


MODE = _selected_mode()
USE_FAKE = MODE == "fake"

if MODE == "fake":
    from .fake_decoder import FakeMegakernelDecoder as Decoder
elif MODE in {"hf", "reference", "hf_reference"}:
    from .hf_reference import HFReferenceDecoder as Decoder
elif MODE in {"real", "megakernel", "cuda", "gpu"}:
    from .megakernel_decoder import MegakernelDecoder as Decoder
else:
    raise ValueError(
        "Unsupported MEGAKERNEL_TTS_MODE="
        f"{MODE!r}. Use fake, hf, or real."
    )


__all__ = ["Decoder", "MODE", "USE_FAKE"]
