"""Factory helpers for decoder selection."""

from __future__ import annotations

from . import Decoder, MODE


class TalkerDecoderLoader:
    """Load the currently selected decoder implementation."""

    def load(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", **kwargs):
        return Decoder(model_name, **kwargs)


def load_decoder(model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", **kwargs):
    """Convenience function used by demos and server code."""
    return TalkerDecoderLoader().load(model_name, **kwargs)


__all__ = ["MODE", "TalkerDecoderLoader", "load_decoder"]
