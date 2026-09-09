"""Runtime adapters for supported multimodal model families."""

from .registry import apply_dqecg, detect_architecture

__all__ = ["apply_dqecg", "detect_architecture"]
