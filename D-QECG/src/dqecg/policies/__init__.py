"""Architecture-owned D-QECG policies.

LLaVA and Qwen2.5-VL intentionally live in separate modules so pruning layers,
entropy windows, and token schedules can evolve independently.
"""

from .base import DecoderPolicy

__all__ = ["DecoderPolicy"]
