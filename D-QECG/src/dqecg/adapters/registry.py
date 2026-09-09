"""Architecture detection and adapter dispatch."""

from __future__ import annotations

from ..config import DQECGConfig


def detect_architecture(model) -> str:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    class_name = type(model).__name__.lower()
    if "qwen2_5_vl" in model_type or "qwen2_5_vl" in class_name:
        return "qwen2_5_vl"
    if "llava" in model_type or "llava" in class_name:
        return "llava"
    raise TypeError(
        f"Cannot detect a supported architecture from {type(model).__name__} "
        f"(model_type={model_type!r})."
    )


def apply_dqecg(model, config: DQECGConfig):
    architecture = (
        detect_architecture(model)
        if config.architecture == "auto"
        else config.architecture
    )
    if getattr(model, "_dqecg_enabled", False):
        existing = getattr(model, "_dqecg_config", None)
        if existing == config:
            return model
        raise RuntimeError(
            "This model already has D-QECG enabled with a different configuration. "
            "Load a fresh model before changing the budget."
        )
    if architecture == "llava":
        from .llava import apply_llava

        return apply_llava(model, config)
    if architecture == "qwen2_5_vl":
        from .qwen2_5_vl import apply_qwen2_5_vl

        return apply_qwen2_5_vl(model, config)
    raise AssertionError(f"Unhandled architecture: {architecture}")
