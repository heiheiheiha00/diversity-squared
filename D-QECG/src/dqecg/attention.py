"""Attention-backend selection for local D-QECG inference."""

from __future__ import annotations

import importlib

from transformers.utils import is_flash_attn_2_available


_SUPPORTED = {"auto", "eager", "sdpa", "flash_attention_2"}


def flash_attention_2_is_usable() -> bool:
    """Reject metadata-only or ABI-broken FlashAttention installations."""

    if not is_flash_attn_2_available():
        return False
    try:
        importlib.import_module("flash_attn_2_cuda")
    except (ImportError, OSError):
        return False
    return True


def resolve_attention_implementation(requested: str | None) -> str:
    """Resolve ``auto`` to FA2 when usable, otherwise native PyTorch SDPA.

    An explicitly requested FlashAttention backend never falls back silently:
    that would make efficiency logs claim FA2 while actually measuring another
    kernel. ``auto`` is the only mode that deliberately falls back to SDPA.
    """

    normalized = "auto" if requested is None else str(requested).strip().lower()
    aliases = {"flash": "flash_attention_2", "fa2": "flash_attention_2"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUPPORTED:
        raise ValueError(
            f"Unsupported attention implementation {requested!r}; "
            f"choose one of {sorted(_SUPPORTED)}."
        )
    if normalized in {"eager", "sdpa"}:
        return normalized
    flash_available = flash_attention_2_is_usable()
    if normalized == "auto":
        return "flash_attention_2" if flash_available else "sdpa"
    if normalized == "flash_attention_2" and not flash_available:
        raise ImportError(
            "attn_implementation=flash_attention_2 was requested, but a "
            "Torch/CUDA-compatible flash-attn build is not importable. Install "
            "a matching wheel with server/install_flash_attention.sh, or use "
            "attn_implementation=auto to fall back explicitly to SDPA."
        )
    return normalized
