"""Public Nuwa-style model wrapper."""

from __future__ import annotations

from .adapters import apply_dqecg
from .config import DQECGConfig


def _apply_architecture(model, *, budget, architecture, debug):
    return apply_dqecg(
        model,
        DQECGConfig(
            budget=int(budget),
            architecture=architecture,
            debug=bool(debug),
        ),
    )


def dqecg_llava(model, budget: int = 128, debug: bool = False):
    """Enable only the formal LLaVA policy on a loaded LLaVA model."""

    return _apply_architecture(
        model, budget=budget, architecture="llava", debug=debug
    )


def dqecg_qwen2_5_vl(model, budget: int = 128, debug: bool = False):
    """Enable only the formal Qwen2.5-VL policy on a loaded Qwen model."""

    return _apply_architecture(
        model, budget=budget, architecture="qwen2_5_vl", debug=debug
    )


def dqecg(model, budget: int = 128, architecture: str = "auto", debug: bool = False):
    """Compatibility entry point; new formal integrations use typed wrappers."""

    return _apply_architecture(
        model,
        budget=budget,
        architecture=architecture,
        debug=debug,
    )
