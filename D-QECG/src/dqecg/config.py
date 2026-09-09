"""Validated runtime configuration and model-specific layer schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DQECGSchedule:
    """Physical token counts and decoder boundaries for one architecture."""

    average_tokens: int
    entry_tokens: int
    after_first_stage: int
    after_second_stage: int
    first_boundary: int
    second_boundary: int
    drop_boundary: int
    layer_count: int

    @property
    def measured_average(self) -> float:
        weighted = (
            self.first_boundary * self.entry_tokens
            + (self.second_boundary - self.first_boundary) * self.after_first_stage
            + (self.drop_boundary - self.second_boundary) * self.after_second_stage
        )
        return weighted / self.layer_count


@dataclass(frozen=True)
class DQECGConfig:
    budget: int = 128
    architecture: str = "auto"
    debug: bool = False
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.architecture not in {"auto", "llava", "qwen2_5_vl"}:
            raise ValueError(f"Unsupported D-QECG architecture: {self.architecture!r}")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int):
            raise TypeError("D-QECG budget must be an integer.")
        if self.budget <= 0:
            raise ValueError("D-QECG budget must be positive.")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def derive_schedule(
    average_tokens: int,
    *,
    layer_count: int,
    patch_count: Optional[int] = None,
) -> DQECGSchedule:
    """Compatibility dispatcher; formal adapters use their own policy module.

    New code must call ``derive_llava_schedule`` or
    ``derive_qwen2_5_vl_schedule`` explicitly. Keeping this wrapper avoids
    breaking older experiment launchers while removing it from both formal
    runtime paths.
    """

    if int(layer_count) == 32:
        from .policies.llava import derive_llava_schedule

        return derive_llava_schedule(
            average_tokens, layer_count=layer_count, patch_count=patch_count
        )
    if int(layer_count) == 28:
        from .policies.qwen2_5_vl import derive_qwen2_5_vl_schedule

        return derive_qwen2_5_vl_schedule(
            average_tokens, layer_count=layer_count, patch_count=patch_count
        )
    raise ValueError(
        "No formal D-QECG architecture policy is registered for "
        f"a {layer_count}-layer decoder."
    )
