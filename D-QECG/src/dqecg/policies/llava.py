"""Formal LLaVA-only pruning, entropy-window, and budget policy."""

from __future__ import annotations

from ..budget import derive_dqecg_budget
from ..config import DQECGSchedule
from .base import DecoderPolicy


LLAVA_LAYER_COUNT = 32
LLAVA_FIRST_PRUNING_LAYER = 9
LLAVA_SECOND_PRUNING_LAYER = 13
LLAVA_DROP_LAYER = 24

# Freeze the current runtime behavior during the architecture split. Future
# LLaVA window changes belong only in this module.
LLAVA_DECODER_POLICY = DecoderPolicy(
    architecture="llava",
    first_entropy_offsets=(-2, -1, 0),
    second_entropy_offsets=(-4, -3, -2, -1, 0),
)


def derive_llava_schedule(
    average_tokens: int,
    *,
    layer_count: int,
    patch_count=None,
) -> DQECGSchedule:
    if int(layer_count) != LLAVA_LAYER_COUNT:
        raise ValueError(
            f"The formal LLaVA policy requires {LLAVA_LAYER_COUNT} decoder layers, "
            f"got {layer_count}."
        )
    budget = derive_dqecg_budget(average_tokens, patch_count=patch_count)
    return DQECGSchedule(
        average_tokens=budget.average_tokens,
        entry_tokens=budget.entry_tokens,
        after_first_stage=budget.after_first_stage,
        after_second_stage=budget.after_second_stage,
        first_boundary=LLAVA_FIRST_PRUNING_LAYER,
        second_boundary=LLAVA_SECOND_PRUNING_LAYER,
        drop_boundary=LLAVA_DROP_LAYER,
        layer_count=LLAVA_LAYER_COUNT,
    )
