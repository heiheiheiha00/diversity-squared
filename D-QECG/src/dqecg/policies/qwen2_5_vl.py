"""Formal Qwen2.5-VL-7B-only pruning, entropy, and budget policy."""

from __future__ import annotations

from ..config import DQECGSchedule
from .base import DecoderPolicy


QWEN2_5_VL_7B_LAYER_COUNT = 28
QWEN2_5_VL_FIRST_PRUNING_LAYER = 8
QWEN2_5_VL_SECOND_PRUNING_LAYER = 11
QWEN2_5_VL_DROP_LAYER = 21

# These windows reproduce the Qwen runtime immediately before the split. They
# are deliberately independent from the LLaVA constants above.
QWEN2_5_VL_DECODER_POLICY = DecoderPolicy(
    architecture="qwen2_5_vl",
    first_entropy_offsets=(-2, -1, 0),
    second_entropy_offsets=(-3, -2, -1, 0),
)

# Qwen keeps its own frozen ratio reference. Changing LLaVA's budget.py later
# therefore cannot silently change Qwen schedules.
_QWEN_REFERENCE_COUNTS = {
    64: (180, 74, 12),
    128: (361, 154, 21),
    192: (542, 223, 34),
}


def _reference_counts(average_tokens: int):
    try:
        average_tokens = int(average_tokens)
    except (TypeError, ValueError) as exc:
        raise TypeError("Qwen D-QECG budget must be an integer.") from exc
    if average_tokens not in _QWEN_REFERENCE_COUNTS:
        raise ValueError("Qwen D-QECG budget must be one of 64, 128 or 192.")
    return average_tokens, _QWEN_REFERENCE_COUNTS[average_tokens]


def derive_qwen2_5_vl_schedule(
    average_tokens: int,
    *,
    layer_count: int,
    patch_count=None,
) -> DQECGSchedule:
    """Derive Qwen's schedule without consulting the LLaVA policy."""

    layer_count = int(layer_count)
    if layer_count != QWEN2_5_VL_7B_LAYER_COUNT:
        raise ValueError(
            "The formal Qwen2.5-VL-7B policy requires "
            f"{QWEN2_5_VL_7B_LAYER_COUNT} decoder layers, got {layer_count}."
        )
    average_tokens, reference = _reference_counts(average_tokens)
    reference_entry, reference_stage1, reference_stage2 = reference
    first_stage_ratio = reference_stage1 / reference_entry
    second_stage_ratio = reference_stage2 / reference_entry
    second_to_first_ratio = reference_stage2 / reference_stage1
    first = QWEN2_5_VL_FIRST_PRUNING_LAYER
    second = QWEN2_5_VL_SECOND_PRUNING_LAYER
    drop = QWEN2_5_VL_DROP_LAYER
    coefficient = (
        first
        + (second - first) * first_stage_ratio
        + (drop - second) * second_stage_ratio
    )
    estimate = average_tokens * layer_count / coefficient
    candidates = []
    for entry in range(max(1, round(estimate) - 8), round(estimate) + 9):
        nominal_stage1 = max(1, round(entry * first_stage_ratio))
        for stage1 in range(max(1, nominal_stage1 - 3), nominal_stage1 + 4):
            nominal_stage2 = max(1, round(stage1 * second_to_first_ratio))
            for stage2 in range(max(1, nominal_stage2 - 1), nominal_stage2 + 2):
                if stage2 <= stage1 <= entry:
                    candidates.append((entry, stage1, stage2))

    def error(candidate):
        entry, stage1, stage2 = candidate
        measured = (
            first * entry
            + (second - first) * stage1
            + (drop - second) * stage2
        ) / layer_count
        ratio_error = abs(stage1 / entry - first_stage_ratio) + abs(
            stage2 / stage1 - second_to_first_ratio
        )
        return abs(measured - average_tokens), ratio_error, entry

    entry, stage1, stage2 = min(candidates, key=error)
    if patch_count is not None and entry > int(patch_count):
        raise ValueError(
            f"Qwen D-QECG entry count {entry} exceeds the "
            f"{int(patch_count)} available visual tokens."
        )
    return DQECGSchedule(
        average_tokens=average_tokens,
        entry_tokens=entry,
        after_first_stage=stage1,
        after_second_stage=stage2,
        first_boundary=first,
        second_boundary=second,
        drop_boundary=drop,
        layer_count=layer_count,
    )
