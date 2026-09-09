"""Fixed D-squared QCEG-IWFC budget schedule.

The method exposes one user-facing hyperparameter: the target average number
of visual tokens per decoder layer, ``B``. The three stage counts are derived
exactly from the fixed schedule in the method specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DQECGBudget:
    """Token counts implied by one target average budget."""

    average_tokens: int
    entry_tokens: int
    after_first_stage: int
    after_second_stage: int

    # Compatibility aliases for old temporary experiment scripts.  The formal
    # method now prunes at layers 9 and 13, so new code must use the generic
    # stage names above.
    @property
    def after_layer_10(self) -> int:
        return self.after_first_stage

    @property
    def after_layer_14(self) -> int:
        return self.after_second_stage

    @property
    def measured_average_tokens(self) -> float:
        total = (
            9 * self.entry_tokens
            + 4 * self.after_first_stage
            + 11 * self.after_second_stage
        )
        return total / 32


def derive_dqecg_budget(
    average_tokens: int,
    *,
    patch_count: Optional[int] = None,
) -> DQECGBudget:
    """Derive ``N``, ``K9`` and ``K13`` from the sole budget input ``B``.

    The supported paper budgets use the exact layer-average schedules selected
    for the formal QCEG-IWFC method.
    """

    if isinstance(average_tokens, bool):
        raise TypeError("D-QECG average budget must be an integer, not bool.")
    if isinstance(average_tokens, float) and not average_tokens.is_integer():
        raise TypeError("D-QECG average budget must be an integer.")
    try:
        average_tokens = int(average_tokens)
    except (TypeError, ValueError) as exc:
        raise TypeError("D-QECG average budget must be an integer.") from exc

    schedules = {
        64: (180, 74, 12),
        128: (361, 154, 21),
        192: (542, 223, 34),
    }
    if average_tokens not in schedules:
        raise ValueError(
            "D-QECG average budget must be one of 64, 128 or 192."
        )
    entry_tokens, after_first_stage, after_second_stage = schedules[average_tokens]
    budget = DQECGBudget(
        average_tokens=average_tokens,
        entry_tokens=entry_tokens,
        after_first_stage=after_first_stage,
        after_second_stage=after_second_stage,
    )
    if not (
        budget.entry_tokens
        >= budget.after_first_stage
        >= budget.after_second_stage
        > 0
    ):
        raise AssertionError("D-QECG derived token counts are not monotonic.")
    if budget.measured_average_tokens != average_tokens:
        raise AssertionError("D-QECG derived counts do not reproduce the target average.")
    if patch_count is not None and budget.entry_tokens > int(patch_count):
        raise ValueError(
            f"D-QECG budget B={average_tokens} requires N={budget.entry_tokens} "
            f"entry tokens, but the vision tower produced only M={int(patch_count)} patches."
        )
    return budget
