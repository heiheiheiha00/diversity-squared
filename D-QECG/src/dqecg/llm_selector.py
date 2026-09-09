"""Formal prompt-conditioned entropy and QCEG token selection."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _sequence_hidden_states(
    hidden_states: torch.Tensor, *, cast_float32: bool = True
) -> torch.Tensor:
    """Normalize hidden states to ``[sequence, hidden]``."""

    if hidden_states.dim() == 3:
        if hidden_states.size(0) != 1:
            raise ValueError("D-QECG currently processes one sample at a time.")
        hidden_states = hidden_states[0]
    if hidden_states.dim() != 2:
        raise ValueError("hidden_states must be [L, D] or [1, L, D].")
    return hidden_states.float() if cast_float32 else hidden_states


def _sanitize_prompt_positions(
    prompt_positions,
    *,
    seq_len: int,
    vision_start: int = None,
    vision_length: int = None,
    visual_positions: torch.Tensor = None,
    device: torch.device,
) -> torch.Tensor:
    if prompt_positions is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if not torch.is_tensor(prompt_positions):
        prompt_positions = torch.tensor(
            prompt_positions, device=device, dtype=torch.long
        )
    else:
        prompt_positions = prompt_positions.to(
            device=device, dtype=torch.long
        ).view(-1)
    if prompt_positions.numel() == 0:
        return prompt_positions
    valid = (prompt_positions >= 0) & (prompt_positions < int(seq_len))
    if visual_positions is not None:
        visual_positions = visual_positions.to(
            device=device, dtype=torch.long
        ).view(-1)
        valid &= ~torch.isin(prompt_positions, visual_positions)
    else:
        vision_end = int(vision_start) + int(vision_length)
        valid &= (prompt_positions < int(vision_start)) | (
            prompt_positions >= vision_end
        )
    return torch.unique(prompt_positions[valid], sorted=True)


@torch.no_grad()
def compute_question_conditioned_entropy(
    hidden_states: torch.Tensor,
    *,
    vision_start: int = None,
    vision_length: int = None,
    visual_positions: torch.Tensor = None,
    question_positions=None,
    eps: float = 1e-6,
    positions_are_valid: bool = False,
) -> torch.Tensor:
    """Compute normalized prompt-conditioned entropy for each visual token.

    ``question_positions`` is retained as an internal compatibility name for
    the runtime state. In the formal method it contains every textual token in
    the post-image user-prompt suffix, including response-format instructions.
    The logits are normalized cosine similarities with no temperature.
    """

    # Keep the full sequence in its native inference dtype. Only selected rows
    # need FP32 for cosine normalization and entropy, avoiding a large temporary
    # allocation at every capture boundary.
    states = _sequence_hidden_states(hidden_states, cast_float32=False)
    device = states.device
    if visual_positions is None:
        vision_start = int(vision_start)
        vision_length = int(vision_length)
        vision_end = vision_start + vision_length
        if vision_length <= 0 or vision_start < 0 or vision_end > states.size(0):
            raise ValueError("Visual token range is outside the hidden-state sequence.")
        visual_positions = torch.arange(vision_start, vision_end, device=device)
    else:
        visual_positions = visual_positions.to(
            device=device, dtype=torch.long
        ).view(-1)
        if visual_positions.numel() < 1:
            raise ValueError("D-QECG requires at least one visual token.")
        if not positions_are_valid:
            valid_visual = (visual_positions >= 0) & (
                visual_positions < states.size(0)
            )
            if not bool(torch.all(valid_visual)):
                raise ValueError(
                    "Visual token positions are outside the hidden-state sequence."
                )

    if positions_are_valid:
        if not torch.is_tensor(question_positions):
            prompt_positions = torch.tensor(
                question_positions, device=device, dtype=torch.long
            )
        else:
            prompt_positions = question_positions.to(
                device=device, dtype=torch.long
            ).view(-1)
    else:
        prompt_positions = _sanitize_prompt_positions(
            question_positions,
            seq_len=states.size(0),
            visual_positions=visual_positions,
            device=device,
        )
    if prompt_positions.numel() < 1:
        raise ValueError(
            "D-squared QCEG-Merge requires post-image user-prompt tokens."
        )

    visual_states = F.normalize(
        states.index_select(0, visual_positions).float(),
        dim=-1,
        eps=float(eps),
    )
    prompt_states = F.normalize(
        states.index_select(0, prompt_positions).float(),
        dim=-1,
        eps=float(eps),
    )
    probabilities = torch.softmax(
        visual_states @ prompt_states.transpose(0, 1),
        dim=-1,
    )
    entropy = -(
        probabilities * torch.log(probabilities + float(eps))
    ).sum(dim=-1)
    denominator = math.log(prompt_positions.numel()) + float(eps)
    return entropy / denominator


@torch.no_grad()
def compute_positive_entropy_reduction(entropies) -> Dict[str, torch.Tensor]:
    """Average positive entropy drops between consecutive boundaries."""

    if isinstance(entropies, (list, tuple)):
        if len(entropies) < 2:
            raise ValueError("At least two entropy boundaries are required.")
        entropy_stack = torch.stack(list(entropies), dim=0)
    else:
        entropy_stack = entropies
    if entropy_stack.dim() != 2 or entropy_stack.size(0) < 2:
        raise ValueError("entropies must have shape [boundaries, visual_tokens].")
    gains = torch.clamp(entropy_stack[:-1] - entropy_stack[1:], min=0.0)
    return {
        "score": gains.mean(dim=0),
        "entropy": entropy_stack,
        "gains": gains,
    }


@torch.no_grad()
def select_visual_tokens_by_entropy_reduction(
    entropies,
    keep_num: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Select deterministic Top-K tokens from equal-weight positive drops."""

    result = compute_positive_entropy_reduction(entropies)
    score = result["score"]
    keep_num = int(keep_num)
    if keep_num < 0 or keep_num > score.numel():
        raise ValueError(
            f"keep_num must be in [0, {score.numel()}], got {keep_num}."
        )
    if keep_num == 0:
        selected = torch.empty(0, dtype=torch.long, device=score.device)
    else:
        # Stable sorting preserves original token order for equal-score ties.
        ranked = torch.argsort(score, descending=True, stable=True)
        selected = torch.sort(ranked[:keep_num].to(dtype=torch.long)).values
    result["selected_indices"] = selected
    return selected, result
