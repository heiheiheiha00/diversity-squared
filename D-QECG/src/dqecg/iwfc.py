"""Importance-weighted feature coverage for formal visual-token selection."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


@torch.no_grad()
def select_visual_tokens_by_iwfc(
    visual_hidden: torch.Tensor,
    qceg_score: torch.Tensor,
    keep_num: int,
) -> torch.Tensor:
    """Select a sorted subset by QCEG-weighted cosine feature coverage."""

    if visual_hidden.dim() != 2:
        raise ValueError("visual_hidden must have shape [visual_tokens, hidden_dim]")
    score = qceg_score.float().view(-1)
    candidate_count = int(visual_hidden.shape[0])
    keep_num = int(keep_num)
    if score.numel() != candidate_count:
        raise ValueError("QCEG score count does not match visual hidden states")
    if not 0 < keep_num <= candidate_count:
        raise ValueError(f"keep_num must be in [1, {candidate_count}]")

    normalized = F.normalize(visual_hidden.float(), dim=-1)
    similarity = torch.clamp(
        normalized @ normalized.transpose(0, 1), min=0.0
    )
    eps = torch.finfo(score.dtype).eps
    weight = 1.0 + score / (score.mean() + eps)
    weight_sum = weight.sum()
    coverage = torch.zeros_like(weight)
    selected_mask = torch.zeros(
        candidate_count, dtype=torch.bool, device=score.device
    )
    selected = []
    optimized_reduction = os.environ.get("DQECG_IWFC_OPTIMIZED", "1") != "0"
    # Reuse one N x N workspace across the greedy iterations.  The original
    # expression allocated both ``gain`` and ``weight * gain`` every step.
    # Weight the reusable workspace in-place before the column reduction.  This
    # avoids the second N x N temporary without routing a transposed strided
    # matrix through cuBLAS SGEMV (which is unstable on some Torch 2.6/cu124
    # builds used with RTX 3090).
    gain_workspace = torch.empty_like(similarity) if optimized_reduction else None
    for _ in range(keep_num):
        if optimized_reduction:
            torch.sub(similarity, coverage[:, None], out=gain_workspace)
            gain_workspace.clamp_(min=0.0)
            gain_workspace.mul_(weight[:, None])
            marginal = gain_workspace.sum(dim=0)
            marginal.div_(weight_sum)
        else:
            gain = torch.clamp(similarity - coverage[:, None], min=0.0)
            marginal = (weight[:, None] * gain).sum(dim=0) / weight_sum
        marginal.masked_fill_(selected_mask, -float("inf"))
        token_index = torch.argmax(marginal)
        selected.append(token_index)
        selected_mask[token_index] = True
        coverage = torch.maximum(coverage, similarity[:, token_index])
    return torch.sort(torch.stack(selected).long()).values
