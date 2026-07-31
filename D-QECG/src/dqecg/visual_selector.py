"""Visual-token selector for the D-QECG pruning module.

This module only selects candidate visual-token indices. It does not remove
tokens from the LLM sequence and does not rebuild attention masks.
"""

from __future__ import annotations

import math
from typing import List, Union

import torch
import torch.nn.functional as F


VisualTokenOutput = Union[torch.Tensor, List[torch.Tensor]]


@torch.no_grad()
def get_projected_visual_tokens_from_original_llava(
    model,
    images: Union[torch.Tensor, List[torch.Tensor]],
) -> VisualTokenOutput:
    """Return original LLaVA visual tokens after vision tower and projector."""
    llava_model = model.get_model() if hasattr(model, "get_model") else model

    vision_tower = (
        model.get_vision_tower()
        if hasattr(model, "get_vision_tower")
        else llava_model.get_vision_tower()
    )
    if not getattr(vision_tower, "is_loaded", True):
        vision_tower.load_model()

    image_features = vision_tower(images)
    if isinstance(image_features, list):
        return [llava_model.mm_projector(features) for features in image_features]

    return llava_model.mm_projector(image_features)


@torch.no_grad()
def select_visual_token_indices(
    visual_tokens: torch.Tensor,
    keep_num: int,
    candidate_ratio: float = 2.0,
    knn: int = 8,
    sort_indices: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Select visual-token indices from projected visual tokens.

    Args:
        visual_tokens: Tensor with shape [N, D] or [B, N, D].
        keep_num: Number of final selected visual tokens.
        candidate_ratio: Candidate multiplier, M = candidate_ratio * keep_num.
        knn: Number of nearest neighbors used for density estimation.
        sort_indices: Whether to sort indices by original visual-token order.
        eps: Tiny random noise scale used to break density ties.

    Returns:
        Tensor with shape [K] for [N, D] input, or [B, K] for batched input.
    """
    if visual_tokens.dim() == 3:
        return torch.stack(
            [
                select_visual_token_indices(
                    visual_tokens=batch_tokens,
                    keep_num=keep_num,
                    candidate_ratio=candidate_ratio,
                    knn=knn,
                    sort_indices=sort_indices,
                    eps=eps,
                )
                for batch_tokens in visual_tokens
            ],
            dim=0,
        )

    if visual_tokens.dim() != 2:
        raise ValueError("visual_tokens must have shape [N, D] or [B, N, D].")

    device = visual_tokens.device
    token_count = visual_tokens.size(0)

    if keep_num <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if keep_num >= token_count:
        return torch.arange(token_count, device=device, dtype=torch.long)

    keep_num = min(int(keep_num), token_count)
    candidate_num = min(max(keep_num, math.ceil(candidate_ratio * keep_num)), token_count)

    # 公式对应：先把 visual token 特征归一化，用 cosine similarity 得到两两相似度，
    # 再转成距离 d(i,j)=1-cos(i,j)，后面的覆盖性和密度都基于这个距离矩阵。
    x = F.normalize(visual_tokens.float(), dim=-1)
    similarity = x @ x.transpose(0, 1)
    distance = 1.0 - similarity
    distance.clamp_(min=0.0, max=2.0)

    candidate_idx = torch.empty(candidate_num, device=device, dtype=torch.long)

    diag = torch.arange(token_count, device=device)
    distance[diag, diag] = float("inf")
    # 公式对应：第一个候选点取“离自己最近邻也尽量远”的点，
    # 即 argmax_i min_{j != i} d(i,j)，保证初始点有较强覆盖性。
    nearest_dist = distance.min(dim=1).values
    first_idx = torch.argmax(nearest_dist)
    distance[diag, diag] = 0.0

    candidate_idx[0] = first_idx
    # 公式对应：min_dist_to_selected[t] = min_{s in S} d(t,s)，
    # 后续每轮选 argmax_t min_dist_to_selected[t]，就是 Max-Min 多样性候选集。
    min_dist_to_selected = distance[first_idx].contiguous()
    min_dist_to_selected[first_idx] = -1.0

    for candidate_pos in range(1, candidate_num):
        next_idx = torch.argmax(min_dist_to_selected)
        candidate_idx[candidate_pos] = next_idx

        # 公式对应：加入新点 s 后，只需增量更新 min(d(t,S), d(t,s))。
        torch.minimum(
            min_dist_to_selected,
            distance[next_idx],
            out=min_dist_to_selected,
        )
        min_dist_to_selected[candidate_idx[: candidate_pos + 1]] = -1.0

    candidate_distance = distance.index_select(0, candidate_idx).index_select(1, candidate_idx)
    candidate_count = candidate_distance.size(0)

    if keep_num >= candidate_count:
        return torch.sort(candidate_idx).values if sort_indices else candidate_idx

    candidate_diag = torch.arange(candidate_count, device=device)
    candidate_distance[candidate_diag, candidate_diag] = float("inf")

    effective_knn = min(int(knn), candidate_count - 1)
    knn_distance = torch.topk(
        candidate_distance,
        k=effective_knn,
        dim=-1,
        largest=False,
    ).values

    candidate_distance[candidate_diag, candidate_diag] = 0.0

    # 公式对应：局部密度 rho_i = exp(-mean_{j in KNN(i)} d(i,j)^2)，
    # KNN 距离越小，rho 越大，表示候选 token 位于更密集区域。
    rho = torch.exp(-(knn_distance * knn_distance).mean(dim=-1))
    if eps > 0:
        rho.add_(torch.rand_like(rho) * eps)

    # 公式对应：delta_i 是 token i 到“密度比它更高的 token”的最小距离；
    # 如果没有更高密度点，则用它到其他点的最大距离。
    higher_density = rho.unsqueeze(0) > rho.unsqueeze(1)
    max_distance = candidate_distance.max()
    delta_matrix = torch.where(
        higher_density,
        candidate_distance,
        torch.full_like(candidate_distance, max_distance),
    )
    delta = delta_matrix.min(dim=-1).values

    has_higher_density = higher_density.any(dim=-1)
    max_dist_to_others = candidate_distance.max(dim=-1).values
    delta = torch.where(has_higher_density, delta, max_dist_to_others)

    # 公式对应：density peak 最终分数 gamma_i = rho_i * delta_i，
    # 取 top-k 作为第一阶段保留的 visual token index。
    score = rho * delta
    selected_local = torch.topk(score, k=keep_num, largest=True).indices
    selected_idx = candidate_idx[selected_local]

    return torch.sort(selected_idx).values if sort_indices else selected_idx


@torch.no_grad()
def select_from_original_llava(
    model,
    images: Union[torch.Tensor, List[torch.Tensor]],
    keep_num: int,
    candidate_ratio: float = 2.0,
    knn: int = 8,
    sort_indices: bool = True,
    eps: float = 1e-6,
) -> VisualTokenOutput:
    """Extract projected visual tokens from original LLaVA and return indices."""
    visual_tokens = get_projected_visual_tokens_from_original_llava(model, images)
    if isinstance(visual_tokens, list):
        return [
            select_visual_token_indices(tokens, keep_num, candidate_ratio, knn, sort_indices, eps)
            for tokens in visual_tokens
        ]

    return select_visual_token_indices(
        visual_tokens=visual_tokens,
        keep_num=keep_num,
        candidate_ratio=candidate_ratio,
        knn=knn,
        sort_indices=sort_indices,
        eps=eps,
    )
