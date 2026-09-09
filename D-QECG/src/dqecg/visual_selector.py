"""Vision-side pivot selection and full-coverage feature aggregation.

The main D-squared QCEG-Merge path uses deterministic farthest-point pivots,
assigns every patch to its nearest pivot, and averages each complete cluster
before the multimodal projector. The older index-only selector remains below
for visualization and ablation compatibility.
"""

from __future__ import annotations

import math
import os
from typing import List, Union

import torch
import torch.nn.functional as F


VisualTokenOutput = Union[torch.Tensor, List[torch.Tensor]]
_DENSE_FPS_MAX_TOKENS = int(
    os.environ.get("DQECG_DENSE_FPS_MAX_TOKENS", "2048")
)
if _DENSE_FPS_MAX_TOKENS < 0:
    raise ValueError("DQECG_DENSE_FPS_MAX_TOKENS must be non-negative")


def _dense_fps_greedy(
    dense_distance: torch.Tensor,
    first_pivot: torch.Tensor,
    output_count: int,
) -> torch.Tensor:
    """Run the exact FPS recurrence on the tensor's current device.

    The recurrence only consumes the already-computed FP32 distance matrix.
    Consequently, moving that matrix to CPU changes neither the distance
    values nor the algorithm.  ``torch.argmax`` uses the first maximum on both
    CPU and CUDA, preserving the deterministic tie rule.
    """

    pivots = torch.empty(
        int(output_count), dtype=torch.long, device=dense_distance.device
    )
    first_pivot = first_pivot.to(device=dense_distance.device, dtype=torch.long)
    pivots[0] = first_pivot
    min_distance = dense_distance[:, first_pivot].clone()
    min_distance.clamp_(min=0.0, max=2.0)
    min_distance[first_pivot] = -1.0

    for pivot_position in range(1, int(output_count)):
        next_pivot = torch.argmax(min_distance)
        pivots[pivot_position] = next_pivot
        torch.minimum(
            min_distance,
            dense_distance[:, next_pivot],
            out=min_distance,
        )
        # Every previously selected entry is already -1 and minimum(-1, d)
        # remains -1 because the clamped distances are non-negative.  Updating
        # only the new pivot is exactly equivalent to rewriting the full pivot
        # prefix, while avoiding a growing advanced-index operation each round.
        min_distance[next_pivot] = -1.0
    return pivots


def _select_dense_fps_pivots(
    dense_distance: torch.Tensor,
    first_pivot: torch.Tensor,
    output_count: int,
) -> torch.Tensor:
    """Select dense FPS pivots with an optional exact CPU launch-reduction path."""

    backend = os.environ.get("DQECG_FPS_GREEDY_BACKEND", "cuda").strip().lower()
    if backend not in {"cuda", "cpu", "verify_cpu", "numpy", "verify_numpy"}:
        raise ValueError(
            "DQECG_FPS_GREEDY_BACKEND must be one of: "
            "cuda, cpu, verify_cpu, numpy, verify_numpy"
        )
    if backend == "cuda" or dense_distance.device.type == "cpu":
        return _dense_fps_greedy(dense_distance, first_pivot, output_count)

    # Copy the exact computed FP32 matrix rather than recomputing similarities
    # on CPU.  This is an execution-placement optimization, not an algorithmic
    # approximation.
    cpu_distance = dense_distance.detach().to(device="cpu", copy=True)
    cpu_first = first_pivot.detach().to(device="cpu", dtype=torch.long)
    if backend in {"numpy", "verify_numpy"}:
        # NumPy avoids hundreds of PyTorch dispatcher/profiler calls for these
        # tiny 576-element reductions.  It consumes the exact copied FP32
        # matrix and follows the same first-maximum argmax and elementwise-min
        # recurrence as the CUDA reference.
        import numpy as np

        distance_array = cpu_distance.numpy()
        pivots_array = np.empty(int(output_count), dtype=np.int64)
        pivot = int(cpu_first)
        pivots_array[0] = pivot
        min_distance = distance_array[:, pivot].copy()
        np.clip(min_distance, 0.0, 2.0, out=min_distance)
        min_distance[pivot] = -1.0
        for pivot_position in range(1, int(output_count)):
            pivot = int(np.argmax(min_distance))
            pivots_array[pivot_position] = pivot
            np.minimum(
                min_distance,
                distance_array[:, pivot],
                out=min_distance,
            )
            min_distance[pivot] = -1.0
        cpu_pivots = torch.from_numpy(pivots_array)
    else:
        cpu_pivots = _dense_fps_greedy(cpu_distance, cpu_first, output_count)
    result = cpu_pivots.to(device=dense_distance.device, dtype=torch.long)

    if backend in {"verify_cpu", "verify_numpy"}:
        cuda_reference = _dense_fps_greedy(
            dense_distance, first_pivot, output_count
        )
        if not torch.equal(result, cuda_reference):
            mismatch = torch.where(result != cuda_reference)[0]
            first_mismatch = int(mismatch[0].detach().cpu())
            raise RuntimeError(
                "CPU and CUDA FPS selected different pivots at selection step "
                f"{first_mismatch}; refusing to use the optimized path."
            )
    return result


def _select_farthest_point_pivots_impl(
    visual_features: torch.Tensor,
    output_count: int,
    eps: float,
    dense_max_tokens: int,
):
    """Return pivots plus reusable normalized/dense similarity tensors."""

    if visual_features.dim() != 2:
        raise ValueError("visual_features must have shape [M, D].")
    patch_count = int(visual_features.size(0))
    output_count = int(output_count)
    if output_count <= 0:
        raise ValueError("output_count must be positive.")
    if output_count > patch_count:
        raise ValueError(
            f"Cannot select N={output_count} pivots from only M={patch_count} patches."
        )
    normalized = F.normalize(visual_features.float(), dim=-1, eps=float(eps))
    normalized_mean = normalized.mean(dim=0, keepdim=True)
    first_pivot = torch.argmin(torch.norm(normalized - normalized_mean, dim=-1))
    # LLaVA has only 576 patches: one dense GEMM is much cheaper than up to
    # 542 sequential 576xD matvec launches, and the resulting 1.27 MiB FP32
    # matrix can be reused for cluster assignment. High-resolution Qwen inputs
    # stay on the original O(M)-memory streaming path.
    dense_distance = None
    if patch_count <= int(dense_max_tokens):
        dense_distance = 1.0 - normalized @ normalized.transpose(0, 1)
        dense_distance.clamp_(min=0.0, max=2.0)
        pivots = _select_dense_fps_pivots(
            dense_distance, first_pivot, output_count
        )
    else:
        pivots = torch.empty(
            output_count, dtype=torch.long, device=visual_features.device
        )
        pivots[0] = first_pivot
        min_distance = 1.0 - normalized @ normalized[first_pivot]
        min_distance.clamp_(min=0.0, max=2.0)
        min_distance[first_pivot] = -1.0
        for pivot_position in range(1, output_count):
            next_pivot = torch.argmax(min_distance)
            pivots[pivot_position] = next_pivot
            next_distance = 1.0 - normalized @ normalized[next_pivot]
            next_distance.clamp_(min=0.0, max=2.0)
            torch.minimum(min_distance, next_distance, out=min_distance)
            min_distance[next_pivot] = -1.0

    return torch.sort(pivots).values, normalized, dense_distance


@torch.no_grad()
def select_farthest_point_pivots(
    visual_features: torch.Tensor,
    output_count: int,
    eps: float = 1e-6,
    dense_max_tokens: int = _DENSE_FPS_MAX_TOKENS,
) -> torch.Tensor:
    """Select deterministic farthest-point pivots from ``[M, D]`` features."""

    if visual_features.dim() != 2:
        raise ValueError("visual_features must have shape [M, D].")
    patch_count = int(visual_features.size(0))
    output_count = int(output_count)
    if output_count <= 0:
        raise ValueError("output_count must be positive.")
    if output_count > patch_count:
        raise ValueError(
            f"Cannot select N={output_count} pivots from only M={patch_count} patches."
        )

    pivots, _, _ = _select_farthest_point_pivots_impl(
        visual_features,
        output_count,
        eps,
        dense_max_tokens,
    )
    return pivots


@torch.no_grad()
def aggregate_visual_features(
    visual_features: torch.Tensor,
    output_count: int,
    eps: float = 1e-6,
    return_details: bool = False,
):
    """Aggregate every patch into its nearest deterministic pivot cluster.

    Args:
        visual_features: ``[M, D]`` or ``[B, M, D]`` vision-tower features.
        output_count: Exact number ``N`` of output clusters.
        eps: Numerical normalization constant fixed by the method.
        return_details: Also return pivot indices and patch assignments.
    """

    if visual_features.dim() == 3:
        outputs = [
            aggregate_visual_features(
                sample,
                output_count=output_count,
                eps=eps,
                return_details=True,
            )
            for sample in visual_features
        ]
        aggregated = torch.stack([item[0] for item in outputs], dim=0)
        pivots = torch.stack([item[1] for item in outputs], dim=0)
        assignments = torch.stack([item[2] for item in outputs], dim=0)
        if return_details:
            return aggregated, pivots, assignments
        return aggregated

    if visual_features.dim() != 2:
        raise ValueError("visual_features must have shape [M, D] or [B, M, D].")

    pivots, normalized, dense_distance = _select_farthest_point_pivots_impl(
        visual_features,
        int(output_count),
        float(eps),
        _DENSE_FPS_MAX_TOKENS,
    )
    if dense_distance is None:
        pivot_features = normalized.index_select(0, pivots)
        distance_to_pivots = 1.0 - normalized @ pivot_features.transpose(0, 1)
    else:
        distance_to_pivots = dense_distance.index_select(1, pivots)
    distance_to_pivots.clamp_(min=0.0, max=2.0)
    assignments = torch.argmin(distance_to_pivots, dim=-1)
    assignments[pivots] = torch.arange(
        int(output_count),
        device=assignments.device,
        dtype=assignments.dtype,
    )

    # Accumulate original (not normalized) vision features, then take the
    # unweighted arithmetic mean required by the method.
    accumulator = torch.zeros(
        (int(output_count), visual_features.size(-1)),
        dtype=visual_features.dtype,
        device=visual_features.device,
    )
    accumulator.index_add_(0, assignments, visual_features)
    counts = torch.bincount(assignments, minlength=int(output_count))
    if torch.any(counts == 0):
        raise AssertionError("Each farthest-point pivot must own at least its own patch.")
    aggregated = accumulator / counts.to(
        device=accumulator.device,
        dtype=accumulator.dtype,
    ).unsqueeze(-1)

    if return_details:
        return aggregated, pivots, assignments
    return aggregated


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
