"""LLM-side secondary visual-token selector for D-squared.

This file implements the second-stage selector only. It captures the target
LLM layer outputs, then returns visual-token-relative indices selected from
tokens that were not chosen by the first visual selector.

It does not prune hidden states, rebuild masks, or modify generation inputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def get_current_llava_decoder_layers(model):
    """Return decoder layers for the current D-squared LLaVA model wrappers."""
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    if hasattr(model, "model"):
        return model.model.layers
    raise AttributeError("Cannot find current LLaVA decoder layers on this model.")


class TargetLayerCapture:
    """Capture attention and output hidden states from one decoder layer."""

    def __init__(self, model, target_layer_idx: int):
        self.model = model
        self.target_layer_idx = int(target_layer_idx)

        self.layers = None
        self.layer = None
        self.self_attn = None
        self.original_attn_forward = None
        self.hook_handle = None

        self.attentions: Optional[torch.Tensor] = None
        self.hidden_states: Optional[torch.Tensor] = None

    def __enter__(self):
        self.layers = get_current_llava_decoder_layers(self.model)
        self.layer = self.layers[self.target_layer_idx]
        if not hasattr(self.layer, "self_attn"):
            raise AttributeError("Target decoder layer has no self_attn module.")

        self.self_attn = self.layer.self_attn
        self.original_attn_forward = self.self_attn.forward
        capture = self

        def wrapped_attn_forward(*args: Any, **kwargs: Any):
            kwargs["output_attentions"] = True
            outputs = capture.original_attn_forward(*args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
                capture.attentions = outputs[1].detach()
            return outputs

        def layer_output_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            capture.hidden_states = hidden.detach()

        self.self_attn.forward = wrapped_attn_forward
        self.hook_handle = self.layer.register_forward_hook(layer_output_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.self_attn is not None and self.original_attn_forward is not None:
            self.self_attn.forward = self.original_attn_forward
        if self.hook_handle is not None:
            self.hook_handle.remove()
        return False


def d_squared_capture_layer_idx(d_squared_agg_layer: int) -> int:
    """Return the layer to capture before pruning at `d_squared_agg_layer`."""
    return max(int(d_squared_agg_layer) - 1, 0)


def _as_1d_long(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    if indices is None or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return indices.to(device=device, dtype=torch.long).view(-1)


def _to_visual_relative_indices(
    indices: torch.Tensor,
    image_token_start_index: int,
    image_token_length: int,
    indices_are_absolute: bool,
    device: torch.device,
) -> torch.Tensor:
    indices = _as_1d_long(indices, device)
    if indices_are_absolute:
        indices = indices - int(image_token_start_index)

    valid = (indices >= 0) & (indices < int(image_token_length))
    return torch.unique(indices[valid], sorted=True)


def _sequence_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize hidden states to [L, D]."""
    if hidden_states.dim() == 3:
        if hidden_states.size(0) != 1:
            raise ValueError("Pass one sample at a time; batched hidden states are not supported here.")
        hidden_states = hidden_states[0]

    if hidden_states.dim() != 2:
        raise ValueError("hidden_states must be [L, D] or [1, L, D].")

    return hidden_states.float()


def _mean_attention(attention: torch.Tensor) -> torch.Tensor:
    """Normalize attention to [L, L] by averaging heads."""
    if attention.dim() == 4:
        if attention.size(0) != 1:
            raise ValueError("Pass one sample at a time; batched attention is not supported here.")
        attention = attention[0]

    if attention.dim() != 3:
        raise ValueError("attention must be [H, L, L] or [1, H, L, L].")

    return attention.float().mean(dim=0)


def _as_token_id_tuple(values) -> Tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().view(-1).tolist()
    if isinstance(values, (int, float)):
        values = [values]
    result = []
    for value in values:
        if value is None:
            continue
        result.append(int(value))
    return tuple(result)


@torch.no_grad()
def compute_query_attention_score(
    attn_mean: torch.Tensor,
    vision_start: int,
    vision_length: int,
    selected_query_positions,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean row-normalized text-to-visual attention score."""
    device = attn_mean.device
    vision_start = int(vision_start)
    vision_length = int(vision_length)

    if not torch.is_tensor(selected_query_positions):
        selected_query_positions = torch.tensor(selected_query_positions, device=device, dtype=torch.long)
    else:
        selected_query_positions = selected_query_positions.to(device=device, dtype=torch.long).view(-1)

    if selected_query_positions.numel() == 0:
        return torch.zeros(vision_length, device=device, dtype=attn_mean.dtype), selected_query_positions

    visual_slice = slice(vision_start, vision_start + vision_length)
    text_to_visual = attn_mean.index_select(0, selected_query_positions)[:, visual_slice]
    text_to_visual = torch.clamp(text_to_visual, min=0.0)
    normalized = text_to_visual / text_to_visual.sum(dim=-1, keepdim=True).clamp_min(eps)
    score = normalized.mean(dim=0)
    return score, selected_query_positions


def build_question_only_positions_by_local_range(
    query_positions,
    local_start: Optional[int],
    local_end: Optional[int],
) -> torch.Tensor:
    """Return query positions for a debug-only local question token range."""
    if local_start is None or local_end is None:
        if torch.is_tensor(query_positions):
            return query_positions.new_empty(0, dtype=torch.long)
        return torch.empty(0, dtype=torch.long)

    if not torch.is_tensor(query_positions):
        query_positions = torch.tensor(query_positions, dtype=torch.long)
    else:
        query_positions = query_positions.view(-1)

    start = max(int(local_start), 0)
    end = min(int(local_end), int(query_positions.numel()) - 1)
    if end < start:
        return query_positions.new_empty(0, dtype=torch.long)
    return query_positions[start : end + 1]


def _sanitize_question_query_positions(question_query_positions, query_positions: torch.Tensor) -> torch.Tensor:
    """Keep only question positions that are valid text queries after the visual block."""
    if question_query_positions is None:
        return query_positions.new_empty(0, dtype=torch.long)
    if not torch.is_tensor(question_query_positions):
        question_query_positions = torch.tensor(
            question_query_positions,
            device=query_positions.device,
            dtype=torch.long,
        )
    else:
        question_query_positions = question_query_positions.to(device=query_positions.device, dtype=torch.long).view(-1)
    if question_query_positions.numel() == 0 or query_positions.numel() == 0:
        return query_positions.new_empty(0, dtype=torch.long)

    min_query = int(query_positions.min().item())
    max_query = int(query_positions.max().item())
    valid = (question_query_positions >= min_query) & (question_query_positions <= max_query)
    return torch.unique(question_query_positions[valid], sorted=True)


def _sanitize_question_positions_for_sequence(
    question_positions,
    *,
    seq_len: int,
    vision_start: int,
    vision_length: int,
    device: torch.device,
) -> torch.Tensor:
    if question_positions is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if not torch.is_tensor(question_positions):
        question_positions = torch.tensor(question_positions, device=device, dtype=torch.long)
    else:
        question_positions = question_positions.to(device=device, dtype=torch.long).view(-1)
    if question_positions.numel() == 0:
        return question_positions
    vision_end = int(vision_start) + int(vision_length)
    valid = (question_positions >= 0) & (question_positions < int(seq_len))
    valid &= (question_positions < int(vision_start)) | (question_positions >= vision_end)
    return torch.unique(question_positions[valid], sorted=True)


@torch.no_grad()
def compute_qceg_simple(
    hidden_states,
    vision_start: int,
    vision_length: int,
    question_positions,
    tau: float = 0.1,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute simple Question-conditioned Cross-layer Entropy Gain."""
    if hidden_states is None or len(hidden_states) < 4:
        raise ValueError("QCEG requires hidden_states[0], [1], [2], [3].")

    device = hidden_states[0].device
    if not torch.is_tensor(question_positions):
        question_positions = torch.tensor(question_positions, device=device, dtype=torch.long)
    else:
        question_positions = question_positions.to(device=device, dtype=torch.long).view(-1)

    if question_positions.numel() < 2:
        raise ValueError("QCEG requires at least 2 question tokens to compute normalized entropy.")

    H_list = []
    vision_start = int(vision_start)
    vision_length = int(vision_length)
    tau = float(tau)

    for layer_idx in range(4):
        X = hidden_states[layer_idx]
        if X.dim() == 3:
            if X.size(0) != 1:
                raise ValueError("QCEG expects one sample at a time.")
            X = X[0]
        if X.dim() != 2:
            raise ValueError("Each QCEG hidden state must be [T, D] or [1, T, D].")

        V_h = X[vision_start : vision_start + vision_length]
        Q_h = X.index_select(0, question_positions)
        V_z = F.normalize(V_h.float(), dim=-1)
        Q_z = F.normalize(Q_h.float(), dim=-1)

        logits = (V_z @ Q_z.transpose(0, 1)) / tau
        p = torch.softmax(logits, dim=-1)
        H = -(p * torch.log(p + eps)).sum(dim=-1)
        denom = torch.log(torch.tensor(p.shape[-1], device=p.device, dtype=p.dtype)) + eps
        H_list.append(H / denom)

    gains = torch.stack(
        [
            torch.clamp(H_list[0] - H_list[1], min=0.0),
            torch.clamp(H_list[1] - H_list[2], min=0.0),
            torch.clamp(H_list[2] - H_list[3], min=0.0),
        ],
        dim=0,
    )
    score = gains.mean(dim=0)
    return {
        "score": score,
        "H": torch.stack(H_list, dim=0),
        "gains": gains,
    }


@torch.no_grad()
def compute_bos_free_query_attention_score(
    attn_mean: torch.Tensor,
    vision_start: int,
    vision_length: int,
    query_positions,
    input_ids: Optional[torch.Tensor] = None,
    special_token_ids=None,
    extra_exclude_token_ids=None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BOS-free text-to-visual attention score."""
    device = attn_mean.device

    if not torch.is_tensor(query_positions):
        query_positions = torch.tensor(query_positions, device=device, dtype=torch.long)
    else:
        query_positions = query_positions.to(device=device, dtype=torch.long).view(-1)

    if query_positions.numel() == 0:
        return torch.zeros(int(vision_length), device=device, dtype=attn_mean.dtype), query_positions

    exclude_ids = set(_as_token_id_tuple(special_token_ids))
    exclude_ids.update(_as_token_id_tuple(extra_exclude_token_ids))

    filtered_query_positions = query_positions
    if input_ids is not None and exclude_ids:
        ids = input_ids[0] if input_ids.dim() == 2 else input_ids
        ids = ids.to(device=device)
        keep = []
        for pos in query_positions.tolist():
            if pos < 0 or pos >= ids.numel():
                keep.append(True)
                continue
            keep.append(int(ids[pos].item()) not in exclude_ids)
        keep = torch.tensor(keep, device=device, dtype=torch.bool)
        filtered_query_positions = query_positions[keep]
        if filtered_query_positions.numel() == 0:
            filtered_query_positions = query_positions

    return compute_query_attention_score(
        attn_mean=attn_mean,
        vision_start=vision_start,
        vision_length=vision_length,
        selected_query_positions=filtered_query_positions,
        eps=eps,
    )


@torch.no_grad()
def select_llm_secondary_visual_token_indices(
    attention: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    first_selected_indices: torch.Tensor,
    keep_num: int,
    image_token_start_index: int,
    image_token_length: int,
    candidate_ratio: float = 2.0,
    top_m: Optional[int] = 8,
    first_indices_are_absolute: bool = False,
    sort_indices: bool = True,
    eps: float = 1e-6,
    input_ids: Optional[torch.Tensor] = None,
    special_token_ids=None,
    extra_exclude_token_ids=None,
    question_local_start: Optional[int] = None,
    question_local_end: Optional[int] = None,
    question_query_positions=None,
    qceg_hidden_states=None,
    second_stage_method: str = "qceg",
    qceg_tau: float = 0.1,
    query_score_mode: str = "question_only",
    return_debug: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Return second-stage visual-token indices.

    Returned indices are relative to the visual-token block. They exclude all
    indices from `first_selected_indices`. The default QCEG path directly uses
    question-conditioned entropy-gain TopK; attention/diversity remains only for
    fallback and ablation modes.
    """
    device = hidden_states.device
    image_token_length = int(image_token_length)
    keep_num = int(keep_num)

    if keep_num <= 0 or image_token_length <= 0:
        selected = torch.empty(0, dtype=torch.long, device=device)
        return (selected, {}) if return_debug else selected

    first_selected = _to_visual_relative_indices(
        first_selected_indices,
        image_token_start_index=image_token_start_index,
        image_token_length=image_token_length,
        indices_are_absolute=first_indices_are_absolute,
        device=device,
    )

    available_mask = torch.ones(image_token_length, device=device, dtype=torch.bool)
    if first_selected.numel() > 0:
        available_mask[first_selected] = False
    available_indices = torch.arange(image_token_length, device=device, dtype=torch.long)[available_mask]

    if available_indices.numel() == 0:
        selected = torch.empty(0, dtype=torch.long, device=device)
        debug = {"first_selected_indices": first_selected}
        return (selected, debug) if return_debug else selected

    keep_num = min(keep_num, available_indices.numel())
    debug: Dict[str, torch.Tensor] = {"first_selected_indices": first_selected, "available_indices": available_indices}

    qceg_result = None
    selection_score_mode = "none"
    if qceg_hidden_states is not None:
        qceg_seq = qceg_hidden_states[0]
        if qceg_seq.dim() == 3:
            qceg_seq_len = int(qceg_seq.size(1))
        else:
            qceg_seq_len = int(qceg_seq.size(0))
        qceg_question_positions = _sanitize_question_positions_for_sequence(
            question_query_positions,
            seq_len=qceg_seq_len,
            vision_start=image_token_start_index,
            vision_length=image_token_length,
            device=device,
        )
    else:
        qceg_question_positions = torch.empty(0, dtype=torch.long, device=device)

    method = str(second_stage_method or "qceg").lower().replace("-", "_")
    if method == "qceg" and qceg_hidden_states is not None and qceg_question_positions.numel() >= 2:
        try:
            qceg_result = compute_qceg_simple(
                hidden_states=qceg_hidden_states,
                vision_start=image_token_start_index,
                vision_length=image_token_length,
                question_positions=qceg_question_positions,
                tau=qceg_tau,
                eps=eps,
            )
        except ValueError:
            qceg_result = None

    if qceg_result is not None:
        selection_score = qceg_result["score"].to(device=device)
        selection_score_mode = "qceg"
        score_for_select = selection_score.clone()
        score_for_select[~available_mask] = -float("inf")
        topk = min(keep_num, int(available_mask.sum().item()))
        selected_ranked_indices = torch.argsort(score_for_select, descending=True)[:topk].to(dtype=torch.long)
        selected_indices = selected_ranked_indices
        selected_indices = torch.sort(selected_indices).values if sort_indices else selected_indices
        if return_debug:
            raw_query_score = torch.zeros(image_token_length, device=device)
            bos_free_score = torch.zeros(image_token_length, device=device)
            question_only_score = torch.zeros(image_token_length, device=device)
            raw_query_positions = torch.empty(0, dtype=torch.long, device=device)
            filtered_query_positions = torch.empty(0, dtype=torch.long, device=device)
            attention_question_positions = torch.empty(0, dtype=torch.long, device=device)
            if attention is not None:
                attn_mean = _mean_attention(attention).to(device=device)
                seq_len = attn_mean.size(-1)
                visual_start = int(image_token_start_index)
                visual_end = visual_start + image_token_length
                query_positions = torch.arange(visual_end, seq_len, device=device, dtype=torch.long)
                raw_query_score, raw_query_positions = compute_query_attention_score(
                    attn_mean=attn_mean,
                    vision_start=visual_start,
                    vision_length=image_token_length,
                    selected_query_positions=query_positions,
                    eps=eps,
                )
                bos_free_score, filtered_query_positions = compute_bos_free_query_attention_score(
                    attn_mean=attn_mean,
                    vision_start=visual_start,
                    vision_length=image_token_length,
                    query_positions=query_positions,
                    input_ids=input_ids,
                    special_token_ids=special_token_ids,
                    extra_exclude_token_ids=extra_exclude_token_ids,
                    eps=eps,
                )
                attention_question_positions = _sanitize_question_query_positions(question_query_positions, query_positions)
                question_only_score, attention_question_positions = compute_query_attention_score(
                    attn_mean=attn_mean,
                    vision_start=visual_start,
                    vision_length=image_token_length,
                    selected_query_positions=attention_question_positions,
                    eps=eps,
                )
            debug.update(
                {
                    "qceg_score": qceg_result["score"],
                    "qceg_H": qceg_result["H"],
                    "qceg_gains": qceg_result["gains"],
                    "qceg_top_indices": selected_ranked_indices,
                    "bos_free_score": bos_free_score,
                    "raw_query_score": raw_query_score,
                    "all_query_score": raw_query_score,
                    "question_only_score": question_only_score,
                    "selection_score": selection_score,
                    "selection_score_mode": selection_score_mode,
                    "filtered_query_positions": filtered_query_positions,
                    "raw_query_positions": raw_query_positions,
                    "all_query_positions": raw_query_positions,
                    "question_query_positions": qceg_question_positions,
                    "attention_question_positions": attention_question_positions,
                    "coarse_candidate_indices": selected_indices,
                    "second_selected_indices": selected_indices,
                }
            )
        return (selected_indices, debug) if return_debug else selected_indices

    if attention is None:
        candidate_indices = available_indices
        debug["bos_free_score"] = torch.zeros(image_token_length, device=device)
        debug["all_query_score"] = torch.zeros(image_token_length, device=device)
        debug["question_only_score"] = torch.zeros(image_token_length, device=device)
        debug["selection_score"] = torch.zeros(image_token_length, device=device)
        debug["selection_score_mode"] = selection_score_mode
        debug["filtered_query_positions"] = torch.empty(0, dtype=torch.long, device=device)
        debug["question_query_positions"] = torch.empty(0, dtype=torch.long, device=device)
    else:
        attn_mean = _mean_attention(attention).to(device=device)
        seq_len = attn_mean.size(-1)
        visual_start = int(image_token_start_index)
        visual_end = visual_start + image_token_length
        if visual_start < 0 or visual_end > seq_len:
            raise ValueError("Visual token range is outside the attention sequence length.")
        query_positions = torch.arange(visual_end, seq_len, device=device, dtype=torch.long)
        explicit_question_positions = _sanitize_question_query_positions(question_query_positions, query_positions)
        if explicit_question_positions.numel() > 0:
            question_query_positions = explicit_question_positions
        else:
            question_query_positions = build_question_only_positions_by_local_range(
                query_positions,
                local_start=question_local_start,
                local_end=question_local_end,
            ).to(device=device, dtype=torch.long)

        bos_free_score = None
        filtered_query_positions = torch.empty(0, dtype=torch.long, device=device)
        raw_query_score = None
        raw_query_positions = query_positions
        question_only_score = None
        normalized_mode = str(query_score_mode or "bos_free").lower().replace("-", "_")
        if normalized_mode in {"question", "question_only"} and question_query_positions.numel() > 0:
            question_only_score, question_query_positions = compute_query_attention_score(
                attn_mean=attn_mean,
                vision_start=visual_start,
                vision_length=image_token_length,
                selected_query_positions=question_query_positions,
                eps=eps,
            )
            selection_score = question_only_score
            selection_score_mode = "question_only"
        elif normalized_mode in {"all", "all_query", "raw", "raw_query"}:
            raw_query_score, raw_query_positions = compute_query_attention_score(
                attn_mean=attn_mean,
                vision_start=visual_start,
                vision_length=image_token_length,
                selected_query_positions=query_positions,
                eps=eps,
            )
            selection_score = raw_query_score
            selection_score_mode = "all_query"
        else:
            bos_free_score, filtered_query_positions = compute_bos_free_query_attention_score(
                attn_mean=attn_mean,
                vision_start=visual_start,
                vision_length=image_token_length,
                query_positions=query_positions,
                input_ids=input_ids,
                special_token_ids=special_token_ids,
                extra_exclude_token_ids=extra_exclude_token_ids,
                eps=eps,
            )
            selection_score = bos_free_score
            selection_score_mode = "bos_free"

        available_scores = selection_score.index_select(0, available_indices)
        candidate_num = min(max(keep_num, math.ceil(float(candidate_ratio) * keep_num)), available_indices.numel())
        candidate_local = torch.topk(available_scores, k=candidate_num, largest=True).indices
        candidate_indices = available_indices[candidate_local]
        if return_debug:
            if bos_free_score is None:
                bos_free_score, filtered_query_positions = compute_bos_free_query_attention_score(
                    attn_mean=attn_mean,
                    vision_start=visual_start,
                    vision_length=image_token_length,
                    query_positions=query_positions,
                    input_ids=input_ids,
                    special_token_ids=special_token_ids,
                    extra_exclude_token_ids=extra_exclude_token_ids,
                    eps=eps,
                )
            if raw_query_score is None:
                raw_query_score, raw_query_positions = compute_query_attention_score(
                    attn_mean=attn_mean,
                    vision_start=visual_start,
                    vision_length=image_token_length,
                    selected_query_positions=query_positions,
                    eps=eps,
                )
            if question_only_score is None:
                question_only_score, question_query_positions = compute_query_attention_score(
                    attn_mean=attn_mean,
                    vision_start=visual_start,
                    vision_length=image_token_length,
                    selected_query_positions=question_query_positions,
                    eps=eps,
                )
            debug.update(
                {
                    "qceg_score": torch.zeros(image_token_length, device=device),
                    "qceg_H": torch.zeros(4, image_token_length, device=device),
                    "qceg_gains": torch.zeros(3, image_token_length, device=device),
                    "bos_free_score": bos_free_score,
                    "raw_query_score": raw_query_score,
                    "all_query_score": raw_query_score,
                    "question_only_score": question_only_score,
                    "selection_score": selection_score,
                    "selection_score_mode": selection_score_mode,
                    "filtered_query_positions": filtered_query_positions,
                    "raw_query_positions": raw_query_positions,
                    "all_query_positions": query_positions,
                    "question_query_positions": question_query_positions,
                    "coarse_candidate_indices": candidate_indices,
                }
            )

    states = _sequence_hidden_states(hidden_states).to(device=device)
    visual_start = int(image_token_start_index)
    visual_end = visual_start + image_token_length
    if visual_start < 0 or visual_end > states.size(0):
        raise ValueError("Visual token range is outside the hidden-state sequence length.")

    visual_states = F.normalize(states[visual_start:visual_end], dim=-1)

    selected = []
    if first_selected.numel() > 0:
        reference_indices = first_selected
    elif attention is not None and candidate_indices.numel() > 0:
        first_idx = candidate_indices[0]
        selected.append(first_idx)
        reference_indices = torch.stack(selected)
    else:
        available_states = visual_states.index_select(0, available_indices)
        available_distance = 1.0 - available_states @ available_states.transpose(0, 1)
        available_distance.clamp_(min=0.0, max=2.0)
        available_diag = torch.arange(available_indices.numel(), device=device)
        available_distance[available_diag, available_diag] = float("inf")
        first_local = torch.argmax(available_distance.min(dim=1).values)
        first_idx = available_indices[first_local]
        selected.append(first_idx)
        reference_indices = torch.stack(selected)

    candidate_states = visual_states.index_select(0, candidate_indices)
    reference_states = visual_states.index_select(0, reference_indices)
    min_dist_to_selected = (1.0 - candidate_states @ reference_states.transpose(0, 1)).min(dim=1).values

    selected_candidate_mask = torch.zeros(candidate_indices.numel(), device=device, dtype=torch.bool)
    if selected:
        selected_candidate_mask |= candidate_indices == selected[0]
        min_dist_to_selected[selected_candidate_mask] = -1.0

    while len(selected) < keep_num:
        next_candidate_pos = torch.argmax(min_dist_to_selected)
        if min_dist_to_selected[next_candidate_pos] < 0:
            break

        next_idx = candidate_indices[next_candidate_pos]
        selected.append(next_idx)
        selected_candidate_mask[next_candidate_pos] = True

        new_distance = 1.0 - candidate_states @ visual_states[next_idx].unsqueeze(-1)
        torch.minimum(min_dist_to_selected, new_distance.squeeze(-1), out=min_dist_to_selected)
        min_dist_to_selected[selected_candidate_mask] = -1.0

    if not selected:
        selected_indices = torch.empty(0, dtype=torch.long, device=device)
        if return_debug:
            debug["second_selected_indices"] = selected_indices
            return selected_indices, debug
        return selected_indices

    selected_indices = torch.stack(selected).to(dtype=torch.long)
    selected_indices = torch.sort(selected_indices).values if sort_indices else selected_indices
    if return_debug:
        debug["second_selected_indices"] = selected_indices
        return selected_indices, debug
    return selected_indices


@torch.no_grad()
def select_llm_secondary_from_capture(
    capture: TargetLayerCapture,
    first_selected_indices: torch.Tensor,
    keep_num: int,
    image_token_start_index: int,
    image_token_length: int,
    candidate_ratio: float = 2.0,
    top_m: Optional[int] = 8,
    first_indices_are_absolute: bool = False,
    sort_indices: bool = True,
    eps: float = 1e-6,
    input_ids: Optional[torch.Tensor] = None,
    special_token_ids=None,
    extra_exclude_token_ids=None,
    question_local_start: Optional[int] = None,
    question_local_end: Optional[int] = None,
    question_query_positions=None,
    qceg_hidden_states=None,
    second_stage_method: str = "qceg",
    qceg_tau: float = 0.1,
    query_score_mode: str = "question_only",
    return_debug: bool = False,
) -> torch.Tensor:
    """Run secondary selection from a populated `TargetLayerCapture`."""
    if capture.attentions is None:
        raise RuntimeError("TargetLayerCapture did not capture attention.")
    if capture.hidden_states is None:
        raise RuntimeError("TargetLayerCapture did not capture hidden states.")

    return select_llm_secondary_visual_token_indices(
        attention=capture.attentions,
        hidden_states=capture.hidden_states,
        first_selected_indices=first_selected_indices,
        keep_num=keep_num,
        image_token_start_index=image_token_start_index,
        image_token_length=image_token_length,
        candidate_ratio=candidate_ratio,
        top_m=top_m,
        first_indices_are_absolute=first_indices_are_absolute,
        sort_indices=sort_indices,
        eps=eps,
        input_ids=input_ids,
        special_token_ids=special_token_ids,
        extra_exclude_token_ids=extra_exclude_token_ids,
        question_local_start=question_local_start,
        question_local_end=question_local_end,
        question_query_positions=question_query_positions,
        qceg_hidden_states=qceg_hidden_states,
        second_stage_method=second_stage_method,
        qceg_tau=qceg_tau,
        query_score_mode=query_score_mode,
        return_debug=return_debug,
    )


@torch.no_grad()
def combine_first_and_second_indices(
    first_selected_indices: torch.Tensor,
    second_selected_indices: torch.Tensor,
    sort_indices: bool = True,
) -> torch.Tensor:
    """Combine first-stage and second-stage visual-token-relative indices."""
    if first_selected_indices.numel() == 0:
        combined = second_selected_indices.to(dtype=torch.long)
    elif second_selected_indices.numel() == 0:
        combined = first_selected_indices.to(dtype=torch.long)
    else:
        combined = torch.cat([first_selected_indices.long(), second_selected_indices.long()])
    return torch.unique(combined, sorted=sort_indices)
