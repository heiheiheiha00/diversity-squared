"""LLM-side secondary visual-token selector for D-squared.

This file implements the second-stage selector only. It captures the target
LLM layer outputs, then returns visual-token-relative indices selected from
tokens that were not chosen by the first visual selector.

It does not prune hidden states, rebuild masks, or modify generation inputs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

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


def _mean_attention(attention: torch.Tensor) -> torch.Tensor:
    """Normalize attention to [L, L] by averaging heads."""
    if attention.dim() == 4:
        if attention.size(0) != 1:
            raise ValueError("Pass one sample at a time; batched attention is not supported here.")
        attention = attention[0]

    if attention.dim() != 3:
        raise ValueError("attention must be [H, L, L] or [1, H, L, L].")

    return attention.float().mean(dim=0)


def _sequence_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize hidden states to [L, D]."""
    if hidden_states.dim() == 3:
        if hidden_states.size(0) != 1:
            raise ValueError("Pass one sample at a time; batched hidden states are not supported here.")
        hidden_states = hidden_states[0]

    if hidden_states.dim() != 2:
        raise ValueError("hidden_states must be [L, D] or [1, L, D].")

    return hidden_states.float()


@torch.no_grad()
def sink_suppressed_residual_attention_scores(
    attention: torch.Tensor,
    image_token_start_index: int,
    image_token_length: int,
    top_m: Optional[int] = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute residual attention score c_i for each visual token."""
    attn = _mean_attention(attention)
    seq_len = attn.size(-1)
    visual_start = int(image_token_start_index)
    visual_end = visual_start + int(image_token_length)

    if visual_start < 0 or visual_end > seq_len:
        raise ValueError("Visual token range is outside the attention sequence length.")

    query_indices = torch.arange(visual_end, seq_len, device=attn.device)
    if query_indices.numel() == 0:
        return torch.zeros(image_token_length, device=attn.device, dtype=attn.dtype)

    text_to_visual = attn.index_select(0, query_indices)[:, visual_start:visual_end]
    text_to_visual = text_to_visual.clamp_min(0.0)
    per_query = text_to_visual / text_to_visual.sum(dim=-1, keepdim=True).clamp_min(eps)

    sink_bias = per_query.mean(dim=0, keepdim=True)
    residual = torch.clamp(per_query - sink_bias, min=0.0)

    effective_top_m = residual.size(0) if top_m is None else min(int(top_m), residual.size(0))
    if effective_top_m <= 0:
        return torch.zeros(image_token_length, device=attn.device, dtype=attn.dtype)

    return torch.topk(residual, k=effective_top_m, dim=0, largest=True).values.mean(dim=0)


@torch.no_grad()
def select_llm_secondary_visual_token_indices(
    attention: torch.Tensor,
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
) -> torch.Tensor:
    """Return second-stage visual-token indices.

    Returned indices are relative to the visual-token block. They exclude all
    indices from `first_selected_indices`.
    """
    device = attention.device
    image_token_length = int(image_token_length)
    keep_num = int(keep_num)

    if keep_num <= 0 or image_token_length <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

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
        return torch.empty(0, dtype=torch.long, device=device)

    keep_num = min(keep_num, available_indices.numel())

    residual_scores = sink_suppressed_residual_attention_scores(
        attention=attention,
        image_token_start_index=image_token_start_index,
        image_token_length=image_token_length,
        top_m=top_m,
        eps=eps,
    )

    available_scores = residual_scores.index_select(0, available_indices)
    candidate_num = min(max(keep_num, math.ceil(candidate_ratio * keep_num)), available_indices.numel())
    candidate_local = torch.topk(available_scores, k=candidate_num, largest=True).indices
    candidate_indices = available_indices[candidate_local]

    states = _sequence_hidden_states(hidden_states).to(device=device)
    visual_start = int(image_token_start_index)
    visual_end = visual_start + image_token_length
    if visual_start < 0 or visual_end > states.size(0):
        raise ValueError("Visual token range is outside the hidden-state sequence length.")

    visual_states = F.normalize(states[visual_start:visual_end], dim=-1)

    selected = []
    if first_selected.numel() > 0:
        reference_indices = first_selected
    else:
        first_local = torch.argmax(available_scores)
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
        return torch.empty(0, dtype=torch.long, device=device)

    selected_indices = torch.stack(selected).to(dtype=torch.long)
    return torch.sort(selected_indices).values if sort_indices else selected_indices


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
