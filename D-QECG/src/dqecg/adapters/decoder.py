"""Shared decoder-loop injection for LLaVA/Llama and Qwen2.5-VL.

Only the decoder loop is replaced. Attention modules, weights, generation and
model-specific RoPE implementations continue to come from Transformers.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..llm_selector import (
    compute_question_conditioned_entropy,
    select_visual_tokens_by_entropy_reduction,
)
from ..iwfc import select_visual_tokens_by_iwfc
from ..profiling import record_stage
from ..pruning import build_visual_position_keep_indices, remap_kept_positions


def _cache_has_tokens(cache) -> bool:
    if cache is None:
        return False
    if isinstance(cache, Cache):
        return cache.get_seq_length() > 0
    return bool(cache) and cache[0][0].shape[-2] > 0


def _trim_cache(cache, keep_indices):
    if cache is None:
        return None
    if isinstance(cache, Cache):
        if not isinstance(cache, DynamicCache) or not hasattr(cache, "layers"):
            raise TypeError("D-QECG requires a mutable DynamicCache during prefill.")
        for layer in cache.layers:
            if layer.keys is not None:
                layer.keys = layer.keys.index_select(-2, keep_indices)
            if layer.values is not None:
                layer.values = layer.values.index_select(-2, keep_indices)
        return cache
    return tuple(
        (
            key_states.index_select(-2, keep_indices),
            value_states.index_select(-2, keep_indices),
        )
        for key_states, value_states in cache
    )


def _select_sequence(tensor, keep_indices):
    if tensor is None:
        return None
    # Llama: [B, S, D] / [B, S]. Qwen mRoPE: [3, B, S, D] / [3, B, S].
    sequence_dim = (
        2 if tensor.dim() >= 3 and tensor.shape[0] in (3, 4) else 1
    )
    return tensor.index_select(sequence_dim, keep_indices)


def _normalize_position_ids(self, position_ids, cache_position, inputs_embeds):
    """Match the native 4.54 LLaMA/Qwen position-id preparation."""

    if self._dqecg_architecture != "qwen2_5_vl":
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        return position_ids

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(
            3, position_ids.shape[0], -1
        )
    # Qwen may prepend text-only positions for packed masking. Rotary embedding
    # consumes only the temporal/height/width triplet.
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        position_ids = position_ids[1:]
    return position_ids


def _create_decoder_causal_mask(
    self,
    attention_mask,
    hidden_states,
    cache_position,
    past_key_values,
    position_ids,
):
    """Build the same mask shape/type as Transformers 4.54 native decoders."""

    if isinstance(attention_mask, dict):
        return attention_mask
    mask_position_ids = (
        position_ids[0]
        if self._dqecg_architecture == "qwen2_5_vl"
        and position_ids.ndim == 3
        else position_ids
    )
    mask_kwargs = {
        "config": self.config,
        "input_embeds": hidden_states,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": mask_position_ids,
    }
    full_mask = create_causal_mask(**mask_kwargs)
    if self._dqecg_architecture != "qwen2_5_vl":
        return full_mask
    masks = {"full_attention": full_mask}
    if getattr(self, "has_sliding_layers", False):
        masks["sliding_attention"] = create_sliding_window_causal_mask(
            **mask_kwargs
        )
    return masks


def _mask_for_layer(causal_mask, decoder_layer):
    if not isinstance(causal_mask, dict):
        return causal_mask
    return causal_mask[decoder_layer.attention_type]


def _layer_position_ids(self, position_ids):
    if self._dqecg_architecture == "qwen2_5_vl" and position_ids.ndim == 3:
        return position_ids[0]
    return position_ids


def _attention_position_ids(self, position_ids):
    """Avoid misclassifying pruned LLaVA tokens as packed FA2 sequences.

    LLaVA's rotary embeddings have already been computed from the preserved
    absolute ``position_ids`` before the decoder loop.  Passing those IDs on to
    Transformers' FlashAttention-2 integration is otherwise interpreted as a
    packed/variable-length layout.  After physical pruning the IDs legitimately
    contain gaps, while the tensors are still one ordinary unpadded sequence;
    forcing the varlen kernel for that representation can access invalid CUDA
    memory.  ``None`` selects the normal fixed-length FA2 kernel without
    changing RoPE.  Qwen keeps its multimodal position IDs unchanged.
    """

    if (
        self._dqecg_architecture == "llava"
        and getattr(self.config, "_attn_implementation", None)
        == "flash_attention_2"
    ):
        return None
    return _layer_position_ids(self, position_ids)


def _attention_mask_for_layer(self, causal_mask, decoder_layer):
    """Keep D-QECG masks internal while selecting fixed-length LLaVA FA2.

    Transformers 4.54 routes FlashAttention-2 through its varlen kernel as
    soon as *any* attention mask is supplied.  D-QECG necessarily constructs a
    fresh 4-D causal mask after each physical token deletion, but the resulting
    batch-size-one sequence is already dense and unpadded.  Passing that mask
    to FA2 therefore misclassifies the representation and can trigger an
    illegal access in ``flash_attn_varlen_func``.  RoPE and pruning continue to
    use the real mask/position IDs above; only the attention backend receives
    ``None`` so it applies its native causal path.
    """

    if (
        self._dqecg_architecture == "llava"
        and getattr(self.config, "_attn_implementation", None)
        == "flash_attention_2"
    ):
        return None
    return _mask_for_layer(causal_mask, decoder_layer)


def _attention_kwargs_for_layer(self, kwargs):
    """Remove packed-sequence metadata from the LLaVA FA2 call boundary."""

    if not (
        self._dqecg_architecture == "llava"
        and getattr(self.config, "_attn_implementation", None)
        == "flash_attention_2"
    ):
        return kwargs
    # Transformers' FA2 helper enters flash_attn_varlen_func when all four of
    # these values are present, even when the explicit attention mask is None.
    # D-QECG prefill is a dense batch-size-one sequence, so these packed-layout
    # hints are both unnecessary and unsafe after physical token deletion.
    return {
        key: value
        for key, value in kwargs.items()
        if key not in {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}
    }


def _debug_to_cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return value.float().cpu() if value.is_floating_point() else value.cpu()
    if isinstance(value, dict):
        return {key: _debug_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_debug_to_cpu(item) for item in value)
    return value


def _state_visual_positions(state, *, device):
    if state.visual_positions is not None:
        return state.visual_positions.to(device=device, dtype=torch.long).view(-1)
    if state.visual_start is None or state.visual_length is None:
        raise RuntimeError("The adapter did not register visual token positions.")
    return torch.arange(
        int(state.visual_start),
        int(state.visual_start) + int(state.visual_length),
        device=device,
        dtype=torch.long,
    )


def _prune_sequence(
    self,
    *,
    selected_visual_indices,
    hidden_states,
    position_ids,
    attention_mask,
    past_key_values,
    position_embeddings,
    use_cache,
):
    state = self._dqecg_state
    visual_positions = _state_visual_positions(state, device=hidden_states.device)
    selected_visual_indices = selected_visual_indices.to(
        device=hidden_states.device, dtype=torch.long
    )
    selected_positions = visual_positions.index_select(0, selected_visual_indices)
    verify_fast_path = os.environ.get("DQECG_VERIFY_PRUNING_FAST_PATH", "0") == "1"
    keep_indices = build_visual_position_keep_indices(
        selected_visual_indices,
        visual_positions=visual_positions,
        seq_length=hidden_states.shape[1],
        device=hidden_states.device,
        validate=False,
    )
    if verify_fast_path:
        reference_keep_indices = build_visual_position_keep_indices(
            selected_visual_indices,
            visual_positions=visual_positions,
            seq_length=hidden_states.shape[1],
            device=hidden_states.device,
            validate=True,
        )
        if not torch.equal(keep_indices, reference_keep_indices):
            raise RuntimeError(
                "Trusted pruning produced different keep indices; refusing the fast path."
            )

    original_next_position_id = state.next_position_id
    if original_next_position_id is None:
        original_next_position_id = int(position_ids.max().item()) + 1

    remapped_question_positions = remap_kept_positions(
        state.question_positions, keep_indices, validate=False
    )
    if verify_fast_path:
        reference_question_positions = remap_kept_positions(
            state.question_positions, keep_indices, validate=True
        )
        if not torch.equal(remapped_question_positions, reference_question_positions):
            raise RuntimeError(
                "Trusted pruning changed remapped question positions; refusing the fast path."
            )
    state.question_positions = remapped_question_positions
    hidden_states = hidden_states.index_select(1, keep_indices)
    position_ids = _select_sequence(position_ids, keep_indices)
    position_embeddings = tuple(
        _select_sequence(embedding, keep_indices)
        for embedding in position_embeddings
    )
    if attention_mask is not None:
        attention_mask = attention_mask.index_select(1, keep_indices)
    if use_cache:
        past_key_values = _trim_cache(past_key_values, keep_indices)
        state.cache_active = True
        state.next_position_id = original_next_position_id

    state.current_visual_indices = state.current_visual_indices.to(
        hidden_states.device
    ).index_select(0, selected_visual_indices)
    remapped_visual_positions = remap_kept_positions(
        selected_positions, keep_indices, validate=False
    )
    if verify_fast_path:
        reference_visual_positions = remap_kept_positions(
            selected_positions, keep_indices, validate=True
        )
        if not torch.equal(remapped_visual_positions, reference_visual_positions):
            raise RuntimeError(
                "Trusted pruning changed remapped visual positions; refusing the fast path."
            )
    state.visual_positions = remapped_visual_positions
    state.visual_start = (
        int(state.visual_positions[0].item()) if state.visual_positions.numel() else None
    )
    state.visual_length = int(selected_visual_indices.numel())
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    causal_mask = _create_decoder_causal_mask(
        self,
        attention_mask,
        hidden_states,
        cache_position,
        # Earlier layers have already populated their cache, but the next
        # layer has no past during a single prefill. Passing the global cache
        # would make 4.54 derive KV length from layer 0 and double the mask.
        None,
        position_ids,
    )
    return (
        hidden_states,
        position_ids,
        attention_mask,
        past_key_values,
        position_embeddings,
        cache_position,
        causal_mask,
    )


def _profiled_prune_sequence(self, *, stage_name, **kwargs):
    """Profile one physical prune only when diagnostics are enabled."""

    with record_stage(stage_name):
        return _prune_sequence(self, **kwargs)


def dqecg_decoder_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """Version-tolerant inference-only decoder loop with physical pruning."""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    state = self._dqecg_state
    prefill = bool(
        state.schedule is not None
        and not _cache_has_tokens(past_key_values)
        and inputs_embeds.shape[1] > 1
    )
    if not prefill:
        return self._dqecg_original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds if input_ids is None else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
    if self.training:
        raise RuntimeError("D-QECG runtime injection is inference-only.")
    if output_attentions:
        raise NotImplementedError(
            "D-QECG physical pruning does not support output_attentions."
        )
    if state.visual_length is None:
        raise RuntimeError("The adapter did not register visual tokens.")
    if state.question_positions is None or len(state.question_positions) < 1:
        raise RuntimeError("D-QECG requires mapped post-image prompt token positions.")
    if len(self.layers) != state.schedule.layer_count:
        raise RuntimeError("Decoder depth changed after D-QECG schedule creation.")

    return_legacy_cache = False
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    elif use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    if self.gradient_checkpointing:
        raise RuntimeError("D-QECG inference does not support gradient checkpointing.")

    if cache_position is None:
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = _normalize_position_ids(
        self, position_ids, cache_position, inputs_embeds
    )
    if state.next_position_id is None:
        # All pruning boundaries preserve the maximum absolute position. Cache
        # it once before the layer loop instead of synchronizing at every prune.
        state.next_position_id = int(position_ids.max().item()) + 1

    hidden_states = inputs_embeds
    # Qwen must preserve placeholder cardinality while its top-level forward
    # scatters visual features. The visual wrapper expands cluster centroids
    # back to M, then this immediate physical selection restores exact N before
    # decoder layer 1. LLaVA already enters here with N tokens.
    if state.original_visual_length and state.original_visual_length != state.visual_length:
        selected = state.pivot_indices.to(hidden_states.device, dtype=torch.long)
        visual_positions = _state_visual_positions(state, device=hidden_states.device)
        selected_positions = visual_positions.index_select(0, selected)
        keep_indices = build_visual_position_keep_indices(
            selected,
            visual_positions=visual_positions,
            seq_length=hidden_states.shape[1],
            device=hidden_states.device,
        )
        state.question_positions = remap_kept_positions(
            state.question_positions, keep_indices
        )
        hidden_states = hidden_states.index_select(1, keep_indices)
        position_ids = _select_sequence(position_ids, keep_indices)
        if attention_mask is not None:
            attention_mask = attention_mask.index_select(1, keep_indices)
        state.visual_positions = remap_kept_positions(selected_positions, keep_indices)
        state.visual_start = int(state.visual_positions[0].item())
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    causal_mask = _create_decoder_causal_mask(
        self,
        attention_mask,
        hidden_states,
        cache_position,
        None,
        position_ids,
    )
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    entropy_history = {}
    selection_debug = {}
    schedule = state.schedule
    policy = self._dqecg_policy
    if policy.architecture != self._dqecg_architecture:
        raise RuntimeError(
            "The injected decoder policy does not match the active architecture."
        )
    first_entropy_layers = policy.first_entropy_layers(schedule)
    second_entropy_layers = policy.second_entropy_layers(schedule)

    capture_layers = set(policy.capture_layers(schedule))
    for layer_index, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if layer_index in capture_layers:
            with record_stage(f"entropy_layer_{layer_index}"):
                entropy_history[layer_index] = compute_question_conditioned_entropy(
                    hidden_states,
                    visual_positions=_state_visual_positions(
                        state, device=hidden_states.device
                    ),
                    question_positions=state.question_positions,
                    eps=self._dqecg_config.eps,
                    positions_are_valid=True,
                )
            if layer_index == schedule.first_boundary:
                history = [entropy_history[index] for index in first_entropy_layers]
                with record_stage("qceg_stage_1"):
                    selected, debug = select_visual_tokens_by_entropy_reduction(
                        history, schedule.after_first_stage
                    )
                state.layer_first_indices = state.current_visual_indices.index_select(
                    0, selected
                ).detach()
                entropy_history = {
                    layer_index: entropy_history[layer_index].index_select(0, selected)
                }
                selection_debug["first_stage"] = debug
                (
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    position_embeddings,
                    cache_position,
                    causal_mask,
                ) = _profiled_prune_sequence(
                    self,
                    stage_name="prune_stage_1",
                    selected_visual_indices=selected,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                )
            elif layer_index == schedule.second_boundary:
                history = [entropy_history[index] for index in second_entropy_layers]
                with record_stage("qceg_stage_2"):
                    _, debug = select_visual_tokens_by_entropy_reduction(
                        history, schedule.after_second_stage
                    )
                visual_hidden = hidden_states[0].index_select(
                    0,
                    _state_visual_positions(
                        state, device=hidden_states.device
                    ),
                )
                with record_stage("iwfc_stage_2"):
                    selected = select_visual_tokens_by_iwfc(
                        visual_hidden,
                        debug["score"],
                        schedule.after_second_stage,
                    )
                debug["qceg_topk_indices"] = debug["selected_indices"]
                debug["selected_indices"] = selected
                debug["selection_method"] = "iwfc"
                state.layer_second_indices = state.current_visual_indices.index_select(
                    0, selected
                ).detach()
                selection_debug["second_stage"] = debug
                (
                    hidden_states,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    position_embeddings,
                    cache_position,
                    causal_mask,
                ) = _profiled_prune_sequence(
                    self,
                    stage_name="prune_stage_2",
                    selected_visual_indices=selected,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    use_cache=use_cache,
                )

        if layer_index == schedule.drop_boundary:
            selected = torch.empty(0, device=hidden_states.device, dtype=torch.long)
            (
                hidden_states,
                position_ids,
                attention_mask,
                past_key_values,
                position_embeddings,
                cache_position,
                causal_mask,
            ) = _profiled_prune_sequence(
                self,
                stage_name="prune_drop",
                selected_visual_indices=selected,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            state.final_visual_indices = state.current_visual_indices.detach()

        with record_stage(f"decoder_layer_{layer_index}"):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=_attention_mask_for_layer(self, causal_mask, decoder_layer),
                position_ids=_attention_position_ids(self, position_ids),
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **_attention_kwargs_for_layer(self, kwargs),
            )
        hidden_states = (
            layer_outputs if isinstance(layer_outputs, torch.Tensor) else layer_outputs[0]
        )

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    if self._dqecg_config.debug:
        selection_debug["schedule"] = state.schedule
        state.debug = _debug_to_cpu(selection_debug)

    next_cache = past_key_values if use_cache else None
    if return_legacy_cache and next_cache is not None:
        next_cache = next_cache.to_legacy_cache()
    if not return_dict:
        return tuple(
            value
            for value in (hidden_states, next_cache, all_hidden_states, all_self_attns)
            if value is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
