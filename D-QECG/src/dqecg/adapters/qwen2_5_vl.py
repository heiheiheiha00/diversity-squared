"""Qwen2.5-VL adapter using upstream Transformers without vendored model code."""

from __future__ import annotations

from types import MethodType

import torch

from ..policies.qwen2_5_vl import (
    QWEN2_5_VL_DECODER_POLICY,
    derive_qwen2_5_vl_schedule,
)
from ..visual_selector import aggregate_visual_features
from .decoder import dqecg_decoder_forward
from .state import DQECGRuntimeState


def _cache_length(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def _qwen_set_question_positions(self, positions) -> None:
    tensor = torch.as_tensor(positions, dtype=torch.long)
    self.model._dqecg_state.question_positions = tensor
    self.model._dqecg_state.original_question_positions = tensor.detach().clone()


def _qwen_clear_question_positions(self) -> None:
    self.model._dqecg_state.clear_question_positions()


def _qwen_reset(self) -> None:
    self.model._dqecg_state.reset_prefill()


def _qwen_visual_forward(self, *args, **kwargs):
    output = self._dqecg_original_forward(*args, **kwargs)
    state = self._dqecg_owner_state
    features = output[0] if isinstance(output, tuple) else output
    if features.dim() != 2:
        raise NotImplementedError("D-QECG Qwen currently supports one flattened image.")
    schedule = derive_qwen2_5_vl_schedule(
        self._dqecg_config.budget,
        layer_count=self._dqecg_decoder_depth,
        patch_count=features.shape[0],
    )
    centroids, pivots, assignments = aggregate_visual_features(
        features.unsqueeze(0),
        output_count=schedule.entry_tokens,
        eps=self._dqecg_config.eps,
        return_details=True,
    )
    # Qwen scatters one feature per image placeholder before calling its
    # decoder. Expand the centroids back to M here; the decoder immediately
    # keeps the sorted pivot positions, so layer 1 still receives exact N.
    expanded = centroids[0].index_select(0, assignments[0]).to(features.dtype)
    state.schedule = schedule
    state.original_visual_length = int(features.shape[0])
    state.visual_length = schedule.entry_tokens
    state.pivot_indices = pivots[0].detach()
    state.patch_assignments = assignments[0].detach()
    state.current_visual_indices = torch.arange(
        schedule.entry_tokens, device=features.device, dtype=torch.long
    )
    if isinstance(output, tuple):
        return (expanded,) + output[1:]
    return expanded


def _qwen_top_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    **kwargs,
):
    """Preserve Qwen's public forward signature for GenerationMixin.

    Transformers inspects this signature to decide whether it must create
    per-step ``position_ids``.  A generic ``*args, **kwargs`` wrapper makes
    generation omit them and breaks Qwen's mRoPE decode path.
    """
    state = self.model._dqecg_state
    prefill = input_ids is not None and input_ids.shape[1] > 1 and pixel_values is not None
    if prefill:
        if pixel_values_videos is not None:
            raise NotImplementedError("D-QECG Qwen video pruning is not implemented.")
        state.reset_prefill()
        visual_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
        if visual_positions.numel() == 0:
            raise RuntimeError("No Qwen image-token block was found.")
        expected = torch.arange(
            visual_positions[0],
            visual_positions[0] + visual_positions.numel(),
            device=visual_positions.device,
        )
        if not torch.equal(visual_positions, expected):
            raise NotImplementedError("D-QECG Qwen currently supports one image block.")
        state.visual_start = int(visual_positions[0].item())
        state.visual_positions = visual_positions.detach().clone()
        if state.question_positions is None:
            # The Qwen branch uses the complete textual user-prompt suffix
            # after the visual block as entropy query Q. Deriving it here
            # preserves exact processor tokenization and mRoPE positions.
            start = int(visual_positions[-1].item()) + 1
            special = {
                value
                for value in (
                    self.config.pad_token_id,
                    self.config.eos_token_id,
                    getattr(self.config, "vision_start_token_id", None),
                    getattr(self.config, "vision_end_token_id", None),
                )
                if value is not None
            }
            positions = [
                index
                for index in range(start, input_ids.shape[1])
                if int(input_ids[0, index].item()) not in special
            ]
            state.question_positions = torch.tensor(
                positions, device=input_ids.device, dtype=torch.long
            )
            state.original_question_positions = state.question_positions.clone()
    elif state.cache_active and input_ids is not None and input_ids.shape[1] == 1:
        cache_length = _cache_length(past_key_values)
        attention_mask = torch.ones(
            (input_ids.shape[0], cache_length + 1),
            device=input_ids.device,
            dtype=torch.long,
        )
        cache_position = torch.arange(
            cache_length, cache_length + 1, device=input_ids.device
        )
        next_position = state.next_position_id or cache_length
        position_ids = torch.full(
            (3, input_ids.shape[0], 1),
            next_position,
            device=input_ids.device,
            dtype=torch.long,
        )
        state.next_position_id = next_position + 1
    return self._dqecg_original_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        second_per_grid_ts=second_per_grid_ts,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )


def apply_qwen2_5_vl(model, config):
    # Transformers 4.54 nests the text decoder below
    # ``model.language_model`` while exposing ``language_model`` and ``visual``
    # as compatibility properties on the conditional-generation wrapper.
    # Older layouts placed decoder layers directly under ``model.model``.
    backbone = getattr(model, "model", None)
    decoder = getattr(model, "language_model", None)
    if decoder is None and backbone is not None:
        decoder = getattr(backbone, "language_model", backbone)
    visual = getattr(model, "visual", None)
    if visual is None and backbone is not None:
        visual = getattr(backbone, "visual", None)
    if decoder is None or visual is None or not hasattr(decoder, "layers"):
        raise TypeError(
            "Expected Qwen2_5_VLForConditionalGeneration with a text decoder "
            "at model.language_model (Transformers 4.54) or model.model."
        )
    state = DQECGRuntimeState()
    # The top-level forward wrapper reaches state through the multimodal
    # backbone, while the injected decoder loop reads it from the text model.
    if backbone is not None:
        backbone._dqecg_state = state
    decoder._dqecg_config = config
    decoder._dqecg_state = state
    decoder._dqecg_architecture = "qwen2_5_vl"
    decoder._dqecg_policy = QWEN2_5_VL_DECODER_POLICY
    decoder._dqecg_original_forward = decoder.forward
    decoder.forward = MethodType(dqecg_decoder_forward, decoder)

    visual._dqecg_original_forward = visual.forward
    visual._dqecg_owner_state = state
    visual._dqecg_config = config
    visual._dqecg_decoder_depth = len(decoder.layers)
    visual.forward = MethodType(_qwen_visual_forward, visual)

    model._dqecg_original_forward = model.forward
    model.forward = MethodType(_qwen_top_forward, model)
    model.set_dqecg_question_positions = MethodType(
        _qwen_set_question_positions, model
    )
    model.clear_dqecg_question_positions = MethodType(
        _qwen_clear_question_positions, model
    )
    model.reset_dqecg = MethodType(_qwen_reset, model)
    model._dqecg_enabled = True
    model._dqecg_config = config
    return model
