"""Instance-local LLaVA adapter following Nuwa's runtime injection pattern."""

from __future__ import annotations

from types import MethodType

import torch

from ..policies.llava import LLAVA_DECODER_POLICY, derive_llava_schedule
from ..profiling import record_stage
from ..visual_selector import aggregate_visual_features
from .decoder import dqecg_decoder_forward
from .state import DQECGRuntimeState


def _cache_length(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def _llava_set_question_positions(self, positions) -> None:
    decoder = self.get_model()
    tensor = torch.as_tensor(positions, dtype=torch.long)
    decoder._dqecg_state.question_positions = tensor
    decoder._dqecg_state.original_question_positions = tensor.detach().clone()


def _llava_clear_question_positions(self) -> None:
    self.get_model()._dqecg_state.clear_question_positions()


def _llava_reset(self) -> None:
    self.get_model()._dqecg_state.reset_prefill()


def allocate_llava_visual_tokens(total_tokens: int, image_count: int) -> tuple[int, ...]:
    """Deterministically distribute one global entry budget across images."""

    total_tokens = int(total_tokens)
    image_count = int(image_count)
    if image_count < 1:
        raise ValueError("image_count must be positive")
    if total_tokens < image_count:
        raise ValueError(
            f"Cannot allocate {total_tokens} visual tokens across {image_count} images."
        )
    base, remainder = divmod(total_tokens, image_count)
    return tuple(base + (index < remainder) for index in range(image_count))


def _normalize_multi_image_tensor(images, *, expected_images: int):
    """Flatten ordinary multi-image containers into one vision-tower batch.

    LLaVA's legacy list path assumes that ``encode_images`` preserves the
    original patch count. D-QECG deliberately changes that count, so regular
    one-tensor-per-image inputs must take the tensor path instead. Any-resolution
    tiled inputs remain untouched because their grouping carries extra meaning.
    """

    if isinstance(images, torch.Tensor):
        if images.ndim == 5 and images.shape[0] == 1:
            candidate = images.flatten(0, 1)
            if candidate.shape[0] == expected_images:
                return candidate
        return images
    if not isinstance(images, list) or len(images) != expected_images:
        return images
    normalized = []
    for image in images:
        if not isinstance(image, torch.Tensor):
            return images
        if image.ndim == 3:
            normalized.append(image)
        elif image.ndim == 4 and image.shape[0] == 1:
            normalized.append(image[0])
        else:
            return images
    if normalized and all(image.shape == normalized[0].shape for image in normalized):
        return torch.stack(normalized, dim=0)
    return images


def _llava_encode_images(self, images):
    decoder = self.get_model()
    state = decoder._dqecg_state
    state.reset_prefill()
    with record_stage("vision_tower"):
        features = decoder.get_vision_tower()(images)
    if features.dim() != 3:
        raise ValueError("LLaVA vision tower must return image features shaped [N, M, D].")
    image_count, patch_count = int(features.shape[0]), int(features.shape[1])
    schedule = derive_llava_schedule(
        decoder._dqecg_config.budget,
        layer_count=len(decoder.layers),
        patch_count=image_count * patch_count,
    )
    token_counts = allocate_llava_visual_tokens(schedule.entry_tokens, image_count)
    projected_features = []
    global_pivots = []
    global_assignments = []
    cluster_offset = 0
    for image_index, output_count in enumerate(token_counts):
        with record_stage("fps_merge"):
            aggregated, pivots, assignments = aggregate_visual_features(
                features[image_index],
                output_count=output_count,
                eps=decoder._dqecg_config.eps,
                return_details=True,
            )
        with record_stage("projector"):
            projected_features.append(decoder.mm_projector(aggregated))
        global_pivots.append(pivots + image_index * patch_count)
        global_assignments.append(assignments + cluster_offset)
        cluster_offset += output_count
    state.schedule = schedule
    state.visual_length = schedule.entry_tokens
    state.original_visual_length = schedule.entry_tokens
    state.visual_token_counts = token_counts
    state.pivot_indices = torch.cat(global_pivots).detach()
    state.patch_assignments = torch.cat(global_assignments).detach()
    state.current_visual_indices = torch.arange(
        schedule.entry_tokens, device=features.device, dtype=torch.long
    )
    if image_count == 1:
        return projected_features[0].unsqueeze(0)
    return projected_features


def _llava_prepare_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    image_sizes=None,
):
    decoder = self.get_model()
    state = decoder._dqecg_state
    if past_key_values is not None and input_ids.shape[1] == 1 and state.cache_active:
        cache_length = _cache_length(past_key_values)
        attention_mask = torch.ones(
            (input_ids.shape[0], cache_length + 1),
            device=input_ids.device,
            dtype=attention_mask.dtype if attention_mask is not None else torch.bool,
        )
        next_position = state.next_position_id or cache_length
        position_ids = torch.full(
            (input_ids.shape[0], 1),
            next_position,
            dtype=torch.long,
            device=input_ids.device,
        )
        state.next_position_id = next_position + 1
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    original_ids = input_ids
    original_mask = attention_mask
    try:
        from llava.constants import IMAGE_TOKEN_INDEX
    except ImportError as exc:
        raise ImportError("The LLaVA package is required for the LLaVA adapter.") from exc
    expected_images = int((original_ids == IMAGE_TOKEN_INDEX).sum().item())
    images = _normalize_multi_image_tensor(images, expected_images=expected_images)
    outputs = self._dqecg_original_prepare_multimodal(
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=image_sizes,
    )
    if images is None or original_ids.shape[1] == 1:
        return outputs
    if original_ids.shape[0] != 1:
        raise NotImplementedError("D-QECG LLaVA evaluation requires batch_size=1.")
    active_ids = original_ids[0]
    if original_mask is not None:
        active_ids = active_ids[original_mask[0].bool()]
    image_positions = torch.where(active_ids == IMAGE_TOKEN_INDEX)[0]
    token_counts = tuple(state.visual_token_counts or ())
    if image_positions.numel() != len(token_counts):
        raise RuntimeError(
            "LLaVA image placeholders and encoded image groups disagree: "
            f"placeholders={image_positions.numel()}, groups={len(token_counts)}."
        )
    output_mask = outputs[2]
    left_padding = 0
    if output_mask is not None:
        active_output = torch.where(output_mask[0].bool())[0]
        left_padding = int(active_output[0].item()) if active_output.numel() else 0
    visual_blocks = []
    cumulative_shift = 0
    for image_position, token_count in zip(image_positions.tolist(), token_counts):
        block_start = left_padding + int(image_position) + cumulative_shift
        visual_blocks.append(
            torch.arange(
                block_start,
                block_start + int(token_count),
                device=original_ids.device,
                dtype=torch.long,
            )
        )
        cumulative_shift += int(token_count) - 1
    state.visual_positions = torch.cat(visual_blocks)
    state.visual_start = int(state.visual_positions[0].item())
    state.visual_length = int(state.visual_positions.numel())
    output_sequence_length = int(outputs[4].shape[1])
    if int(state.visual_positions[-1].item()) >= output_sequence_length:
        raise RuntimeError("Expanded LLaVA visual positions exceed the output sequence.")
    if state.question_positions is not None:
        positions = state.question_positions.to(original_ids.device).view(-1)
        valid = (
            (positions >= 0)
            & (positions < output_sequence_length)
            & ~torch.isin(positions, state.visual_positions)
        )
        if not bool(torch.all(valid)):
            # Never reuse or partially filter invalid positions: derive the
            # current sample's post-image prompt suffix below.
            state.clear_question_positions()
    if state.question_positions is None:
        # The formal method uses every textual user-prompt token after the
        # first image as entropy query Q. Map positions only after image
        # placeholders have expanded to their actual token counts.
        raw_start = int(image_positions[0].item()) + 1
        special = {
            value
            for value in (
                getattr(self.config, "bos_token_id", None),
                getattr(self.config, "eos_token_id", None),
                getattr(self.config, "pad_token_id", None),
            )
            if value is not None
        }
        image_position_list = image_positions.tolist()
        positions = []
        for index in range(raw_start, active_ids.shape[0]):
            token_id = int(active_ids[index].item())
            if token_id == IMAGE_TOKEN_INDEX or token_id in special:
                continue
            shift = sum(
                int(token_count) - 1
                for image_position, token_count in zip(image_position_list, token_counts)
                if image_position < index
            )
            positions.append(left_padding + index + shift)
        state.question_positions = torch.tensor(
            positions, device=original_ids.device, dtype=torch.long
        )
        state.original_question_positions = state.question_positions.clone()
    return outputs


def _llava_prepare_generation(
    self,
    input_ids,
    past_key_values=None,
    inputs_embeds=None,
    **kwargs,
):
    result = self._dqecg_original_prepare_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    state = self.get_model()._dqecg_state
    cache = result.get("past_key_values")
    if state.cache_active and cache is not None:
        input_ids = result.get("input_ids")
        query_length = 1 if input_ids is None else input_ids.shape[1]
        cache_length = _cache_length(cache)
        result["cache_position"] = torch.arange(
            cache_length,
            cache_length + query_length,
            device=(input_ids.device if input_ids is not None else next(self.parameters()).device),
        )
    return result


def apply_llava(model, config):
    decoder = model.get_model() if hasattr(model, "get_model") else model.model
    if not hasattr(decoder, "layers") or not hasattr(decoder, "get_vision_tower"):
        raise TypeError("Expected a loaded LLaVA causal language model.")
    state = DQECGRuntimeState()
    decoder._dqecg_config = config
    decoder._dqecg_state = state
    decoder._dqecg_architecture = "llava"
    decoder._dqecg_policy = LLAVA_DECODER_POLICY
    # The sequence remains dense after physical pruning.  Tell the vendored
    # Llama attention boundary not to reinterpret gapped absolute RoPE ids as
    # FlashAttention packed-sequence metadata during native decode steps.
    decoder.config._dqecg_force_dense_attention = True
    decoder._dqecg_original_forward = decoder.forward
    decoder.forward = MethodType(dqecg_decoder_forward, decoder)

    model._dqecg_original_prepare_multimodal = model.prepare_inputs_labels_for_multimodal
    model.prepare_inputs_labels_for_multimodal = MethodType(
        _llava_prepare_multimodal, model
    )
    model.encode_images = MethodType(_llava_encode_images, model)
    if hasattr(model, "prepare_inputs_for_generation"):
        model._dqecg_original_prepare_generation = model.prepare_inputs_for_generation
        model.prepare_inputs_for_generation = MethodType(
            _llava_prepare_generation, model
        )
    model.set_dqecg_question_positions = MethodType(
        _llava_set_question_positions, model
    )
    model.clear_dqecg_question_positions = MethodType(
        _llava_clear_question_positions, model
    )
    model.reset_dqecg = MethodType(_llava_reset, model)
    model._dqecg_enabled = True
    model._dqecg_config = config
    # Compatibility for existing scripts; operational logic reads runtime state.
    model.config.use_d_squared = True
    model.config.d_squared_budget = config.budget
    return model
