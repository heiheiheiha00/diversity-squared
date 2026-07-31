#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    _prepare_4d_causal_attention_mask_with_cache_position,
)
from transformers.utils import logging

from dqecg.pruning import (
    build_d_squared_keep_indices,
    build_d_squared_visual_token_indices,
)
from dqecg.llm_selector import select_llm_secondary_visual_token_indices

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


logger = logging.get_logger(__name__)


def _cache_has_tokens(past_key_values):
    if past_key_values is None:
        return False
    if isinstance(past_key_values, Cache):
        return past_key_values.get_seq_length() > 0
    return len(past_key_values) > 0 and past_key_values[0][0].shape[-2] > 0


def _cache_sequence_length(past_key_values):
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, Cache):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].shape[-2])


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, **kwargs):
        super(LlavaLlamaModel, self).__init__(config)
        self.reset_d_squared()

    def reset_d_squared(self):
        self.use_d_squared = bool(getattr(self.config, "use_d_squared", False))
        self.d_squared_sys_length = getattr(self.config, "d_squared_sys_length", None)
        self.d_squared_image_token_length = getattr(self.config, "d_squared_image_token_length", None)
        self.d_squared_visual_keep_count = getattr(self.config, "d_squared_visual_keep_count", None)
        self.d_squared_llm_keep_count = getattr(self.config, "d_squared_llm_keep_count", None)
        self.d_squared_agg_layer = getattr(self.config, "d_squared_agg_layer", None)
        self.d_squared_inplace = bool(getattr(self.config, "d_squared_inplace", False))
        self.d_squared_skip_visual_selector = bool(getattr(self.config, "d_squared_skip_visual_selector", False))
        self.d_squared_static_kv_cache = bool(getattr(self.config, "d_squared_static_kv_cache", False))
        self.reset_d_squared_runtime_state()
        self.d_squared_question_query_positions = None

    def reset_d_squared_runtime_state(self):
        self.d_squared_cache_active = False
        self.d_squared_cached_sequence_length = None
        self.d_squared_next_position_id = None
        self.d_squared_first_visual_indices = None
        self.d_squared_second_visual_indices = None
        self.d_squared_final_visual_indices = None
        self.d_squared_expanded_input_ids = None
        self.d_squared_qceg_hidden_states = None
        self.d_squared_last_selection_debug = None

    @staticmethod
    def _d_squared_cache_has_tokens(cache):
        if cache is None:
            return False
        if isinstance(cache, Cache):
            return cache.get_seq_length() > 0
        return len(cache) > 0 and cache[0][0].shape[-2] > 0

    @staticmethod
    def _d_squared_trim_cache(cache, keep_indices):
        if cache is None:
            return None
        if isinstance(cache, Cache):
            if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
                raise TypeError("D-QECG static KV cache requires a mutable DynamicCache.")
            cache.key_cache = [key_states.index_select(-2, keep_indices) for key_states in cache.key_cache]
            cache.value_cache = [value_states.index_select(-2, keep_indices) for value_states in cache.value_cache]
            return cache
        return tuple(
            (key_states.index_select(-2, keep_indices), value_states.index_select(-2, keep_indices))
            for key_states, value_states in cache
        )

    def _d_squared_qceg_has_valid_question_positions(self):
        question_positions = self.d_squared_question_query_positions
        if question_positions is None:
            return False
        if isinstance(question_positions, torch.Tensor):
            return question_positions.numel() >= 2
        return len(question_positions) >= 2

    @staticmethod
    def _d_squared_debug_to_cpu(debug):
        if debug is None:
            return None
        if isinstance(debug, torch.Tensor):
            return debug.detach().float().cpu() if debug.is_floating_point() else debug.detach().cpu()
        if isinstance(debug, dict):
            return {key: LlavaLlamaModel._d_squared_debug_to_cpu(value) for key, value in debug.items()}
        if isinstance(debug, (list, tuple)):
            return type(debug)(LlavaLlamaModel._d_squared_debug_to_cpu(value) for value in debug)
        return debug

    def _d_squared_special_token_ids(self):
        ids = []
        for value in (
            getattr(self.config, "bos_token_id", None),
            getattr(self.config, "eos_token_id", None),
            getattr(self.config, "pad_token_id", None),
        ):
            if value is not None:
                ids.append(int(value))
        return tuple(dict.fromkeys(ids))

    def _d_squared_validate_config(self, use_cache):
        if not self.use_d_squared:
            return
        if self.d_squared_agg_layer is None or self.d_squared_agg_layer <= 0:
            raise RuntimeError("D-QECG requires `d_squared_agg_layer > 0`.")
        if self.d_squared_sys_length is None or self.d_squared_image_token_length is None:
            raise RuntimeError("D-QECG requires system and image token lengths.")
        if (not self.d_squared_skip_visual_selector and self.d_squared_visual_keep_count is None) or self.d_squared_llm_keep_count is None:
            raise RuntimeError("D-QECG requires first- and second-stage visual keep counts.")
        if self.d_squared_inplace and use_cache and not self.d_squared_static_kv_cache:
            raise RuntimeError("D-QECG token drop with KV cache requires `d_squared_static_kv_cache=True`.")
        if (
            (not self.d_squared_inplace or not self.d_squared_static_kv_cache)
            and getattr(self.config, "_attn_implementation", "eager") == "flash_attention_2"
        ):
            raise RuntimeError(
                "Dynamic-mask D-QECG is incompatible with FlashAttention 2. "
                "Use in-place token drop with `d_squared_static_kv_cache=true`, "
                "or set `attn_implementation=eager`."
            )

    def _d_squared_set_selector_attention_mode(self, use_eager):
        if self.d_squared_agg_layer is None:
            return
        selector_layer_index = int(self.d_squared_agg_layer) - 1
        if selector_layer_index < 0 or selector_layer_index >= len(self.layers):
            raise RuntimeError("D-QECG aggregation layer is outside the LLaMA decoder.")

        saved_flash_attention = getattr(self, "_d_squared_selector_flash_attention", None)
        if not use_eager:
            if saved_flash_attention is not None:
                self.layers[selector_layer_index].self_attn = saved_flash_attention
                self._d_squared_selector_flash_attention = None
            return

        selector_attention = self.layers[selector_layer_index].self_attn
        if not isinstance(selector_attention, LlamaFlashAttention2):
            return

        eager_attention = LlamaAttention(self.config, layer_idx=selector_attention.layer_idx)
        eager_attention.load_state_dict(selector_attention.state_dict())
        eager_attention.to(device=selector_attention.q_proj.weight.device, dtype=selector_attention.q_proj.weight.dtype)
        eager_attention.train(selector_attention.training)
        self._d_squared_selector_flash_attention = selector_attention
        self.layers[selector_layer_index].self_attn = eager_attention

    def _d_squared_select_second_visual_indices(self, attention, hidden_states, target_layer):
        selector_kwargs = {
            "attention": attention,
            "hidden_states": hidden_states,
            "first_selected_indices": self.d_squared_first_visual_indices,
            "keep_num": self.d_squared_llm_keep_count,
            "image_token_start_index": self.d_squared_sys_length,
            "image_token_length": self.d_squared_image_token_length,
            "first_indices_are_absolute": False,
            "input_ids": self.d_squared_expanded_input_ids,
            "special_token_ids": self._d_squared_special_token_ids(),
            "extra_exclude_token_ids": (-200,),
            "question_local_start": getattr(self.config, "d_squared_question_local_start", None),
            "question_local_end": getattr(self.config, "d_squared_question_local_end", None),
            "question_query_positions": self.d_squared_question_query_positions,
            "qceg_hidden_states": self.d_squared_qceg_hidden_states,
            "second_stage_method": getattr(self.config, "d_squared_second_stage_method", "qceg"),
            "qceg_tau": getattr(self.config, "d_squared_qceg_tau", 0.1),
            "query_score_mode": getattr(self.config, "d_squared_query_score_mode", "question_only"),
        }
        if getattr(self.config, "d_squared_dump_selection_debug", False):
            second_visual_indices, selection_debug = select_llm_secondary_visual_token_indices(
                **selector_kwargs,
                return_debug=True,
            )
            selection_debug["visual_token_start"] = int(self.d_squared_sys_length)
            selection_debug["visual_token_end"] = int(
                self.d_squared_sys_length + self.d_squared_image_token_length
            )
            selection_debug["target_layer"] = int(target_layer)
            self.d_squared_last_selection_debug = self._d_squared_debug_to_cpu(selection_debug)
            return second_visual_indices
        return select_llm_secondary_visual_token_indices(**selector_kwargs)

    @staticmethod
    def _d_squared_eager_causal_mask(attention_mask, input_tensor, cache_position, past_key_values):
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        target_length = attention_mask.shape[-1] if attention_mask is not None else past_length + input_tensor.shape[1]
        return _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=input_tensor.shape[1],
            target_length=target_length,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
            min_dtype=torch.finfo(input_tensor.dtype).min,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

    def forward(
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
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        d_squared_prefill = bool(
            self.use_d_squared
            and not self._d_squared_cache_has_tokens(past_key_values)
            and inputs_embeds.shape[1] > 1
        )
        self._d_squared_validate_config(use_cache)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None or d_squared_prefill:
            if d_squared_prefill:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        d_squared_attention_layer = int(self.d_squared_agg_layer) - 1 if d_squared_prefill else None
        d_squared_second_stage_method = getattr(self.config, "d_squared_second_stage_method", "qceg")
        d_squared_requires_selector_attention = bool(
            d_squared_prefill
            and (
                getattr(self.config, "d_squared_dump_selection_debug", False)
                or str(d_squared_second_stage_method).lower().replace("-", "_") != "qceg"
                or not self._d_squared_qceg_has_valid_question_positions()
            )
        )
        self.d_squared_qceg_hidden_states = [] if d_squared_prefill else None
        previous_layer_attention = None
        active_causal_mask = causal_mask
        active_cache_position = cache_position
        active_position_embeddings = position_embeddings
        dynamic_selection_mask = None

        self._d_squared_set_selector_attention_mode(d_squared_requires_selector_attention)
        try:
            for layer_index, decoder_layer in enumerate(self.layers):
                if self.d_squared_qceg_hidden_states is not None and len(self.d_squared_qceg_hidden_states) < 4:
                    self.d_squared_qceg_hidden_states.append(hidden_states.detach())
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_output_attentions = bool(
                    output_attentions
                    or (d_squared_requires_selector_attention and layer_index == d_squared_attention_layer)
                )

                if d_squared_prefill and self.d_squared_inplace and layer_index == self.d_squared_agg_layer:
                    second_visual_indices = self._d_squared_select_second_visual_indices(
                        previous_layer_attention,
                        hidden_states,
                        target_layer=layer_index - 1,
                    )
                    self.d_squared_second_visual_indices = second_visual_indices.detach()
                    keep_indices = build_d_squared_keep_indices(
                        first_visual_indices=self.d_squared_first_visual_indices,
                        second_visual_indices=second_visual_indices,
                        image_token_start_index=self.d_squared_sys_length,
                        image_token_length=self.d_squared_image_token_length,
                        seq_length=hidden_states.shape[1],
                        device=hidden_states.device,
                    )
                    self.d_squared_final_visual_indices = (
                        keep_indices[
                            (keep_indices >= self.d_squared_sys_length)
                            & (keep_indices < self.d_squared_sys_length + self.d_squared_image_token_length)
                        ]
                        - self.d_squared_sys_length
                    ).detach()

                    original_next_position_id = int(position_ids.max().item()) + 1
                    hidden_states = hidden_states.index_select(1, keep_indices)
                    position_ids = position_ids.index_select(1, keep_indices)
                    active_position_embeddings = tuple(
                        position_embedding.index_select(1, keep_indices)
                        for position_embedding in active_position_embeddings
                    )
                    active_cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

                    if use_cache and self.d_squared_static_kv_cache:
                        past_key_values = self._d_squared_trim_cache(past_key_values, keep_indices)
                        self.d_squared_cache_active = True
                        self.d_squared_cached_sequence_length = int(hidden_states.shape[1])
                        self.d_squared_next_position_id = original_next_position_id

                    active_causal_mask = self._update_causal_mask(
                        None,
                        hidden_states,
                        active_cache_position,
                        past_key_values,
                        layer_output_attentions,
                    )

                elif d_squared_prefill and not self.d_squared_inplace and layer_index == self.d_squared_agg_layer:
                    second_visual_indices = self._d_squared_select_second_visual_indices(
                        previous_layer_attention,
                        hidden_states,
                        target_layer=layer_index - 1,
                    )
                    self.d_squared_second_visual_indices = second_visual_indices.detach()
                    keep_visual_indices = build_d_squared_visual_token_indices(
                        first_visual_indices=self.d_squared_first_visual_indices,
                        second_visual_indices=second_visual_indices,
                        image_token_start_index=self.d_squared_sys_length,
                        image_token_length=self.d_squared_image_token_length,
                        device=hidden_states.device,
                    )
                    self.d_squared_final_visual_indices = (keep_visual_indices - self.d_squared_sys_length).detach()
                    if attention_mask is None:
                        dynamic_selection_mask = torch.ones(
                            hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
                        )
                    else:
                        dynamic_selection_mask = attention_mask.clone()
                    dynamic_selection_mask[:, self.d_squared_sys_length : self.d_squared_sys_length + self.d_squared_image_token_length] = False
                    dynamic_selection_mask[:, keep_visual_indices] = True
                    active_causal_mask = self._update_causal_mask(
                        dynamic_selection_mask,
                        hidden_states,
                        active_cache_position,
                        past_key_values,
                        layer_output_attentions,
                    )

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        (
                            self._d_squared_eager_causal_mask(
                                attention_mask,
                                hidden_states,
                                active_cache_position,
                                past_key_values,
                            )
                            if d_squared_requires_selector_attention
                            and layer_index == d_squared_attention_layer
                            and isinstance(decoder_layer.self_attn, LlamaAttention)
                            else active_causal_mask
                        ),
                        position_ids,
                        past_key_values,
                        layer_output_attentions,
                        use_cache,
                        active_cache_position,
                        active_position_embeddings,
                    )
                else:
                    layer_causal_mask = active_causal_mask
                    if (
                        d_squared_requires_selector_attention
                        and layer_index == d_squared_attention_layer
                        and isinstance(decoder_layer.self_attn, LlamaAttention)
                    ):
                        layer_causal_mask = self._d_squared_eager_causal_mask(
                            attention_mask,
                            hidden_states,
                            active_cache_position,
                            past_key_values,
                        )
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=layer_causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=layer_output_attentions,
                        use_cache=use_cache,
                        cache_position=active_cache_position,
                        position_embeddings=active_position_embeddings,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if layer_output_attentions else 1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                previous_layer_attention = layer_outputs[1] if layer_output_attentions else None
        finally:
            self._d_squared_set_selector_attention_mode(False)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(value for value in [hidden_states, next_cache, all_hidden_states, all_self_attns] if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def generate(self, *args, **kwargs):
        lmms_eval_generation = "image_sizes" in kwargs
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        input_token_len = input_ids.shape[1] if lmms_eval_generation and input_ids is not None else None

        outputs = super().generate(*args, **kwargs)

        # lmms-eval decodes `generate()` outputs directly. Original LLaVA input_ids
        # include IMAGE_TOKEN_INDEX=-200, so return only newly generated ids there.
        if lmms_eval_generation and input_token_len is not None and isinstance(outputs, torch.Tensor):
            outputs = outputs[:, input_token_len:]

        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        multimodal_prepare = inputs_embeds is None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        d_squared_model = self.get_model()
        if getattr(d_squared_model, "d_squared_cache_active", False) and position_ids is not None:
            cache_position = position_ids[0]
        elif multimodal_prepare:
            cache_position = None
            position_ids = None

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        if past_key_values is not None and kwargs.get("cache_position") is None:
            cache_length = _cache_sequence_length(past_key_values)
            kwargs["cache_position"] = torch.arange(
                cache_length,
                cache_length + input_ids.shape[1],
                device=input_ids.device,
            )
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None and not _cache_has_tokens(past_key_values):
            _inputs['images'] = images
        if image_sizes is not None and not _cache_has_tokens(past_key_values):
            _inputs['image_sizes'] = image_sizes
        return _inputs

AutoConfig.register("llava", LlavaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, exist_ok=True)
