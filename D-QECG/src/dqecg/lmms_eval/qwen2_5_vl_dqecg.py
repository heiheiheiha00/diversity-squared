"""Qwen2.5-VL adapter compatible with lmms-eval 0.3.4."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from dqecg import dqecg_qwen2_5_vl
from dqecg.attention import resolve_attention_implementation


@register_model("qwen2_5_vl_dqecg")
class Qwen2_5_VLDQECG(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Union[int, str] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "auto",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        system_prompt: str = "You are a helpful assistant.",
        d_squared_budget: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected Qwen D-QECG arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("D-QECG Qwen evaluation currently requires batch_size=1.")
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            resolved_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            resolved_map = device_map or device
        resolved_attention = resolve_attention_implementation(attn_implementation)
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": resolved_map,
            "attn_implementation": resolved_attention,
        }
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained, **model_kwargs
        ).eval()
        self._model = dqecg_qwen2_5_vl(
            self._model,
            budget=int(d_squared_budget),
        )
        self._model._dqecg_attention_implementation = resolved_attention
        print(
            f"[D-QECG] attention_implementation={resolved_attention}; "
            "entropy_query=post_image_user_prompt",
            flush=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            pretrained, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self._tokenizer = self.processor.tokenizer
        self.system_prompt = system_prompt
        self.use_cache = bool(use_cache)
        self.batch_size_per_gpu = 1
        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in {
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            }:
                raise RuntimeError("Only FSDP and multi-GPU data parallelism are supported.")
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self.model.config, "max_position_embeddings", 32768)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Qwen2.5-VL D-QECG loglikelihood is not implemented.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        def collate(item):
            tokens = self.tokenizer.encode(item[0])
            return -len(tokens), item[0]

        reordered = utils.Collator(
            [request.args for request in requests], collate, grouping=True
        )
        chunks = reordered.get_batched(n=1, batch_fn=None)
        progress = tqdm(
            total=len(requests), disable=self.rank != 0, desc="Model Responding"
        )
        for chunk in chunks:
            request_args = chunk[0]
            context, generation_kwargs, doc_to_visual, doc_id, task, split = request_args
            self.model.clear_dqecg_question_positions()
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if not isinstance(visuals, list):
                visuals = [visuals]
            content = [
                {"type": "image", "image": visual.convert("RGB")}
                for visual in visuals
            ]
            content.append({"type": "text", "text": context.replace("<image>", "").strip()})
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # This adapter is intentionally image-only. The visuals have
            # already been materialized as PIL images by lmms-eval, so the
            # video/network helpers from qwen-vl-utils (and their PyAV/FFmpeg
            # dependency) are unnecessary here.
            image_inputs = [item["image"] for item in content if item["type"] == "image"]
            video_inputs = None
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            kwargs = dict(generation_kwargs)
            kwargs.pop("until", None)
            kwargs.setdefault("max_new_tokens", 1024)
            kwargs.setdefault("temperature", 0)
            kwargs.setdefault("top_p", None)
            kwargs.setdefault("num_beams", 1)
            kwargs["do_sample"] = kwargs["temperature"] > 0
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    use_cache=self.use_cache,
                    **kwargs,
                )
            trimmed = generated[:, inputs.input_ids.shape[1] :]
            output = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            results.append(output)
            self.cache_hook.add_partial(
                "generate_until", (context, generation_kwargs), output
            )
            progress.update(1)
        progress.close()
        return reordered.get_original(results)

    def generate_until_multi_round(self, requests):
        raise NotImplementedError
