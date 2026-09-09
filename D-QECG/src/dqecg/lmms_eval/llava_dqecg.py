"""Thin D-QECG wrapper around lmms-eval 0.3.4's native LLaVA adapter."""

from __future__ import annotations

from types import MethodType

from lmms_eval.api.registry import register_model
from lmms_eval.models.llava import Llava

from dqecg import dqecg_llava
from dqecg.attention import resolve_attention_implementation


def _generate_with_prompt_query(self, *args, **kwargs):
    """Use all textual user-prompt tokens after the image as entropy query Q."""

    # lmms-eval 0.3.4 passes temperature=0 for greedy decoding. Transformers
    # 4.54 treats it as an inapplicable sampling flag and emits a warning.
    if kwargs.get("temperature") == 0:
        kwargs.pop("temperature")
    # Query positions belong to exactly one sample. The multimodal adapter
    # derives the current positions after expanding the actual image tokens.
    self.clear_dqecg_question_positions()
    return self._dqecg_original_generate(*args, **kwargs)


@register_model("llava_dqecg")
class LlavaDQECG(Llava):
    def __init__(
        self, d_squared_budget=128, attn_implementation="auto", **kwargs
    ):
        resolved_attention = resolve_attention_implementation(attn_implementation)
        kwargs["attn_implementation"] = resolved_attention
        kwargs.setdefault("use_cache", True)
        super().__init__(**kwargs)
        # Base lmms-eval may already have wrapped the model with Accelerate.
        # Mutate the unwrapped instance and keep the distributed wrapper intact.
        runtime_model = dqecg_llava(
            self.model,
            budget=int(d_squared_budget),
        )
        if runtime_model is self._model:
            self._model = runtime_model
        self._config = runtime_model.config
        runtime_model._dqecg_attention_implementation = resolved_attention
        runtime_model._dqecg_original_generate = runtime_model.generate
        runtime_model.generate = MethodType(_generate_with_prompt_query, runtime_model)
        print(
            f"[D-QECG] attention_implementation={resolved_attention}; "
            "entropy_query=post_image_user_prompt",
            flush=True,
        )
