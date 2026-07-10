#!/usr/bin/env python3
"""Patch official lmms-eval's LLaVA wrapper for this D-squared repo."""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "_D_SQUARED_LMMS_EVAL_PATCH = True"
MODELS_INIT_PATCH_MARKER = "_D_SQUARED_MODELS_INIT_PATCH = True"


HELPER_BLOCK = r'''

_D_SQUARED_LMMS_EVAL_PATCH = True
_D_SQUARED_MODEL_ARG_KEYS = {
    "use_d_squared",
    "d_squared_inplace",
    "d_squared_skip_visual_selector",
    "d_squared_dump_selection_debug",
    "d_squared_static_kv_cache",
    "d_squared_sys_length",
    "d_squared_image_token_length",
    "d_squared_visual_keep_count",
    "d_squared_llm_keep_count",
    "d_squared_agg_layer",
    "d_squared_query_score_mode",
    "d_squared_second_stage_method",
    "d_squared_qceg_tau",
}


def _d_squared_to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _d_squared_pop_model_args(kwargs):
    args = {}
    kwargs.pop("attn_implementation", None)
    for key in list(kwargs.keys()):
        if key in _D_SQUARED_MODEL_ARG_KEYS:
            args[key] = kwargs.pop(key)
    return args


def _d_squared_trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _d_squared_question_positions_from_prompt(prompt, user_message, tokenizer, model_config):
    try:
        from D_squared_method.question_span import (
            char_spans_to_original_token_positions,
            map_original_positions_to_expanded,
        )
    except Exception as exc:
        eval_logger.warning(f"D-squared question span helpers are unavailable: {exc}")
        return []

    message_start = prompt.find(user_message)
    if message_start < 0:
        return []

    spans = []
    cursor = 0
    parts = user_message.split(DEFAULT_IMAGE_TOKEN)
    for index, part in enumerate(parts):
        start = cursor
        end = cursor + len(part)
        trimmed_start, trimmed_end = _d_squared_trim_span(user_message, start, end)
        if trimmed_end > trimmed_start:
            spans.append((message_start + trimmed_start, message_start + trimmed_end))
        cursor = end
        if index < len(parts) - 1:
            cursor += len(DEFAULT_IMAGE_TOKEN)

    token_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().view(-1).tolist()
    else:
        token_ids = list(token_ids)
    try:
        image_token_position = token_ids.index(IMAGE_TOKEN_INDEX)
    except ValueError:
        return []

    original_positions = char_spans_to_original_token_positions(
        prompt,
        spans,
        tokenizer=tokenizer,
        tokenizer_image_token=tokenizer_image_token,
        image_token_index=IMAGE_TOKEN_INDEX,
    )
    return map_original_positions_to_expanded(
        original_positions,
        image_token_position=image_token_position,
        visual_token_count=int(getattr(model_config, "d_squared_image_token_length", 576)),
    )
'''


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}.")
    return text.replace(old, new, 1)


def patch_text(text: str) -> str:
    already_patched = PATCH_MARKER in text
    old_deepspeed_import = '''from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

'''
    if old_deepspeed_import in text:
        text = text.replace(
            old_deepspeed_import,
            '''def is_deepspeed_zero3_enabled():
    return False


def set_hf_deepspeed_config(*args, **kwargs):
    return None


def unset_hf_deepspeed_config(*args, **kwargs):
    return None


''',
            1,
        )

    old_llava_import_error = '''except ImportError:
    eval_logger.error("LLaVA is not installed. Please install LLaVA to use this model.")
'''
    if old_llava_import_error in text:
        text = text.replace(
            old_llava_import_error,
            '''except ImportError as exc:
    raise ImportError("LLaVA is not installed or failed to import. Check PYTHONPATH and local D_squared_method/src/LLaVA.") from exc
''',
            1,
        )

    text = text.replace("            self._word_size = 1\n", "            self._world_size = 1\n")
    text = text.replace(
        "def _d_squared_pop_model_args(kwargs):\n    args = {}\n    for key in list(kwargs.keys()):",
        "def _d_squared_pop_model_args(kwargs):\n    args = {}\n    kwargs.pop(\"attn_implementation\", None)\n    for key in list(kwargs.keys()):",
    )
    if '"d_squared_static_kv_cache"' not in text:
        text = text.replace(
            '    "d_squared_dump_selection_debug",\n',
            '    "d_squared_dump_selection_debug",\n    "d_squared_static_kv_cache",\n',
            1,
        )

    if already_patched:
        text = text.replace(
            '        d_squared_model_args = _d_squared_pop_model_args(kwargs)\n'
            '        use_cache = _d_squared_to_bool(use_cache)\n'
            '        d_squared_enabled = _d_squared_to_bool(d_squared_model_args.get("use_d_squared", False))\n'
            '        if d_squared_enabled:\n'
            '            use_cache = False\n'
            '            attn_implementation = "eager"\n'
            '        # Do not use other kwargs for now\n',
            '        d_squared_model_args = _d_squared_pop_model_args(kwargs)\n'
            '        use_cache = _d_squared_to_bool(use_cache)\n'
            '        # Do not use other kwargs for now\n',
            1,
        )
        text = text.replace(
            '        llava_model_args.update(d_squared_model_args)\n'
            '        if d_squared_enabled:\n'
            '            llava_model_args["attn_implementation"] = "eager"\n',
            '        llava_model_args.update(d_squared_model_args)\n',
            1,
        )
        text = text.replace(
            '                    output_attentions=bool(getattr(self._config, "use_d_squared", False)),\n',
            '                    output_attentions=bool(getattr(self._config, "d_squared_dump_selection_debug", False)),\n',
            1,
        )
        return text

    text = replace_once(
        text,
        '\n\n@register_model("llava")\n',
        HELPER_BLOCK + '\n\n@register_model("llava")\n',
        label="class decorator",
    )

    text = replace_once(
        text,
        '        # Do not use kwargs for now\n        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"\n',
        '        d_squared_model_args = _d_squared_pop_model_args(kwargs)\n'
        '        use_cache = _d_squared_to_bool(use_cache)\n'
        '        # Do not use other kwargs for now\n'
        '        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"\n',
        label="kwargs assertion",
    )
    if "llava_model_args" in text:
        text = replace_once(
            text,
            '        if "use_flash_attention_2" in kwargs:\n            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]\n',
            '        if "use_flash_attention_2" in kwargs:\n            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]\n'
            '        llava_model_args.update(d_squared_model_args)\n',
            label="llava model args",
        )
    else:
        text = replace_once(
            text,
            "        ) = load_pretrained_model(pretrained, None, get_model_name_from_path(pretrained), device_map=self._device)\n",
            "        ) = load_pretrained_model(\n"
            "            pretrained,\n"
            "            None,\n"
            "            get_model_name_from_path(pretrained),\n"
            "            device_map=self._device,\n"
            "            **d_squared_model_args,\n"
            "        )\n",
            label="old llava load_pretrained_model",
        )
    text = replace_once(
        text,
        "            question_input = []\n\n            for visual, context in zip(",
        "            question_input = []\n            d_squared_question_positions = []\n\n            for visual, context in zip(",
        label="question input list",
    )
    text = replace_once(
        text,
        "                prompt_question = conv.get_prompt()\n                question_input.append(prompt_question)\n",
        "                prompt_question = conv.get_prompt()\n"
        "                if getattr(self._config, \"use_d_squared\", False):\n"
        "                    d_squared_question_positions.append(\n"
        "                        _d_squared_question_positions_from_prompt(prompt_question, question, self.tokenizer, self._config)\n"
        "                    )\n"
        "                question_input.append(prompt_question)\n",
        label="question position capture",
    )
    old_pre_generate_anchor = (
        "            # These steps are not in LLaVA's original code, but are necessary for generation to work\n"
        "            # TODO: attention to this major generation step...\n"
        "            try:\n"
    )
    if old_pre_generate_anchor not in text:
        old_pre_generate_anchor = (
            "            # These steps are not in LLaVA's original code, but are necessary for generation to work\n"
            "            # TODO: pay attention to this major generation step...\n"
            "            try:\n"
        )
    text = replace_once(
        text,
        old_pre_generate_anchor,
        "            if getattr(self._config, \"use_d_squared\", False):\n"
        "                if len(input_ids_list) != 1:\n"
        "                    raise RuntimeError(\"D-squared lmms-eval path requires batch_size=1.\")\n"
        "                self.model.model.d_squared_question_query_positions = torch.tensor(\n"
        "                    d_squared_question_positions[0], device=self.device, dtype=torch.long\n"
        "                )\n"
        + old_pre_generate_anchor,
        label="pre-generate question positions",
    )
    text = replace_once(
        text,
        "                    max_new_tokens=gen_kwargs[\"max_new_tokens\"],\n                    use_cache=self.use_cache,\n",
        "                    max_new_tokens=gen_kwargs[\"max_new_tokens\"],\n"
        "                    use_cache=self.use_cache,\n"
        "                    output_attentions=bool(getattr(self._config, \"d_squared_dump_selection_debug\", False)),\n",
        label="generate kwargs",
    )
    return text


def patch_models_init(root: Path) -> None:
    target = root / "lmms_eval" / "models" / "__init__.py"
    if not target.exists():
        return

    text = target.read_text(encoding="utf-8")
    if MODELS_INIT_PATCH_MARKER in text:
        print(f"Already patched: {target}")
        return

    backup = target.with_suffix(target.suffix + ".d_squared_backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    target.write_text(
        '''"""Model imports for the D-squared AutoDL lmms-eval run."""

_D_SQUARED_MODELS_INIT_PATCH = True

# Import the only model needed here explicitly. The upstream 0.1.0 file silently
# swallowed ImportError from optional models, which can leave MODEL_REGISTRY
# empty and hide the real LLaVA import failure.
from .llava import Llava  # noqa: F401
''',
        encoding="utf-8",
    )
    print(f"Patched: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmms-eval-dir", required=True, help="Path to the cloned official lmms-eval repo.")
    args = parser.parse_args()

    root = Path(args.lmms_eval_dir)
    candidates = [
        root / "lmms_eval" / "models" / "simple" / "llava.py",
        root / "lmms_eval" / "models" / "llava.py",
    ]
    target = next((path for path in candidates if path.exists()), None)
    if target is None:
        raise FileNotFoundError(
            "Cannot find official LLaVA wrapper. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    original = target.read_text(encoding="utf-8")
    patched = patch_text(original)
    if patched == original:
        print(f"Already patched: {target}")
        patch_models_init(root)
        return
    backup = target.with_suffix(target.suffix + ".d_squared_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup: {backup}")
    patch_models_init(root)


if __name__ == "__main__":
    main()
