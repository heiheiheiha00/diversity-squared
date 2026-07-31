#!/usr/bin/env python3
"""Benchmark one LLaVA sample repeatedly and report synchronized mean latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from dqecg.question_span import (
    build_llava_prompt_with_question_spans,
    char_spans_to_original_token_positions,
    map_original_positions_to_expanded,
)
try:
    from .latency_utils import latency_report, measure_call
except ImportError:
    from latency_utils import latency_report, measure_call
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one image/question once, warm up the model, then time model.generate "
            "for the same sample repeatedly with CUDA synchronization."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-file", required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Question text for the sample.")
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing the question.")
    parser.add_argument("--output-path", default="single_sample_latency.json")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use-dqecg", "--use-d-squared", dest="use_d_squared", action="store_true")
    parser.add_argument("--d-squared-inplace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-squared-static-kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-squared-skip-visual-selector", action="store_true")
    parser.add_argument("--d-squared-sys-length", type=int, default=36)
    parser.add_argument("--d-squared-image-token-length", type=int, default=576)
    parser.add_argument("--d-squared-visual-keep-count", type=int, default=32)
    parser.add_argument("--d-squared-llm-keep-count", type=int, default=32)
    parser.add_argument("--d-squared-agg-layer", type=int, default=3)
    parser.add_argument("--d-squared-qceg-tau", type=float, default=0.1)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative.")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive.")
    if args.load_8bit and args.load_4bit:
        parser.error("Choose at most one of --load-8bit and --load-4bit.")
    if args.use_d_squared and args.d_squared_inplace and args.use_cache and not args.d_squared_static_kv_cache:
        parser.error("D-QECG inplace generation with cache requires --d-squared-static-kv-cache.")
    return args


def infer_conv_mode(model_name: str) -> str:
    lower = model_name.lower()
    if "llama-2" in lower:
        return "llava_llama_2"
    if "v1" in lower:
        return "llava_v1"
    if "mpt" in lower:
        return "mpt"
    return "llava_v0"


def load_question(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt.strip()
    return Path(args.prompt_file).read_text(encoding="utf-8").strip()


def build_prompt_and_question_positions(question, *, model, tokenizer, conv_mode, visual_token_count):
    prompt, char_spans = build_llava_prompt_with_question_spans(
        question,
        model_config=model.config,
        conv_templates=conv_templates,
        conv_mode=conv_mode,
        image_placeholder=DEFAULT_IMAGE_TOKEN,
        default_image_token=DEFAULT_IMAGE_TOKEN,
        default_im_start_token=DEFAULT_IM_START_TOKEN,
        default_im_end_token=DEFAULT_IM_END_TOKEN,
    )
    token_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors=None,
    )
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().view(-1).tolist()
    else:
        token_ids = list(token_ids)
    try:
        image_token_position = token_ids.index(IMAGE_TOKEN_INDEX)
    except ValueError as exc:
        raise RuntimeError("The rendered prompt does not contain the image token.") from exc

    original_positions = char_spans_to_original_token_positions(
        prompt,
        char_spans,
        tokenizer=tokenizer,
        tokenizer_image_token=tokenizer_image_token,
        image_token_index=IMAGE_TOKEN_INDEX,
    )
    expanded_positions = map_original_positions_to_expanded(
        original_positions,
        image_token_position=image_token_position,
        visual_token_count=visual_token_count,
    )
    return prompt, expanded_positions


def configure_d_squared(model, args: argparse.Namespace) -> None:
    config = model.config
    config.use_d_squared = bool(args.use_d_squared)
    config.d_squared_inplace = bool(args.d_squared_inplace)
    config.d_squared_static_kv_cache = bool(args.d_squared_static_kv_cache)
    config.d_squared_skip_visual_selector = bool(args.d_squared_skip_visual_selector)
    config.d_squared_sys_length = int(args.d_squared_sys_length)
    config.d_squared_image_token_length = int(args.d_squared_image_token_length)
    config.d_squared_visual_keep_count = int(args.d_squared_visual_keep_count)
    config.d_squared_llm_keep_count = int(args.d_squared_llm_keep_count)
    config.d_squared_agg_layer = int(args.d_squared_agg_layer)
    config.d_squared_second_stage_method = "qceg"
    config.d_squared_qceg_tau = float(args.d_squared_qceg_tau)
    config.d_squared_query_score_mode = "question_only"
    config.d_squared_dump_selection_debug = False
    config.use_cache = bool(args.use_cache)
    if hasattr(model, "model") and hasattr(model.model, "reset_d_squared"):
        model.model.reset_d_squared()


def main() -> None:
    args = parse_args()
    question = load_question(args)
    if not question:
        raise ValueError("The question is empty.")

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    configure_d_squared(model, args)

    conv_mode = args.conv_mode or infer_conv_mode(model_name)
    if conv_mode not in conv_templates:
        raise KeyError(f"Unknown conversation mode: {conv_mode}")
    prompt, question_positions = build_prompt_and_question_positions(
        question,
        model=model,
        tokenizer=tokenizer,
        conv_mode=conv_mode,
        visual_token_count=args.d_squared_image_token_length,
    )
    if args.use_d_squared and len(question_positions) < 2:
        raise RuntimeError(
            "QCEG needs at least two mapped question tokens; choose a longer question "
            "or inspect the prompt/tokenizer mapping."
        )

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    image = Image.open(args.image_file).convert("RGB")
    image_args = SimpleNamespace(image_aspect_ratio="pad")
    image_tensor = process_images([image], image_processor, image_args)
    if isinstance(image_tensor, list):
        image_tensor = [tensor.to(model.device, dtype=torch.float16) for tensor in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    question_position_tensor = torch.tensor(
        question_positions,
        device=model.device,
        dtype=torch.long,
    )

    def prepare_sample_state() -> None:
        if not args.use_d_squared:
            return
        model.model.reset_d_squared()
        model.model.d_squared_question_query_positions = question_position_tensor

    def generate_once():
        return model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=None,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=args.use_cache,
            output_attentions=False,
            return_dict_in_generate=False,
        )

    print(
        f"Warmup: {args.warmup_runs} runs; timed: {args.repeats} runs; "
        f"scope=model.generate; mode={'dqecg' if args.use_d_squared else 'baseline'}"
    )
    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            prepare_sample_state()
            generate_once()

        latencies = []
        output_ids = None
        for index in range(args.repeats):
            prepare_sample_state()
            output_ids, elapsed = measure_call(generate_once, device=model.device)
            latencies.append(elapsed)
            if (index + 1) % 10 == 0 or index + 1 == args.repeats:
                print(f"timed {index + 1}/{args.repeats}: {elapsed * 1000.0:.3f} ms")

    generated_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    report = latency_report(latencies)
    report.update(
        {
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "scope_detail": (
                "model.generate only; includes vision tower, projector, prefill and decode; "
                "excludes model loading, disk I/O, prompt tokenization and CPU image preprocessing"
            ),
        }
    )
    result = {
        "model_path": args.model_path,
        "mode": "dqecg" if args.use_d_squared else "baseline",
        "question": question,
        "generated_text": generated_text,
        "generated_tokens": int(generated_ids.numel()),
        "latency": report,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": args.use_cache,
            "attn_implementation": args.attn_implementation,
            "conv_mode": conv_mode,
            "batch_size": 1,
        },
        "d_squared": {
            "enabled": args.use_d_squared,
            "inplace": args.d_squared_inplace,
            "static_kv_cache": args.d_squared_static_kv_cache,
            "skip_visual_selector": args.d_squared_skip_visual_selector,
            "sys_length": args.d_squared_sys_length,
            "image_token_length": args.d_squared_image_token_length,
            "visual_keep_count": args.d_squared_visual_keep_count,
            "llm_keep_count": args.d_squared_llm_keep_count,
            "agg_layer": args.d_squared_agg_layer,
            "second_stage_method": "qceg",
            "qceg_tau": args.d_squared_qceg_tau,
            "question_token_count": len(question_positions),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean sample latency: {report['mean_seconds'] * 1000.0:.3f} ms")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
