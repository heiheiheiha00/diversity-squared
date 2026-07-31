"""Dump D-QECG selection internals for later local visualization.

Expected cases JSONL fields:
  {"id": "case_0001", "image": "path/to/image.jpg", "question": "..."}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "visualize" / "outputs" / "qceg_debug"
LOCAL_IMPORT_PATHS = [
    ROOT / "src",
    ROOT / "src" / "LLaVA",
]
for import_path in reversed(LOCAL_IMPORT_PATHS):
    if import_path.exists():
        sys.path.insert(0, str(import_path))

from dqecg.question_span import (
    build_llava_prompt_with_question_spans,
    char_spans_to_original_token_positions,
    map_original_positions_to_expanded,
)


def import_llava_modules():
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_PLACEHOLDER": IMAGE_PLACEHOLDER,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "SeparatorStyle": SeparatorStyle,
        "conv_templates": conv_templates,
        "get_model_name_from_path": get_model_name_from_path,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


def iter_cases(path: Path) -> Iterable[Dict[str, str]]:
    base_dir = path.resolve().parent
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            missing = {"id", "image", "question"} - set(case)
            if missing:
                raise ValueError(f"{path}:{line_no} is missing keys: {sorted(missing)}")
            original_image = case["image"]
            case["image_relpath"] = original_image
            image_path = Path(original_image)
            if not image_path.is_absolute():
                case["image"] = str(base_dir / image_path)
            yield case


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def parse_patch_grid(value: str | None, visual_token_count: int) -> Tuple[int, int]:
    if value:
        parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("--patch-grid must look like 24,24 or 24x24.")
        if parts[0] * parts[1] != visual_token_count:
            raise ValueError("--patch-grid does not match visual token count.")
        return parts[0], parts[1]
    side = int(math.sqrt(visual_token_count))
    if side * side != visual_token_count:
        raise ValueError("Cannot infer square patch grid; pass --patch-grid.")
    return side, side


def find_visual_span(input_ids: torch.Tensor, image_token_index: int, visual_token_count: int) -> Tuple[int, int]:
    image_positions = torch.where(input_ids[0] == image_token_index)[0]
    if image_positions.numel() != 1:
        raise ValueError(f"Expected exactly one image token, found {image_positions.numel()}.")
    visual_start = int(image_positions[0].item())
    return visual_start, visual_start + int(visual_token_count)


def model_dtype(model, device: str):
    if device == "cpu":
        return torch.float32
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def configure_d_squared(
    model,
    *,
    visual_start: int,
    visual_token_count: int,
    visual_keep_count: int,
    llm_keep_count: int,
    agg_layer: int,
    inplace: bool,
    question_local_start: int | None,
    question_local_end: int | None,
    query_score_mode: str,
    second_stage_method: str,
    qceg_tau: float,
) -> None:
    model.config.use_d_squared = True
    model.config.d_squared_inplace = bool(inplace)
    model.config.d_squared_sys_length = int(visual_start)
    model.config.d_squared_image_token_length = int(visual_token_count)
    model.config.d_squared_visual_keep_count = int(visual_keep_count)
    model.config.d_squared_llm_keep_count = int(llm_keep_count)
    model.config.d_squared_agg_layer = int(agg_layer)
    model.config.d_squared_dump_selection_debug = True
    model.config.d_squared_question_local_start = question_local_start
    model.config.d_squared_question_local_end = question_local_end
    model.config.d_squared_query_score_mode = str(query_score_mode)
    model.config.d_squared_second_stage_method = str(second_stage_method)
    model.config.d_squared_qceg_tau = float(qceg_tau)
    if hasattr(model, "model") and hasattr(model.model, "reset_d_squared"):
        model.model.reset_d_squared()


def _to_int_list(value) -> List[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    return [int(item) for item in value]


def build_query_token_debug(input_ids, tokenizer, dump) -> List[Dict[str, object]]:
    if input_ids is None or dump is None:
        return []
    ids = input_ids.detach().cpu()
    if ids.dim() == 2:
        ids = ids[0]

    all_positions = _to_int_list(dump.get("all_query_positions"))
    bos_free_positions = set(_to_int_list(dump.get("filtered_query_positions")))
    question_positions = set(_to_int_list(dump.get("question_query_positions")))
    rows = []
    for local_i, abs_pos in enumerate(all_positions):
        tok_id = int(ids[abs_pos].item()) if 0 <= abs_pos < ids.numel() else None
        if tok_id is None:
            tok_str = "<out_of_range>"
        elif tok_id < 0:
            tok_str = f"<special:{tok_id}>"
        else:
            tok_str = tokenizer.decode([tok_id])
        rows.append(
            {
                "local": int(local_i),
                "abs_pos": int(abs_pos),
                "token_id": tok_id,
                "token": tok_str,
                "bos_free": "KEEP" if abs_pos in bos_free_positions else "DROP",
                "question_only": "KEEP" if abs_pos in question_positions else "DROP",
            }
        )
    return rows


def run_case(case, args, tokenizer, model, image_processor, conv_mode: str, llava) -> Dict[str, object]:
    image = load_image(case["image"])
    image_tensor = llava["process_images"]([image], image_processor, model.config).to(
        model.device,
        dtype=model_dtype(model, args.device),
    )
    prompt, question_char_spans = build_llava_prompt_with_question_spans(
        case["question"],
        model_config=model.config,
        conv_templates=llava["conv_templates"],
        conv_mode=conv_mode,
        image_placeholder=llava["IMAGE_PLACEHOLDER"],
        default_image_token=llava["DEFAULT_IMAGE_TOKEN"],
        default_im_start_token=llava["DEFAULT_IM_START_TOKEN"],
        default_im_end_token=llava["DEFAULT_IM_END_TOKEN"],
    )
    input_ids = llava["tokenizer_image_token"](
        prompt, tokenizer, llava["IMAGE_TOKEN_INDEX"], return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    visual_start, visual_end = find_visual_span(input_ids, llava["IMAGE_TOKEN_INDEX"], args.visual_token_count)
    question_original_positions = []
    question_expanded_positions = []
    if args.question_source == "auto":
        question_original_positions = char_spans_to_original_token_positions(
            prompt,
            question_char_spans,
            tokenizer=tokenizer,
            tokenizer_image_token=llava["tokenizer_image_token"],
            image_token_index=llava["IMAGE_TOKEN_INDEX"],
        )
        question_expanded_positions = map_original_positions_to_expanded(
            question_original_positions,
            image_token_position=visual_start,
            visual_token_count=args.visual_token_count,
        )
    configure_d_squared(
        model,
        visual_start=visual_start,
        visual_token_count=args.visual_token_count,
        visual_keep_count=args.visual_keep_count,
        llm_keep_count=args.llm_keep_count,
        agg_layer=args.d_squared_agg_layer,
        inplace=args.d_squared_inplace,
        question_local_start=args.question_local_start if args.question_source == "local" else None,
        question_local_end=args.question_local_end if args.question_source == "local" else None,
        query_score_mode=args.query_score_mode,
        second_stage_method=args.second_stage_method,
        qceg_tau=args.qceg_tau,
    )
    llama_model = getattr(model, "model", None)
    if args.question_source == "auto" and llama_model is not None:
        llama_model.d_squared_question_query_positions = torch.tensor(
            question_expanded_positions,
            device=model.device,
            dtype=torch.long,
        )

    with torch.inference_mode():
        model(
            input_ids=input_ids,
            images=image_tensor,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    dump = getattr(llama_model, "d_squared_last_selection_debug", None)
    if not dump:
        raise RuntimeError("Selection debug dump was not produced. Check output_attentions and D-QECG config.")
    expanded_input_ids = getattr(llama_model, "d_squared_expanded_input_ids", None)
    query_token_debug = build_query_token_debug(expanded_input_ids, tokenizer, dump)

    return {
        "case_id": case["id"],
        "image_path": case["image"],
        "image_relpath": case.get("image_relpath", case["image"]),
        "question": case["question"],
        "image_size": (int(image.height), int(image.width)),
        "patch_grid": parse_patch_grid(args.patch_grid, args.visual_token_count),
        "visual_token_start": int(visual_start),
        "visual_token_end": int(visual_end),
        "visual_token_count": int(args.visual_token_count),
        "first_selected_indices": getattr(llama_model, "d_squared_first_visual_indices", None).detach().cpu(),
        "second_selected_indices": getattr(llama_model, "d_squared_second_visual_indices", None).detach().cpu(),
        "final_keep_indices": getattr(llama_model, "d_squared_final_visual_indices", None).detach().cpu(),
        "bos_free_score": dump.get("bos_free_score"),
        "raw_query_score": dump.get("raw_query_score"),
        "all_query_score": dump.get("all_query_score", dump.get("raw_query_score")),
        "question_only_score": dump.get("question_only_score"),
        "qceg_score": dump.get("qceg_score"),
        "qceg_H": dump.get("qceg_H"),
        "qceg_gains": dump.get("qceg_gains"),
        "qceg_top_indices": dump.get("qceg_top_indices"),
        "coarse_candidate_indices": dump.get("coarse_candidate_indices"),
        "filtered_query_positions": dump.get("filtered_query_positions"),
        "all_query_positions": dump.get("all_query_positions"),
        "question_query_positions": dump.get("question_query_positions"),
        "question_source": args.question_source,
        "query_score_mode": args.query_score_mode,
        "second_stage_method": args.second_stage_method,
        "qceg_tau": float(args.qceg_tau),
        "question_char_spans": question_char_spans,
        "question_original_positions": torch.tensor(question_original_positions, dtype=torch.long),
        "question_expanded_positions": torch.tensor(question_expanded_positions, dtype=torch.long),
        "selection_score_mode": dump.get("selection_score_mode"),
        "selection_score": dump.get("selection_score"),
        "expanded_input_ids": expanded_input_ids.detach().cpu() if expanded_input_ids is not None else None,
        "query_token_debug": query_token_debug,
        "target_layer": dump.get("target_layer"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--visual-token-count", type=int, default=576)
    parser.add_argument("--visual-keep-count", type=int, default=144)
    parser.add_argument("--llm-keep-count", type=int, default=144)
    parser.add_argument("--d-squared-agg-layer", type=int, default=3)
    parser.add_argument("--d-squared-inplace", action="store_true")
    parser.add_argument("--patch-grid", type=str, default=None)
    parser.add_argument("--query-score-mode", choices=["question_only", "bos_free", "all_query"], default="question_only")
    parser.add_argument("--second-stage-method", choices=["qceg", "question_only", "bos_free", "all_query"], default="qceg")
    parser.add_argument("--qceg-tau", type=float, default=0.1)
    parser.add_argument("--question-source", choices=["auto", "local", "none"], default="auto")
    parser.add_argument("--question-local-start", type=int, default=2)
    parser.add_argument("--question-local-end", type=int, default=7)
    args = parser.parse_args()

    llava = import_llava_modules()
    llava["disable_torch_init"]()
    model_name = args.model_name or llava["get_model_name_from_path"](args.model_path)
    tokenizer, model, image_processor, _ = llava["load_pretrained_model"](
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in iter_cases(args.cases):
        result = run_case(case, args, tokenizer, model, image_processor, args.conv_mode, llava)
        output_path = args.output_dir / f"{case['id']}.pt"
        torch.save(result, output_path)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
