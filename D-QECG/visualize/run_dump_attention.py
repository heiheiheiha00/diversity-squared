"""Dump raw and residual visual-token scores for the D-QECG sink experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def find_dqecg_root() -> Path:
    candidate = Path(os.environ.get("DQECG_ROOT", ROOT)).expanduser().resolve()
    if (candidate / "src" / "LLaVA" / "llava" / "__init__.py").exists():
        return candidate
    raise RuntimeError(f"Cannot find local LLaVA package under DQECG_ROOT={candidate}.")


DQECG_ROOT = find_dqecg_root()
LOCAL_IMPORT_PATHS = [
    DQECG_ROOT / "src",
    DQECG_ROOT / "src" / "LLaVA",
    DQECG_ROOT / "src" / "transformers" / "src",
]
for import_path in reversed(LOCAL_IMPORT_PATHS):
    if import_path.exists():
        sys.path.insert(0, str(import_path))

DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_END_TOKEN = None
DEFAULT_IM_START_TOKEN = None
IMAGE_PLACEHOLDER = None
IMAGE_TOKEN_INDEX = None
SeparatorStyle = None
conv_templates = None
get_model_name_from_path = None
process_images = None
tokenizer_image_token = None
load_pretrained_model = None
disable_torch_init = None


DEFAULT_DUMP_LAYERS = "0,1,2,8,9"
DEFAULT_EARLY_LAYERS = (0, 1, 2)
DEFAULT_DEEP_LAYERS = (8, 9)


def import_llava_modules() -> None:
    """Import LLaVA after CLI parsing so --help stays lightweight."""
    global DEFAULT_IMAGE_TOKEN
    global DEFAULT_IM_END_TOKEN
    global DEFAULT_IM_START_TOKEN
    global IMAGE_PLACEHOLDER
    global IMAGE_TOKEN_INDEX
    global SeparatorStyle
    global conv_templates
    global get_model_name_from_path
    global process_images
    global tokenizer_image_token
    global load_pretrained_model
    global disable_torch_init

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN as _DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN as _DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN as _DEFAULT_IM_START_TOKEN,
        IMAGE_PLACEHOLDER as _IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX as _IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle as _SeparatorStyle, conv_templates as _conv_templates
    from llava.mm_utils import (
        get_model_name_from_path as _get_model_name_from_path,
        process_images as _process_images,
        tokenizer_image_token as _tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model as _load_pretrained_model
    from llava.utils import disable_torch_init as _disable_torch_init

    DEFAULT_IMAGE_TOKEN = _DEFAULT_IMAGE_TOKEN
    DEFAULT_IM_END_TOKEN = _DEFAULT_IM_END_TOKEN
    DEFAULT_IM_START_TOKEN = _DEFAULT_IM_START_TOKEN
    IMAGE_PLACEHOLDER = _IMAGE_PLACEHOLDER
    IMAGE_TOKEN_INDEX = _IMAGE_TOKEN_INDEX
    SeparatorStyle = _SeparatorStyle
    conv_templates = _conv_templates
    get_model_name_from_path = _get_model_name_from_path
    process_images = _process_images
    tokenizer_image_token = _tokenizer_image_token
    load_pretrained_model = _load_pretrained_model
    disable_torch_init = _disable_torch_init


def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def merged_layer_ids(*layer_groups: Sequence[int]) -> List[int]:
    layers = []
    seen = set()
    for group in layer_groups:
        for layer_id in group:
            layer_id = int(layer_id)
            if layer_id not in seen:
                layers.append(layer_id)
                seen.add(layer_id)
    return layers


def parse_patch_grid(value: str | None, visual_token_count: int) -> Tuple[int, int]:
    if value:
        parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("--patch-grid must look like 24,24 or 24x24.")
        return parts[0], parts[1]

    side = int(math.sqrt(visual_token_count))
    if side * side != visual_token_count:
        raise ValueError(
            f"Cannot infer a square patch grid for {visual_token_count} visual tokens. "
            "Pass --patch-grid H,W explicitly."
        )
    return side, side


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
            image_path = Path(case["image"])
            if not image_path.is_absolute():
                case["image"] = str(base_dir / image_path)
            yield case


def load_image(image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def infer_conv_mode(model_name: str, requested: str | None) -> str:
    if "llama-2" in model_name.lower():
        inferred = "llava_llama_2"
    elif "v1" in model_name.lower():
        inferred = "llava_v1"
    elif "mpt" in model_name.lower():
        inferred = "mpt"
    else:
        inferred = "llava_v0"
    return requested or inferred


def resolve_model_name(model_path: str, requested: str | None) -> str:
    if requested:
        return requested

    model_name = get_model_name_from_path(model_path)
    if "llava" in model_name.lower():
        return model_name

    path = Path(model_path)
    for part in path.parts:
        if part.startswith("models--") and "llava" in part.lower():
            return part[len("models--") :].replace("--", "/")

    return model_name


def build_prompt(question: str, model, conv_mode: str) -> str:
    image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in question:
        if getattr(model.config, "mm_use_im_start_end", False):
            question = re.sub(IMAGE_PLACEHOLDER, image_token, question)
        else:
            question = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
    else:
        question = (
            f"{image_token}\n{question}"
            if getattr(model.config, "mm_use_im_start_end", False)
            else f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        )

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def model_dtype(model, device: str) -> torch.dtype:
    if "cuda" not in device:
        return torch.float32
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def find_visual_span(input_ids: torch.Tensor, visual_token_count: int) -> Tuple[int, int]:
    image_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if image_positions.numel() != 1:
        raise ValueError(f"Expected exactly one image token, found {image_positions.numel()}.")
    visual_start = int(image_positions[0].item())
    return visual_start, visual_start + int(visual_token_count)


def compute_text_to_visual_score(
    attention: torch.Tensor,
    visual_start: int,
    visual_end: int,
    eps: float = 1e-6,
    return_matrix: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if attention.dim() == 4:
        if attention.size(0) != 1:
            raise ValueError("Batch size must be 1 while dumping attention.")
        attention = attention[0]
    if attention.dim() != 3:
        raise ValueError("Attention must be [heads, seq, seq] or [1, heads, seq, seq].")

    attn_mean = attention.float().mean(dim=0)
    seq_len = attn_mean.size(-1)
    if visual_start < 0 or visual_end > seq_len:
        raise ValueError(
            f"Visual span [{visual_start}, {visual_end}) is outside attention sequence length {seq_len}."
        )

    query_indices = torch.arange(visual_end, seq_len, device=attn_mean.device)
    if query_indices.numel() == 0:
        score = torch.zeros(visual_end - visual_start, dtype=attn_mean.dtype, device=attn_mean.device)
        matrix = torch.empty(0, visual_end - visual_start, dtype=attn_mean.dtype, device=attn_mean.device)
        matrix = matrix.cpu() if return_matrix else None
        return score, query_indices.cpu(), matrix, matrix

    text_to_visual = attn_mean.index_select(0, query_indices)[:, visual_start:visual_end].clamp_min(0.0)
    normalized = text_to_visual / text_to_visual.sum(dim=-1, keepdim=True).clamp_min(eps)
    matrix = normalized.detach().float().cpu() if return_matrix else None
    raw_matrix = text_to_visual.detach().float().cpu() if return_matrix else None
    return normalized.mean(dim=0).detach().float().cpu(), query_indices.cpu(), matrix, raw_matrix


def mean_scores(scores: Dict[int, torch.Tensor], layer_ids: Sequence[int], name: str) -> torch.Tensor:
    missing = [layer_id for layer_id in layer_ids if layer_id not in scores]
    if missing:
        raise ValueError(f"Missing {name} layers {missing}; include them in --dump-layers.")
    return torch.stack([scores[layer_id] for layer_id in layer_ids], dim=0).mean(dim=0)


def mean_matrices(matrices: Dict[int, torch.Tensor], layer_ids: Sequence[int], name: str) -> torch.Tensor:
    missing = [layer_id for layer_id in layer_ids if layer_id not in matrices]
    if missing:
        raise ValueError(f"Missing {name} layers {missing}; include them in --dump-layers.")
    return torch.stack([matrices[layer_id] for layer_id in layer_ids], dim=0).mean(dim=0)


def decode_query_tokens(
    tokenizer,
    input_ids: torch.Tensor,
    query_indices: torch.Tensor,
    visual_start: int,
    visual_end: int,
) -> Tuple[torch.Tensor, List[str]]:
    if query_indices is None or query_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long), []
    visual_token_count = int(visual_end - visual_start)
    original_indices = query_indices.detach().cpu().long() - visual_token_count + 1
    seq_len = int(input_ids.size(1))
    if original_indices.numel() and (int(original_indices.min()) < 0 or int(original_indices.max()) >= seq_len):
        raise IndexError(
            "Cannot map expanded attention query indices back to input_ids. "
            f"expanded range=({int(query_indices.min())}, {int(query_indices.max())}), "
            f"mapped range=({int(original_indices.min())}, {int(original_indices.max())}), "
            f"input_ids length={seq_len}, visual span=[{visual_start}, {visual_end})."
        )
    ids = input_ids[0].detach().cpu().index_select(0, original_indices)
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
    return ids, [str(token) for token in tokens]


def configure_d_squared(
    model,
    *,
    enabled: bool,
    visual_start: int,
    visual_token_count: int,
    visual_keep_count: int,
    llm_keep_count: int,
    agg_layer: int,
    inplace: bool,
    dump_residual: bool,
) -> None:
    model.config.use_d_squared = bool(enabled)
    model.config.d_squared_inplace = bool(inplace)
    model.config.d_squared_sys_length = int(visual_start)
    model.config.d_squared_image_token_length = int(visual_token_count)
    model.config.d_squared_visual_keep_count = int(visual_keep_count)
    model.config.d_squared_llm_keep_count = int(llm_keep_count)
    model.config.d_squared_agg_layer = int(agg_layer)
    model.config.d_squared_dump_residual = bool(dump_residual)
    if hasattr(model, "model") and hasattr(model.model, "reset_d_squared"):
        model.model.reset_d_squared()


def run_case(case, args, tokenizer, model, image_processor, conv_mode: str) -> Dict[str, object]:
    image = load_image(case["image"])
    image_tensor = process_images([image], image_processor, model.config).to(
        model.device,
        dtype=model_dtype(model, args.device),
    )

    prompt = build_prompt(case["question"], model, conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(model.device)
    visual_start, visual_end = find_visual_span(input_ids, args.visual_token_count)

    patch_grid = parse_patch_grid(args.patch_grid, args.visual_token_count)
    common = {
        "case_id": case["id"],
        "image_path": case["image"],
        "image_relpath": case.get("image_relpath", case["image"]),
        "question": case["question"],
        "visual_token_start": int(visual_start),
        "visual_token_end": int(visual_end),
        "image_size": (int(image.height), int(image.width)),
        "patch_grid": tuple(map(int, patch_grid)),
    }

    if args.dump_attn:
        configure_d_squared(
            model,
            enabled=False,
            visual_start=visual_start,
            visual_token_count=args.visual_token_count,
            visual_keep_count=args.visual_keep_count,
            llm_keep_count=args.llm_keep_count,
            agg_layer=args.d_squared_agg_layer,
            inplace=False,
            dump_residual=False,
        )
    elif args.dump_residual:
        configure_d_squared(
            model,
            enabled=True,
            visual_start=visual_start,
            visual_token_count=args.visual_token_count,
            visual_keep_count=args.visual_keep_count,
            llm_keep_count=args.llm_keep_count,
            agg_layer=args.d_squared_agg_layer,
            inplace=args.d_squared_inplace,
            dump_residual=True,
        )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    if args.dump_attn:
        if outputs.attentions is None:
            raise RuntimeError("No attentions were returned. Disable flash attention or use eager attention.")

        raw_score_by_layer = {}
        text_to_visual_by_layer = {}
        text_to_visual_raw_by_layer = {}
        query_indices = None
        early_layers = parse_int_list(args.early_layers)
        deep_layers = parse_int_list(args.deep_layers)
        dump_layers = merged_layer_ids(parse_int_list(args.dump_layers), early_layers, deep_layers)
        for layer_id in dump_layers:
            if layer_id >= len(outputs.attentions):
                raise ValueError(f"Layer {layer_id} is out of range for {len(outputs.attentions)} attention layers.")
            score, query_indices, matrix, raw_matrix = compute_text_to_visual_score(
                outputs.attentions[layer_id],
                visual_start,
                visual_end,
                return_matrix=args.dump_text_contrib,
            )
            raw_score_by_layer[int(layer_id)] = score
            if matrix is not None:
                text_to_visual_by_layer[int(layer_id)] = matrix
            if raw_matrix is not None:
                text_to_visual_raw_by_layer[int(layer_id)] = raw_matrix

        query_token_ids, query_tokens = decode_query_tokens(tokenizer, input_ids, query_indices, visual_start, visual_end)
        result = {
            **common,
            "query_indices": query_indices,
            "query_token_ids": query_token_ids,
            "query_tokens": query_tokens,
            "early_layers": early_layers,
            "deep_layers": deep_layers,
            "early_raw_score": mean_scores(raw_score_by_layer, early_layers, "early raw"),
            "deep_raw_score": mean_scores(raw_score_by_layer, deep_layers, "deep raw"),
            "raw_score_by_layer": raw_score_by_layer,
        }
        if args.dump_text_contrib:
            early_text_to_visual_raw = mean_matrices(
                text_to_visual_raw_by_layer,
                early_layers,
                "early raw text-to-visual contribution",
            )
            deep_text_to_visual_raw = mean_matrices(
                text_to_visual_raw_by_layer,
                deep_layers,
                "deep raw text-to-visual contribution",
            )
            result.update(
                {
                    "text_to_visual_by_layer": text_to_visual_by_layer,
                    "text_to_visual_raw_by_layer": text_to_visual_raw_by_layer,
                    "early_text_to_visual": mean_matrices(text_to_visual_by_layer, early_layers, "early text contribution"),
                    "deep_text_to_visual": mean_matrices(text_to_visual_by_layer, deep_layers, "deep text contribution"),
                    "early_text_to_visual_raw": early_text_to_visual_raw,
                    "deep_text_to_visual_raw": deep_text_to_visual_raw,
                    "early_raw_attention_sum": early_text_to_visual_raw.sum(dim=0),
                    "deep_raw_attention_sum": deep_text_to_visual_raw.sum(dim=0),
                }
            )

        return result

    if args.dump_residual:
        llama_model = getattr(model, "model", None)
        residual_dump = getattr(llama_model, "d_squared_last_residual_dump", None)
        if not residual_dump or "residual_score" not in residual_dump:
            raise RuntimeError("D-QECG residual dump was not produced. Check one-stage setup and output_attentions.")

        result = {
            **common,
            "query_indices": residual_dump.get("query_indices"),
            "residual_score": residual_dump["residual_score"],
            "sink_bias": residual_dump.get("sink_bias"),
            "coarse_candidate_indices": residual_dump.get("coarse_candidate_indices"),
            "first_selected_indices": residual_dump.get("first_selected_indices"),
            "second_selected_indices": residual_dump.get("second_selected_indices"),
            "target_layer": residual_dump.get("target_layer"),
        }
        final_indices = getattr(llama_model, "d_squared_final_visual_indices", None)
        if final_indices is not None:
            result["final_keep_indices"] = final_indices.detach().cpu()
        return result

    raise RuntimeError("Choose either --dump-attn or --dump-residual.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dump-attn", action="store_true")
    parser.add_argument("--dump-residual", action="store_true")
    parser.add_argument("--dump-text-contrib", action="store_true")
    parser.add_argument("--disable-prune", action="store_true")
    parser.add_argument("--enable-residual-filter", action="store_true")
    parser.add_argument("--dump-layers", type=str, default=DEFAULT_DUMP_LAYERS)
    parser.add_argument("--early-layers", type=str, default=",".join(map(str, DEFAULT_EARLY_LAYERS)))
    parser.add_argument("--deep-layers", type=str, default=",".join(map(str, DEFAULT_DEEP_LAYERS)))
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--visual-token-count", type=int, default=576)
    parser.add_argument("--patch-grid", type=str, default=None)
    parser.add_argument("--visual-keep-count", type=int, default=144)
    parser.add_argument("--llm-keep-count", type=int, default=144)
    parser.add_argument("--d-squared-agg-layer", type=int, default=3)
    parser.add_argument("--d-squared-inplace", action="store_true")
    args = parser.parse_args()

    if args.dump_attn == args.dump_residual:
        raise ValueError("Specify exactly one of --dump-attn or --dump-residual.")
    if args.dump_text_contrib and not args.dump_attn:
        raise ValueError("--dump-text-contrib is only valid together with --dump-attn.")
    if args.dump_attn and not args.disable_prune:
        raise ValueError("--dump-attn must be paired with --disable-prune for the no-prune baseline.")
    if args.dump_residual and not args.enable_residual_filter:
        raise ValueError("--dump-residual must be paired with --enable-residual-filter.")

    import_llava_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disable_torch_init()

    model_name = resolve_model_name(args.model_path, args.model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model.eval()
    if image_processor is None:
        raise RuntimeError(
            "image_processor is None. The checkpoint was not loaded as a LLaVA model. "
            "Pass --model-name llava-v1.5-7b or use a --model-path whose directory name contains 'llava'."
        )
    conv_mode = infer_conv_mode(model_name, args.conv_mode)

    for case in iter_cases(args.cases):
        result = run_case(case, args, tokenizer, model, image_processor, conv_mode)
        output_path = args.output_dir / f"{case['id']}.pt"
        torch.save(result, output_path)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
