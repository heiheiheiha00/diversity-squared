"""Dump CLIP vision-tower self-attention maps before LLaVA sends tokens to the LLM."""

from __future__ import annotations

import argparse
import json
import math
import os
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


get_model_name_from_path = None
load_pretrained_model = None
process_images = None
disable_torch_init = None


def import_llava_modules() -> None:
    """Import LLaVA after CLI parsing so --help remains lightweight."""
    global get_model_name_from_path
    global load_pretrained_model
    global process_images
    global disable_torch_init

    from llava.mm_utils import get_model_name_from_path as _get_model_name_from_path
    from llava.mm_utils import process_images as _process_images
    from llava.model.builder import load_pretrained_model as _load_pretrained_model
    from llava.utils import disable_torch_init as _disable_torch_init

    get_model_name_from_path = _get_model_name_from_path
    load_pretrained_model = _load_pretrained_model
    process_images = _process_images
    disable_torch_init = _disable_torch_init


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


def model_dtype(model, device: str) -> torch.dtype:
    if "cuda" not in device:
        return torch.float32
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def parse_int_list(value: str | None) -> List[int] | None:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_indices(indices: Sequence[int] | None, count: int) -> List[int]:
    if indices is None:
        return list(range(count))
    normalized = []
    for index in indices:
        index = int(index)
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise ValueError(f"Index {index} is outside 0..{count - 1}.")
        if index not in normalized:
            normalized.append(index)
    return normalized


def parse_patch_grid(value: str | None, visual_token_count: int) -> Tuple[int, int]:
    if value:
        parts = [int(part.strip()) for part in value.replace("x", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("--patch-grid must look like 24,24 or 24x24.")
        if parts[0] * parts[1] != visual_token_count:
            raise ValueError(
                f"--patch-grid {parts[0]}x{parts[1]} does not match {visual_token_count} visual tokens."
            )
        return parts[0], parts[1]

    side = int(math.sqrt(visual_token_count))
    if side * side != visual_token_count:
        raise ValueError(
            f"Cannot infer a square patch grid for {visual_token_count} visual tokens. "
            "Pass --patch-grid H,W explicitly."
        )
    return side, side


def vision_tower_module(model):
    tower = model.get_vision_tower()
    if not getattr(tower, "is_loaded", True):
        tower.load_model()
    return tower


@torch.no_grad()
def compute_cls_to_patch_attention(attentions, layers: Sequence[int] | None, heads: Sequence[int] | None) -> Tuple[torch.Tensor, List[int], List[int]]:
    if not attentions:
        raise RuntimeError("Vision tower did not return attentions. Pass output_attentions=True.")

    layer_ids = normalize_indices(layers, len(attentions))
    scores = []
    resolved_heads = None
    for layer_id in layer_ids:
        attention = attentions[layer_id]
        if attention.dim() != 4 or attention.size(0) != 1:
            raise ValueError("Expected CLIP attention with shape [1, heads, seq, seq].")
        head_ids = normalize_indices(heads, attention.size(1))
        if resolved_heads is None:
            resolved_heads = head_ids
        elif resolved_heads != head_ids:
            raise ValueError("Head selection resolved differently across layers.")
        cls_to_patch = attention[0].index_select(0, torch.tensor(head_ids, device=attention.device))[:, 0, 1:]
        scores.append(cls_to_patch.float().mean(dim=0))

    score = torch.stack(scores, dim=0).mean(dim=0)
    return score.detach().cpu(), layer_ids, (resolved_heads or [])


def run_case(case, args, model, image_processor) -> Dict[str, object]:
    image = load_image(case["image"])
    image_tensor = process_images([image], image_processor, model.config).to(
        model.device,
        dtype=model_dtype(model, args.device),
    )

    tower = vision_tower_module(model)
    with torch.inference_mode():
        outputs = tower.vision_tower(
            image_tensor.to(device=tower.device, dtype=tower.dtype),
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    score, layer_ids, head_ids = compute_cls_to_patch_attention(
        outputs.attentions,
        parse_int_list(args.layers),
        parse_int_list(args.heads),
    )
    patch_grid = parse_patch_grid(args.patch_grid, int(score.numel()))

    return {
        "case_id": case["id"],
        "image_path": case["image"],
        "image_relpath": case.get("image_relpath", case["image"]),
        "question": case["question"],
        "answer": case.get("answer"),
        "category": case.get("category"),
        "source": case.get("source"),
        "image_size": (int(image.height), int(image.width)),
        "patch_grid": tuple(map(int, patch_grid)),
        "vision_attention_score": score.float(),
        "layers": layer_ids,
        "heads": head_ids,
        "score_type": "clip_cls_to_patch_attention",
        "visual_token_count": int(score.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("visualize/outputs/visual_tower_attention"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--layers", type=str, default="-1")
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--patch-grid", type=str, default=None)
    args = parser.parse_args()

    import_llava_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disable_torch_init()

    model_name = resolve_model_name(args.model_path, args.model_name)
    _, model, image_processor, _ = load_pretrained_model(
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

    for case in iter_cases(args.cases):
        result = run_case(case, args, model, image_processor)
        output_path = args.output_dir / f"{case['id']}.pt"
        torch.save(result, output_path)
        print(f"saved {output_path} layers={result['layers']} heads={result['heads']}")


if __name__ == "__main__":
    main()
