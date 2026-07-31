"""Dump first-stage KNN feature-selection scores for visual-token analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
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
get_projected_visual_tokens_from_original_llava = None
load_pretrained_model = None
process_images = None
select_visual_token_indices = None
disable_torch_init = None


def import_llava_modules() -> None:
    """Import LLaVA only after CLI parsing so --help remains lightweight."""
    global get_model_name_from_path
    global get_projected_visual_tokens_from_original_llava
    global load_pretrained_model
    global process_images
    global select_visual_token_indices
    global disable_torch_init

    from dqecg.visual_selector import (
        get_projected_visual_tokens_from_original_llava as _get_projected_visual_tokens,
        select_visual_token_indices as _select_visual_token_indices,
    )
    from llava.mm_utils import get_model_name_from_path as _get_model_name_from_path
    from llava.mm_utils import process_images as _process_images
    from llava.model.builder import load_pretrained_model as _load_pretrained_model
    from llava.utils import disable_torch_init as _disable_torch_init

    get_model_name_from_path = _get_model_name_from_path
    get_projected_visual_tokens_from_original_llava = _get_projected_visual_tokens
    load_pretrained_model = _load_pretrained_model
    process_images = _process_images
    select_visual_token_indices = _select_visual_token_indices
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


@torch.no_grad()
def compute_knn_density_peak_debug(
    visual_tokens: torch.Tensor,
    keep_num: int,
    candidate_ratio: float = 2.0,
    knn: int = 8,
    eps: float = 0.0,
) -> Dict[str, torch.Tensor | int | float]:
    """Reproduce visual_selector.py and retain intermediate density-peak scores."""
    if visual_tokens.dim() == 3:
        if visual_tokens.size(0) != 1:
            raise NotImplementedError("This dump script supports batch size 1.")
        visual_tokens = visual_tokens[0]
    if visual_tokens.dim() != 2:
        raise ValueError("visual_tokens must have shape [N, D] or [1, N, D].")

    device = visual_tokens.device
    token_count = int(visual_tokens.size(0))
    keep_num = min(max(int(keep_num), 0), token_count)
    if keep_num <= 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return {
            "candidate_indices": empty_long,
            "rho": empty_float,
            "delta": empty_float,
            "candidate_scores": empty_float,
            "selected_indices_by_score": empty_long,
            "selected_indices": empty_long,
            "selected_scores_by_score": empty_float,
            "score_map": torch.zeros(token_count, dtype=torch.float32, device=device),
            "selected_mask": torch.zeros(token_count, dtype=torch.bool, device=device),
            "candidate_ratio": float(candidate_ratio),
            "knn": int(knn),
            "eps": float(eps),
            "keep_num": int(keep_num),
        }

    candidate_num = min(max(keep_num, math.ceil(candidate_ratio * keep_num)), token_count)

    x = F.normalize(visual_tokens.float(), dim=-1)
    similarity = x @ x.transpose(0, 1)
    distance = 1.0 - similarity
    distance.clamp_(min=0.0, max=2.0)

    candidate_idx = torch.empty(candidate_num, device=device, dtype=torch.long)

    diag = torch.arange(token_count, device=device)
    distance[diag, diag] = float("inf")
    nearest_dist = distance.min(dim=1).values
    first_idx = torch.argmax(nearest_dist)
    distance[diag, diag] = 0.0

    candidate_idx[0] = first_idx
    min_dist_to_selected = distance[first_idx].contiguous()
    min_dist_to_selected[first_idx] = -1.0

    for candidate_pos in range(1, candidate_num):
        next_idx = torch.argmax(min_dist_to_selected)
        candidate_idx[candidate_pos] = next_idx
        torch.minimum(min_dist_to_selected, distance[next_idx], out=min_dist_to_selected)
        min_dist_to_selected[candidate_idx[: candidate_pos + 1]] = -1.0

    candidate_distance = distance.index_select(0, candidate_idx).index_select(1, candidate_idx)
    candidate_count = int(candidate_distance.size(0))

    if keep_num >= candidate_count:
        candidate_scores = torch.ones(candidate_count, dtype=torch.float32, device=device)
        selected_by_score = candidate_idx
        selected_scores = candidate_scores
        selected_sorted = torch.sort(selected_by_score).values
        score_map = torch.zeros(token_count, dtype=torch.float32, device=device)
        score_map[candidate_idx] = candidate_scores
        selected_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
        selected_mask[selected_by_score] = True
        return {
            "candidate_indices": candidate_idx,
            "rho": candidate_scores.clone(),
            "delta": candidate_scores.clone(),
            "candidate_scores": candidate_scores,
            "selected_indices_by_score": selected_by_score,
            "selected_indices": selected_sorted,
            "selected_scores_by_score": selected_scores,
            "score_map": score_map,
            "selected_mask": selected_mask,
            "candidate_ratio": float(candidate_ratio),
            "knn": int(knn),
            "eps": float(eps),
            "keep_num": int(keep_num),
        }

    candidate_diag = torch.arange(candidate_count, device=device)
    candidate_distance[candidate_diag, candidate_diag] = float("inf")

    effective_knn = min(int(knn), candidate_count - 1)
    if effective_knn <= 0:
        raise ValueError("knn must be positive when more than one candidate token is available.")
    knn_distance = torch.topk(candidate_distance, k=effective_knn, dim=-1, largest=False).values

    candidate_distance[candidate_diag, candidate_diag] = 0.0

    rho = torch.exp(-(knn_distance * knn_distance).mean(dim=-1))
    if eps > 0:
        rho.add_(torch.rand_like(rho) * eps)

    higher_density = rho.unsqueeze(0) > rho.unsqueeze(1)
    max_distance = candidate_distance.max()
    delta_matrix = torch.where(
        higher_density,
        candidate_distance,
        torch.full_like(candidate_distance, max_distance),
    )
    delta = delta_matrix.min(dim=-1).values

    has_higher_density = higher_density.any(dim=-1)
    max_dist_to_others = candidate_distance.max(dim=-1).values
    delta = torch.where(has_higher_density, delta, max_dist_to_others)

    score = rho * delta
    selected_local = torch.topk(score, k=keep_num, largest=True).indices
    selected_by_score = candidate_idx[selected_local]
    selected_scores = score[selected_local]
    selected_sorted = torch.sort(selected_by_score).values

    score_map = torch.zeros(token_count, dtype=torch.float32, device=device)
    score_map[candidate_idx] = score
    selected_mask = torch.zeros(token_count, dtype=torch.bool, device=device)
    selected_mask[selected_by_score] = True

    return {
        "candidate_indices": candidate_idx,
        "rho": rho,
        "delta": delta,
        "candidate_scores": score,
        "selected_indices_by_score": selected_by_score,
        "selected_indices": selected_sorted,
        "selected_scores_by_score": selected_scores,
        "score_map": score_map,
        "selected_mask": selected_mask,
        "candidate_ratio": float(candidate_ratio),
        "knn": int(knn),
        "effective_knn": int(effective_knn),
        "eps": float(eps),
        "keep_num": int(keep_num),
    }


def to_cpu_debug(debug: Dict[str, object]) -> Dict[str, object]:
    result = {}
    for key, value in debug.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
        else:
            result[key] = value
    return result


def selected_records(data: Dict[str, object]) -> List[Dict[str, object]]:
    rows, cols = tuple(map(int, data["patch_grid"]))
    candidate_indices = data["candidate_indices"].tolist()
    local_by_index = {int(index): local for local, index in enumerate(candidate_indices)}
    selected = data["selected_indices_by_score"].tolist()
    scores = data["candidate_scores"].tolist()
    rho = data["rho"].tolist()
    delta = data["delta"].tolist()
    records = []
    for rank, index in enumerate(selected, start=1):
        index = int(index)
        local = local_by_index[index]
        records.append(
            {
                "case_id": data["case_id"],
                "rank": rank,
                "index": index,
                "row": index // cols,
                "col": index % cols,
                "score": float(scores[local]),
                "rho": float(rho[local]),
                "delta": float(delta[local]),
                "is_candidate": True,
                "is_selected": True,
            }
        )
    return records


def save_selected_csv(path: Path, records: List[Dict[str, object]]) -> None:
    fieldnames = ["case_id", "rank", "index", "row", "col", "score", "rho", "delta", "is_candidate", "is_selected"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_summary_json(path: Path, data: Dict[str, object], records: List[Dict[str, object]]) -> None:
    summary = {
        "case_id": data["case_id"],
        "image_path": data["image_path"],
        "image_relpath": data["image_relpath"],
        "question": data["question"],
        "image_size": list(data["image_size"]),
        "patch_grid": list(data["patch_grid"]),
        "visual_token_count": int(data["visual_token_count"]),
        "keep_num": int(data["keep_num"]),
        "candidate_ratio": float(data["candidate_ratio"]),
        "knn": int(data["knn"]),
        "eps": float(data["eps"]),
        "consistency_ok": bool(data.get("consistency_ok", False)),
        "candidate_indices": [int(value) for value in data["candidate_indices"].tolist()],
        "selected_indices": [int(value) for value in data["selected_indices"].tolist()],
        "selected_tokens": records,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_case(case, args, model, image_processor) -> Dict[str, object]:
    image = load_image(case["image"])
    image_tensor = process_images([image], image_processor, model.config).to(
        model.device,
        dtype=model_dtype(model, args.device),
    )

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    with torch.inference_mode():
        visual_tokens = get_projected_visual_tokens_from_original_llava(model, image_tensor)

    if isinstance(visual_tokens, list):
        if len(visual_tokens) != 1:
            raise NotImplementedError("This dump script supports one image per case.")
        visual_tokens = visual_tokens[0]
    if visual_tokens.dim() == 3 and visual_tokens.size(0) == 1:
        visual_tokens = visual_tokens[0]

    visual_token_count = int(visual_tokens.size(0))
    patch_grid = parse_patch_grid(args.patch_grid, visual_token_count)

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    debug = compute_knn_density_peak_debug(
        visual_tokens=visual_tokens,
        keep_num=args.keep_num,
        candidate_ratio=args.candidate_ratio,
        knn=args.knn,
        eps=args.eps,
    )

    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    reference = select_visual_token_indices(
        visual_tokens,
        keep_num=args.keep_num,
        candidate_ratio=args.candidate_ratio,
        knn=args.knn,
        sort_indices=True,
        eps=args.eps,
    )
    if reference.dim() == 2:
        reference = reference[0]
    consistency_ok = torch.equal(debug["selected_indices"], reference)
    if args.check_consistency and not consistency_ok:
        raise RuntimeError(f"{case['id']}: debug selection does not match visual_selector.py.")

    result = {
        **to_cpu_debug(debug),
        "case_id": case["id"],
        "image_path": case["image"],
        "image_relpath": case.get("image_relpath", case["image"]),
        "question": case["question"],
        "answer": case.get("answer"),
        "category": case.get("category"),
        "source": case.get("source"),
        "image_size": (int(image.height), int(image.width)),
        "patch_grid": tuple(map(int, patch_grid)),
        "visual_token_count": int(visual_token_count),
        "reference_selected_indices": reference.detach().cpu(),
        "consistency_ok": bool(consistency_ok),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("visualize/output/knn_feature_selection"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--keep-num", type=int, default=144)
    parser.add_argument("--candidate-ratio", type=float, default=2.0)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patch-grid", type=str, default=None)
    parser.add_argument("--check-consistency", action="store_true", default=True)
    parser.add_argument("--no-check-consistency", dest="check_consistency", action="store_false")
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

    all_records = []
    for case in iter_cases(args.cases):
        result = run_case(case, args, model, image_processor)
        records = selected_records(result)
        all_records.extend(records)

        output_path = args.output_dir / f"{case['id']}.pt"
        torch.save(result, output_path)
        save_selected_csv(args.output_dir / f"{case['id']}_selected_tokens.csv", records)
        save_summary_json(args.output_dir / f"{case['id']}.json", result, records)
        print(f"saved {output_path} consistency_ok={result['consistency_ok']}")

    if all_records:
        save_selected_csv(args.output_dir / "selected_tokens_all_cases.csv", all_records)


if __name__ == "__main__":
    main()
