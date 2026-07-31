"""Dump Diversity -> APS visual-token selector internals for visualization."""

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
    """Import LLaVA after CLI parsing so --help stays lightweight."""
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


def pairwise_cosine_distance(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokens.dim() != 2:
        raise ValueError("tokens must have shape [N, D].")
    x = F.normalize(tokens.float(), dim=-1)
    distance = 1.0 - x @ x.transpose(0, 1)
    distance.clamp_(min=0.0, max=2.0)
    return x, distance


@torch.no_grad()
def pure_diversity_select_debug(
    visual_tokens: torch.Tensor,
    select_num: int,
    distance: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if visual_tokens.dim() != 2:
        raise ValueError("visual_tokens must have shape [N, D].")

    device = visual_tokens.device
    token_count = visual_tokens.size(0)
    select_num = int(select_num)

    if select_num <= 0 or token_count <= 0:
        return {"selected_indices": torch.empty(0, dtype=torch.long, device=device)}
    if select_num >= token_count:
        return {"selected_indices": torch.arange(token_count, device=device, dtype=torch.long)}

    if distance is None:
        _, distance = pairwise_cosine_distance(visual_tokens)
    elif distance.shape != (token_count, token_count):
        raise ValueError("distance must have shape [N, N].")

    selected_idx = torch.empty(select_num, device=device, dtype=torch.long)

    diag = torch.arange(token_count, device=device)
    distance_no_self = distance.clone()
    distance_no_self[diag, diag] = float("inf")
    first_idx = torch.argmax(distance_no_self.min(dim=1).values)

    selected_idx[0] = first_idx
    min_dist_to_selected = distance[first_idx].contiguous()
    min_dist_to_selected[first_idx] = -1.0

    for selected_pos in range(1, select_num):
        next_idx = torch.argmax(min_dist_to_selected)
        selected_idx[selected_pos] = next_idx
        torch.minimum(min_dist_to_selected, distance[next_idx], out=min_dist_to_selected)
        min_dist_to_selected[selected_idx[: selected_pos + 1]] = -1.0

    return {"selected_indices": selected_idx}


@torch.no_grad()
def adaptive_peak_support_debug(
    tokens: torch.Tensor,
    keep_num: int,
    knn: int = 8,
    beta: float = 1.0,
    eps: float = 0.0,
    distance: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor | int | float | bool]:
    if tokens.dim() != 2:
        raise ValueError("tokens must have shape [M, D].")

    device = tokens.device
    token_count = tokens.size(0)
    keep_num = int(keep_num)
    diag = torch.arange(token_count, device=device)

    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_float = torch.empty(0, dtype=torch.float32, device=device)

    if keep_num <= 0 or token_count <= 0:
        return {
            "selected_local_indices": empty_long,
            "peak_indices": empty_long,
            "rho": empty_float,
            "delta": empty_float,
            "score": empty_float,
            "assigned_peak": empty_long,
            "member_dist": empty_float,
            "threshold": empty_float,
            "support_mask": torch.empty(0, dtype=torch.bool, device=device),
            "peak_support_indices": empty_long,
            "fallback_no_peak": False,
        }

    if distance is None:
        _, distance = pairwise_cosine_distance(tokens)
    elif distance.shape != (token_count, token_count):
        raise ValueError("distance must have shape [M, M].")

    if keep_num >= token_count:
        return {
            "selected_local_indices": diag,
            "peak_indices": diag,
            "rho": torch.ones(token_count, dtype=torch.float32, device=device),
            "delta": torch.ones(token_count, dtype=torch.float32, device=device),
            "score": torch.ones(token_count, dtype=torch.float32, device=device),
            "assigned_peak": diag,
            "member_dist": torch.zeros(token_count, dtype=torch.float32, device=device),
            "threshold": torch.zeros(token_count, dtype=torch.float32, device=device),
            "support_mask": torch.ones(token_count, dtype=torch.bool, device=device),
            "peak_support_indices": diag,
            "fallback_no_peak": False,
        }

    distance_no_self = distance.clone()
    distance_no_self[diag, diag] = float("inf")

    effective_knn = min(max(int(knn), 1), token_count - 1)
    knn_result = torch.topk(distance_no_self, k=effective_knn, dim=-1, largest=False)
    knn_dist = knn_result.values
    neighbor_idx = knn_result.indices

    rho = torch.exp(-(knn_dist * knn_dist).mean(dim=-1))
    if eps > 0:
        rho = rho + torch.rand_like(rho) * eps

    higher_density = rho.unsqueeze(0) > rho.unsqueeze(1)
    max_distance = distance.max()
    delta_matrix = torch.where(higher_density, distance, torch.full_like(distance, max_distance))
    delta = delta_matrix.min(dim=-1).values

    has_higher_density = higher_density.any(dim=-1)
    max_dist_to_others = distance.max(dim=-1).values
    delta = torch.where(has_higher_density, delta, max_dist_to_others)

    score = rho * delta
    neighbor_score = score[neighbor_idx]
    is_peak = (score.unsqueeze(-1) >= neighbor_score).all(dim=-1)
    is_peak = is_peak & (score >= score.mean())
    peak_idx = torch.where(is_peak)[0]

    if peak_idx.numel() == 0:
        selected_idx = torch.topk(score, k=keep_num, largest=True).indices
        return {
            "selected_local_indices": selected_idx,
            "peak_indices": empty_long,
            "rho": rho,
            "delta": delta,
            "score": score,
            "assigned_peak": torch.full((token_count,), -1, dtype=torch.long, device=device),
            "member_dist": torch.full((token_count,), float("nan"), dtype=score.dtype, device=device),
            "threshold": torch.full((token_count,), float("nan"), dtype=score.dtype, device=device),
            "support_mask": torch.zeros(token_count, dtype=torch.bool, device=device),
            "peak_support_indices": empty_long,
            "fallback_no_peak": True,
            "effective_knn": int(effective_knn),
        }

    dist_to_peaks = distance.index_select(1, peak_idx)
    assign = torch.argmin(dist_to_peaks, dim=1)
    member_dist = dist_to_peaks[diag, assign]

    peak_count = peak_idx.numel()
    count = torch.bincount(assign, minlength=peak_count).to(dtype=member_dist.dtype)
    sum_dist = torch.zeros(peak_count, device=device, dtype=member_dist.dtype)
    sum_dist.scatter_add_(0, assign, member_dist)
    mean_dist = sum_dist / count.clamp_min(1.0)

    centered = member_dist - mean_dist.index_select(0, assign)
    sum_sq = torch.zeros(peak_count, device=device, dtype=member_dist.dtype)
    sum_sq.scatter_add_(0, assign, centered * centered)
    std_dist = torch.sqrt(sum_sq / count.clamp_min(1.0))

    threshold = mean_dist.index_select(0, assign) + float(beta) * std_dist.index_select(0, assign)
    support_mask = member_dist <= threshold + float(eps)
    peak_support_idx = torch.where(support_mask)[0]

    if peak_support_idx.numel() == 0:
        selected_idx = torch.topk(score, k=keep_num, largest=True).indices
    elif peak_support_idx.numel() >= keep_num:
        support_score = score.index_select(0, peak_support_idx)
        top_local = torch.topk(support_score, k=keep_num, largest=True).indices
        selected_idx = peak_support_idx.index_select(0, top_local)
    else:
        selected_idx = peak_support_idx
        remaining_mask = torch.ones(token_count, dtype=torch.bool, device=device)
        remaining_mask[selected_idx] = False
        remaining_idx = diag[remaining_mask]
        need = keep_num - selected_idx.numel()
        extra_local = torch.topk(score.index_select(0, remaining_idx), k=need, largest=True).indices
        extra_idx = remaining_idx.index_select(0, extra_local)
        selected_idx = torch.cat((selected_idx, extra_idx), dim=0)

    return {
        "selected_local_indices": selected_idx,
        "peak_indices": peak_idx,
        "rho": rho,
        "delta": delta,
        "score": score,
        "assigned_peak": assign,
        "member_dist": member_dist,
        "threshold": threshold,
        "support_mask": support_mask,
        "peak_support_indices": peak_support_idx,
        "peak_member_count": count,
        "peak_mean_dist": mean_dist,
        "peak_std_dist": std_dist,
        "fallback_no_peak": False,
        "effective_knn": int(effective_knn),
    }


@torch.no_grad()
def diversity_then_aps_debug(
    visual_tokens: torch.Tensor,
    keep_num: int,
    candidate_ratio: float = 2.0,
    knn: int = 8,
    beta: float = 1.0,
    eps: float = 0.0,
) -> Dict[str, object]:
    if visual_tokens.dim() == 3:
        if visual_tokens.size(0) != 1:
            raise NotImplementedError("This dump script supports batch size 1.")
        visual_tokens = visual_tokens[0]
    if visual_tokens.dim() != 2:
        raise ValueError("visual_tokens must have shape [N, D] or [1, N, D].")

    token_count = visual_tokens.size(0)
    keep_num = min(max(int(keep_num), 0), token_count)
    candidate_num = min(max(math.ceil(float(candidate_ratio) * keep_num), keep_num), token_count)
    _, distance = pairwise_cosine_distance(visual_tokens)

    diversity = pure_diversity_select_debug(visual_tokens, candidate_num, distance=distance)
    candidate_idx = diversity["selected_indices"]
    candidate_tokens = visual_tokens.index_select(0, candidate_idx)
    candidate_distance = distance.index_select(0, candidate_idx).index_select(1, candidate_idx)
    aps = adaptive_peak_support_debug(
        tokens=candidate_tokens,
        keep_num=keep_num,
        knn=knn,
        beta=beta,
        eps=eps,
        distance=candidate_distance,
    )

    local_selected = aps["selected_local_indices"]
    selected_by_score = candidate_idx.index_select(0, local_selected)
    selected_sorted = torch.sort(selected_by_score).values
    peak_indices = aps["peak_indices"]
    peak_token_indices = candidate_idx.index_select(0, peak_indices) if peak_indices.numel() else peak_indices

    return {
        **aps,
        "candidate_indices": candidate_idx,
        "selected_indices_by_aps": selected_by_score,
        "selected_indices": selected_sorted,
        "peak_token_indices": peak_token_indices,
        "candidate_ratio": float(candidate_ratio),
        "candidate_num": int(candidate_num),
        "keep_num": int(keep_num),
        "knn": int(knn),
        "beta": float(beta),
        "eps": float(eps),
        "visual_token_count": int(token_count),
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
    _, cols = tuple(map(int, data["patch_grid"]))
    candidate_indices = data["candidate_indices"].tolist()
    local_by_index = {int(index): local for local, index in enumerate(candidate_indices)}
    peak_tokens = set(int(value) for value in data["peak_token_indices"].tolist())
    selected = data["selected_indices_by_aps"].tolist()
    score = data["score"].tolist()
    rho = data["rho"].tolist()
    delta = data["delta"].tolist()
    assigned_peak = data["assigned_peak"].tolist()
    peak_indices = data["peak_indices"].tolist()
    records = []
    for rank, index in enumerate(selected, start=1):
        index = int(index)
        local = local_by_index[index]
        peak_ord = int(assigned_peak[local]) if assigned_peak and assigned_peak[local] >= 0 else -1
        assigned_peak_index = int(candidate_indices[peak_indices[peak_ord]]) if peak_ord >= 0 else -1
        records.append(
            {
                "case_id": data["case_id"],
                "index": index,
                "row": index // cols,
                "col": index % cols,
                "rank": rank,
                "is_peak": index in peak_tokens,
                "is_candidate": True,
                "score": float(score[local]),
                "rho": float(rho[local]),
                "delta": float(delta[local]),
                "assigned_peak_index": assigned_peak_index,
            }
        )
    return records


def save_selected_csv(path: Path, records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "index",
        "row",
        "col",
        "rank",
        "is_peak",
        "is_candidate",
        "score",
        "rho",
        "delta",
        "assigned_peak_index",
    ]
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
        "candidate_num": int(data["candidate_num"]),
        "knn": int(data["knn"]),
        "beta": float(data["beta"]),
        "eps": float(data["eps"]),
        "consistency_ok": bool(data.get("consistency_ok", False)),
        "fallback_no_peak": bool(data.get("fallback_no_peak", False)),
        "candidate_indices": [int(value) for value in data["candidate_indices"].tolist()],
        "peak_token_indices": [int(value) for value in data["peak_token_indices"].tolist()],
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
    debug = diversity_then_aps_debug(
        visual_tokens=visual_tokens,
        keep_num=args.keep_num,
        candidate_ratio=args.candidate_ratio,
        knn=args.knn,
        beta=args.beta,
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
        beta=args.beta,
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
    parser.add_argument("--output-dir", type=Path, default=Path("visualize/outputs/aps_selector"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--keep-num", type=int, default=144)
    parser.add_argument("--candidate-ratio", type=float, default=2.0)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
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
        save_selected_csv(args.output_dir / f"{case['id']}_D_selected_tokens.csv", records)
        save_summary_json(args.output_dir / f"{case['id']}.json", result, records)
        print(f"saved {output_path} consistency_ok={result['consistency_ok']}")

    if all_records:
        save_selected_csv(args.output_dir / "D_selected_tokens_all_cases.csv", all_records)


if __name__ == "__main__":
    main()
