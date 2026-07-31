"""Visualize cumulative-gain scores from raw text-to-vision attention matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Rectangle


def load_pt(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def tensor_2d(value, name: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing {name}; re-dump with --dump-text-contrib after raw-A support was added.")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be [text_tokens, visual_tokens], got {tuple(tensor.shape)}.")
    return tensor.float()


def token_labels(data: Dict[str, object], text_count: int) -> List[str]:
    tokens = data.get("query_tokens")
    if isinstance(tokens, list) and len(tokens) == text_count:
        return [str(token).replace("▁", "").replace("Ġ", "") for token in tokens]
    return [str(i) for i in range(text_count)]


def patch_grid(data: Dict[str, object], visual_count: int) -> Tuple[int, int]:
    grid = data.get("patch_grid", None)
    if grid:
        return int(grid[0]), int(grid[1])
    side = int(round(np.sqrt(visual_count)))
    if side * side != visual_count:
        raise ValueError(f"Cannot infer square patch grid for {visual_count} visual tokens.")
    return side, side


def row_col(index: int, grid: Tuple[int, int]) -> Tuple[int, int]:
    return int(index) // grid[1], int(index) % grid[1]


def patch_box(index: int, grid: Tuple[int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    row, col = row_col(index, grid)
    patch_w = width / grid[1]
    patch_h = height / grid[0]
    return col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h


def load_background(data: Dict[str, object], image_root: Path | None) -> Image.Image:
    image_path = str(data.get("image_path", "") or "")
    image_relpath = str(data.get("image_relpath", "") or "")
    candidates = []
    if image_path:
        candidates.append(Path(image_path))
    if image_root is not None and image_relpath:
        candidates.append(image_root / image_relpath)
    if image_root is not None and image_path:
        candidates.append(image_root / Path(image_path).name)
    for candidate in candidates:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    height, width = data.get("image_size", (336, 336))
    return Image.new("RGB", (int(width), int(height)), (245, 245, 245))


def compute_scores(A: np.ndarray, early_k: int = 2, gate_percentile: float = 10.0, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    if A.ndim != 2:
        raise ValueError(f"A must be [T,V], got {A.shape}.")
    T, _ = A.shape
    A = np.maximum(A.astype(np.float64), 0.0)
    raw_score = A.sum(axis=0)
    pi = A / np.maximum(raw_score[None, :], eps)
    C = np.cumsum(pi, axis=0)
    k = min(max(int(early_k), 0), T - 1)
    L = 1.0 - C[k, :]
    H = -(pi * np.log(pi + eps)).sum(axis=0)
    H_norm = H / (np.log(max(T, 2)) + eps)
    threshold = np.percentile(raw_score, float(gate_percentile))
    M = (raw_score > threshold).astype(np.float32)
    final_score = L * H_norm * M
    return {
        "raw_score": raw_score,
        "L": L,
        "H_norm": H_norm,
        "M": M,
        "final_score": final_score,
        "pi": pi,
        "C": C,
        "gate_threshold": np.array([threshold], dtype=np.float64),
        "early_k": np.array([k], dtype=np.int64),
    }


def rank_array(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(score) + 1)
    return ranks


def top_tokens(score: np.ndarray, top_k: int) -> List[int]:
    k = min(int(top_k), score.shape[0])
    return [int(i) for i in np.argsort(-score, kind="mergesort")[:k]]


def save_token_list(path: Path, token_ids: Iterable[int], score: np.ndarray) -> None:
    lines = [f"{idx}\t{float(score[idx]):.10g}" for idx in token_ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_grid(
    path: Path,
    image: Image.Image,
    token_ids: List[int],
    score: np.ndarray,
    grid: Tuple[int, int],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, max(4.5, 7.2 * image.height / max(image.width, 1))))
    ax.imshow(image)
    colors = plt.get_cmap("tab20")
    for rank, token_id in enumerate(token_ids[:64], start=1):
        x0, y0, x1, y1 = patch_box(token_id, grid, image.width, image.height)
        color = colors((rank - 1) % 20)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=(color[0], color[1], color[2], 0.18),
                edgecolor=color,
                linewidth=1.4,
            )
        )
        if rank <= 12:
            ax.text(
                x0 + 1,
                y0 + 10,
                str(rank),
                fontsize=7,
                color="white",
                bbox={"boxstyle": "round,pad=0.14", "facecolor": color, "edgecolor": "none", "alpha": 0.95},
            )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_rank_bar(path: Path, score: np.ndarray, token_ids: List[int], title: str) -> None:
    shown = token_ids[:20]
    values = [float(score[idx]) for idx in shown]
    labels = [str(idx) for idx in shown]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(np.arange(len(shown)), values, color="#4c78a8", alpha=0.86)
    ax.set_xticks(np.arange(len(shown)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("visual token id")
    ax.set_ylabel("score")
    ax.grid(True, axis="y", alpha=0.24)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_curves(
    path: Path,
    curve: np.ndarray,
    token_ids: List[int],
    labels: List[str],
    title: str,
    ylabel: str,
    max_lines: int = 8,
) -> None:
    shown = token_ids[:max_lines]
    x = np.arange(curve.shape[0])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    cmap = plt.get_cmap("tab10")
    for i, token_id in enumerate(shown):
        ax.plot(x, curve[:, token_id], marker="o", linewidth=1.8, markersize=4.0, color=cmap(i), label=f"v{token_id}")
    ax.set_title(title)
    ax.set_xlabel("text token index")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="best", fontsize=8, framealpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{idx}:{token}" for idx, token in enumerate(labels)], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_score_table(path: Path, scores: Dict[str, np.ndarray], tops: Dict[str, List[int]], grid: Tuple[int, int]) -> None:
    ranks = {
        "raw": rank_array(scores["raw_score"]),
        "L": rank_array(scores["L"]),
        "H": rank_array(scores["H_norm"]),
        "final": rank_array(scores["final_score"]),
    }
    top_sets = {name: set(values) for name, values in tops.items()}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "token_id",
                "row",
                "col",
                "raw_attention",
                "L_score",
                "entropy_score",
                "hard_gate",
                "final_score",
                "raw_rank",
                "L_rank",
                "entropy_rank",
                "final_rank",
                "is_top_raw",
                "is_top_L",
                "is_top_H",
                "is_top_final",
            ],
        )
        writer.writeheader()
        for token_id in range(scores["raw_score"].shape[0]):
            row, col = row_col(token_id, grid)
            writer.writerow(
                {
                    "token_id": token_id,
                    "row": row,
                    "col": col,
                    "raw_attention": float(scores["raw_score"][token_id]),
                    "L_score": float(scores["L"][token_id]),
                    "entropy_score": float(scores["H_norm"][token_id]),
                    "hard_gate": int(scores["M"][token_id]),
                    "final_score": float(scores["final_score"][token_id]),
                    "raw_rank": int(ranks["raw"][token_id]),
                    "L_rank": int(ranks["L"][token_id]),
                    "entropy_rank": int(ranks["H"][token_id]),
                    "final_rank": int(ranks["final"][token_id]),
                    "is_top_raw": int(token_id in top_sets["raw"]),
                    "is_top_L": int(token_id in top_sets["L"]),
                    "is_top_H": int(token_id in top_sets["H"]),
                    "is_top_final": int(token_id in top_sets["final"]),
                }
            )


def overlap_report(tops: Dict[str, List[int]]) -> Dict[str, object]:
    raw = set(tops["raw"])
    final = set(tops["final"])
    L = set(tops["L"])
    H = set(tops["H"])
    pairs = {
        "raw_final": raw & final,
        "raw_L": raw & L,
        "raw_H": raw & H,
        "L_H": L & H,
        "L_final": L & final,
        "H_final": H & final,
    }
    return {name: sorted(values) for name, values in pairs.items()}


def save_overlap_report(path: Path, report: Dict[str, List[int]], top_k: int) -> None:
    lines = []
    for name, values in report.items():
        lines.append(f"{name}_count: {len(values)} / {top_k}")
        lines.append(f"{name}_tokens: {','.join(map(str, values))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_case(path: Path, args) -> Dict[str, object]:
    data = load_pt(path)
    case_id = str(data.get("case_id", path.stem))
    case_dir = args.save_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    A = tensor_2d(data.get(args.matrix_key), args.matrix_key).numpy()
    scores = compute_scores(A, early_k=args.early_k, gate_percentile=args.gate_percentile, eps=args.eps)
    visual_count = A.shape[1]
    grid = patch_grid(data, visual_count)
    image = load_background(data, args.image_root)
    labels = token_labels(data, A.shape[0])

    tops = {
        "raw": top_tokens(scores["raw_score"], args.top_k),
        "L": top_tokens(scores["L"], args.top_k),
        "H": top_tokens(scores["H_norm"], args.top_k),
        "final": top_tokens(scores["final_score"], args.top_k),
    }

    draw_grid(case_dir / "raw_topK_grid.png", image, tops["raw"], scores["raw_score"], grid, "Raw Attention Top-K")
    draw_grid(case_dir / "L_topK_grid.png", image, tops["L"], scores["L"], grid, "L Score Top-K")
    draw_grid(case_dir / "H_topK_grid.png", image, tops["H"], scores["H_norm"], grid, "Entropy Score Top-K")
    draw_grid(case_dir / "final_topK_grid.png", image, tops["final"], scores["final_score"], grid, "Final Score Top-K")

    draw_rank_bar(case_dir / "raw_topK_rank_bar.png", scores["raw_score"], tops["raw"], "Raw Attention Top Tokens")
    draw_rank_bar(case_dir / "L_topK_rank_bar.png", scores["L"], tops["L"], "L Score Top Tokens")
    draw_rank_bar(case_dir / "H_topK_rank_bar.png", scores["H_norm"], tops["H"], "Entropy Score Top Tokens")
    draw_rank_bar(case_dir / "final_topK_rank_bar.png", scores["final_score"], tops["final"], "Final Score Top Tokens")

    draw_curves(case_dir / "raw_topK_curves.png", A, tops["raw"], labels, "Raw Top-K Raw Attention Curves", "raw attention")
    draw_curves(case_dir / "L_topK_cumulative_curves.png", scores["C"], tops["L"], labels, "L Top-K Cumulative Curves", "C_v(t)")
    draw_curves(case_dir / "H_topK_pi_curves.png", scores["pi"], tops["H"], labels, "Entropy Top-K Contribution Curves", "pi_v(t)")
    draw_curves(case_dir / "final_topK_raw_curves.png", A, tops["final"], labels, "Final Top-K Raw Attention Curves", "raw attention")
    draw_curves(case_dir / "final_topK_cumulative_curves.png", scores["C"], tops["final"], labels, "Final Top-K Cumulative Curves", "C_v(t)")
    draw_curves(case_dir / "final_topK_pi_curves.png", scores["pi"], tops["final"], labels, "Final Top-K Contribution Curves", "pi_v(t)")

    save_token_list(case_dir / "top_raw_tokens.txt", tops["raw"], scores["raw_score"])
    save_token_list(case_dir / "top_L_tokens.txt", tops["L"], scores["L"])
    save_token_list(case_dir / "top_H_tokens.txt", tops["H"], scores["H_norm"])
    save_token_list(case_dir / "top_final_tokens.txt", tops["final"], scores["final_score"])
    save_score_table(case_dir / "visual_token_score_table.csv", scores, tops, grid)
    report = overlap_report(tops)
    save_overlap_report(case_dir / "token_overlap_report.txt", report, args.top_k)

    return {
        "case_id": case_id,
        "text_tokens": int(A.shape[0]),
        "visual_tokens": int(A.shape[1]),
        "gate_threshold": float(scores["gate_threshold"][0]),
        "raw_final_overlap": len(report["raw_final"]),
        "raw_L_overlap": len(report["raw_L"]),
        "raw_H_overlap": len(report["raw_H"]),
        "L_final_overlap": len(report["L_final"]),
        "H_final_overlap": len(report["H_final"]),
    }


def save_summary(save_dir: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "text_tokens",
        "visual_tokens",
        "gate_threshold",
        "raw_final_overlap",
        "raw_L_overlap",
        "raw_H_overlap",
        "L_final_overlap",
        "H_final_overlap",
    ]
    with (save_dir / "summary_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if rows:
        lines = [f"cases: {len(rows)}"]
        for key in fieldnames[4:]:
            values = np.array([row[key] for row in rows], dtype=np.float64)
            lines.append(f"mean_{key}: {values.mean():.4f}")
        (save_dir / "summary_overlap.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=Path("visualize/outputs/sink_residual/no_prune_text_contrib_raw"))
    parser.add_argument("--save-dir", type=Path, default=Path("visualize/figures/cumulative_gain_vis"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--matrix-key", type=str, default="early_text_to_visual_raw")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--early-k", type=int, default=2)
    parser.add_argument("--gate-percentile", type=float, default=10.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.dump_dir.glob("*.pt")):
        row = process_case(path, args)
        rows.append(row)
        print(
            f"{row['case_id']}: raw_final={row['raw_final_overlap']} "
            f"raw_L={row['raw_L_overlap']} raw_H={row['raw_H_overlap']}"
        )
    save_summary(args.save_dir, rows)


if __name__ == "__main__":
    main()
