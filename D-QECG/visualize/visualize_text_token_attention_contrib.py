"""Plot which text tokens contribute to high-scoring visual tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Rectangle


DEFAULT_TARGET_REGION_TOKENS = {
    "case_0001": [207, 255, 302, 374, 445],
    "case_0002": [321, 345, 372, 399, 421],
    "case_0003": [132, 156, 180, 204, 228],
    "case_0004": [250, 299, 348, 398, 443],
    "case_0005": [196, 221, 269, 317, 413],
}


def load_pt(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def tensor_1d(value, name: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing {name}. Re-run dump with --dump-text-contrib.")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.float().view(-1)


def tensor_2d(value, name: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing {name}. Re-run dump with --dump-text-contrib.")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2D [text_tokens, visual_tokens], got {tuple(tensor.shape)}.")
    return tensor.float()


def token_labels(data: Dict[str, object], text_count: int) -> List[str]:
    tokens = data.get("query_tokens")
    if isinstance(tokens, list) and len(tokens) == text_count:
        return [str(token).replace("▁", "").replace("Ġ", "") for token in tokens]
    return [str(index) for index in range(text_count)]


def visual_row_col(index: int, grid) -> tuple[int, int]:
    rows, cols = int(grid[0]), int(grid[1])
    if index < 0 or index >= rows * cols:
        raise ValueError(f"Visual token {index} is outside {rows}x{cols}.")
    return index // cols, index % cols


def patch_box(index: int, grid, width: int, height: int) -> tuple[float, float, float, float]:
    rows, cols = int(grid[0]), int(grid[1])
    row, col = visual_row_col(index, grid)
    patch_w = width / cols
    patch_h = height / rows
    return col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h


def load_background(data: Dict[str, object], image_root: Path | None) -> Image.Image:
    candidates = []
    image_path = str(data.get("image_path", "") or "")
    image_relpath = str(data.get("image_relpath", "") or "")
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


def score_rank(score: torch.Tensor, index: int) -> int:
    order = torch.argsort(score, descending=True)
    match = torch.where(order == int(index))[0]
    return int(match[0].item()) + 1 if match.numel() else -1


def contribution_for_visual(matrix: torch.Tensor, visual_index: int) -> torch.Tensor:
    column = matrix[:, int(visual_index)].clamp_min(0.0)
    return column / column.sum().clamp_min(1e-12)


def save_case_csv(
    path: Path,
    case_id: str,
    visual_indices: List[int],
    matrix: torch.Tensor,
    labels: List[str],
    grid,
    raw_score: torch.Tensor,
    residual_score: torch.Tensor | None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "visual_index",
                "visual_row",
                "visual_col",
                "visual_rank",
                "early_raw_score",
                "residual_score",
                "residual_rank",
                "text_rank",
                "text_token",
                "contribution",
            ],
        )
        writer.writeheader()
        for visual_rank, visual_index in enumerate(visual_indices, start=1):
            row, col = visual_row_col(visual_index, grid)
            contrib = contribution_for_visual(matrix, visual_index)
            residual_value = float(residual_score[visual_index]) if residual_score is not None else ""
            residual_rank = score_rank(residual_score, visual_index) if residual_score is not None else ""
            for text_rank, value in enumerate(contrib.tolist()):
                writer.writerow(
                    {
                        "case_id": case_id,
                        "visual_index": int(visual_index),
                        "visual_row": row,
                        "visual_col": col,
                        "visual_rank": visual_rank,
                        "early_raw_score": float(raw_score[visual_index]),
                        "residual_score": residual_value,
                        "residual_rank": residual_rank,
                        "text_rank": text_rank,
                        "text_token": labels[text_rank],
                        "contribution": float(value),
                    }
                )


def draw_case_plot(
    path: Path,
    case_id: str,
    question: str,
    group_name: str,
    visual_indices: List[int],
    raw_score: torch.Tensor,
    matrix: torch.Tensor,
    labels: List[str],
    grid,
) -> None:
    x = np.arange(matrix.size(0))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.get_cmap("tab10")
    for line_id, visual_index in enumerate(visual_indices):
        row, col = visual_row_col(visual_index, grid)
        contrib = contribution_for_visual(matrix, visual_index).numpy()
        label = f"v{visual_index} r{row}c{col} raw={float(raw_score[visual_index]):.4g}"
        ax.plot(x, contrib, marker="o", linewidth=1.8, markersize=4.2, color=colors(line_id), label=label)

    ax.set_title(f"{case_id} {group_name}: text-token contribution curves\n{question}")
    ax.set_xlabel("text token index")
    ax.set_ylabel("contribution share for this visual token")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{idx}:{token}" for idx, token in enumerate(labels)], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_single_visual_plot(
    path: Path,
    case_id: str,
    question: str,
    group_name: str,
    visual_index: int,
    visual_rank: int,
    raw_score: torch.Tensor,
    matrix: torch.Tensor,
    labels: List[str],
    grid,
) -> None:
    row, col = visual_row_col(visual_index, grid)
    contrib = contribution_for_visual(matrix, visual_index).numpy()
    x = np.arange(matrix.size(0))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(x, contrib, marker="o", linewidth=2.0, markersize=5.0, color="#1f77b4")
    ax.fill_between(x, contrib, 0, alpha=0.18, color="#1f77b4")
    ax.set_title(
        f"{case_id} {group_name} visual token {visual_index} rank {visual_rank} r{row}c{col} raw={float(raw_score[visual_index]):.4g}\n"
        f"{question}"
    )
    ax.set_xlabel("text token index")
    ax.set_ylabel("contribution share for this visual token")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.28)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{idx}:{token}" for idx, token in enumerate(labels)], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_position_residual_panel(
    path: Path,
    case_id: str,
    question: str,
    group_name: str,
    image: Image.Image,
    visual_indices: List[int],
    raw_score: torch.Tensor,
    residual_score: torch.Tensor | None,
    grid,
) -> None:
    colors = [plt.get_cmap("tab10")(i) for i in range(len(visual_indices))]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax_img, ax_bar = axes
    ax_img.imshow(image)
    width, height = image.size
    for rank, visual_index in enumerate(visual_indices, start=1):
        row, col = visual_row_col(visual_index, grid)
        x0, y0, x1, y1 = patch_box(visual_index, grid, width, height)
        color = colors[rank - 1]
        ax_img.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=(color[0], color[1], color[2], 0.22),
                edgecolor=color,
                linewidth=2.4,
            )
        )
        ax_img.text(
            x0 + 2,
            y0 + 12,
            str(rank),
            color="white",
            fontsize=9,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": color, "edgecolor": "none", "alpha": 0.95},
        )
        ax_img.text(
            x0 + 2,
            y1 - 3,
            f"{visual_index}",
            color="white",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "edgecolor": "none", "alpha": 0.6},
        )
    ax_img.set_title("position in original image")
    ax_img.axis("off")

    labels = []
    residual_values = []
    raw_values = []
    for rank, visual_index in enumerate(visual_indices, start=1):
        row, col = visual_row_col(visual_index, grid)
        labels.append(f"{rank}:v{visual_index}\nr{row}c{col}")
        raw_values.append(float(raw_score[visual_index]))
        residual_values.append(float(residual_score[visual_index]) if residual_score is not None else np.nan)

    x = np.arange(len(visual_indices))
    ax_bar.bar(x - 0.18, raw_values, width=0.36, label="early raw", color="#4c78a8", alpha=0.85)
    if residual_score is not None:
        ax_bar.bar(x + 0.18, residual_values, width=0.36, label="residual", color="#f58518", alpha=0.85)
        for i, visual_index in enumerate(visual_indices):
            ax_bar.text(
                i + 0.18,
                residual_values[i],
                f"rank {score_rank(residual_score, visual_index)}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
            )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_title("early raw score vs residual score")
    ax_bar.set_ylabel("score")
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(fontsize=8)
    fig.suptitle(f"{case_id} {group_name}: positions and residual scores\n{question}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_visual_group(
    case_dir: Path,
    case_id: str,
    question: str,
    group_id: str,
    group_name: str,
    visual_indices: List[int],
    raw_score: torch.Tensor,
    residual_score: torch.Tensor | None,
    matrix: torch.Tensor,
    labels: List[str],
    grid,
    image: Image.Image,
) -> None:
    draw_case_plot(
        case_dir / f"{group_id}_text_token_contribution_curves.png",
        case_id,
        question,
        group_name,
        visual_indices,
        raw_score,
        matrix,
        labels,
        grid,
    )
    for visual_rank, visual_index in enumerate(visual_indices, start=1):
        row, col = visual_row_col(visual_index, grid)
        draw_single_visual_plot(
            case_dir / f"{group_id}_visual_{visual_rank:02d}_idx_{visual_index:04d}_r{row:02d}c{col:02d}_curve.png",
            case_id,
            question,
            group_name,
            visual_index,
            visual_rank,
            raw_score,
            matrix,
            labels,
            grid,
        )
    draw_position_residual_panel(
        case_dir / f"{group_id}_position_residual.png",
        case_id,
        question,
        group_name,
        image,
        visual_indices,
        raw_score,
        residual_score,
        grid,
    )
    save_case_csv(
        case_dir / f"{group_id}_text_token_contribution.csv",
        case_id,
        visual_indices,
        matrix,
        labels,
        grid,
        raw_score,
        residual_score,
    )


def process_case(
    path: Path,
    save_dir: Path,
    top_visual: int,
    residual_dir: Path | None = None,
    image_root: Path | None = None,
    tail_count: int = 5,
) -> Dict[str, object]:
    data = load_pt(path)
    case_id = str(data.get("case_id", path.stem))
    case_dir = save_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    residual_score = None
    if residual_dir is not None:
        residual_path = residual_dir / path.name
        if residual_path.exists():
            residual_score = tensor_1d(load_pt(residual_path).get("residual_score"), "residual_score")

    raw_score = tensor_1d(data.get("early_raw_score"), "early_raw_score")
    matrix = tensor_2d(data.get("early_text_to_visual"), "early_text_to_visual")
    if matrix.size(1) != raw_score.numel():
        raise ValueError(
            f"{path} has early_text_to_visual shape {tuple(matrix.shape)} but early_raw_score length {raw_score.numel()}."
        )

    top_k = min(int(top_visual), raw_score.numel())
    visual_indices = [int(value) for value in torch.topk(raw_score, k=top_k, largest=True).indices.tolist()]
    labels = token_labels(data, matrix.size(0))
    grid = data.get("patch_grid", (24, 24))
    question = str(data.get("question", ""))
    image = load_background(data, image_root)

    draw_case_plot(
        case_dir / "text_token_contribution_curves.png",
        case_id,
        question,
        "top raw",
        visual_indices,
        raw_score,
        matrix,
        labels,
        grid,
    )
    for visual_rank, visual_index in enumerate(visual_indices, start=1):
        row, col = visual_row_col(visual_index, grid)
        draw_single_visual_plot(
            case_dir / f"visual_{visual_rank:02d}_idx_{visual_index:04d}_r{row:02d}c{col:02d}_curve.png",
            case_id,
            question,
            "top raw",
            visual_index,
            visual_rank,
            raw_score,
            matrix,
            labels,
            grid,
        )
    draw_position_residual_panel(
        case_dir / "selected_visual_tokens_position_residual.png",
        case_id,
        question,
        "top raw",
        image,
        visual_indices,
        raw_score,
        residual_score,
        grid,
    )
    save_case_csv(
        case_dir / "text_token_contribution.csv",
        case_id,
        visual_indices,
        matrix,
        labels,
        grid,
        raw_score,
        residual_score,
    )

    visual_count = int(raw_score.numel())
    tail_start = max(0, visual_count - int(tail_count))
    tail_indices = list(range(tail_start, visual_count))
    render_visual_group(
        case_dir,
        case_id,
        question,
        "tail_last_tokens",
        "tail last tokens",
        tail_indices,
        raw_score,
        residual_score,
        matrix,
        labels,
        grid,
        image,
    )

    target_indices = DEFAULT_TARGET_REGION_TOKENS.get(case_id)
    if target_indices:
        render_visual_group(
            case_dir,
            case_id,
            question,
            "target_region",
            "target region tokens",
            target_indices,
            raw_score,
            residual_score,
            matrix,
            labels,
            grid,
            image,
        )
    return {"case_id": case_id, "visual_indices": visual_indices, "text_count": int(matrix.size(0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=Path("visualize/outputs/sink_residual/no_prune"))
    parser.add_argument("--save-dir", type=Path, default=Path("visualize/figures/text_token_attention_contrib"))
    parser.add_argument("--residual-dir", type=Path, default=Path("visualize/outputs/sink_residual/with_residual"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--top-visual", type=int, default=5)
    parser.add_argument("--tail-count", type=int, default=5)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.dump_dir.glob("*.pt"))[: args.num_cases]:
        result = process_case(path, args.save_dir, args.top_visual, args.residual_dir, args.image_root, args.tail_count)
        rows.append(result)
        print(f"{result['case_id']}: top visual tokens {result['visual_indices']} text_count={result['text_count']}")

    with (args.save_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "text_count", "visual_indices"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "text_count": row["text_count"],
                    "visual_indices": ",".join(map(str, row["visual_indices"])),
                }
            )


if __name__ == "__main__":
    main()
