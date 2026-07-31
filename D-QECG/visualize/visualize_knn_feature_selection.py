"""Visualize first-stage KNN feature-selection scores and selected tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle


def load_pt(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def tensor_1d(value, name: str, dtype=torch.float32) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing {name}.")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.to(dtype=dtype).view(-1)


def patch_grid(data: Dict[str, object]) -> Tuple[int, int]:
    grid = data.get("patch_grid", (24, 24))
    return int(grid[0]), int(grid[1])


def load_background(path: str, image_size, image_root: Path | None = None, image_relpath: str | None = None) -> Image.Image:
    candidates = [Path(path)] if path else []
    if image_root is not None and image_relpath:
        candidates.append(image_root / image_relpath)
    if image_root is not None and path:
        candidates.append(image_root / Path(path).name)

    for candidate in candidates:
        if candidate.exists():
            return Image.open(candidate).convert("RGB")

    height, width = image_size
    return Image.new("RGB", (int(width), int(height)), (245, 245, 245))


def normalize_to_0_1(
    score: torch.Tensor,
    low_quantile: float = 0.0,
    high_quantile: float = 0.985,
    gamma: float = 0.7,
) -> np.ndarray:
    array = score.detach().float().cpu().numpy()
    if array.size == 0:
        return array
    min_value = float(np.nanquantile(array, low_quantile))
    max_value = float(np.nanquantile(array, high_quantile))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    normalized = np.clip((array - min_value) / (max_value - min_value), 0.0, 1.0)
    if gamma > 0:
        normalized = normalized ** gamma
    return normalized.astype(np.float32)


def score_map(data: Dict[str, object]) -> torch.Tensor:
    if "score_map" in data:
        return tensor_1d(data["score_map"], "score_map")

    token_count = int(data.get("visual_token_count", 576))
    score = torch.zeros(token_count, dtype=torch.float32)
    candidate_indices = tensor_1d(data["candidate_indices"], "candidate_indices", dtype=torch.long)
    candidate_scores = tensor_1d(data["candidate_scores"], "candidate_scores")
    score[candidate_indices] = candidate_scores
    return score


def score_to_grid(
    score: torch.Tensor,
    grid: Tuple[int, int],
    low_quantile: float,
    high_quantile: float,
    gamma: float,
) -> np.ndarray:
    rows, cols = grid
    values = normalize_to_0_1(score, low_quantile=low_quantile, high_quantile=high_quantile, gamma=gamma)
    if values.size != rows * cols:
        raise ValueError(f"Score length {values.size} does not match patch grid {rows}x{cols}.")
    return values.reshape(rows, cols)


def faded_image_array(image: Image.Image, background_alpha: float) -> np.ndarray:
    base = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(base * background_alpha + (1.0 - background_alpha), 0.0, 1.0)


def overlay_patch_heatmap(
    image: Image.Image,
    heat: np.ndarray,
    title: str,
    output_path: Path,
    background_alpha: float,
    patch_alpha: float,
    cmap_name: str,
    draw_grid: bool,
) -> None:
    rows, cols = heat.shape
    width, height = image.size
    patch_w = width / cols
    patch_h = height / rows
    cmap = plt.get_cmap(cmap_name)

    fig_w = 6.4
    fig_h = max(4.0, fig_w * height / max(width, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(faded_image_array(image, background_alpha))
    for row in range(rows):
        for col in range(cols):
            value = float(heat[row, col])
            alpha = 0.0 if value <= 0 else min(0.95, max(0.06, patch_alpha * (value ** 1.15)))
            edge = (1, 1, 1, 0.45) if draw_grid else (1, 1, 1, 0)
            ax.add_patch(
                Rectangle(
                    (col * patch_w, row * patch_h),
                    patch_w,
                    patch_h,
                    facecolor=cmap(value),
                    edgecolor=edge,
                    linewidth=0.35 if draw_grid else 0,
                    alpha=alpha,
                )
            )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_selected_tokens(
    image: Image.Image,
    selected_indices: torch.Tensor,
    grid: Tuple[int, int],
    output_path: Path,
    background_alpha: float,
    draw_index_labels: bool,
) -> None:
    rows, cols = grid
    width, height = image.size
    patch_w = width / cols
    patch_h = height / rows
    canvas = Image.fromarray(np.uint8(faded_image_array(image, background_alpha) * 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    selected = set(int(value) for value in selected_indices.view(-1).tolist())

    try:
        font = ImageFont.truetype("arial.ttf", max(8, int(min(patch_w, patch_h) * 0.38)))
    except OSError:
        font = ImageFont.load_default()

    for index in selected:
        row = index // cols
        col = index % cols
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        x0 = col * patch_w
        y0 = row * patch_h
        x1 = (col + 1) * patch_w
        y1 = (row + 1) * patch_h
        draw.rectangle((x0, y0, x1, y1), fill=(0, 128, 255, 118), outline=(0, 50, 130, 255), width=2)
        if draw_index_labels:
            label = str(index)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            tx = x0 + max(1, (patch_w - text_w) / 2)
            ty = y0 + max(1, (patch_h - text_h) / 2)
            draw.rectangle((tx - 1, ty - 1, tx + text_w + 1, ty + text_h + 1), fill=(255, 255, 255, 160))
            draw.text((tx, ty), label, fill=(0, 20, 60, 255), font=font)

    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * height / max(width, 1))))
    ax.imshow(canvas)
    ax.set_title("B. Selected Visual Tokens")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_panel(paths: List[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    widths, heights = zip(*(image.size for image in images))
    cell_w = max(widths)
    cell_h = max(heights)
    panel = Image.new("RGB", (cell_w * len(images), cell_h), (255, 255, 255))
    for idx, image in enumerate(images):
        panel.paste(image.resize((cell_w, cell_h)), (idx * cell_w, 0))
    panel.save(output_path)


def selected_records(data: Dict[str, object]) -> List[Dict[str, object]]:
    rows, cols = patch_grid(data)
    candidate_indices = tensor_1d(data["candidate_indices"], "candidate_indices", dtype=torch.long).tolist()
    local_by_index = {int(index): local for local, index in enumerate(candidate_indices)}
    selected = tensor_1d(data["selected_indices_by_score"], "selected_indices_by_score", dtype=torch.long).tolist()
    scores = tensor_1d(data["candidate_scores"], "candidate_scores").tolist()
    rho = tensor_1d(data["rho"], "rho").tolist()
    delta = tensor_1d(data["delta"], "delta").tolist()

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


def visualize_case(
    dump_path: Path,
    save_root: Path,
    image_root: Path | None,
    background_alpha: float,
    patch_alpha: float,
    heat_low_quantile: float,
    heat_high_quantile: float,
    heat_gamma: float,
    cmap_name: str,
    draw_grid: bool,
    draw_index_labels: bool,
) -> List[Dict[str, object]]:
    data = load_pt(dump_path)
    case_id = str(data.get("case_id", dump_path.stem))
    save_dir = save_root / case_id
    save_dir.mkdir(parents=True, exist_ok=True)

    grid = patch_grid(data)
    image_size = tuple(map(int, data.get("image_size", (336, 336))))
    image = load_background(data.get("image_path", ""), image_size, image_root, data.get("image_relpath"))
    selected_indices = tensor_1d(data["selected_indices"], "selected_indices", dtype=torch.long)
    score = score_map(data)

    a_path = save_dir / "A_knn_score_heatmap.png"
    b_path = save_dir / "B_selected_tokens.png"
    c_path = save_dir / "C_score_selected_panel.png"
    records_path = save_dir / "selected_tokens.csv"

    overlay_patch_heatmap(
        image,
        score_to_grid(score, grid, heat_low_quantile, heat_high_quantile, heat_gamma),
        "A. KNN Score = rho * delta",
        a_path,
        background_alpha,
        patch_alpha,
        cmap_name,
        draw_grid,
    )
    draw_selected_tokens(image, selected_indices, grid, b_path, background_alpha, draw_index_labels)
    save_panel([a_path, b_path], c_path)

    records = selected_records(data)
    save_selected_csv(records_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--background-alpha", type=float, default=0.30)
    parser.add_argument("--patch-alpha", type=float, default=0.90)
    parser.add_argument("--heat-low-quantile", type=float, default=0.0)
    parser.add_argument("--heat-high-quantile", type=float, default=0.985)
    parser.add_argument("--heat-gamma", type=float, default=0.7)
    parser.add_argument("--cmap", type=str, default="YlOrRd")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--draw-index-labels", action="store_true")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for dump_path in sorted(args.dump_dir.glob("*.pt")):
        records = visualize_case(
            dump_path,
            args.save_dir,
            args.image_root,
            args.background_alpha,
            args.patch_alpha,
            args.heat_low_quantile,
            args.heat_high_quantile,
            args.heat_gamma,
            args.cmap,
            not args.no_grid,
            args.draw_index_labels,
        )
        all_records.extend(records)
        print(f"saved figures for {dump_path.stem}")

    if all_records:
        save_selected_csv(args.save_dir / "selected_tokens_all_cases.csv", all_records)


if __name__ == "__main__":
    main()
