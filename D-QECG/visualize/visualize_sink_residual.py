"""Visualize raw attention and residual coarse scores for sink suppression."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle


def load_pt(path: Path):
    return torch.load(path, map_location="cpu")


def tensor_1d(value, name: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing {name}.")
    tensor = value.detach().float().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value).float()
    return tensor.view(-1)


def normalize_to_0_1(
    score: torch.Tensor,
    low_quantile: float = 0.05,
    high_quantile: float = 0.98,
    gamma: float = 0.6,
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


def patch_grid(data) -> Tuple[int, int]:
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
    base = np.asarray(image).astype(np.float32) / 255.0
    faded = np.clip(base * background_alpha + (1.0 - background_alpha), 0.0, 1.0)
    rows, cols = heat.shape
    width, height = image.size
    patch_w = width / cols
    patch_h = height / rows
    cmap = plt.get_cmap(cmap_name)

    fig_w = 6
    fig_h = max(4, fig_w * height / max(width, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(faded)
    for row in range(rows):
        for col in range(cols):
            value = float(heat[row, col])
            color = cmap(value)
            alpha = min(0.95, max(0.02, patch_alpha * (value ** 1.35)))
            edge = (1, 1, 1, 0.38) if draw_grid else (1, 1, 1, 0)
            ax.add_patch(
                Rectangle(
                    (col * patch_w, row * patch_h),
                    patch_w,
                    patch_h,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=0.35 if draw_grid else 0,
                    alpha=alpha,
                )
            )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_suppressed_tokens(
    image: Image.Image,
    mask: torch.Tensor,
    grid: Tuple[int, int],
    output_path: Path,
    background_alpha: float,
) -> None:
    rows, cols = grid
    base = np.asarray(image).astype(np.float32) / 255.0
    faded = np.uint8(np.clip(base * background_alpha + (1.0 - background_alpha), 0.0, 1.0) * 255)
    canvas = Image.fromarray(faded)
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    patch_w = width / cols
    patch_h = height / rows

    mask = mask.view(-1).bool().cpu()
    if mask.numel() != rows * cols:
        raise ValueError(f"Mask length {mask.numel()} does not match patch grid {rows}x{cols}.")

    for idx, selected in enumerate(mask.tolist()):
        if not selected:
            continue
        row = idx // cols
        col = idx % cols
        x0 = col * patch_w
        y0 = row * patch_h
        x1 = (col + 1) * patch_w
        y1 = (row + 1) * patch_h
        draw.rectangle((x0, y0, x1, y1), fill=(230, 0, 0, 120), outline=(255, 0, 0, 245), width=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(canvas)
    ax.set_title("D. Suppressed Raw-high Tokens")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_scatter(raw_score: torch.Tensor, residual_score: torch.Tensor, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(raw_score.numpy(), residual_score.numpy(), s=14, alpha=0.72, edgecolors="none")
    ax.set_xlabel("Early raw attention score")
    ax.set_ylabel("Residual coarse score")
    ax.set_title("Raw vs Residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_panel(paths, output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    widths, heights = zip(*(image.size for image in images))
    cell_w = max(widths)
    cell_h = max(heights)
    panel = Image.new("RGB", (cell_w * 2, cell_h * 2), (255, 255, 255))
    for idx, image in enumerate(images):
        row = idx // 2
        col = idx % 2
        panel.paste(image.resize((cell_w, cell_h)), (col * cell_w, row * cell_h))
    panel.save(output_path)


def visualize_case(
    no_prune_path: Path,
    residual_path: Path,
    save_root: Path,
    raw_top_q: float,
    res_low_q: float,
    image_root: Path | None = None,
    background_alpha: float = 0.34,
    patch_alpha: float = 0.88,
    heat_low_quantile: float = 0.05,
    heat_high_quantile: float = 0.98,
    heat_gamma: float = 1.0,
    cmap_name: str = "YlOrRd",
    draw_grid: bool = True,
) -> None:
    raw = load_pt(no_prune_path)
    residual = load_pt(residual_path)
    case_id = str(raw.get("case_id", no_prune_path.stem))
    save_dir = save_root / case_id
    save_dir.mkdir(parents=True, exist_ok=True)

    early_raw = tensor_1d(raw.get("early_raw_score"), "early_raw_score")
    deep_raw = tensor_1d(raw.get("deep_raw_score"), "deep_raw_score")
    residual_score = tensor_1d(residual.get("residual_score"), "residual_score")
    if early_raw.numel() != residual_score.numel():
        raise ValueError(f"{case_id}: raw and residual score lengths differ.")

    grid = patch_grid(raw)
    image_size = tuple(map(int, raw.get("image_size", (336, 336))))
    image = load_background(raw.get("image_path", ""), image_size, image_root, raw.get("image_relpath"))
    image_size = (image.height, image.width)

    a_path = save_dir / "A_early_raw_attention.png"
    b_path = save_dir / "B_deep_raw_attention.png"
    c_path = save_dir / "C_residual_score.png"
    d_path = save_dir / "D_suppressed_tokens.png"

    overlay_patch_heatmap(
        image,
        score_to_grid(early_raw, grid, heat_low_quantile, heat_high_quantile, heat_gamma),
        "A. Early Raw Attention",
        a_path,
        background_alpha,
        patch_alpha,
        cmap_name,
        draw_grid,
    )
    overlay_patch_heatmap(
        image,
        score_to_grid(deep_raw, grid, heat_low_quantile, heat_high_quantile, heat_gamma),
        "B. Deep Raw Attention",
        b_path,
        background_alpha,
        patch_alpha,
        cmap_name,
        draw_grid,
    )
    overlay_patch_heatmap(
        image,
        score_to_grid(residual_score, grid, heat_low_quantile, heat_high_quantile, heat_gamma),
        "C. Residual Coarse Score",
        c_path,
        background_alpha,
        patch_alpha,
        cmap_name,
        draw_grid,
    )

    raw_thr = torch.quantile(early_raw, float(raw_top_q))
    res_thr = torch.quantile(residual_score, float(res_low_q))
    suppressed_mask = (early_raw >= raw_thr) & (residual_score <= res_thr)
    draw_suppressed_tokens(image, suppressed_mask, grid, d_path, background_alpha)
    save_panel([a_path, b_path, c_path, d_path], save_dir / "ABCD_panel.png")
    save_scatter(early_raw, residual_score, save_dir / "scatter_raw_vs_residual.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prune-dir", type=Path, required=True)
    parser.add_argument("--residual-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--raw-top-quantile", type=float, default=0.75)
    parser.add_argument("--res-low-quantile", type=float, default=0.50)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--background-alpha", type=float, default=0.34)
    parser.add_argument("--patch-alpha", type=float, default=0.88)
    parser.add_argument("--heat-low-quantile", type=float, default=0.05)
    parser.add_argument("--heat-high-quantile", type=float, default=0.98)
    parser.add_argument("--heat-gamma", type=float, default=1.0)
    parser.add_argument("--cmap", type=str, default="YlOrRd")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for no_prune_path in sorted(args.no_prune_dir.glob("*.pt")):
        residual_path = args.residual_dir / no_prune_path.name
        if not residual_path.exists():
            print(f"skip {no_prune_path.stem}: missing {residual_path}")
            continue
        visualize_case(
            no_prune_path,
            residual_path,
            args.save_dir,
            args.raw_top_quantile,
            args.res_low_quantile,
            args.image_root,
            args.background_alpha,
            args.patch_alpha,
            args.heat_low_quantile,
            args.heat_high_quantile,
            args.heat_gamma,
            args.cmap,
            not args.no_grid,
        )
        print(f"saved figures for {no_prune_path.stem}")


if __name__ == "__main__":
    main()
