"""Visualize CLIP vision-tower attention against KNN/APS selected visual tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib.patches import Rectangle


def load_pt(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def tensor_1d(value, name: str, dtype=torch.float32) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing {name}.")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.to(dtype=dtype).view(-1)


def tensor_set(value) -> set[int]:
    tensor = value.detach().cpu().view(-1) if isinstance(value, torch.Tensor) else torch.tensor(value).view(-1)
    return {int(item) for item in tensor.tolist()}


def ranked_selector_set(data: Dict[str, object], selector_name: str, top_k: int) -> set[int]:
    rank_keys = {
        "knn": ("selected_indices_by_score", "selected_indices"),
        "aps": ("selected_indices_by_aps", "selected_indices"),
    }
    for key in rank_keys.get(selector_name, ("selected_indices",)):
        if key in data and data[key] is not None:
            tensor = data[key].detach().cpu().view(-1) if isinstance(data[key], torch.Tensor) else torch.tensor(data[key]).view(-1)
            return {int(value) for value in tensor[:top_k].tolist()}
    raise KeyError(f"Missing ranked selected indices for {selector_name}.")


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


def normalize_to_0_1(score: torch.Tensor, low_q: float, high_q: float, gamma: float) -> np.ndarray:
    array = score.detach().float().cpu().numpy()
    if array.size == 0:
        return array
    lo = float(np.nanquantile(array, low_q))
    hi = float(np.nanquantile(array, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    normalized = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    if gamma > 0:
        normalized = normalized**gamma
    return normalized.astype(np.float32)


def faded_image_array(image: Image.Image, background_alpha: float) -> np.ndarray:
    base = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(base * background_alpha + (1.0 - background_alpha), 0.0, 1.0)


def faded_image(image: Image.Image, background_alpha: float) -> Image.Image:
    return Image.fromarray(np.uint8(faded_image_array(image, background_alpha) * 255))


def patch_box(index: int, grid: Tuple[int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    _, cols = grid
    row = index // cols
    col = index % cols
    patch_w = width / cols
    patch_h = height / grid[0]
    return col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h


def draw_attention_heatmap(
    image: Image.Image,
    score: torch.Tensor,
    grid: Tuple[int, int],
    output_path: Path,
    *,
    title: str,
    background_alpha: float,
    patch_alpha: float,
    low_q: float,
    high_q: float,
    gamma: float,
    cmap_name: str,
) -> None:
    rows, cols = grid
    heat = normalize_to_0_1(score, low_q, high_q, gamma).reshape(rows, cols)
    width, height = image.size
    patch_w = width / cols
    patch_h = height / rows
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * height / max(width, 1))))
    ax.imshow(faded_image_array(image, background_alpha))
    for row in range(rows):
        for col in range(cols):
            value = float(heat[row, col])
            alpha = min(0.88, max(0.18, patch_alpha))
            ax.add_patch(
                Rectangle(
                    (col * patch_w, row * patch_h),
                    patch_w,
                    patch_h,
                    facecolor=cmap(value),
                    edgecolor=(1, 1, 1, 0.35),
                    linewidth=0.3,
                    alpha=alpha,
                )
            )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_selection(
    image: Image.Image,
    indices: Iterable[int],
    grid: Tuple[int, int],
    output_path: Path,
    *,
    title: str,
    background_alpha: float,
    fill,
    outline,
    width: int,
) -> Image.Image:
    canvas = faded_image(image, background_alpha)
    draw = ImageDraw.Draw(canvas, "RGBA")
    img_w, img_h = canvas.size
    for index in sorted(indices):
        draw.rectangle(patch_box(index, grid, img_w, img_h), fill=fill, outline=outline, width=width)
    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * img_h / max(img_w, 1))))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return canvas


def draw_overlay(
    image: Image.Image,
    score: torch.Tensor,
    selected_indices: set[int],
    grid: Tuple[int, int],
    output_path: Path,
    args,
    *,
    title: str,
    outline=(0.0, 0.22, 0.62, 1.0),
) -> Image.Image:
    rows, cols = grid
    heat = normalize_to_0_1(score, args.heat_low_quantile, args.heat_high_quantile, args.heat_gamma).reshape(rows, cols)
    img_w, img_h = image.size
    patch_w = img_w / cols
    patch_h = img_h / rows
    cmap = plt.get_cmap(args.cmap)

    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * img_h / max(img_w, 1))))
    ax.imshow(faded_image_array(image, args.background_alpha))
    for row in range(rows):
        for col in range(cols):
            value = float(heat[row, col])
            alpha = min(0.88, max(0.18, args.patch_alpha))
            ax.add_patch(
                Rectangle(
                    (col * patch_w, row * patch_h),
                    patch_w,
                    patch_h,
                    facecolor=cmap(value),
                    edgecolor=(1, 1, 1, 0.25),
                    linewidth=0.25,
                    alpha=alpha,
                )
            )
    for index in sorted(selected_indices):
        x0, y0, x1, y1 = patch_box(index, grid, img_w, img_h)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=(1, 1, 1, 0),
                edgecolor=outline,
                linewidth=0.85,
            )
        )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    canvas = faded_image(image, args.background_alpha)
    return canvas


def draw_diff(
    image: Image.Image,
    overlap: set[int],
    selected_only: set[int],
    attn_only: set[int],
    grid: Tuple[int, int],
    output_path: Path,
    background_alpha: float,
    title: str,
) -> Image.Image:
    canvas = faded_image(image, background_alpha)
    draw = ImageDraw.Draw(canvas, "RGBA")
    img_w, img_h = canvas.size
    styles = [
        (overlap, (32, 170, 80, 118), (0, 110, 42, 255), 2),
        (selected_only, (25, 116, 230, 116), (0, 55, 155, 255), 2),
        (attn_only, (245, 132, 32, 116), (185, 72, 0, 255), 2),
    ]
    for indices, fill, outline, line_width in styles:
        for index in sorted(indices):
            draw.rectangle(patch_box(index, grid, img_w, img_h), fill=fill, outline=outline, width=line_width)
    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * img_h / max(img_w, 1))))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return canvas


def selector_overlap_stats(case_id: str, selector_name: str, top_attention: set[int], selected: set[int], top_k: int) -> Dict[str, object]:
    overlap = selected & top_attention
    selected_only = selected - top_attention
    attention_only = top_attention - selected
    union = selected | top_attention
    return {
        "case_id": case_id,
        "selector": selector_name,
        "top_k": top_k,
        "selected_count": len(selected),
        "attention_top_count": len(top_attention),
        "overlap": len(overlap),
        "selected_only": len(selected_only),
        "attention_only": len(attention_only),
        "union": len(union),
        "overlap_rate_vs_selected": len(overlap) / max(len(selected), 1),
        "overlap_rate_vs_attention": len(overlap) / max(len(top_attention), 1),
        "jaccard": len(overlap) / max(len(union), 1),
        "selected_only_indices": sorted(selected_only),
        "attention_only_indices": sorted(attention_only),
        "overlap_indices": sorted(overlap),
        "top_attention_indices": sorted(top_attention),
    }


def save_stats_txt(path: Path, stats: Dict[str, Dict[str, object]]) -> None:
    lines = [
        f"case_id: {next(iter(stats.values()))['case_id'] if stats else ''}",
        "Legend for diff images: green=overlap, blue=selector-only, orange=attention-only",
    ]
    for name, item in stats.items():
        lines.extend(
            [
                "",
                f"[{name}]",
                f"top_k: {item['top_k']}",
                f"selected_count: {item['selected_count']}",
                f"attention_top_count: {item['attention_top_count']}",
                f"overlap: {item['overlap']}",
                f"selected_only: {item['selected_only']}",
                f"attention_only: {item['attention_only']}",
                f"union: {item['union']}",
                f"overlap_rate_vs_selected: {item['overlap_rate_vs_selected']:.6f}",
                f"overlap_rate_vs_attention: {item['overlap_rate_vs_attention']:.6f}",
                f"jaccard: {item['jaccard']:.6f}",
                "selected_only_indices: " + ",".join(map(str, item["selected_only_indices"])),
                "attention_only_indices: " + ",".join(map(str, item["attention_only_indices"])),
                "overlap_indices: " + ",".join(map(str, item["overlap_indices"])),
                "top_attention_indices: " + ",".join(map(str, item["top_attention_indices"])),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_case(
    attn_path: Path,
    knn_path: Path,
    aps_path: Path | None,
    save_root: Path,
    image_root: Path | None,
    args,
) -> List[Dict[str, object]]:
    attn = load_pt(attn_path)
    knn = load_pt(knn_path)
    aps = load_pt(aps_path) if aps_path and aps_path.exists() else None
    case_id = str(attn.get("case_id", attn_path.stem))
    save_dir = save_root / case_id
    save_dir.mkdir(parents=True, exist_ok=True)

    score = tensor_1d(attn["vision_attention_score"], "vision_attention_score")
    top_k = min(int(args.top_k), score.numel())
    top_attention = set(int(value) for value in torch.topk(score, k=top_k, largest=True).indices.tolist())
    knn_set = ranked_selector_set(knn, "knn", top_k)
    aps_set = ranked_selector_set(aps, "aps", top_k) if aps is not None else set()
    knn_stats = selector_overlap_stats(case_id, "knn", top_attention, knn_set, top_k)
    stats_by_name = {"knn": knn_stats}
    if aps is not None:
        stats_by_name["aps"] = selector_overlap_stats(case_id, "aps", top_attention, aps_set, top_k)

    grid = patch_grid(attn)
    image_size = tuple(map(int, attn.get("image_size", (336, 336))))
    image = load_background(attn.get("image_path", ""), image_size, image_root, attn.get("image_relpath"))

    draw_attention_heatmap(
        image,
        score,
        grid,
        save_dir / "A_visual_tower_attention.png",
        title="A. CLIP Vision Tower Attention Score",
        background_alpha=args.background_alpha,
        patch_alpha=args.patch_alpha,
        low_q=args.heat_low_quantile,
        high_q=args.heat_high_quantile,
        gamma=args.heat_gamma,
        cmap_name=args.cmap,
    )
    draw_selection(
        image,
        knn_set,
        grid,
        save_dir / "B_knn_selected_tokens.png",
        title="B. KNN Selected Tokens",
        background_alpha=args.background_alpha,
        fill=(25, 116, 230, 112),
        outline=(0, 55, 155, 255),
        width=2,
    )
    if aps is not None:
        draw_selection(
            image,
            aps_set,
            grid,
            save_dir / "C_aps_selected_tokens.png",
            title="C. APS Selected Tokens",
            background_alpha=args.background_alpha,
            fill=(145, 78, 180, 112),
            outline=(92, 34, 128, 255),
            width=2,
        )
    draw_overlay(
        image,
        score,
        knn_set,
        grid,
        save_dir / "D_attention_knn_overlay.png",
        args,
        title="D. Attention Score + KNN Tokens",
        outline=(0.0, 0.22, 0.62, 1.0),
    )
    if aps is not None:
        draw_overlay(
            image,
            score,
            aps_set,
            grid,
            save_dir / "E_attention_aps_overlay.png",
            args,
            title="E. Attention Score + APS Tokens",
            outline=(0.36, 0.13, 0.50, 1.0),
        )
    draw_diff(
        image,
        set(knn_stats["overlap_indices"]),
        set(knn_stats["selected_only_indices"]),
        set(knn_stats["attention_only_indices"]),
        grid,
        save_dir / "F_top_attention_vs_knn_diff.png",
        args.background_alpha,
        "F. Top Attention vs KNN Difference",
    )
    if aps is not None:
        aps_stats = stats_by_name["aps"]
        draw_diff(
            image,
            set(aps_stats["overlap_indices"]),
            set(aps_stats["selected_only_indices"]),
            set(aps_stats["attention_only_indices"]),
            grid,
            save_dir / "G_top_attention_vs_aps_diff.png",
            args.background_alpha,
            "G. Top Attention vs APS Difference",
        )
    save_stats_txt(save_dir / "stats.txt", stats_by_name)
    return list(stats_by_name.values())


def save_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "selector",
        "top_k",
        "selected_count",
        "attention_top_count",
        "overlap",
        "selected_only",
        "attention_only",
        "union",
        "overlap_rate_vs_selected",
        "overlap_rate_vs_attention",
        "jaccard",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def save_summary_txt(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("No cases compared.\n", encoding="utf-8")
        return
    lines = [
        f"cases: {len({row['case_id'] for row in rows})}",
        "Legend for diff images: green=overlap, blue=selector-only, orange=attention-only",
    ]
    for selector in sorted({str(row["selector"]) for row in rows}):
        selector_rows = [row for row in rows if row["selector"] == selector]
        overlap = np.array([row["overlap"] for row in selector_rows], dtype=np.float32)
        selected_only = np.array([row["selected_only"] for row in selector_rows], dtype=np.float32)
        attn_only = np.array([row["attention_only"] for row in selector_rows], dtype=np.float32)
        jaccard = np.array([row["jaccard"] for row in selector_rows], dtype=np.float32)
        lines.extend(
            [
                "",
                f"[{selector}]",
                f"rows: {len(selector_rows)}",
                f"mean_overlap: {overlap.mean():.4f}",
                f"min_overlap: {overlap.min():.0f}",
                f"max_overlap: {overlap.max():.0f}",
                f"mean_selected_only: {selected_only.mean():.4f}",
                f"mean_attention_only: {attn_only.mean():.4f}",
                f"mean_jaccard: {jaccard.mean():.6f}",
                f"min_jaccard: {jaccard.min():.6f}",
                f"max_jaccard: {jaccard.max():.6f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--knn-dir", type=Path, default=Path("visualize/outputs/knn_feature_selection"))
    parser.add_argument("--aps-dir", type=Path, default=Path("visualize/outputs/aps_selector"))
    parser.add_argument("--save-dir", type=Path, default=Path("visualize/figures/visual_tower_vs_knn"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=144)
    parser.add_argument("--background-alpha", type=float, default=0.72)
    parser.add_argument("--patch-alpha", type=float, default=0.68)
    parser.add_argument("--heat-low-quantile", type=float, default=0.0)
    parser.add_argument("--heat-high-quantile", type=float, default=1.0)
    parser.add_argument("--heat-gamma", type=float, default=0.75)
    parser.add_argument("--cmap", type=str, default="YlOrRd")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for attn_path in sorted(args.attention_dir.glob("*.pt")):
        knn_path = args.knn_dir / attn_path.name
        if not knn_path.exists():
            print(f"skip {attn_path.stem}: missing {knn_path}")
            continue
        aps_path = args.aps_dir / attn_path.name if args.aps_dir else None
        stats_rows = compare_case(attn_path, knn_path, aps_path, args.save_dir, args.image_root, args)
        rows.extend(stats_rows)
        compact = " ".join(
            f"{item['selector']}:overlap={item['overlap']} jaccard={item['jaccard']:.3f}" for item in stats_rows
        )
        print(f"{attn_path.stem}: {compact}")

    save_summary_csv(args.save_dir / "summary_stats.csv", rows)
    save_summary_txt(args.save_dir / "summary_stats.txt", rows)


if __name__ == "__main__":
    main()
