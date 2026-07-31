"""Compare KNN top-k selected visual tokens with APS selected tokens."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def load_pt(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def tensor_set(value) -> set[int]:
    tensor = value.detach().cpu().view(-1) if isinstance(value, torch.Tensor) else torch.tensor(value).view(-1)
    return {int(item) for item in tensor.tolist()}


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


def faded_image(image: Image.Image, background_alpha: float) -> Image.Image:
    base = np.asarray(image).astype(np.float32) / 255.0
    faded = np.clip(base * background_alpha + (1.0 - background_alpha), 0.0, 1.0)
    return Image.fromarray(np.uint8(faded * 255))


def patch_box(index: int, grid: Tuple[int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    _, cols = grid
    row = index // cols
    col = index % cols
    patch_w = width / cols
    patch_h = height / grid[0]
    return col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h


def draw_selection(
    image: Image.Image,
    indices: Iterable[int],
    grid: Tuple[int, int],
    *,
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
    return canvas


def draw_diff(
    image: Image.Image,
    overlap: set[int],
    knn_only: set[int],
    aps_only: set[int],
    grid: Tuple[int, int],
    background_alpha: float,
) -> Image.Image:
    canvas = faded_image(image, background_alpha)
    draw = ImageDraw.Draw(canvas, "RGBA")
    img_w, img_h = canvas.size

    styles = [
        (overlap, (32, 170, 80, 118), (0, 110, 42, 255), 2),
        (knn_only, (245, 132, 32, 116), (185, 72, 0, 255), 2),
        (aps_only, (25, 116, 230, 116), (0, 55, 155, 255), 2),
    ]
    for indices, fill, outline, line_width in styles:
        for index in sorted(indices):
            draw.rectangle(patch_box(index, grid, img_w, img_h), fill=fill, outline=outline, width=line_width)
    return canvas


def save_panel(knn_img: Image.Image, aps_img: Image.Image, diff_img: Image.Image, output_path: Path, stats: Dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        "KNN Feature Selection",
        "APS Selector",
        (
            "Overlap / Difference\n"
            f"same={stats['overlap']}  KNN-only={stats['knn_only']}  APS-only={stats['aps_only']}  "
            f"J={stats['jaccard']:.3f}"
        ),
    ]
    for ax, image, title in zip(axes, [knn_img, aps_img, diff_img], titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_stats_txt(path: Path, stats: Dict[str, object]) -> None:
    lines = [
        f"case_id: {stats['case_id']}",
        f"knn_count: {stats['knn_count']}",
        f"aps_count: {stats['aps_count']}",
        f"overlap: {stats['overlap']}",
        f"knn_only: {stats['knn_only']}",
        f"aps_only: {stats['aps_only']}",
        f"union: {stats['union']}",
        f"overlap_rate_vs_knn: {stats['overlap_rate_vs_knn']:.6f}",
        f"overlap_rate_vs_aps: {stats['overlap_rate_vs_aps']:.6f}",
        f"jaccard: {stats['jaccard']:.6f}",
        "knn_only_indices: " + ",".join(map(str, stats["knn_only_indices"])),
        "aps_only_indices: " + ",".join(map(str, stats["aps_only_indices"])),
        "overlap_indices: " + ",".join(map(str, stats["overlap_indices"])),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_case(
    knn_path: Path,
    aps_path: Path,
    save_root: Path,
    image_root: Path | None,
    background_alpha: float,
) -> Dict[str, object]:
    knn = load_pt(knn_path)
    aps = load_pt(aps_path)
    case_id = str(knn.get("case_id", knn_path.stem))
    save_dir = save_root / case_id
    save_dir.mkdir(parents=True, exist_ok=True)

    knn_set = tensor_set(knn["selected_indices"])
    aps_set = tensor_set(aps["selected_indices"])
    overlap = knn_set & aps_set
    knn_only = knn_set - aps_set
    aps_only = aps_set - knn_set
    union = knn_set | aps_set

    stats = {
        "case_id": case_id,
        "knn_count": len(knn_set),
        "aps_count": len(aps_set),
        "overlap": len(overlap),
        "knn_only": len(knn_only),
        "aps_only": len(aps_only),
        "union": len(union),
        "overlap_rate_vs_knn": len(overlap) / max(len(knn_set), 1),
        "overlap_rate_vs_aps": len(overlap) / max(len(aps_set), 1),
        "jaccard": len(overlap) / max(len(union), 1),
        "knn_only_indices": sorted(knn_only),
        "aps_only_indices": sorted(aps_only),
        "overlap_indices": sorted(overlap),
    }

    grid = patch_grid(knn)
    image_size = tuple(map(int, knn.get("image_size", (336, 336))))
    image = load_background(knn.get("image_path", ""), image_size, image_root, knn.get("image_relpath"))

    knn_img = draw_selection(
        image,
        knn_set,
        grid,
        background_alpha=background_alpha,
        fill=(245, 132, 32, 112),
        outline=(185, 72, 0, 255),
        width=2,
    )
    aps_img = draw_selection(
        image,
        aps_set,
        grid,
        background_alpha=background_alpha,
        fill=(25, 116, 230, 112),
        outline=(0, 55, 155, 255),
        width=2,
    )
    diff_img = draw_diff(image, overlap, knn_only, aps_only, grid, background_alpha=background_alpha)

    save_panel(knn_img, aps_img, diff_img, save_dir / "topk_vs_aps_panel.png", stats)
    diff_img.save(save_dir / "topk_vs_aps_diff.png")
    save_stats_txt(save_dir / "stats.txt", stats)
    return stats


def save_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "knn_count",
        "aps_count",
        "overlap",
        "knn_only",
        "aps_only",
        "union",
        "overlap_rate_vs_knn",
        "overlap_rate_vs_aps",
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
    overlap = np.array([row["overlap"] for row in rows], dtype=np.float32)
    knn_only = np.array([row["knn_only"] for row in rows], dtype=np.float32)
    aps_only = np.array([row["aps_only"] for row in rows], dtype=np.float32)
    jaccard = np.array([row["jaccard"] for row in rows], dtype=np.float32)
    lines = [
        f"cases: {len(rows)}",
        f"mean_overlap: {overlap.mean():.4f}",
        f"min_overlap: {overlap.min():.0f}",
        f"max_overlap: {overlap.max():.0f}",
        f"mean_knn_only: {knn_only.mean():.4f}",
        f"mean_aps_only: {aps_only.mean():.4f}",
        f"mean_jaccard: {jaccard.mean():.6f}",
        f"min_jaccard: {jaccard.min():.6f}",
        f"max_jaccard: {jaccard.max():.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn-dir", type=Path, default=Path("visualize/outputs/knn_feature_selection"))
    parser.add_argument("--aps-dir", type=Path, default=Path("visualize/outputs/aps_selector"))
    parser.add_argument("--save-dir", type=Path, default=Path("visualize/figures/topk_vs_aps"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--background-alpha", type=float, default=0.62)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for knn_path in sorted(args.knn_dir.glob("*.pt")):
        aps_path = args.aps_dir / knn_path.name
        if not aps_path.exists():
            print(f"skip {knn_path.stem}: missing {aps_path}")
            continue
        stats = compare_case(knn_path, aps_path, args.save_dir, args.image_root, args.background_alpha)
        rows.append(stats)
        print(
            f"{stats['case_id']}: overlap={stats['overlap']} "
            f"knn_only={stats['knn_only']} aps_only={stats['aps_only']} "
            f"jaccard={stats['jaccard']:.3f}"
        )

    save_summary_csv(args.save_dir / "summary_stats.csv", rows)
    save_summary_txt(args.save_dir / "summary_stats.txt", rows)


if __name__ == "__main__":
    main()
