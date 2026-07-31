"""Visualize Diversity -> APS feature peaks, selected tokens, and peak clusters."""

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


def faded_image_array(image: Image.Image, background_alpha: float) -> np.ndarray:
    base = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(base * background_alpha + (1.0 - background_alpha), 0.0, 1.0)


def patch_box(index: int, grid: Tuple[int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    rows, cols = grid
    row = index // cols
    col = index % cols
    patch_w = width / cols
    patch_h = height / rows
    return col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h


def draw_token_overlay(
    image: Image.Image,
    indices: torch.Tensor,
    grid: Tuple[int, int],
    output_path: Path,
    title: str,
    background_alpha: float,
    fill_rgba: Tuple[int, int, int, int],
    outline_rgba: Tuple[int, int, int, int],
    outline_width: int,
    label_indices: bool,
) -> None:
    width, height = image.size
    rows, cols = grid
    canvas = Image.fromarray(np.uint8(faded_image_array(image, background_alpha) * 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", max(8, int(min(width / cols, height / rows) * 0.34)))
    except OSError:
        font = ImageFont.load_default()

    for rank, index in enumerate(indices.view(-1).tolist(), start=1):
        index = int(index)
        row = index // cols
        col = index % cols
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        x0, y0, x1, y1 = patch_box(index, grid, width, height)
        draw.rectangle((x0, y0, x1, y1), fill=fill_rgba, outline=outline_rgba, width=outline_width)
        if label_indices:
            label = str(rank)
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            tx = x0 + max(1, (x1 - x0 - text_w) / 2)
            ty = y0 + max(1, (y1 - y0 - text_h) / 2)
            draw.rectangle((tx - 1, ty - 1, tx + text_w + 1, ty + text_h + 1), fill=(255, 255, 255, 185))
            draw.text((tx, ty), label, fill=(20, 20, 20, 255), font=font)

    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * height / max(width, 1))))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_peak_clusters(
    image: Image.Image,
    data: Dict[str, object],
    grid: Tuple[int, int],
    output_path: Path,
    background_alpha: float,
    cluster_alpha: float,
    cluster_mode: str,
) -> None:
    width, height = image.size
    rows, cols = grid
    canvas = Image.fromarray(np.uint8(faded_image_array(image, background_alpha) * 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    candidate_indices = tensor_1d(data["candidate_indices"], "candidate_indices", dtype=torch.long)
    peak_indices = tensor_1d(data["peak_indices"], "peak_indices", dtype=torch.long)
    assigned_peak = tensor_1d(data["assigned_peak"], "assigned_peak", dtype=torch.long)
    peak_token_indices = tensor_1d(data["peak_token_indices"], "peak_token_indices", dtype=torch.long)
    selected_indices = set(int(value) for value in tensor_1d(data["selected_indices"], "selected_indices", dtype=torch.long))

    cmap = plt.get_cmap("tab20")
    for local, token_index in enumerate(candidate_indices.tolist()):
        if local >= assigned_peak.numel():
            continue
        peak_ord = int(assigned_peak[local])
        if peak_ord < 0:
            continue
        token_index = int(token_index)
        row = token_index // cols
        col = token_index % cols
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        if cluster_mode == "selected" and token_index not in selected_indices:
            draw.rectangle(
                patch_box(token_index, grid, width, height),
                fill=(255, 255, 255, 0),
                outline=(80, 80, 80, 42),
                width=1,
            )
            continue

        color = cmap(peak_ord % 20)
        rgba = tuple(int(channel * 255) for channel in color[:3]) + (int(cluster_alpha * 255),)
        outline = tuple(int(channel * 255) for channel in color[:3]) + (220 if cluster_mode == "selected" else 190,)
        draw.rectangle(
            patch_box(token_index, grid, width, height),
            fill=rgba,
            outline=outline,
            width=2 if cluster_mode == "selected" else 1,
        )

    for peak_ord, token_index in enumerate(peak_token_indices.tolist()):
        token_index = int(token_index)
        color = cmap(peak_ord % 20)
        outline = tuple(int(channel * 255) for channel in color[:3]) + (255,)
        draw.rectangle(patch_box(token_index, grid, width, height), fill=(255, 255, 255, 0), outline=outline, width=4)

    fig, ax = plt.subplots(figsize=(6.4, max(4.0, 6.4 * height / max(width, 1))))
    ax.imshow(canvas)
    title = "C. Selected Tokens by APS Peak Cluster" if cluster_mode == "selected" else "C. APS Peak Clusters"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def selected_records(data: Dict[str, object]) -> List[Dict[str, object]]:
    _, cols = patch_grid(data)
    candidate_indices = tensor_1d(data["candidate_indices"], "candidate_indices", dtype=torch.long).tolist()
    local_by_index = {int(index): local for local, index in enumerate(candidate_indices)}
    peak_indices = tensor_1d(data["peak_indices"], "peak_indices", dtype=torch.long).tolist()
    peak_tokens = set(int(value) for value in tensor_1d(data["peak_token_indices"], "peak_token_indices", dtype=torch.long))
    selected = tensor_1d(data["selected_indices_by_aps"], "selected_indices_by_aps", dtype=torch.long).tolist()
    score = tensor_1d(data["score"], "score").tolist()
    rho = tensor_1d(data["rho"], "rho").tolist()
    delta = tensor_1d(data["delta"], "delta").tolist()
    assigned_peak = tensor_1d(data["assigned_peak"], "assigned_peak", dtype=torch.long).tolist()

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


def visualize_case(
    dump_path: Path,
    save_root: Path,
    image_root: Path | None,
    background_alpha: float,
    cluster_alpha: float,
    cluster_mode: str,
    label_peaks: bool,
) -> List[Dict[str, object]]:
    data = load_pt(dump_path)
    case_id = str(data.get("case_id", dump_path.stem))
    save_dir = save_root / case_id
    save_dir.mkdir(parents=True, exist_ok=True)

    grid = patch_grid(data)
    image_size = tuple(map(int, data.get("image_size", (336, 336))))
    image = load_background(data.get("image_path", ""), image_size, image_root, data.get("image_relpath"))
    peak_token_indices = tensor_1d(data["peak_token_indices"], "peak_token_indices", dtype=torch.long)
    selected_indices = tensor_1d(data["selected_indices"], "selected_indices", dtype=torch.long)

    a_path = save_dir / "A_feature_peaks.png"
    b_path = save_dir / "B_selected_tokens.png"
    c_path = save_dir / "C_peak_clusters.png"
    d_path = save_dir / "D_selected_tokens.csv"

    draw_token_overlay(
        image,
        peak_token_indices,
        grid,
        a_path,
        "A. APS Feature Peaks",
        background_alpha,
        fill_rgba=(255, 220, 0, 85),
        outline_rgba=(210, 90, 0, 255),
        outline_width=4,
        label_indices=label_peaks,
    )
    draw_token_overlay(
        image,
        selected_indices,
        grid,
        b_path,
        "B. Final Selected Tokens",
        background_alpha,
        fill_rgba=(0, 128, 255, 112),
        outline_rgba=(0, 50, 130, 255),
        outline_width=2,
        label_indices=False,
    )
    draw_peak_clusters(image, data, grid, c_path, background_alpha, cluster_alpha, cluster_mode)

    records = selected_records(data)
    save_selected_csv(d_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("visualize/figures/aps_selector"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--background-alpha", type=float, default=0.58)
    parser.add_argument("--cluster-alpha", type=float, default=0.34)
    parser.add_argument("--cluster-mode", choices=("selected", "all"), default="selected")
    parser.add_argument("--label-peaks", action="store_true")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for dump_path in sorted(args.dump_dir.glob("*.pt")):
        records = visualize_case(
            dump_path,
            args.save_dir,
            args.image_root,
            args.background_alpha,
            args.cluster_alpha,
            args.cluster_mode,
            args.label_peaks,
        )
        all_records.extend(records)
        print(f"saved figures for {dump_path.stem}")

    if all_records:
        save_selected_csv(args.save_dir / "D_selected_tokens_all_cases.csv", all_records)


if __name__ == "__main__":
    main()
