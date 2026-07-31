"""Visualize D-QECG selection debug dumps on the original images."""

from __future__ import annotations

import argparse
import io
import csv
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # Local visualization can run without PyTorch.
    torch = None
from PIL import Image, ImageDraw, ImageFont


TOOL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOOL_ROOT.parent
DEFAULT_DUMP_DIR = WORKSPACE_ROOT / "visualize" / "outputs" / "qceg_debug"
DEFAULT_SAVE_DIR = WORKSPACE_ROOT / "visualize" / "figures" / "qceg_debug"


@dataclass
class _StorageType:
    name: str
    dtype: np.dtype


@dataclass
class _Storage:
    array: np.ndarray


_TORCH_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    array = storage.array
    offset = int(storage_offset)
    size = tuple(int(item) for item in size)
    stride = tuple(int(item) for item in stride)
    if not size:
        return array[offset].copy()
    view = np.lib.stride_tricks.as_strided(
        array[offset:],
        shape=size,
        strides=tuple(item * array.dtype.itemsize for item in stride),
    )
    return np.array(view)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class _TorchlessUnpickler(pickle.Unpickler):
    def __init__(self, handle, archive: zipfile.ZipFile, root: str):
        super().__init__(handle)
        self.archive = archive
        self.root = root

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch" and name in _TORCH_STORAGE_DTYPES:
            return _StorageType(name=name, dtype=_TORCH_STORAGE_DTYPES[name])
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or not saved_id:
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        typename = saved_id[0].decode("utf-8") if isinstance(saved_id[0], bytes) else saved_id[0]
        if typename != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id type: {typename!r}")
        _, storage_type, key, _location, numel = saved_id
        key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        payload = self.archive.read(f"{self.root}/data/{key}")
        array = np.frombuffer(payload, dtype=storage_type.dtype, count=int(numel))
        return _Storage(array=array)


def _torchless_load(path: Path) -> Dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist() if "/" in name}
        if len(roots) != 1:
            raise ValueError(f"unexpected torch archive layout in {path}")
        root = next(iter(roots))
        payload = archive.read(f"{root}/data.pkl")
        return _TorchlessUnpickler(io.BytesIO(payload), archive, root).load()


def load_pt(path: Path) -> Dict[str, object]:
    if torch is not None:
        return torch.load(path, map_location="cpu")
    return _torchless_load(path)


def tensor_1d(value, name: str, dtype=None) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=dtype or np.float32)
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = array.reshape(-1)
    return array.astype(dtype, copy=False) if dtype is not None else array.astype(np.float32, copy=False)


def tensor_2d(value, name: str, dtype=None) -> np.ndarray:
    if value is None:
        return np.empty((0, 0), dtype=dtype or np.float32)
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(dtype, copy=False) if dtype is not None else array.astype(np.float32, copy=False)


def tensor_set(value) -> set[int]:
    tensor = tensor_1d(value, "indices", dtype=np.int64)
    return {int(item) for item in tensor.tolist()}


def top_indices(score, top_k: int) -> List[int]:
    tensor = tensor_1d(score, "score")
    if tensor.size == 0:
        return []
    k = min(int(top_k), tensor.size)
    return [int(i) for i in np.argsort(tensor)[-k:][::-1].tolist()]


def patch_grid(data: Dict[str, object]) -> Tuple[int, int]:
    grid = data.get("patch_grid", (24, 24))
    return int(grid[0]), int(grid[1])


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


def faded_image(image: Image.Image, alpha: float) -> Image.Image:
    base = np.asarray(image).astype(np.float32) / 255.0
    faded = np.clip(base * alpha + (1.0 - alpha), 0.0, 1.0)
    return Image.fromarray(np.uint8(faded * 255))


def draw_selection(
    image: Image.Image,
    indices: Iterable[int],
    grid: Tuple[int, int],
    *,
    fill,
    outline,
    background_alpha: float,
    label_first: int = 0,
    title: str = "",
) -> Image.Image:
    canvas = faded_image(image, background_alpha)
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    for rank, index in enumerate(sorted({int(i) for i in indices}), start=1):
        draw.rectangle(patch_box(index, grid, width, height), fill=fill, outline=outline, width=2)
        if label_first and rank <= label_first:
            x0, y0, _, _ = patch_box(index, grid, width, height)
            draw.text((x0 + 2, y0 + 2), str(rank), fill=(0, 0, 0, 255))
    if title:
        draw.rectangle((0, 0, width, 24), fill=(255, 255, 255, 215))
        draw.text((8, 5), title, fill=(0, 0, 0, 255))
    return canvas


def draw_stage_overlay(
    image: Image.Image,
    first: set[int],
    second: set[int],
    final: set[int],
    grid: Tuple[int, int],
    background_alpha: float,
    second_label: str = "S2 LLM",
) -> Image.Image:
    canvas = faded_image(image, background_alpha)
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    second_only = second - first
    first_only = first - second
    overlap = first & second
    for indices, fill, outline in [
        (final - first - second, (45, 180, 90, 92), (15, 120, 55, 255)),
        (first_only, (245, 132, 32, 105), (185, 72, 0, 255)),
        (second_only, (25, 116, 230, 105), (0, 55, 155, 255)),
        (overlap, (40, 190, 90, 130), (0, 120, 50, 255)),
    ]:
        for index in sorted(indices):
            draw.rectangle(patch_box(index, grid, width, height), fill=fill, outline=outline, width=2)
    draw.rectangle((0, 0, width, 42), fill=(255, 255, 255, 220))
    draw.text((8, 5), f"orange=S1 visual, blue={second_label}, green=overlap/other final", fill=(0, 0, 0, 255))
    draw.text((8, 23), f"S1={len(first)}  S2={len(second)}  final={len(final)}", fill=(0, 0, 0, 255))
    return canvas


def draw_score_curve(
    score,
    top_ids: Iterable[int],
    title: str,
    color,
    *,
    x_label: str = "visual token id",
) -> Image.Image:
    values = tensor_1d(score, "score")
    width, height = 900, 260
    margin_l, margin_r, margin_t, margin_b = 54, 18, 34, 34
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((margin_l, 10), title, fill=(0, 0, 0, 255))
    draw.line((margin_l, height - margin_b, width - margin_r, height - margin_b), fill=(40, 40, 40, 255), width=1)
    draw.line((margin_l, margin_t, margin_l, height - margin_b), fill=(40, 40, 40, 255), width=1)
    if values.size == 0:
        draw.text((margin_l + 12, margin_t + 30), "no score in dump", fill=(120, 120, 120, 255))
        return canvas

    vmax = float(np.max(values))
    vmin = float(np.min(values))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = margin_l + np.arange(values.size) * (plot_w / max(values.size - 1, 1))
    ys = height - margin_b - ((values - vmin) / (vmax - vmin)) * plot_h
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    for token_id in top_ids:
        if 0 <= int(token_id) < values.size:
            x = float(xs[int(token_id)])
            draw.line((x, margin_t, x, height - margin_b), fill=(color[0], color[1], color[2], 55), width=1)
    draw.text((margin_l, height - 24), x_label, fill=(0, 0, 0, 255))
    draw.text((8, margin_t), f"max={vmax:.4g}", fill=(0, 0, 0, 255))
    draw.text((8, height - margin_b - 10), f"min={vmin:.4g}", fill=(0, 0, 0, 255))
    return canvas


def normalize_values(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    if values.size == 0 or not finite.any():
        return np.zeros_like(values, dtype=np.float32), 0.0, 0.0
    vmin = float(np.min(values[finite]))
    vmax = float(np.max(values[finite]))
    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values, dtype=np.float32), vmin, vmax
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[finite] = np.clip((values[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    return normalized, vmin, vmax


def heat_color(value: float):
    value = float(np.clip(value, 0.0, 1.0))
    if value < 0.5:
        t = value / 0.5
        r = int(35 + 220 * t)
        g = int(90 + 120 * t)
        b = int(190 - 130 * t)
    else:
        t = (value - 0.5) / 0.5
        r = 255
        g = int(210 - 95 * t)
        b = int(60 - 45 * t)
    a = int(45 + 155 * value)
    return r, g, b, a


def draw_score_heatmap(
    image: Image.Image,
    score,
    grid: Tuple[int, int],
    *,
    title: str,
    background_alpha: float,
) -> Image.Image:
    values = tensor_1d(score, "score")
    canvas = faded_image(image, background_alpha)
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    if values.size == 0:
        draw.rectangle((0, 0, width, 24), fill=(255, 255, 255, 220))
        draw.text((8, 5), f"{title}: no score in dump", fill=(0, 0, 0, 255))
        return canvas
    normalized, vmin, vmax = normalize_values(values)
    token_count = min(values.size, grid[0] * grid[1])
    for index in range(token_count):
        draw.rectangle(patch_box(index, grid, width, height), fill=heat_color(float(normalized[index])))
    draw.rectangle((0, 0, width, 42), fill=(255, 255, 255, 220))
    draw.text((8, 5), title, fill=(0, 0, 0, 255))
    draw.text((8, 23), f"min={vmin:.4g}  max={vmax:.4g}", fill=(0, 0, 0, 255))
    return canvas


def draw_qceg_entropy_curves(
    H,
    *,
    qceg_top: List[int],
    raw_top: List[int],
    question_top: List[int],
    first: set[int],
    second: set[int],
    final: set[int],
    score,
) -> Image.Image:
    H = tensor_2d(H, "qceg_H")
    values = tensor_1d(score, "qceg_score")
    width, height = 920, 360
    margin_l, margin_r, margin_t, margin_b = 64, 26, 38, 42
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((margin_l, 12), "QCEG normalized entropy across hidden states", fill=(0, 0, 0, 255))
    draw.line((margin_l, height - margin_b, width - margin_r, height - margin_b), fill=(40, 40, 40, 255))
    draw.line((margin_l, margin_t, margin_l, height - margin_b), fill=(40, 40, 40, 255))
    if H.shape[0] < 4 or H.shape[1] == 0:
        draw.text((margin_l + 12, margin_t + 40), "no QCEG entropy in dump", fill=(120, 120, 120, 255))
        return canvas

    blocked = set(first) | set(second) | set(final)
    background = None
    if values.size:
        for idx in np.argsort(values).tolist():
            if int(idx) not in blocked:
                background = int(idx)
                break
        if background is None:
            background = int(np.argsort(values)[0])

    reps = OrderedDict()
    if qceg_top:
        reps[f"QCEG top token {qceg_top[0]}"] = (int(qceg_top[0]), (220, 70, 35, 255))
    if raw_top:
        reps[f"raw sink token {raw_top[0]}"] = (int(raw_top[0]), (80, 80, 80, 255))
    if background is not None:
        reps[f"background token {background}"] = (int(background), (40, 120, 210, 255))
    if question_top:
        reps[f"target proxy token {question_top[0]}"] = (int(question_top[0]), (40, 155, 90, 255))

    finite = H[np.isfinite(H)]
    ymin = float(np.min(finite)) if finite.size else 0.0
    ymax = float(np.max(finite)) if finite.size else 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = [margin_l + i * (plot_w / 3.0) for i in range(4)]
    for label, (token_id, color) in reps.items():
        if not (0 <= token_id < H.shape[1]):
            continue
        ys = height - margin_b - ((H[:4, token_id] - ymin) / (ymax - ymin)) * plot_h
        points = [(float(x), float(y)) for x, y in zip(xs, ys)]
        draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    for i, x in enumerate(xs):
        draw.text((x - 8, height - 28), f"H{i}", fill=(0, 0, 0, 255))
    legend_y = margin_t + 8
    for label, (_token_id, color) in reps.items():
        draw.rectangle((width - 285, legend_y + 3, width - 273, legend_y + 15), fill=color)
        draw.text((width - 266, legend_y), label, fill=(0, 0, 0, 255))
        legend_y += 22
    draw.text((8, margin_t), f"max={ymax:.4g}", fill=(0, 0, 0, 255))
    draw.text((8, height - margin_b - 8), f"min={ymin:.4g}", fill=(0, 0, 0, 255))
    return canvas


def save_panel(images: List[Tuple[str, Image.Image]], path: Path) -> None:
    if not images:
        return
    thumb_w = max(img.width for _, img in images)
    thumb_h = max(img.height for _, img in images)
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    panel = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for idx, (title, img) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * thumb_w
        y = row * (thumb_h + 28)
        panel.paste(img.convert("RGB"), (x, y + 28))
        draw.text((x + 8, y + 8), title, fill=(0, 0, 0))
    panel.save(path)


def save_tokens_csv(path: Path, data: Dict[str, object], grid: Tuple[int, int]) -> None:
    raw_score = tensor_1d(data.get("raw_query_score"), "raw_query_score")
    bos_score = tensor_1d(data.get("bos_free_score"), "bos_free_score")
    question_score = tensor_1d(data.get("question_only_score"), "question_only_score")
    first = tensor_set(data.get("first_selected_indices"))
    second = tensor_set(data.get("second_selected_indices"))
    final = tensor_set(data.get("final_keep_indices"))
    count = max(raw_score.size, bos_score.size, question_score.size, len(final))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "token_id",
                "row",
                "col",
                "raw_query_score",
                "bos_free_score",
                "question_only_score",
                "is_first_stage",
                "is_second_stage",
                "is_final_keep",
            ],
        )
        writer.writeheader()
        for token_id in range(count):
            row, col = row_col(token_id, grid)
            writer.writerow(
                {
                    "token_id": token_id,
                    "row": row,
                    "col": col,
                    "raw_query_score": float(raw_score[token_id]) if token_id < raw_score.size else "",
                    "bos_free_score": float(bos_score[token_id]) if token_id < bos_score.size else "",
                    "question_only_score": (
                        float(question_score[token_id]) if token_id < question_score.size else ""
                    ),
                    "is_first_stage": int(token_id in first),
                    "is_second_stage": int(token_id in second),
                    "is_final_keep": int(token_id in final),
                }
            )


def save_qceg_score_table(path: Path, data: Dict[str, object], grid: Tuple[int, int], qceg_top: List[int]) -> None:
    score = tensor_1d(data.get("qceg_score"), "qceg_score")
    H = tensor_2d(data.get("qceg_H"), "qceg_H")
    gains = tensor_2d(data.get("qceg_gains"), "qceg_gains")
    count = max(score.size, H.shape[1] if H.ndim == 2 else 0, gains.shape[1] if gains.ndim == 2 else 0)
    ranks = np.full(count, "", dtype=object)
    if score.size:
        order = np.argsort(score)[::-1]
        for rank, token_id in enumerate(order.tolist(), start=1):
            if token_id < count:
                ranks[token_id] = rank
    qceg_top_set = {int(item) for item in qceg_top}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "token_id",
                "row",
                "col",
                "H0",
                "H1",
                "H2",
                "H3",
                "gain_0_1",
                "gain_1_2",
                "gain_2_3",
                "qceg_score",
                "qceg_rank",
                "is_top_qceg",
            ],
        )
        writer.writeheader()
        for token_id in range(count):
            row, col = row_col(token_id, grid)
            writer.writerow(
                {
                    "token_id": token_id,
                    "row": row,
                    "col": col,
                    "H0": float(H[0, token_id]) if H.shape[0] > 0 and token_id < H.shape[1] else "",
                    "H1": float(H[1, token_id]) if H.shape[0] > 1 and token_id < H.shape[1] else "",
                    "H2": float(H[2, token_id]) if H.shape[0] > 2 and token_id < H.shape[1] else "",
                    "H3": float(H[3, token_id]) if H.shape[0] > 3 and token_id < H.shape[1] else "",
                    "gain_0_1": float(gains[0, token_id]) if gains.shape[0] > 0 and token_id < gains.shape[1] else "",
                    "gain_1_2": float(gains[1, token_id]) if gains.shape[0] > 1 and token_id < gains.shape[1] else "",
                    "gain_2_3": float(gains[2, token_id]) if gains.shape[0] > 2 and token_id < gains.shape[1] else "",
                    "qceg_score": float(score[token_id]) if token_id < score.size else "",
                    "qceg_rank": ranks[token_id] if token_id < len(ranks) else "",
                    "is_top_qceg": int(token_id in qceg_top_set),
                }
            )


def save_token_list(path: Path, token_ids: List[int], score) -> None:
    score = tensor_1d(score, "score")
    lines = []
    for token_id in token_ids:
        value = float(score[token_id]) if 0 <= token_id < score.size else 0.0
        lines.append(f"{token_id}\t{value:.10g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_query_debug(path: Path, data: Dict[str, object]) -> None:
    rows = data.get("query_token_debug") or []
    lines = [
        "local\tabs_pos\ttoken_id\tbos_free\tquestion_only\ttoken",
    ]
    for row in rows:
        lines.append(
            f"{row.get('local')}\t{row.get('abs_pos')}\t{row.get('token_id')}\t"
            f"{row.get('bos_free')}\t{row.get('question_only')}\t{row.get('token')!r}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_overlap_report(path: Path, groups: Dict[str, List[int]]) -> None:
    def line(name_a: str, name_b: str) -> List[str]:
        set_a = {int(x) for x in groups.get(name_a, [])}
        set_b = {int(x) for x in groups.get(name_b, [])}
        inter = sorted(set_a & set_b)
        denom = len(set_a) if set_a else 0
        return [
            f"{name_a}_{name_b}_count: {len(inter)} / {denom}",
            f"{name_a}_{name_b}_tokens: {inter}",
        ]

    lines = []
    for name_a, name_b in [("all", "bos_free"), ("all", "question"), ("bos_free", "question")]:
        if name_a in groups and name_b in groups:
            lines.extend(line(name_a, name_b))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_case(path: Path, args) -> None:
    data = load_pt(path)
    case_id = str(data.get("case_id", path.stem))
    save_dir = args.save_dir / case_id
    save_dir.mkdir(parents=True, exist_ok=True)

    image = load_background(data, args.image_root)
    grid = patch_grid(data)
    first = tensor_set(data.get("first_selected_indices"))
    second = tensor_set(data.get("second_selected_indices"))
    final = tensor_set(data.get("final_keep_indices"))
    selection_mode = str(data.get("selection_score_mode") or data.get("query_score_mode") or "bos_free")
    second_label = f"S2 {selection_mode} LLM"
    all_score = data.get("all_query_score", data.get("raw_query_score"))
    raw_top = top_indices(all_score, args.top_k)
    bos_top = top_indices(data.get("bos_free_score"), args.top_k)
    question_top = top_indices(data.get("question_only_score"), args.top_k)
    qceg_score = data.get("qceg_score")
    qceg_top = top_indices(qceg_score, args.top_k)
    qceg_gains = tensor_2d(data.get("qceg_gains"), "qceg_gains")
    coarse = tensor_set(data.get("coarse_candidate_indices"))

    raw_img = draw_selection(
        image,
        raw_top,
        grid,
        fill=(160, 160, 160, 100),
        outline=(80, 80, 80, 255),
        background_alpha=args.background_alpha,
        label_first=12,
        title=f"all query top-{len(raw_top)}",
    )
    bos_img = draw_selection(
        image,
        bos_top,
        grid,
        fill=(25, 116, 230, 115),
        outline=(0, 55, 155, 255),
        background_alpha=args.background_alpha,
        label_first=12,
        title=f"BOS-free query top-{len(bos_top)}",
    )
    question_img = draw_selection(
        image,
        question_top,
        grid,
        fill=(40, 170, 100, 115),
        outline=(0, 115, 55, 255),
        background_alpha=args.background_alpha,
        label_first=12,
        title=f"question-only query top-{len(question_top)}",
    )
    first_img = draw_selection(
        image,
        first,
        grid,
        fill=(245, 132, 32, 105),
        outline=(185, 72, 0, 255),
        background_alpha=args.background_alpha,
        title="S1 visual selection",
    )
    second_img = draw_selection(
        image,
        second,
        grid,
        fill=(25, 116, 230, 105),
        outline=(0, 55, 155, 255),
        background_alpha=args.background_alpha,
        title=f"{second_label} selection",
    )
    final_img = draw_stage_overlay(image, first, second, final, grid, args.background_alpha, second_label=second_label)
    coarse_img = draw_selection(
        image,
        coarse,
        grid,
        fill=(120, 80, 200, 75),
        outline=(80, 40, 145, 230),
        background_alpha=args.background_alpha,
        title=f"{selection_mode} coarse candidate pool",
    )
    qceg_top_img = draw_selection(
        image,
        qceg_top,
        grid,
        fill=(220, 70, 35, 115),
        outline=(150, 35, 15, 255),
        background_alpha=args.background_alpha,
        label_first=12,
        title=f"QCEG score top-{len(qceg_top)}",
    )
    qceg_score_img = draw_score_heatmap(
        image,
        qceg_score,
        grid,
        title="QCEG score heatmap",
        background_alpha=args.background_alpha,
    )
    qceg_gain_imgs = []
    for gain_idx, name in enumerate(["g1 H0-H1", "g2 H1-H2", "g3 H2-H3"]):
        gain_score = qceg_gains[gain_idx] if gain_idx < qceg_gains.shape[0] else np.empty(0, dtype=np.float32)
        qceg_gain_imgs.append(
            draw_score_heatmap(
                image,
                gain_score,
                grid,
                title=f"QCEG gain {name}",
                background_alpha=args.background_alpha,
            )
        )

    raw_img.save(save_dir / "raw_topK_grid.png")
    raw_img.save(save_dir / "all_query_topK_grid.png")
    bos_img.save(save_dir / "bos_free_query_topK_grid.png")
    question_img.save(save_dir / "question_only_topK_grid.png")
    first_img.save(save_dir / "first_stage_tokens.png")
    second_img.save(save_dir / "second_stage_tokens.png")
    final_img.save(save_dir / "final_selected_tokens.png")
    coarse_img.save(save_dir / "selection_candidate_pool.png")
    qceg_top_img.save(save_dir / "qceg_topK_grid.png")
    qceg_score_img.save(save_dir / "qceg_score_grid.png")
    qceg_gain_imgs[0].save(save_dir / "qceg_gain_g1_grid.png")
    qceg_gain_imgs[1].save(save_dir / "qceg_gain_g2_grid.png")
    qceg_gain_imgs[2].save(save_dir / "qceg_gain_g3_grid.png")
    draw_qceg_entropy_curves(
        data.get("qceg_H"),
        qceg_top=qceg_top,
        raw_top=raw_top,
        question_top=question_top,
        first=first,
        second=second,
        final=final,
        score=qceg_score,
    ).save(save_dir / "qceg_entropy_curves.png")
    save_panel(
        [
            ("raw top-K", raw_img),
            ("BOS-free top-K", bos_img),
            ("question-only top-K", question_img),
            ("QCEG score top-K", qceg_top_img),
            ("QCEG score heatmap", qceg_score_img),
            ("S1 visual", first_img),
            ("S2 LLM", second_img),
            ("final keep", final_img),
            ("coarse candidates", coarse_img),
        ],
        save_dir / "selected_tokens_panel.png",
    )
    draw_score_curve(all_score, raw_top, "all query score over visual tokens", (90, 90, 90, 255)).save(
        save_dir / "all_query_topK_curves.png"
    )
    draw_score_curve(data.get("bos_free_score"), bos_top, "BOS-free query score over visual tokens", (25, 116, 230, 255)).save(
        save_dir / "bos_free_query_topK_curves.png"
    )
    draw_score_curve(
        data.get("question_only_score"),
        question_top,
        "question-only query score over visual tokens",
        (40, 170, 100, 255),
    ).save(save_dir / "question_only_topK_curves.png")
    save_token_list(save_dir / "top_raw_query_tokens.txt", raw_top, all_score)
    save_token_list(save_dir / "top_all_tokens.txt", raw_top, all_score)
    save_token_list(save_dir / "top_bos_free_query_tokens.txt", bos_top, data.get("bos_free_score"))
    save_token_list(save_dir / "top_question_only_tokens.txt", question_top, data.get("question_only_score"))
    save_token_list(save_dir / "qceg_topK_tokens.txt", qceg_top, qceg_score)
    save_query_debug(save_dir / "query_filter_debug.txt", data)
    save_overlap_report(
        save_dir / "query_filter_overlap_report.txt",
        {"all": raw_top, "bos_free": bos_top, "question": question_top},
    )
    save_tokens_csv(save_dir / "selected_token_scores.csv", data, grid)
    save_qceg_score_table(save_dir / "qceg_score_table.csv", data, grid, qceg_top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--background-alpha", type=float, default=0.62)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.dump_dir.glob("*.pt")):
        process_case(path, args)
        print(f"visualized {path.stem}")


if __name__ == "__main__":
    main()
