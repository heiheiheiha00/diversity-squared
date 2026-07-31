"""Small helpers for synchronized per-sample inference latency measurement."""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Dict, Iterable, Tuple, TypeVar


T = TypeVar("T")


def _is_cuda_device(device) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def synchronize(device=None) -> None:
    """Wait for queued CUDA work so wall-clock timing is not asynchronous."""
    if _is_cuda_device(device):
        import torch

        torch.cuda.synchronize(device)


def measure_call(call: Callable[[], T], device=None) -> Tuple[T, float]:
    """Measure one callable after synchronizing the selected CUDA device."""
    synchronize(device)
    start = time.perf_counter()
    result = call()
    synchronize(device)
    return result, time.perf_counter() - start


def _percentile(sorted_values, percentile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_latencies(latencies: Iterable[float]) -> Dict[str, float]:
    """Return JSON-serializable per-sample latency statistics in seconds."""
    values = [float(value) for value in latencies]
    if not values:
        raise ValueError("At least one latency value is required.")
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("Latency values must be finite and non-negative.")

    ordered = sorted(values)
    return {
        "num_samples": len(values),
        "mean_seconds": float(statistics.fmean(values)),
        "median_seconds": float(statistics.median(values)),
        "p95_seconds": _percentile(ordered, 0.95),
        "min_seconds": float(ordered[0]),
        "max_seconds": float(ordered[-1]),
    }


def latency_report(latencies: Iterable[float], *, scope: str = "model.generate") -> Dict[str, object]:
    """Build the output object used by evaluation and benchmark scripts."""
    values = [float(value) for value in latencies]
    return {
        "unit": "seconds_per_sample",
        "scope": scope,
        "per_sample_seconds": values,
        **summarize_latencies(values),
    }
