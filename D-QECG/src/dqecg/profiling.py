"""Opt-in profiler ranges for diagnosing D-QECG prefill overhead."""

from __future__ import annotations

import os
from contextlib import nullcontext

from torch.profiler import record_function


def record_stage(name: str):
    """Return a profiler range only when stage diagnostics are requested."""

    if os.environ.get("DQECG_STAGE_PROFILE", "0") == "1":
        return record_function(f"dqecg::{name}")
    return nullcontext()
