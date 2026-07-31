#!/usr/bin/env python3
"""Verify that the files required for MME evaluation are present."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "server/install_lmms_eval_only.sh",
    "server/patch_lmms_eval_llava.py",
    "server/run_mme.sh",
    "server/run_full_mme_comparison.sh",
    "server/summarize_mme_scores.py",
    "src/LLaVA/llava/model/llava_arch.py",
    "src/LLaVA/llava/model/language_model/llava_llama.py",
    "src/dqecg/question_span.py",
    "src/dqecg/pruning.py",
    "src/dqecg/llm_selector.py",
)


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(
            "Incomplete D-QECG evaluation bundle:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    for relative in REQUIRED:
        if relative.endswith(".py"):
            path = ROOT / relative
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print(f"D-QECG evaluation bundle: ok ({len(REQUIRED)} required files)")


if __name__ == "__main__":
    main()

