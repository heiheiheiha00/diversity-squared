#!/usr/bin/env python3
"""Compare baseline and D-QECG metrics from lmms-eval result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from numbers import Number
from pathlib import Path


def find_result(path: Path) -> tuple[Path, dict]:
    candidates: list[tuple[float, Path, dict]] = []
    for candidate in path.rglob("*.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
            candidates.append((candidate.stat().st_mtime, candidate, payload))
    if not candidates:
        raise FileNotFoundError(f"No lmms-eval result JSON found under {path}")
    _, result_path, payload = max(candidates, key=lambda item: item[0])
    return result_path, payload


def extract_metrics(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = payload["results"]
    preferred = [name for name in results if name.lower() == "mme"]
    task_names = preferred or [name for name in results if "mme" in name.lower()]
    if not task_names:
        task_names = list(results)

    for task_name in task_names:
        task_metrics = results.get(task_name)
        if not isinstance(task_metrics, dict):
            continue
        for metric_name, value in task_metrics.items():
            if isinstance(value, bool) or not isinstance(value, Number):
                continue
            if "stderr" in metric_name.lower():
                continue
            label = metric_name.split(",", 1)[0]
            if len(task_names) > 1:
                label = f"{task_name}/{label}"
            metrics[label] = float(value)

    perception = next(
        (
            value
            for name, value in metrics.items()
            if "perception" in name.lower() and "score" in name.lower()
        ),
        None,
    )
    cognition = next(
        (
            value
            for name, value in metrics.items()
            if "cognition" in name.lower() and "score" in name.lower()
        ),
        None,
    )
    if perception is not None and cognition is not None:
        metrics.setdefault("mme_total_score", perception + cognition)
    if not metrics:
        raise ValueError("The result JSON contains no numeric MME metrics")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--dqecg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_file, baseline_payload = find_result(args.baseline_dir)
    dqecg_file, dqecg_payload = find_result(args.dqecg_dir)
    baseline = extract_metrics(baseline_payload)
    dqecg = extract_metrics(dqecg_payload)
    metric_names = sorted(set(baseline) | set(dqecg))

    rows = []
    for metric in metric_names:
        baseline_value = baseline.get(metric)
        dqecg_value = dqecg.get(metric)
        delta = (
            dqecg_value - baseline_value
            if baseline_value is not None and dqecg_value is not None
            else None
        )
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "dqecg": dqecg_value,
                "delta": delta,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_result": str(baseline_file),
        "dqecg_result": str(dqecg_file),
        "metrics": rows,
    }
    (args.output_dir / "mme_score_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (args.output_dir / "mme_score_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "baseline", "dqecg", "delta"]
        )
        writer.writeheader()
        writer.writerows(rows)

    def display(value: float | None) -> str:
        return "" if value is None else f"{value:.6f}"

    markdown = [
        "# MME score comparison",
        "",
        f"- Baseline result: `{baseline_file}`",
        f"- D-QECG result: `{dqecg_file}`",
        "",
        "| Metric | Baseline | D-QECG | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| {row['metric']} | {display(row['baseline'])} | "
        f"{display(row['dqecg'])} | {display(row['delta'])} |"
        for row in rows
    )
    (args.output_dir / "mme_score_comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    print("\n".join(markdown))
    print(f"\nReports saved under: {args.output_dir}")


if __name__ == "__main__":
    main()

