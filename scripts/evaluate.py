#!/usr/bin/env python3
"""Standalone evaluation script: load results JSON files and produce
aggregated comparison tables across experiments.

Usage:
    python evaluate.py --results-dir outputs/results
    python evaluate.py --results-dir outputs/results --format markdown
    python evaluate.py --results-dir outputs/results --format latex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


METRIC_KEYS = [
    "macro_auroc",
    "macro_f1",
    "balanced_accuracy",
    "macro_auprc",
    "brier_score",
    "ece",
    "excluded_to_hot_confusion",
]

METRIC_LABELS = {
    "macro_auroc": "AUROC",
    "macro_f1": "macro-F1",
    "balanced_accuracy": "BAcc",
    "macro_auprc": "AUPRC",
    "brier_score": "Brier",
    "ece": "ECE",
    "excluded_to_hot_confusion": "Excl→Hot",
}

# Higher is better for these; lower is better for the rest
HIGHER_IS_BETTER = {"macro_auroc", "macro_f1", "balanced_accuracy", "macro_auprc"}


def load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    """Find and load all results.json files under results_dir."""
    entries: list[dict[str, Any]] = []
    for rfile in sorted(results_dir.rglob("results.json")):
        with rfile.open("r", encoding="utf-8") as f:
            data = json.load(f)
        entries.append(data)
    return entries


def aggregate_per_experiment(data: dict[str, Any]) -> dict[str, Any]:
    """Re-aggregate from per_fold data to get mean +/- std per metric."""
    per_fold = data.get("per_fold", [])
    if not per_fold:
        return data.get("aggregated", {})

    agg: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = [
            r["final_metrics"].get(key, float("nan"))
            for r in per_fold
        ]
        values = [v for v in values if not np.isnan(v)]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"] = float("nan")
    return agg


def format_tsv(experiments: list[dict[str, Any]]) -> str:
    """Format results as a TSV comparison table."""
    header_parts = ["experiment_id", "model_family", "n_seeds×folds"]
    for key in METRIC_KEYS:
        header_parts.append(f"{METRIC_LABELS.get(key, key)}")
    lines = ["\t".join(header_parts)]

    for exp in experiments:
        agg = aggregate_per_experiment(exp)
        exp_id = exp.get("experiment_id", "?")
        model_family = exp.get("aggregated", {}).get("model_family", "?")
        n_runs = len(exp.get("per_fold", []))
        row = [exp_id, model_family, str(n_runs)]
        for key in METRIC_KEYS:
            mean = agg.get(f"{key}_mean", float("nan"))
            std = agg.get(f"{key}_std", float("nan"))
            if np.isnan(mean):
                row.append("N/A")
            else:
                row.append(f"{mean:.4f}±{std:.4f}")
        lines.append("\t".join(row))
    return "\n".join(lines)


def format_markdown(experiments: list[dict[str, Any]]) -> str:
    """Format results as a Markdown comparison table."""
    header = "| Experiment | Model |"
    sep = "| --- | --- |"
    for key in METRIC_KEYS:
        label = METRIC_LABELS.get(key, key)
        arrow = "↑" if key in HIGHER_IS_BETTER else "↓"
        header += f" {label} {arrow} |"
        sep += " --- |"

    lines = [header, sep]

    # Find best values for bolding
    best: dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = []
        for exp in experiments:
            agg = aggregate_per_experiment(exp)
            m = agg.get(f"{key}_mean", float("nan"))
            if not np.isnan(m):
                vals.append(m)
        if vals:
            best[key] = max(vals) if key in HIGHER_IS_BETTER else min(vals)

    for exp in experiments:
        agg = aggregate_per_experiment(exp)
        exp_id = exp.get("experiment_id", "?")
        model_family = exp.get("aggregated", {}).get("model_family", "?")
        row = f"| {exp_id} | {model_family} |"
        for key in METRIC_KEYS:
            mean = agg.get(f"{key}_mean", float("nan"))
            std = agg.get(f"{key}_std", float("nan"))
            if np.isnan(mean):
                row += " N/A |"
            else:
                cell = f"{mean:.4f}±{std:.4f}"
                if key in best and abs(mean - best[key]) < 1e-6:
                    cell = f"**{cell}**"
                row += f" {cell} |"
        lines.append(row)
    return "\n".join(lines)


def format_latex(experiments: list[dict[str, Any]]) -> str:
    """Format results as a LaTeX tabular."""
    n_metrics = len(METRIC_KEYS)
    col_spec = "ll" + "c" * n_metrics
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    header = "Experiment & Model"
    for key in METRIC_KEYS:
        label = METRIC_LABELS.get(key, key).replace("→", "$\\rightarrow$")
        arrow = "$\\uparrow$" if key in HIGHER_IS_BETTER else "$\\downarrow$"
        header += f" & {label} {arrow}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for exp in experiments:
        agg = aggregate_per_experiment(exp)
        exp_id = exp.get("experiment_id", "?").replace("_", "\\_")
        model_family = exp.get("aggregated", {}).get("model_family", "?").replace("_", "\\_")
        row = f"{exp_id} & {model_family}"
        for key in METRIC_KEYS:
            mean = agg.get(f"{key}_mean", float("nan"))
            std = agg.get(f"{key}_std", float("nan"))
            if np.isnan(mean):
                row += " & N/A"
            else:
                row += f" & {mean:.4f}$\\pm${std:.4f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and compare experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Experiment/core_code/outputs/results"),
        help="Root directory containing per-experiment results",
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "markdown", "latex"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output to file instead of stdout",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    experiments = load_all_results(results_dir)
    if not experiments:
        print(f"No results.json files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(experiments)} experiments", file=sys.stderr)

    if args.format == "tsv":
        output = format_tsv(experiments)
    elif args.format == "latex":
        output = format_latex(experiments)
    else:
        output = format_markdown(experiments)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
