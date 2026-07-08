#!/usr/bin/env python3
"""Experiment orchestration script.

Runs the full experimental protocol in sequence:
  Phase 1: Run all baseline experiments (default configs)
  Phase 2: Run HGT-TIME hyperparameter search (Optuna)
  Phase 3: Run HGT-TIME with best hyperparameters
  Phase 4: Cross-dataset validation (if external data available)
  Phase 5: Generate comparison report

Usage:
    # Full pipeline
    python run_experiments.py --phases 1 2 3 5

    # Baselines only
    python run_experiments.py --phases 1 --seeds 42 123 2026

    # HGT search + final run only
    python run_experiments.py --phases 2 3 --hparam-trials 30

    # Comparison report only
    python run_experiments.py --phases 5
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def run_cmd(cmd: list[str], label: str, log_dir: Path) -> int:
    """Run a command, stream output, and log to file."""
    log_file = log_dir / f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.info(f"[{label}] Running: {' '.join(cmd)}")
    logger.info(f"[{label}] Log: {log_file}")

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"# {label}\n# cmd: {' '.join(cmd)}\n# started: {datetime.now().isoformat()}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()
        elapsed = time.time() - t0
        lf.write(f"\n# finished: {datetime.now().isoformat()}\n# elapsed: {elapsed:.1f}s\n# exit code: {proc.returncode}\n")

    if proc.returncode != 0:
        logger.error(f"[{label}] FAILED (exit code {proc.returncode})")
    else:
        logger.info(f"[{label}] DONE in {elapsed:.1f}s")
    return proc.returncode


def build_seed_args(seeds: list[int]) -> list[str]:
    """Render argparse-compatible seed arguments."""
    if not seeds:
        return []
    return ["--seeds", *[str(seed) for seed in seeds]]


def phase1_baselines(project_root: Path, seeds: list[int], log_dir: Path) -> dict[str, Any]:
    """Phase 1: Run all baseline experiments from the registry."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Running baseline experiments")
    logger.info("=" * 60)

    registry_path = project_root / "Experiment" / "core_code" / "configs" / "experiment_registry.yaml"
    with registry_path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    results: dict[str, Any] = {"phase": 1, "experiments": {}}
    baseline_exps = [e for e in registry.get("experiments", []) if e.get("role") == "baseline"]

    for exp in baseline_exps:
        config_path = project_root / exp["config"]
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, skipping {exp['id']}")
            results["experiments"][exp["id"]] = {"status": "skipped", "reason": "config not found"}
            continue

        cmd = [
            sys.executable,
            str(project_root / "Experiment" / "core_code" / "scripts" / "train_eval_pipeline.py"),
            "--config", str(config_path),
        ] + build_seed_args(seeds)

        rc = run_cmd(cmd, f"baseline_{exp['id']}", log_dir)
        results["experiments"][exp["id"]] = {
            "status": "success" if rc == 0 else "failed",
            "exit_code": rc,
        }

    return results


def phase2_hparam_search(
    project_root: Path, n_trials: int, log_dir: Path, model_configs: list[str] | None = None
) -> dict[str, Any]:
    """Phase 2: Run Optuna hyperparameter search for specified models."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Hyperparameter search")
    logger.info("=" * 60)

    if model_configs is None:
        model_configs = ["Experiment/core_code/configs/hgt_time.default.yaml"]

    results: dict[str, Any] = {"phase": 2, "searches": {}}

    for cfg_rel in model_configs:
        cfg_path = project_root / cfg_rel
        if not cfg_path.exists():
            logger.warning(f"Config not found: {cfg_path}")
            continue

        with cfg_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        model_family = config.get("model_family", "unknown")

        cmd = [
            sys.executable,
            str(project_root / "Experiment" / "core_code" / "scripts" / "hparam_search.py"),
            "--config", str(cfg_path),
            "--search-config", str(project_root / "Experiment" / "core_code" / "configs" / "hyperparameter_search_strategy.yaml"),
            "--n-trials", str(n_trials),
        ]

        rc = run_cmd(cmd, f"hparam_{model_family}", log_dir)
        results["searches"][model_family] = {
            "status": "success" if rc == 0 else "failed",
            "exit_code": rc,
        }

    return results


def phase3_main_with_best(project_root: Path, seeds: list[int], log_dir: Path) -> dict[str, Any]:
    """Phase 3: Run main model with best hyperparameters (or default if search not done)."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Running main model with best hyperparameters")
    logger.info("=" * 60)

    results: dict[str, Any] = {"phase": 3, "experiments": {}}

    # Check if best config from search exists
    best_config_path = (
        project_root
        / "Experiment" / "core_code" / "outputs" / "results"
        / "hparam_search" / "hgt_time_search_best_config.yaml"
    )
    if best_config_path.exists():
        config_path = best_config_path
        logger.info(f"Using best config from search: {config_path}")
    else:
        config_path = project_root / "Experiment" / "core_code" / "configs" / "hgt_time.default.yaml"
        logger.info(f"No search results found, using default config: {config_path}")

    cmd = [
        sys.executable,
        str(project_root / "Experiment" / "core_code" / "scripts" / "train_eval_pipeline.py"),
        "--config", str(config_path),
    ] + build_seed_args(seeds)

    rc = run_cmd(cmd, "main_hgt_time", log_dir)
    results["experiments"]["EXP-M01-HGT"] = {
        "status": "success" if rc == 0 else "failed",
        "exit_code": rc,
        "config_used": str(config_path),
    }

    return results


def phase4_cross_dataset(project_root: Path, seeds: list[int], log_dir: Path) -> dict[str, Any]:
    """Phase 4: Cross-dataset validation."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Cross-dataset validation")
    logger.info("=" * 60)

    script = project_root / "Experiment" / "core_code" / "scripts" / "cross_dataset_validate.py"
    if not script.exists():
        logger.warning("cross_dataset_validate.py not found, skipping phase 4")
        return {"phase": 4, "status": "skipped"}

    # Check if external graph data exists
    external_dirs = list((project_root / "Experiment" / "core_code" / "outputs" / "hetero_graph").glob("*"))
    external_dirs = [d for d in external_dirs if d.is_dir() and "mock" not in d.name]

    if len(external_dirs) < 2:
        logger.warning("Insufficient external datasets for cross-dataset validation, skipping")
        return {"phase": 4, "status": "skipped", "reason": "insufficient external data"}

    results: dict[str, Any] = {"phase": 4, "validations": {}}
    for ext_dir in external_dirs:
        graphs_dir = ext_dir / "graphs"
        if not graphs_dir.exists() or not list(graphs_dir.glob("*.pt")):
            continue

        cmd = [
            sys.executable, str(script),
            "--model-dir", str(project_root / "Experiment" / "core_code" / "outputs" / "results" / "EXP-M01-HGT"),
            "--external-graphs", str(graphs_dir),
        ]
        rc = run_cmd(cmd, f"cross_val_{ext_dir.name}", log_dir)
        results["validations"][ext_dir.name] = {
            "status": "success" if rc == 0 else "failed",
            "exit_code": rc,
        }

    return results


def phase5_report(project_root: Path, log_dir: Path) -> dict[str, Any]:
    """Phase 5: Generate comparison report."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Generating comparison reports")
    logger.info("=" * 60)

    results_dir = project_root / "Experiment" / "core_code" / "outputs" / "results"
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"phase": 5, "reports": []}

    for fmt, ext in [("markdown", "md"), ("tsv", "tsv"), ("latex", "tex")]:
        output_file = report_dir / f"comparison_table.{ext}"
        cmd = [
            sys.executable,
            str(project_root / "Experiment" / "core_code" / "scripts" / "evaluate.py"),
            "--results-dir", str(results_dir),
            "--format", fmt,
            "--output", str(output_file),
        ]
        rc = run_cmd(cmd, f"report_{fmt}", log_dir)
        if rc == 0:
            results["reports"].append(str(output_file))

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Experiment orchestration for TIME classification")
    parser.add_argument(
        "--phases",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
        help="Phases to run: 1=baselines, 2=hparam search, 3=main model, 4=cross-dataset, 5=report",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--hparam-trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()

    project_root = discover_project_root(Path("."))
    log_dir = project_root / "Experiment" / "core_code" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_log: dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "phases_requested": args.phases,
        "seeds": args.seeds,
        "results": {},
    }

    t0 = time.time()

    if 1 in args.phases:
        run_log["results"]["phase1"] = phase1_baselines(project_root, args.seeds, log_dir)

    if 2 in args.phases:
        run_log["results"]["phase2"] = phase2_hparam_search(project_root, args.hparam_trials, log_dir)

    if 3 in args.phases:
        run_log["results"]["phase3"] = phase3_main_with_best(project_root, args.seeds, log_dir)

    if 4 in args.phases:
        run_log["results"]["phase4"] = phase4_cross_dataset(project_root, args.seeds, log_dir)

    if 5 in args.phases:
        run_log["results"]["phase5"] = phase5_report(project_root, log_dir)

    run_log["finished_at"] = datetime.now().isoformat()
    run_log["total_elapsed_s"] = time.time() - t0

    # Save run log
    run_log_path = log_dir / f"run_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with run_log_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, default=str)
    logger.info(f"Run log saved to {run_log_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT RUN COMPLETE")
    logger.info(f"Total time: {run_log['total_elapsed_s']:.1f}s")
    for phase_key, phase_result in run_log["results"].items():
        phase_status = phase_result.get("status", "completed")
        logger.info(f"  {phase_key}: {phase_status}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
