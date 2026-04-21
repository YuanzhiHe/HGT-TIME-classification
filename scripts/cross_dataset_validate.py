#!/usr/bin/env python3
"""Cross-dataset validation: evaluate trained models on external graph data.

Loads best checkpoints from a completed experiment directory and evaluates
them on an external set of heterogeneous graphs. Reports per-checkpoint
and ensemble metrics.

Usage:
    python cross_dataset_validate.py \
        --model-dir outputs/results/EXP-M01-HGT \
        --external-graphs outputs/hetero_graph/external_breast__hetero_v1/graphs \
        --config configs/hgt_time.default.yaml

    python cross_dataset_validate.py \
        --model-dir outputs/results/EXP-B03-GCN \
        --external-graphs outputs/hetero_graph/tcga_brca__hetero_v1/graphs \
        --config configs/baseline_homo_gcn.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise SystemExit("Could not locate project root via repository structure")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Cross-dataset validation for TIME models")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with checkpoints/")
    parser.add_argument("--external-graphs", type=Path, required=True, help="Path to external graph .pt files")
    parser.add_argument("--config", type=Path, default=None, help="Model config YAML (auto-detected if absent)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: model-dir/cross_validation.json)")
    args = parser.parse_args()

    project_root = discover_project_root(args.model_dir)
    sys.path.insert(0, str(project_root / "Experiment" / "core_code"))

    from torch_geometric.loader import DataLoader
    from scripts.train_eval_pipeline import build_model, compute_metrics

    # Load external graphs
    ext_dir = args.external_graphs
    if not ext_dir.exists():
        logger.error(f"External graphs directory not found: {ext_dir}")
        sys.exit(1)

    graph_paths = sorted(ext_dir.glob("*.pt"))
    if not graph_paths:
        logger.error(f"No .pt files found under {ext_dir}")
        sys.exit(1)

    ext_graphs = [torch.load(p, weights_only=False) for p in graph_paths]
    logger.info(f"Loaded {len(ext_graphs)} external graphs from {ext_dir}")

    ext_loader = DataLoader(ext_graphs, batch_size=args.batch_size, shuffle=False)

    # Determine config
    if args.config is not None:
        config_path = project_root / args.config if not args.config.is_absolute() else args.config
    else:
        # Try to find config from results.json
        results_json = args.model_dir / "results.json"
        if results_json.exists():
            with results_json.open("r", encoding="utf-8") as f:
                exp_results = json.load(f)
            # Try to infer model_family
            model_family = exp_results.get("aggregated", {}).get("model_family", "hgt_time")
            config_path = project_root / "Experiment" / "core_code" / "configs" / "hgt_time.default.yaml"
        else:
            logger.error("No --config and no results.json found for auto-detection")
            sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    model_family = config.get("model_family", "hgt_time")
    num_classes = config.get("model", {}).get("num_classes", 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoints
    ckpt_dir = args.model_dir / "checkpoints"
    if not ckpt_dir.exists():
        logger.error(f"No checkpoints directory in {args.model_dir}")
        sys.exit(1)

    ckpt_paths = sorted(ckpt_dir.glob("best_*.pt"))
    if not ckpt_paths:
        logger.error(f"No checkpoint files found in {ckpt_dir}")
        sys.exit(1)

    logger.info(f"Found {len(ckpt_paths)} checkpoints")

    # Evaluate each checkpoint
    per_checkpoint_results: list[dict[str, Any]] = []
    ensemble_probs_sum: np.ndarray | None = None
    all_y_true: list[int] | None = None

    for ckpt_path in ckpt_paths:
        logger.info(f"Evaluating {ckpt_path.name}")

        model = build_model(model_family, config.get("model", {}), ext_graphs[0])
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        y_true_list: list[int] = []
        y_prob_list: list[np.ndarray] = []

        with torch.no_grad():
            for batch in ext_loader:
                batch = batch.to(device)
                outputs = model(batch)
                logits = outputs["graph_logits"]
                probs = F.softmax(logits, dim=-1)

                if hasattr(batch, "y_graph") and batch.y_graph is not None:
                    y = batch.y_graph.view(-1).long()
                else:
                    y = torch.zeros(logits.size(0), dtype=torch.long)

                y_true_list.extend(y.cpu().numpy().tolist())
                y_prob_list.extend(probs.cpu().numpy())

        y_true_arr = np.array(y_true_list)
        y_prob_arr = np.array(y_prob_list)

        metrics = compute_metrics(y_true_arr, y_prob_arr, num_classes=num_classes)
        per_checkpoint_results.append({
            "checkpoint": ckpt_path.name,
            "metrics": metrics,
        })

        # Accumulate for ensemble
        if ensemble_probs_sum is None:
            ensemble_probs_sum = y_prob_arr.copy()
            all_y_true = y_true_list.copy()
        else:
            ensemble_probs_sum += y_prob_arr

        logger.info(
            f"  {ckpt_path.name} | AUROC={metrics['macro_auroc']:.4f} "
            f"F1={metrics['macro_f1']:.4f} BAcc={metrics['balanced_accuracy']:.4f}"
        )

    # Ensemble (average probabilities)
    ensemble_metrics: dict[str, float] = {}
    if ensemble_probs_sum is not None and all_y_true is not None:
        ensemble_probs = ensemble_probs_sum / len(ckpt_paths)
        ensemble_metrics = compute_metrics(np.array(all_y_true), ensemble_probs, num_classes=num_classes)
        logger.info(
            f"  ENSEMBLE | AUROC={ensemble_metrics['macro_auroc']:.4f} "
            f"F1={ensemble_metrics['macro_f1']:.4f} BAcc={ensemble_metrics['balanced_accuracy']:.4f}"
        )

    # Aggregate per-checkpoint metrics
    agg: dict[str, Any] = {}
    metric_keys = list(per_checkpoint_results[0]["metrics"].keys()) if per_checkpoint_results else []
    for key in metric_keys:
        values = [r["metrics"][key] for r in per_checkpoint_results if not np.isnan(r["metrics"].get(key, float("nan")))]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))

    # Save results
    output_path = args.output or (args.model_dir / "cross_validation.json")
    result = {
        "external_data": str(ext_dir),
        "num_external_graphs": len(ext_graphs),
        "num_checkpoints": len(ckpt_paths),
        "model_family": model_family,
        "per_checkpoint": per_checkpoint_results,
        "aggregated": agg,
        "ensemble": ensemble_metrics,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
