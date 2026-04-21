#!/usr/bin/env python3
"""Out-of-Distribution (OOD) evaluation for domain-generalizable TIME typing.

Implements two OOD evaluation protocols:

1. **Cross-section transfer**: Train on Section 1, evaluate on Section 2 (and vice versa).
   Uses existing checkpoints from the training pipeline.

2. **Leave-one-platform-out (LOPO)**: Train on all platforms except one, evaluate on
   the held-out platform. Requires multi-platform HEST-1k data.

Both protocols report per-domain and aggregated metrics (macro-AUROC, macro-F1,
balanced accuracy) and produce structured JSON output for downstream analysis.

Usage:
    # Cross-section transfer
    python ood_evaluation.py \\
        --protocol cross_section \\
        --train-graphs outputs/hetero_graph/visium_breast__hetero_v1/graphs \\
        --test-graphs outputs/hetero_graph/section2__hetero_v1/graphs \\
        --checkpoints outputs/results/EXP-M01-HGT/checkpoints \\
        --config configs/hgt_time.default.yaml

    # Leave-one-platform-out
    python ood_evaluation.py \\
        --protocol lopo \\
        --graphs-dir outputs/hetero_graph/multi_platform/graphs \\
        --config configs/hgt_time_domain_gen.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def load_graphs(graphs_dir: Path) -> list:
    """Load all .pt graph files from a directory."""
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    graphs = [torch.load(p, weights_only=False) for p in graph_paths]
    logger.info(f"Loaded {len(graphs)} graphs from {graphs_dir}")
    return graphs


def get_platform_label(graph) -> str:
    """Extract platform label from a graph's metadata."""
    for attr in ("platform_id", "platform", "spatial_variant"):
        val = getattr(graph, attr, None)
        if val is not None:
            return str(val) if not isinstance(val, (list, tuple)) else str(val[0])
    return "unknown"


def get_section_label(graph) -> str:
    """Extract section/cohort label from a graph's metadata."""
    for attr in ("cohort_id", "section_id", "slide_id"):
        val = getattr(graph, attr, None)
        if val is not None:
            return str(val) if not isinstance(val, (list, tuple)) else str(val[0])
    return "unknown"


def compute_ood_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 3
) -> dict[str, float]:
    """Compute OOD evaluation metrics."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    y_pred = np.argmax(y_prob, axis=1)
    metrics: dict[str, float] = {}

    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        if len(np.unique(y_true)) >= 2:
            metrics["macro_auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        else:
            metrics["macro_auroc"] = float("nan")
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    metrics["n_samples"] = int(len(y_true))
    metrics["class_distribution"] = {
        str(c): int((y_true == c).sum()) for c in range(num_classes)
    }

    return metrics


@torch.no_grad()
def evaluate_on_graphs(
    model: torch.nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, float]:
    """Run model on a list of graphs and compute metrics."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_y_true = []
    all_y_prob = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        logits = outputs["graph_logits"]
        probs = F.softmax(logits, dim=-1)

        if hasattr(batch, "y_graph") and batch.y_graph is not None:
            y = batch.y_graph.view(-1).long()
        else:
            y = torch.zeros(logits.size(0), dtype=torch.long)

        all_y_true.extend(y.cpu().numpy().tolist())
        all_y_prob.extend(probs.cpu().numpy())

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)
    return compute_ood_metrics(y_true, y_prob)


def cross_section_eval(
    train_graphs_dir: Path,
    test_graphs_dir: Path,
    checkpoints_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Cross-section transfer evaluation.

    Loads trained checkpoints (from train section), evaluates on test section.
    Reports per-checkpoint and ensemble metrics.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_eval_pipeline import build_model

    test_graphs = load_graphs(test_graphs_dir)
    train_graphs_sample = load_graphs(train_graphs_dir)[:1]  # Just for model init

    if not test_graphs:
        raise SystemExit(f"No test graphs found in {test_graphs_dir}")

    # Find checkpoints
    ckpt_paths = sorted(checkpoints_dir.glob("best_fold*.pt"))
    if not ckpt_paths:
        ckpt_paths = sorted(checkpoints_dir.glob("*.pt"))
    logger.info(f"Found {len(ckpt_paths)} checkpoints in {checkpoints_dir}")

    per_checkpoint_metrics = []
    all_probs = []  # For ensemble

    for ckpt_path in ckpt_paths:
        model = build_model(
            config.get("model_family", "hgt_time"),
            config.get("model", {}),
            train_graphs_sample[0],
        )
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)

        metrics = evaluate_on_graphs(model, test_graphs, device)
        metrics["checkpoint"] = ckpt_path.name
        per_checkpoint_metrics.append(metrics)

        # Collect predictions for ensemble
        model.eval()
        loader = DataLoader(test_graphs, batch_size=4, shuffle=False)
        ckpt_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                outputs = model(batch)
                probs = F.softmax(outputs["graph_logits"], dim=-1)
                ckpt_probs.extend(probs.cpu().numpy())
        all_probs.append(np.array(ckpt_probs))

    # Ensemble: average probabilities across checkpoints
    y_true = []
    loader = DataLoader(test_graphs, batch_size=4, shuffle=False)
    for batch in loader:
        if hasattr(batch, "y_graph") and batch.y_graph is not None:
            y_true.extend(batch.y_graph.view(-1).long().numpy().tolist())
        else:
            y_true.extend([0] * batch.num_graphs)
    y_true = np.array(y_true)

    ensemble_probs = np.mean(all_probs, axis=0) if all_probs else np.array([])
    ensemble_metrics = compute_ood_metrics(y_true, ensemble_probs) if len(ensemble_probs) > 0 else {}

    # Aggregate per-checkpoint stats
    aurocs = [m["macro_auroc"] for m in per_checkpoint_metrics if not np.isnan(m["macro_auroc"])]
    f1s = [m["macro_f1"] for m in per_checkpoint_metrics]
    bal_accs = [m["balanced_accuracy"] for m in per_checkpoint_metrics]

    return {
        "protocol": "cross_section",
        "train_source": str(train_graphs_dir),
        "test_target": str(test_graphs_dir),
        "n_checkpoints": len(ckpt_paths),
        "n_test_graphs": len(test_graphs),
        "ensemble_metrics": ensemble_metrics,
        "per_checkpoint_summary": {
            "macro_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "macro_auroc_std": float(np.std(aurocs)) if aurocs else None,
            "macro_auroc_range": [float(np.min(aurocs)), float(np.max(aurocs))] if aurocs else None,
            "macro_f1_mean": float(np.mean(f1s)),
            "balanced_acc_mean": float(np.mean(bal_accs)),
        },
        "per_checkpoint_metrics": per_checkpoint_metrics,
    }


def leave_one_platform_out_eval(
    graphs_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Leave-one-platform-out (LOPO) evaluation.

    For each unique platform in the dataset:
        1. Hold out all graphs from that platform as OOD test set.
        2. Train on all other platforms using the specified config.
        3. Evaluate on the held-out platform.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_eval_pipeline import build_model, train_one_fold, set_seed

    if seeds is None:
        seeds = [42]

    all_graphs = load_graphs(graphs_dir)
    if not all_graphs:
        raise SystemExit(f"No graphs found in {graphs_dir}")

    # Group by platform
    platform_groups: dict[str, list] = defaultdict(list)
    for g in all_graphs:
        plat = get_platform_label(g)
        platform_groups[plat].append(g)

    platforms = sorted(platform_groups.keys())
    logger.info(f"Found {len(platforms)} platforms: {platforms}")
    for plat in platforms:
        logger.info(f"  {plat}: {len(platform_groups[plat])} graphs")

    if len(platforms) < 2:
        logger.warning("LOPO requires at least 2 platforms. Skipping.")
        return {"protocol": "lopo", "error": "fewer than 2 platforms"}

    lopo_results = []

    for held_out_platform in platforms:
        logger.info(f"\n=== LOPO: holding out {held_out_platform} ===")

        test_graphs = platform_groups[held_out_platform]
        train_graphs = []
        for plat in platforms:
            if plat != held_out_platform:
                train_graphs.extend(platform_groups[plat])

        logger.info(f"  Train: {len(train_graphs)} graphs, Test: {len(test_graphs)} graphs")

        platform_result = {
            "held_out_platform": held_out_platform,
            "n_train": len(train_graphs),
            "n_test": len(test_graphs),
            "seed_results": [],
        }

        for seed in seeds:
            set_seed(seed)
            model = build_model(
                config.get("model_family", "domain_generalized"),
                config.get("model", {}),
                train_graphs[0],
            )

            train_loader = DataLoader(train_graphs, batch_size=config.get("runtime", {}).get("batch_size", 4), shuffle=True)
            test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)

            # Simple train on all train data (no CV, just one split)
            fold_result = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                config=config,
                device=device,
                fold=0,
                seed=seed,
            )

            # Evaluate on held-out platform
            ood_metrics = evaluate_on_graphs(model, test_graphs, device)
            ood_metrics["seed"] = seed
            platform_result["seed_results"].append(ood_metrics)

        # Aggregate across seeds
        aurocs = [r["macro_auroc"] for r in platform_result["seed_results"]
                  if not np.isnan(r["macro_auroc"])]
        platform_result["aggregated"] = {
            "macro_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "macro_auroc_std": float(np.std(aurocs)) if aurocs else None,
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in platform_result["seed_results"]])),
            "balanced_acc_mean": float(np.mean([r["balanced_accuracy"] for r in platform_result["seed_results"]])),
        }

        lopo_results.append(platform_result)

    # Overall LOPO summary
    all_aurocs = [r["aggregated"]["macro_auroc_mean"]
                  for r in lopo_results if r["aggregated"]["macro_auroc_mean"] is not None]
    return {
        "protocol": "lopo",
        "n_platforms": len(platforms),
        "platforms": platforms,
        "overall_macro_auroc_mean": float(np.mean(all_aurocs)) if all_aurocs else None,
        "per_platform_results": lopo_results,
    }


def main():
    parser = argparse.ArgumentParser(description="OOD evaluation for TIME typing")
    parser.add_argument("--protocol", required=True, choices=["cross_section", "lopo"],
                        help="Evaluation protocol")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])

    # Cross-section specific
    parser.add_argument("--train-graphs", default=None, help="Training graphs dir (cross_section)")
    parser.add_argument("--test-graphs", default=None, help="Test graphs dir (cross_section)")
    parser.add_argument("--checkpoints", default=None, help="Checkpoints dir (cross_section)")

    # LOPO specific
    parser.add_argument("--graphs-dir", default=None, help="All graphs dir (lopo)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.protocol == "cross_section":
        if not all([args.train_graphs, args.test_graphs, args.checkpoints]):
            parser.error("cross_section requires --train-graphs, --test-graphs, --checkpoints")
        results = cross_section_eval(
            Path(args.train_graphs), Path(args.test_graphs),
            Path(args.checkpoints), config, device,
        )
    elif args.protocol == "lopo":
        if not args.graphs_dir:
            parser.error("lopo requires --graphs-dir")
        results = leave_one_platform_out_eval(
            Path(args.graphs_dir), config, device, seeds=args.seeds,
        )

    # Output
    output_path = args.output or f"ood_eval_{args.protocol}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    if args.protocol == "cross_section":
        em = results.get("ensemble_metrics", {})
        print(f"\nCross-section transfer ensemble: "
              f"AUROC={em.get('macro_auroc', 'N/A'):.3f}, "
              f"F1={em.get('macro_f1', 'N/A'):.3f}, "
              f"BalAcc={em.get('balanced_accuracy', 'N/A'):.3f}")
    elif args.protocol == "lopo":
        print(f"\nLOPO overall: AUROC={results.get('overall_macro_auroc_mean', 'N/A')}")
        for pr in results.get("per_platform_results", []):
            agg = pr["aggregated"]
            print(f"  {pr['held_out_platform']}: "
                  f"AUROC={agg['macro_auroc_mean']}, n={pr['n_test']}")


if __name__ == "__main__":
    main()
