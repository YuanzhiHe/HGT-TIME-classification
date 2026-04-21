#!/usr/bin/env python3
"""Cross-platform evaluation: train on Visium, evaluate on Xenium.

Trains HGT-TIME on Visium HEST-1k graphs (or any source platform),
then evaluates the frozen model on Xenium HEST-1k graphs (or any target platform).
Both platforms must share the same cell.x feature dimension (50-dim by default).

Usage:
    python cross_platform_eval.py \
        --train-graphs-dir outputs/hetero_graph/hest1k_visium__hetero_v1/graphs \
        --test-graphs-dir outputs/hetero_graph/hest1k_xenium__hetero_v1/graphs \
        --config configs/hgt_time_lopo_hest1k.yaml \
        --output-dir outputs/results/EXP-CROSS-PLATFORM \
        --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_eval_pipeline import (
    build_model,
    compute_metrics,
    evaluate_model,
    set_seed,
    train_one_fold,
)

logger = logging.getLogger(__name__)


def load_graphs(graphs_dir: Path, keep_classes: list[int] | None = None) -> list:
    """Load .pt graph files, optionally filter and remap classes."""
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    if not graph_paths:
        raise SystemExit(f"No graph files found under {graphs_dir}")
    graphs = [torch.load(p, weights_only=False) for p in graph_paths]

    # Filter uncertain
    graphs = [
        g for g in graphs
        if getattr(g, "label_mask", None) is None
        or bool(torch.as_tensor(g.label_mask).view(-1)[0].item())
    ]

    if keep_classes is not None:
        keep_set = set(keep_classes)
        graphs = [g for g in graphs if int(g.y_graph[0].item()) in keep_set]
        remap = {old: new for new, old in enumerate(sorted(keep_set))}
        for g in graphs:
            old_label = int(g.y_graph[0].item())
            g.y_graph[0] = remap[old_label]

    return graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform evaluation")
    parser.add_argument("--train-graphs-dir", type=Path, required=True)
    parser.add_argument("--test-graphs-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import yaml

    with args.config.open() as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep_classes = config.get("split", {}).get("keep_classes", None)
    num_classes = config.get("model", {}).get("num_classes", 3)
    batch_size = config.get("runtime", {}).get("batch_size", 4)

    logger.info("Loading train graphs (source platform)...")
    train_graphs = load_graphs(args.train_graphs_dir, keep_classes)
    logger.info(f"  {len(train_graphs)} train graphs loaded")
    train_labels = Counter(int(g.y_graph[0].item()) for g in train_graphs)
    logger.info(f"  Train label distribution: {dict(train_labels)}")

    logger.info("Loading test graphs (target platform)...")
    test_graphs = load_graphs(args.test_graphs_dir, keep_classes)
    logger.info(f"  {len(test_graphs)} test graphs loaded")
    test_labels = Counter(int(g.y_graph[0].item()) for g in test_graphs)
    logger.info(f"  Test label distribution: {dict(test_labels)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in args.seeds:
        set_seed(seed)
        logger.info(f"--- Seed {seed} ---")

        # Train on all source platform graphs
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        # Use 20% of train for validation (early stopping)
        n_val = max(int(len(train_graphs) * 0.2), 1)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_graphs))
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]
        inner_train = [train_graphs[i] for i in train_indices]
        inner_val = [train_graphs[i] for i in val_indices]

        train_loader = DataLoader(inner_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(inner_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        model = build_model(config.get("model_family", "hgt_time"), config.get("model", {}), train_graphs[0])
        ckpt_dir = args.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        fold_result = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=ckpt_dir,
            fold=0,
            seed=seed,
        )
        logger.info(
            f"  Train complete: val_AUROC={fold_result['best_val_auroc']:.4f}, "
            f"epochs={fold_result['epochs_trained']}"
        )

        # Evaluate on target platform (cross-platform transfer)
        test_metrics, test_loss = evaluate_model(model, test_loader, config, device)
        logger.info(
            f"  Cross-platform test: AUROC={test_metrics.get('macro_auroc', 0):.4f} "
            f"F1={test_metrics.get('macro_f1', 0):.4f} "
            f"BAcc={test_metrics.get('balanced_accuracy', 0):.4f}"
        )

        all_results.append({
            "seed": seed,
            "train_val_auroc": fold_result["best_val_auroc"],
            "train_epochs": fold_result["epochs_trained"],
            "test_metrics": test_metrics,
            "test_loss": test_loss,
        })

    # Aggregate
    test_aurocs = [r["test_metrics"]["macro_auroc"] for r in all_results]
    test_f1s = [r["test_metrics"]["macro_f1"] for r in all_results]
    test_baccs = [r["test_metrics"]["balanced_accuracy"] for r in all_results]

    aggregated = {
        "macro_auroc_mean": float(np.mean(test_aurocs)),
        "macro_auroc_std": float(np.std(test_aurocs)),
        "macro_f1_mean": float(np.mean(test_f1s)),
        "macro_f1_std": float(np.std(test_f1s)),
        "balanced_accuracy_mean": float(np.mean(test_baccs)),
        "balanced_accuracy_std": float(np.std(test_baccs)),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Cross-Platform Transfer Results ({len(args.seeds)} seeds)")
    logger.info(f"  AUROC = {aggregated['macro_auroc_mean']:.4f} ± {aggregated['macro_auroc_std']:.4f}")
    logger.info(f"  F1    = {aggregated['macro_f1_mean']:.4f} ± {aggregated['macro_f1_std']:.4f}")
    logger.info(f"  BAcc  = {aggregated['balanced_accuracy_mean']:.4f} ± {aggregated['balanced_accuracy_std']:.4f}")

    results_path = args.output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump({
            "experiment": "cross_platform_transfer",
            "train_platform": str(args.train_graphs_dir),
            "test_platform": str(args.test_graphs_dir),
            "n_train_graphs": len(train_graphs),
            "n_test_graphs": len(test_graphs),
            "aggregated": aggregated,
            "per_seed": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
