#!/usr/bin/env python3
"""Post-hoc calibration analysis for the Information Fusion revision.

Reloads the per-fold best checkpoints saved by train_eval_pipeline, reproduces the
deterministic StratifiedGroupKFold splits (no shuffle -> identical folds), dumps
per-sample probabilities, and reports:

  (a) default argmax balanced metrics (should match published Tables), and
  (b) class-prior-corrected decision-rule balanced metrics.

Prior correction is leakage-free: posteriors are divided by the class prior estimated
from the TRAINING labels of each fold, then renormalised, then argmax. This removes the
majority-class-default bias without changing AUROC, directly answering R2.4/R4 ("is the
MLP's high-AUROC/low-balanced-accuracy a mere operating-point artifact?").

Also emits pooled confusion matrices per experiment (R3.7).

Usage:
  python reviewer_calibration.py --config <path> [--config ...] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

THIS = Path(__file__).resolve()
CORE = THIS.parents[1]  # Experiment/core_code
PROJECT_ROOT = THIS.parents[3]
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "scripts"))

from train_eval_pipeline import build_model, _get_labels  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def prepare_graphs(config: dict):
    graphs_dir = PROJECT_ROOT / config["input"]["graphs_dir"]
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    graphs = [torch.load(p, weights_only=False) for p in graph_paths]
    # filter label_mask=false
    kept = []
    for g in graphs:
        m = getattr(g, "label_mask", None)
        if m is None or bool(torch.as_tensor(m).view(-1)[0].item()):
            kept.append(g)
    graphs = kept
    keep_classes = config.get("split", {}).get("keep_classes")
    if keep_classes is not None:
        keep_set = set(keep_classes)
        graphs = [g for g in graphs if int(g.y_graph[0].item()) in keep_set]
        remap = {old: new for new, old in enumerate(sorted(keep_set))}
        for g in graphs:
            g.y_graph[0] = remap[int(g.y_graph[0].item())]
    return graphs


def get_split_arrays(graphs, split_unit):
    groups, labels = [], []
    for g in graphs:
        gv = getattr(g, split_unit, None)
        if gv is None:
            gv = getattr(g, "graph_id", "unknown_0")
        groups.append(str(gv))
        labels.append(int(g.y_graph[0].item()))
    return groups, labels


def prior_correct(probs: np.ndarray, train_prior: np.ndarray) -> np.ndarray:
    adj = probs / np.clip(train_prior, 1e-8, None)[None, :]
    adj = adj / adj.sum(axis=1, keepdims=True)
    return adj


def metrics_from(y_true, y_prob, num_classes):
    y_pred = np.argmax(y_prob, axis=1)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if num_classes == 2:
            out["macro_auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            out["macro_auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_auroc"] = float("nan")
    return out


def run_one(config_path: Path, device):
    config = load_config(config_path)
    exp_id = config["experiment_id"]
    model_family = config.get("model_family", "homogeneous_graph")
    num_classes = config.get("model", {}).get("num_classes", 3)
    split_cfg = config.get("split", {})
    n_folds = split_cfg.get("n_folds", 5)
    split_unit = split_cfg.get("unit", "patient_id")
    batch_size = config.get("runtime", {}).get("batch_size", 4)
    ckpt_dir = PROJECT_ROOT / config.get("output_root", "outputs/results") / exp_id / "checkpoints"

    graphs = prepare_graphs(config)
    groups, labels = get_split_arrays(graphs, split_unit)
    seeds = [42, 123, 2026]

    # Match the pipeline protocol: compute metrics PER FOLD, then average over all
    # fold-seed combinations. Confusion matrices are pooled across all folds/seeds.
    per_fold_default, per_fold_prior = [], []
    pool_y, pool_pdef, pool_pprior = [], [], []

    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    fold_splits = list(sgkf.split(graphs, labels, groups))

    for seed in seeds:
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            ckpt = ckpt_dir / f"best_fold{fold}_seed{seed}.pt"
            if not ckpt.exists():
                continue
            val_graphs = [graphs[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            cnt = Counter(train_labels)
            train_prior = np.array([cnt.get(c, 0) for c in range(num_classes)], dtype=float)
            train_prior = train_prior / train_prior.sum()

            model = build_model(model_family, config.get("model", {}), graphs[0])
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state)
            model.to(device).eval()

            loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
            ys, ps = [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    out = model(batch)
                    logits = out["graph_logits"]
                    prob = torch.softmax(logits, dim=-1).cpu().numpy()
                    y = _get_labels(batch, logits.size(0), device).cpu().numpy()
                    ys.extend(y.tolist())
                    ps.extend(prob)
            ys = np.array(ys)
            ps = np.array(ps)
            ps_prior = prior_correct(ps, train_prior)

            per_fold_default.append(metrics_from(ys, ps, num_classes))
            per_fold_prior.append(metrics_from(ys, ps_prior, num_classes))
            pool_y.extend(ys.tolist())
            pool_pdef.extend(ps)
            pool_pprior.extend(ps_prior)

    def agg(lst):
        keys = lst[0].keys()
        return {k: [float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst]))] for k in keys}

    yv = np.array(pool_y)
    cm_default = confusion_matrix(yv, np.argmax(np.array(pool_pdef), axis=1),
                                  labels=list(range(num_classes))).tolist()
    cm_prior = confusion_matrix(yv, np.argmax(np.array(pool_pprior), axis=1),
                                labels=list(range(num_classes))).tolist()
    per_seed_default, per_seed_prior = per_fold_default, per_fold_prior

    return {
        "experiment_id": exp_id,
        "model_family": model_family,
        "num_classes": num_classes,
        "default_argmax": agg(per_seed_default),
        "prior_corrected": agg(per_seed_prior),
        "confusion_default_pooled": cm_default,
        "confusion_prior_pooled": cm_prior,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for cfg in args.config:
        cfg = cfg if cfg.is_absolute() else PROJECT_ROOT / cfg
        print(f"[calib] {cfg}")
        try:
            results.append(run_one(cfg, device))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            results.append({"config": str(cfg), "error": str(e)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[calib] wrote {args.out}")
    for r in results:
        if "error" in r:
            continue
        d, p = r["default_argmax"], r["prior_corrected"]
        print(f"\n{r['experiment_id']} ({r['num_classes']}-class)")
        print(f"  default : AUROC {d['macro_auroc'][0]:.3f} F1 {d['macro_f1'][0]:.3f} BAcc {d['balanced_accuracy'][0]:.3f}")
        print(f"  prior   : AUROC {p['macro_auroc'][0]:.3f} F1 {p['macro_f1'][0]:.3f} BAcc {p['balanced_accuracy'][0]:.3f}")


if __name__ == "__main__":
    main()
