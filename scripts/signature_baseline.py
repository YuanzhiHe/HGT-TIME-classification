#!/usr/bin/env python3
"""Biological-signature baseline for the three-class TIME benchmark (Section 1).

Answers reviewer R3.10: "a pathway/signature-based classifier using the same
biological priors". Rather than learning region representations from the
heterogeneous graph, this baseline collapses each region to a small, hand-built
immune-signature feature vector derived ONLY from the curated biological priors
already encoded in the graph (immune-target gene mask, gene mean-expression,
immune-signaling pathway flag, pathway member-expression). A multinomial logistic
regression then classifies Hot / Excluded / Cold.

Protocol matches train_eval_pipeline.py / reviewer_calibration.py exactly:
  - Deterministic StratifiedGroupKFold(n_splits=5), one group per region
    (groups = graph_id), so the fold split is identical across seeds.
  - Seeds [42, 123, 2026] repeat the classifier fit; the split is unchanged, so
    the 15 fold-seed combinations are aggregated as mean +/- std.
  - Metrics follow compute_metrics(): argmax for macro-F1 / balanced accuracy,
    roc_auc_score(multi_class="ovr", average="macro") for macro-AUROC.
  - StandardScaler is fit on the training fold only (leakage-free).

Feature vector (7 features, all from biological priors):
  1. immune_gene_mean_expr        mean log1p expr over immune-target genes
                                  (gene nodes with target_pos_mask=True)          [prior (a)]
  2. all_gene_mean_expr           mean log1p expr over all gene nodes             [prior (b)]
  3. all_gene_max_expr            max log1p expr over all gene nodes              [prior (b)]
  4. immune_frac_above_region_med fraction of immune-target genes whose expr
                                  exceeds the region's median gene expr          [prior (c)]
  5. immune_enrichment            immune_gene_mean_expr - all_gene_mean_expr
                                  (immune contrast vs. background)               [prior (a)-(b)]
  6. immune_pathway_expr_mean     mean pathway member-expression (pathway.x col 3,
                                  sample_member_expr_mean) over immune-signaling
                                  pathways (pathway target_pos_mask=True)         [prior (d)]
  7. immune_pathway_fraction      fraction of pathway nodes flagged immune-
                                  signaling (pathway.x col 5, immune_pathway_flag) [prior (d)]

Gene feature column 0 == sample_mean_log1p_expr; pathway feature column 3 ==
sample_member_expr_mean, column 5 == immune_pathway_flag (verified equal to
pathway target_pos_mask). Node ordering differs between graphs, so every feature
is computed via the per-graph boolean masks, never fixed indices.

Usage:
    python3 signature_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

THIS = Path(__file__).resolve()
CORE = THIS.parents[1]  # Experiment/core_code
PROJECT_ROOT = THIS.parents[3]

GRAPHS_DIR = (
    PROJECT_ROOT
    / "outputs/hetero_graph"
    / "visium_breast_regions__hetero_v1/graphs"
)
OUT_DIR = PROJECT_ROOT / "outputs/results/EXP-B09-SIGNATURE"

EXPERIMENT_ID = "EXP-B09-SIGNATURE"
NUM_CLASSES = 3
N_FOLDS = 5
SEEDS = [42, 123, 2026]

GENE_EXPR_COL = 0        # sample_mean_log1p_expr
PATHWAY_MEMBER_EXPR_COL = 3  # sample_member_expr_mean
PATHWAY_IMMUNE_FLAG_COL = 5  # immune_pathway_flag

FEATURE_NAMES = [
    "immune_gene_mean_expr",
    "all_gene_mean_expr",
    "all_gene_max_expr",
    "immune_frac_above_region_med",
    "immune_enrichment",
    "immune_pathway_expr_mean",
    "immune_pathway_fraction",
]


def build_features(graph) -> np.ndarray:
    """Collapse one region HeteroData into the 7-D immune-signature vector."""
    gene_expr = graph["gene"].x[:, GENE_EXPR_COL].numpy().astype(np.float64)
    gene_immune = graph["gene"].target_pos_mask.numpy().astype(bool)

    immune_expr = gene_expr[gene_immune]
    region_median = float(np.median(gene_expr))

    immune_gene_mean = float(immune_expr.mean()) if immune_expr.size else 0.0
    all_gene_mean = float(gene_expr.mean())
    all_gene_max = float(gene_expr.max())
    immune_frac_above = (
        float(np.mean(immune_expr > region_median)) if immune_expr.size else 0.0
    )
    immune_enrichment = immune_gene_mean - all_gene_mean

    pathway_x = graph["pathway"].x.numpy().astype(np.float64)
    pathway_immune = graph["pathway"].target_pos_mask.numpy().astype(bool)
    if pathway_immune.any():
        immune_pathway_expr_mean = float(
            pathway_x[pathway_immune, PATHWAY_MEMBER_EXPR_COL].mean()
        )
    else:
        immune_pathway_expr_mean = 0.0
    immune_pathway_fraction = (
        float(pathway_x[:, PATHWAY_IMMUNE_FLAG_COL].mean())
        if pathway_x.shape[0] > 0
        else 0.0
    )

    return np.array(
        [
            immune_gene_mean,
            all_gene_mean,
            all_gene_max,
            immune_frac_above,
            immune_enrichment,
            immune_pathway_expr_mean,
            immune_pathway_fraction,
        ],
        dtype=np.float64,
    )


def load_dataset():
    graph_paths = sorted(GRAPHS_DIR.glob("*.pt"))
    if not graph_paths:
        raise SystemExit(f"No graph files found under {GRAPHS_DIR}")

    feats, labels, groups = [], [], []
    dropped = 0
    for path in graph_paths:
        g = torch.load(path, weights_only=False)
        # Match the pipeline: drop label_mask=false regions before splitting.
        lm = g._global_store.get("label_mask", None)
        if lm is not None and not bool(torch.as_tensor(lm).view(-1)[0].item()):
            dropped += 1
            continue
        feats.append(build_features(g))
        labels.append(int(g._global_store["y_graph"].view(-1)[0].item()))
        groups.append(str(g._global_store["graph_id"]))

    X = np.vstack(feats)
    y = np.array(labels, dtype=int)
    if dropped:
        print(f"[signature] dropped {dropped} regions with label_mask=false")
    print(f"[signature] loaded {X.shape[0]} regions, {X.shape[1]} features")
    return X, y, groups


def metrics_from(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Same conventions as train_eval_pipeline.compute_metrics()."""
    y_pred = np.argmax(y_prob, axis=1)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["macro_auroc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["macro_auroc"] = 0.5
    return out


def main() -> None:
    X, y, groups = load_dataset()

    # Deterministic fold split, computed once (matches reviewer_calibration.py).
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)
    fold_splits = list(sgkf.split(X, y, groups))

    per_fold = []
    for seed in SEEDS:
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_va_s = scaler.transform(X_va)

            # Newer sklearn removed the `multi_class` kwarg; with the lbfgs
            # solver, multiclass LogisticRegression is natively multinomial.
            clf = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                random_state=seed,
            )
            clf.fit(X_tr_s, y_tr)

            # Map predict_proba columns to the full 0..NUM_CLASSES-1 label space.
            prob = np.zeros((X_va_s.shape[0], NUM_CLASSES), dtype=np.float64)
            prob[:, clf.classes_] = clf.predict_proba(X_va_s)

            m = metrics_from(y_va, prob)
            m["fold"] = fold
            m["seed"] = seed
            per_fold.append(m)
            print(
                f"[signature] seed={seed} fold={fold} "
                f"AUROC={m['macro_auroc']:.4f} F1={m['macro_f1']:.4f} "
                f"BAcc={m['balanced_accuracy']:.4f}"
            )

    def mean_std(key):
        vals = [r[key] for r in per_fold]
        return float(np.mean(vals)), float(np.std(vals))

    auroc_mean, auroc_std = mean_std("macro_auroc")
    f1_mean, f1_std = mean_std("macro_f1")
    bacc_mean, bacc_std = mean_std("balanced_accuracy")

    aggregated = {
        "macro_auroc_mean": auroc_mean,
        "macro_auroc_std": auroc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "balanced_accuracy_mean": bacc_mean,
        "balanced_accuracy_std": bacc_std,
        "experiment_id": EXPERIMENT_ID,
        "model_family": "signature_logreg",
        "seeds": SEEDS,
        "n_folds": N_FOLDS,
        "n_fold_seed_combos": len(per_fold),
        "feature_names": FEATURE_NAMES,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": EXPERIMENT_ID,
                "aggregated": aggregated,
                "per_fold": per_fold,
                "feature_names": FEATURE_NAMES,
            },
            f,
            indent=2,
        )
    print(f"[signature] results saved to {results_path}")
    print(
        f"[signature] SUMMARY (mean+/-std over {len(per_fold)} fold-seed combos): "
        f"macro-AUROC {auroc_mean:.3f}+/-{auroc_std:.3f} | "
        f"macro-F1 {f1_mean:.3f}+/-{f1_std:.3f} | "
        f"balanced-acc {bacc_mean:.3f}+/-{bacc_std:.3f}"
    )


if __name__ == "__main__":
    main()
