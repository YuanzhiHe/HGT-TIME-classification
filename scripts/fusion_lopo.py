#!/usr/bin/env python3
"""Cross-patient (leave-one-sample-out) modality ablation on HEST Visium fusion3 graphs.

Same graphs, same split; only the fused streams change, isolating each modality's contribution:
  graph-only | +expression | +morphology | +expression+morphology
Binary Hot vs Cold. Reports AUROC / macro-F1 / balanced accuracy (mean over folds x seeds).
"""
from __future__ import annotations
import glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

GRAPHS = CORE / "outputs/hetero_graph/hest1k_visium__fusion3/graphs"
KEEP = {0, 2}
SEEDS = [42, 123, 2026]

BASE_MODEL = dict(hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, num_classes=2,
                  pheno_dim=4, use_pheno_head=True, use_ranking_heads=True, fusion_mode="gate")
BASE_LOSS = dict(classification_weight=1.0, phenotype_weight=0.3, region_weight=0.0,
                 ranking_weight=0.1, label_smoothing=0.05, class_weights=None)
RUNTIME = dict(batch_size=4, epochs=100, learning_rate=1e-3, weight_decay=1e-5,
               patience=15, scheduler="cosine", scheduler_kwargs={"T_max": 100})

VARIANTS = {
    "graph-only": dict(pretrain_dim=0, morph_dim=0),
    "graph+expr": dict(pretrain_dim=256, morph_dim=0),
    "graph+morph": dict(pretrain_dim=0, morph_dim=768),
    "graph+expr+morph": dict(pretrain_dim=256, morph_dim=768),
}


def load_graphs():
    gs = []
    for p in sorted(glob.glob(str(GRAPHS / "*.pt"))):
        g = torch.load(p, weights_only=False)
        y = int(g.y_graph[0].item()) if hasattr(g, "y_graph") else int(g["_global_store"]["y_graph"].item())
        if y not in KEEP:
            continue
        g.y_graph = torch.tensor([0 if y == 0 else 1])
        g.sample = str(g["cell"].node_id[0]).split("__")[0]
        gs.append(g)
    return gs


def run_variant(name, streams, graphs, groups):
    cfg = {"model_family": "fusion_hgt", "runtime": RUNTIME, "loss": BASE_LOSS,
           "model": {**BASE_MODEL, **streams}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uniq = sorted(set(groups))
    per = []
    ckdir = CORE / "outputs/results" / f"EXP-FUSLOPO-{name}" / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        P.set_seed(seed)
        for fold, held in enumerate(uniq):
            tr = [g for g, s in zip(graphs, groups) if s != held]
            va = [g for g, s in zip(graphs, groups) if s == held]
            if len(set(int(g.y_graph[0]) for g in va)) < 2:
                continue
            model = build_model("fusion_hgt", cfg["model"], graphs[0])
            r = train_one_fold(model=model, train_loader=DataLoader(tr, batch_size=4, shuffle=True),
                               val_loader=DataLoader(va, batch_size=4, shuffle=False),
                               config=cfg, device=device, checkpoint_dir=ckdir, fold=fold, seed=seed)
            per.append(r["final_metrics"])
    def agg(k):
        return float(np.mean([m[k] for m in per])), float(np.std([m[k] for m in per]))
    return {k: agg(k) for k in ["macro_auroc", "macro_f1", "balanced_accuracy"]}


def main():
    graphs = load_graphs()
    groups = [g.sample for g in graphs]
    print("samples:", dict(Counter(groups)), "| n=", len(graphs))
    print(f"{'variant':20s} AUROC        F1           BAcc")
    for name, streams in VARIANTS.items():
        r = run_variant(name, streams, graphs, groups)
        print(f"{name:20s} {r['macro_auroc'][0]:.3f}±{r['macro_auroc'][1]:.2f}  "
              f"{r['macro_f1'][0]:.3f}±{r['macro_f1'][1]:.2f}  "
              f"{r['balanced_accuracy'][0]:.3f}±{r['balanced_accuracy'][1]:.2f}")


if __name__ == "__main__":
    main()
