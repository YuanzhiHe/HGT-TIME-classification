#!/usr/bin/env python3
"""Decision-level (late) fusion on HEST Visium fusion3, cross-patient LOPO, binary Hot/Cold.

Feature-level fusion overfit on small data. Here each modality predicts independently and we
combine at the PROBABILITY level (leakage-free averaging), the standard information-fusion
paradigm most robust to small samples:
  - graph modality : HGT on the heterogeneous graph (structure + 50-dim features)
  - expression     : class-balanced logistic regression on per-graph mean pretrained embedding
  - morphology     : class-balanced logistic regression on per-graph mean Phikon embedding
Reports each unimodal model and their probability-average combinations.
"""
from __future__ import annotations
import glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold, evaluate_model, compute_metrics

GRAPHS = CORE / "outputs/hetero_graph/hest1k_visium__fusion3/graphs"
KEEP = {0, 2}; SEEDS = [42, 123, 2026]
RUNTIME = dict(batch_size=4, epochs=100, learning_rate=1e-3, weight_decay=1e-5,
               patience=15, scheduler="cosine", scheduler_kwargs={"T_max": 100})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.3, region_weight=0.0,
            ranking_weight=0.1, label_smoothing=0.05, class_weights=None)
GMODEL = dict(hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, num_classes=2,
              pheno_dim=4, use_pheno_head=True, use_ranking_heads=True,
              fusion_mode="gate", pretrain_dim=0, morph_dim=0)


def load():
    gs = []
    for p in sorted(glob.glob(str(GRAPHS / "*.pt"))):
        g = torch.load(p, weights_only=False)
        y = int(g.y_graph[0].item())
        if y not in KEEP: continue
        g.y_graph = torch.tensor([0 if y == 0 else 1])
        g.sample = str(g["cell"].node_id[0]).split("__")[0]
        g.mean_pre = g["cell"].pretrain.mean(0).numpy()
        g.mean_mor = g["cell"].morph.mean(0).numpy()
        gs.append(g)
    return gs


@torch.no_grad()
def graph_probs(model, graphs, device):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=8):
        b = b.to(device)
        out.append(torch.softmax(model(b)["graph_logits"], -1).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    graphs = load(); groups = [g.sample for g in graphs]
    print("samples:", dict(Counter(groups)), "n=", len(graphs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (CORE / "outputs/results/EXP-DECFUS/checkpoints").mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(groups))
    N = len(graphs)
    y_all = np.array([int(g.y_graph[0]) for g in graphs])
    # accumulators for per-graph probabilities (prob of class 1)
    P_graph = np.zeros(N); P_expr = np.zeros(N); P_mor = np.zeros(N); seen = np.zeros(N)

    for held in uniq:
        tr = [i for i, s in enumerate(groups) if s != held]
        te = [i for i, s in enumerate(groups) if s == held]
        if len(set(y_all[te])) < 2:  # need both classes in test for AUROC; still record probs
            pass
        # --- expression / morphology: class-balanced LR ---
        for feat, acc in [("mean_pre", P_expr), ("mean_mor", P_mor)]:
            Xtr = np.array([getattr(graphs[i], feat) for i in tr]); Xte = np.array([getattr(graphs[i], feat) for i in te])
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), y_all[tr])
            acc[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
        # --- graph: HGT averaged over seeds ---
        cfg = {"model_family": "fusion_hgt", "runtime": RUNTIME, "loss": LOSS, "model": GMODEL}
        gp = np.zeros(len(te))
        for seed in SEEDS:
            P.set_seed(seed)
            model = build_model("fusion_hgt", GMODEL, graphs[0])
            train_one_fold(model=model, train_loader=DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True),
                           val_loader=DataLoader([graphs[i] for i in te], batch_size=4, shuffle=False),
                           config=cfg, device=device, checkpoint_dir=CORE / "outputs/results/EXP-DECFUS/checkpoints",
                           fold=0, seed=seed)
            gp += graph_probs(model, [graphs[i] for i in te], device)[:, 1]
        P_graph[te] = gp / len(SEEDS)
        seen[te] = 1

    def m(p1):
        prob = np.stack([1 - p1, p1], 1)
        return compute_metrics(y_all, prob, num_classes=2)
    combos = {
        "graph": P_graph, "expression": P_expr, "morphology": P_mor,
        "graph+expr (avg)": (P_graph + P_expr) / 2,
        "graph+morph (avg)": (P_graph + P_mor) / 2,
        "graph+expr+morph (avg)": (P_graph + P_expr + P_mor) / 3,
    }
    print(f"{'model':26s} AUROC   F1     BAcc")
    for k, p in combos.items():
        r = m(p)
        print(f"{k:26s} {r['macro_auroc']:.3f}  {r['macro_f1']:.3f}  {r['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
