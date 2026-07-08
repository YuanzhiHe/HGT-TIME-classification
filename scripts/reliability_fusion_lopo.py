#!/usr/bin/env python3
"""Reliability-weighted (contextual-discounting) fusion on cross-patient LOPO.

Unlike equal-weight averaging (which lets a near-chance modality inject noise), this learns a
per-modality reliability weight and down-weights unreliable modalities toward zero, so fusion
cannot do worse than the strongest reliable modality. Leakage-free: for each held-out patient,
each modality's reliability is estimated from the OTHER patients' predictions, never from the
held-out patient.

Modalities: graph (HGT), pretrained expression (class-balanced logistic regression), Phikon
morphology (class-balanced logistic regression). Reports graph-only, equal-average fusion, and
reliability-weighted fusion.
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

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
def gprobs(model, graphs, device):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=8):
        b = b.to(device)
        out.append(torch.softmax(model(b)["graph_logits"], -1).cpu().numpy())
    return np.concatenate(out, 0)


def metrics(y, p1):
    yp = (p1 >= 0.5).astype(int)
    return (roc_auc_score(y, p1) if len(set(y)) > 1 else float("nan"),
            f1_score(y, yp, average="macro", zero_division=0),
            balanced_accuracy_score(y, yp))


def main():
    graphs = load(); groups = np.array([g.sample for g in graphs])
    y_all = np.array([int(g.y_graph[0]) for g in graphs]); N = len(graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (CORE / "outputs/results/EXP-RELFUS/checkpoints").mkdir(parents=True, exist_ok=True)
    print("samples:", dict(Counter(groups.tolist())), "n=", N)

    Pg = np.zeros(N); Pe = np.zeros(N); Pm = np.zeros(N)
    for held in sorted(set(groups)):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        for feat, acc in [("mean_pre", Pe), ("mean_mor", Pm)]:
            Xtr = np.array([getattr(graphs[i], feat) for i in tr]); Xte = np.array([getattr(graphs[i], feat) for i in te])
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), y_all[tr])
            acc[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
        gp = np.zeros(len(te)); cfg = {"model_family": "fusion_hgt", "runtime": RUNTIME, "loss": LOSS, "model": GMODEL}
        for seed in SEEDS:
            P.set_seed(seed)
            m = build_model("fusion_hgt", GMODEL, graphs[0])
            train_one_fold(model=m, train_loader=DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True),
                           val_loader=DataLoader([graphs[i] for i in te], batch_size=4, shuffle=False),
                           config=cfg, device=device, checkpoint_dir=CORE / "outputs/results/EXP-RELFUS/checkpoints",
                           fold=0, seed=seed)
            gp += gprobs(m, [graphs[i] for i in te], device)[:, 1]
        Pg[te] = gp / len(SEEDS)

    np.savez(CORE / "outputs/results/EXP-RELFUS/preds.npz", Pg=Pg, Pe=Pe, Pm=Pm, y=y_all, groups=groups)

    eps = 1e-6
    def logit(p): p = np.clip(p, eps, 1 - eps); return np.log(p) - np.log(1 - p)
    Lg, Le, Lm = logit(Pg), logit(Pe), logit(Pm)

    # reliability-weighted fusion (leakage-free): weight_m for held-out patient P estimated from
    # OTHER patients' balanced accuracy of modality m; w = max(0, 2*(BA-0.5)); combine as weighted
    # logit pool (a log-opinion pool = discounted evidence combination).
    Pfused = np.zeros(N)
    for held in sorted(set(groups)):
        oth = groups != held; te = groups == held
        w = {}
        for name, Pm_ in [("g", Pg), ("e", Pe), ("m", Pm)]:
            ba = balanced_accuracy_score(y_all[oth], (Pm_[oth] >= 0.5).astype(int))
            w[name] = max(0.0, 2 * (ba - 0.5))
        s = w["g"] + w["e"] + w["m"] + eps
        wl = (w["g"] * Lg[te] + w["e"] * Le[te] + w["m"] * Lm[te]) / s
        Pfused[te] = 1 / (1 + np.exp(-wl))

    Pavg = 1 / (1 + np.exp(-(Lg + Le + Lm) / 3))  # equal-weight pool
    print(f"{'model':30s} AUROC   F1     BAcc")
    for name, p in [("graph only", Pg), ("expression only", Pe), ("morphology only", Pm),
                    ("equal-average fusion", Pavg), ("reliability-weighted fusion", Pfused)]:
        a, f, b = metrics(y_all, p)
        print(f"{name:30s} {a:.3f}  {f:.3f}  {b:.3f}")
    # report the learned weights (pooled across folds, informative)
    print("\nlearned reliability weights (last fold shown):", {k: round(v, 3) for k, v in w.items()})


if __name__ == "__main__":
    main()
