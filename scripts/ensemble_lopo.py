#!/usr/bin/env python3
"""Cross-patient ensemble of distinct architectures + leakage-free threshold calibration.

Prior fusion attempts combined weak external modalities. Here we instead ensemble the strong models
that operate on the same graph data but have complementary error structures: the heterogeneous graph
(HGT, best balanced accuracy) and a non-graph MLP on pooled cell features (best AUROC in Table
lopo). Ensembling distinct architectures is an evidence-backed domain-generalization lever. We also
test a leakage-free balanced-accuracy-optimal decision threshold (tuned on the other patients).
"""
from __future__ import annotations
import glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

GRAPHS = CORE / "outputs/hetero_graph/hest1k_visium__fusion3/graphs"
KEEP = {0, 2}; SEEDS = [42, 123, 2026]
GMODEL = dict(hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, num_classes=2, pheno_dim=4,
              use_pheno_head=True, use_ranking_heads=True, fusion_mode="gate", pretrain_dim=0, morph_dim=0)
RUNTIME = dict(batch_size=4, epochs=100, learning_rate=1e-3, weight_decay=1e-5, patience=15,
               scheduler="cosine", scheduler_kwargs={"T_max": 100})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.3, region_weight=0.0,
            ranking_weight=0.1, label_smoothing=0.05, class_weights=None)


def load():
    gs = []
    for p in sorted(glob.glob(str(GRAPHS / "*.pt"))):
        g = torch.load(p, weights_only=False)
        y = int(g.y_graph[0].item())
        if y not in KEEP: continue
        g.y_graph = torch.tensor([0 if y == 0 else 1])
        g.sample = str(g["cell"].node_id[0]).split("__")[0]
        x = g["cell"].x
        g.pool = torch.cat([x.mean(0), x.std(0)]).numpy()   # 100-dim non-graph feature
        gs.append(g)
    return gs


@torch.no_grad()
def gprobs(model, graphs, device):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=8):
        out.append(torch.softmax(model(b.to(device))["graph_logits"], -1).cpu().numpy())
    return np.concatenate(out, 0)[:, 1]


def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    return roc_auc_score(y, p), f1_score(y, yp, average="macro"), balanced_accuracy_score(y, yp)


def best_thr(y, p):
    ts = np.linspace(0.1, 0.9, 81)
    return ts[np.argmax([balanced_accuracy_score(y, (p >= t).astype(int)) for t in ts])]


def main():
    graphs = load(); groups = np.array([g.sample for g in graphs]); y = np.array([int(g.y_graph[0]) for g in graphs])
    N = len(graphs); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (CORE / "outputs/results/EXP-ENS/checkpoints").mkdir(parents=True, exist_ok=True)
    print("samples:", dict(Counter(groups.tolist())), "n=", N)
    Pg = np.zeros(N); Pm = np.zeros(N); Xpool = np.stack([g.pool for g in graphs])

    for held in sorted(set(groups.tolist())):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        # non-graph MLP on pooled cell features
        sc = StandardScaler().fit(Xpool[tr])
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=800, alpha=1e-3, random_state=0)
        mlp.fit(sc.transform(Xpool[tr]), y[tr])
        Pm[te] = mlp.predict_proba(sc.transform(Xpool[te]))[:, 1]
        # HGT graph
        gp = np.zeros(len(te))
        for seed in SEEDS:
            P.set_seed(seed)
            mdl = build_model("fusion_hgt", GMODEL, graphs[0])
            cfg = {"model_family": "fusion_hgt", "runtime": RUNTIME, "loss": LOSS, "model": GMODEL}
            train_one_fold(model=mdl, train_loader=DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True),
                           val_loader=DataLoader([graphs[i] for i in te], batch_size=4, shuffle=False),
                           config=cfg, device=device, checkpoint_dir=CORE / "outputs/results/EXP-ENS/checkpoints",
                           fold=0, seed=seed)
            gp += gprobs(mdl, [graphs[i] for i in te], device)
        Pg[te] = gp / len(SEEDS)
    np.savez(CORE / "outputs/results/EXP-ENS/preds.npz", Pg=Pg, Pm=Pm, y=y, groups=groups)

    # leakage-free optimal threshold per model (tuned on other patients), applied to held-out
    def lofo_thr_metric(p):
        yp = np.zeros(N)
        for held in sorted(set(groups.tolist())):
            oth = groups != held; te = groups == held
            t = best_thr(y[oth], p[oth]); yp[te] = (p[te] >= t).astype(int)
        return roc_auc_score(y, p), f1_score(y, yp, average="macro"), balanced_accuracy_score(y, yp)

    Pens = (Pg + Pm) / 2
    print(f"\n{'model':32s} AUROC   F1     BAcc(0.5)  BAcc(opt-thr)")
    for name, p in [("non-graph MLP", Pm), ("HGT graph", Pg), ("ensemble MLP+HGT (avg)", Pens)]:
        a, f, b = metrics(y, p); _, _, bo = lofo_thr_metric(p)
        print(f"{name:32s} {a:.3f}  {f:.3f}  {b:.3f}      {bo:.3f}")


if __name__ == "__main__":
    main()
