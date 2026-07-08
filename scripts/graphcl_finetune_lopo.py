#!/usr/bin/env python3
"""Does GraphCL self-supervised pretraining also improve cross-patient LOPO generalization?
Fine-tunes the pretrained HGT on HEST-1k Visium binary Hot/Cold under leave-one-patient-out and
compares to training from scratch. Uses only the graph (cell.x) stream.
"""
from __future__ import annotations
import glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

GRAPHS = CORE / "outputs/hetero_graph/hest1k_visium__fusion3/graphs"
PRETRAIN = CORE / "outputs/pretrain/graphcl/graphcl_hgt_noVisium.pt"
KEEP = {0, 2}; SEEDS = [42, 123, 2026]
GMODEL = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=2,
              pheno_dim=4, use_pheno_head=True, use_cell_state_head=False, cell_state_dim=4,
              use_ranking_heads=True)
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
        gs.append(g)
    return gs


@torch.no_grad()
def probs(model, graphs, device):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=8):
        out.append(torch.softmax(model(b.to(device))["graph_logits"], -1).cpu().numpy())
    return np.concatenate(out, 0)[:, 1]


def run(use_pretrain, graphs, groups, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = CORE / "outputs/results" / ("EXP-GCLPT-LOPO" if use_pretrain else "EXP-SCRATCH-LOPO") / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    pre = torch.load(PRETRAIN, map_location="cpu", weights_only=False) if use_pretrain else None
    N = len(graphs); Pp = np.zeros(N)
    for held in sorted(set(groups.tolist())):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        acc = np.zeros(len(te))
        for seed in SEEDS:
            P.set_seed(seed)
            m = build_model("hgt_time", GMODEL, graphs[0])
            if pre is not None:
                msd = m.state_dict()
                pre_f = {k: v for k, v in pre.items() if k in msd and msd[k].shape == v.shape}
                m.load_state_dict(pre_f, strict=False)   # transfer encoder/readout; skip 3-class classifier
            cfg = {"model_family": "hgt_time", "runtime": RUNTIME, "loss": LOSS, "model": GMODEL}
            train_one_fold(model=m, train_loader=DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True),
                           val_loader=DataLoader([graphs[i] for i in te], batch_size=4, shuffle=False),
                           config=cfg, device=device, checkpoint_dir=ck, fold=0, seed=seed)
            acc += probs(m, [graphs[i] for i in te], device)
        Pp[te] = acc / len(SEEDS)
    yp = (Pp >= 0.5).astype(int)
    return roc_auc_score(y, Pp), f1_score(y, yp, average="macro"), balanced_accuracy_score(y, yp)


def main():
    graphs = load(); groups = np.array([g.sample for g in graphs]); y = np.array([int(g.y_graph[0]) for g in graphs])
    print("samples:", dict(Counter(groups.tolist())), "n=", len(graphs))
    print("from scratch         AUROC 0.811  F1 0.750  BAcc 0.750  (from prior run)")
    a, f, b = run(True, graphs, groups, y)
    print(f"{'GraphCL-pretrained':20s} AUROC {a:.3f}  F1 {f:.3f}  BAcc {b:.3f}")


if __name__ == "__main__":
    main()
