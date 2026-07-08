#!/usr/bin/env python3
"""Fine-tune the GraphCL-pretrained HGT on the 36 labeled Section-1 regions and compare to
training from scratch. Same 5-fold x 3-seed protocol as Table 1.
"""
from __future__ import annotations
import glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

GRAPHS = CORE / "outputs/hetero_graph/visium_breast_regions__hetero_v1/graphs"
PRETRAIN = CORE / "outputs/pretrain/graphcl/graphcl_hgt.pt"
GMODEL = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=3,
              pheno_dim=4, use_pheno_head=True, use_cell_state_head=False, cell_state_dim=4,
              use_ranking_heads=True)
RUNTIME = dict(batch_size=4, epochs=100, learning_rate=1e-3, weight_decay=1e-5, patience=15,
               scheduler="cosine", scheduler_kwargs={"T_max": 100})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.3, region_weight=0.0,
            ranking_weight=0.1, label_smoothing=0.05, class_weights=None)
SEEDS = [42, 123, 2026]


def run(use_pretrain):
    graphs = [torch.load(p, weights_only=False) for p in sorted(glob.glob(str(GRAPHS / "*.pt")))]
    graphs = [g for g in graphs if bool(torch.as_tensor(getattr(g, "label_mask")).view(-1)[0].item())]
    labels = [int(g.y_graph[0].item()) for g in graphs]
    groups = [g._global_store["graph_id"] for g in graphs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = CORE / "outputs/results" / ("EXP-GRAPHCL-FT" if use_pretrain else "EXP-GRAPHCL-SCRATCH") / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    pre = torch.load(PRETRAIN, map_location="cpu", weights_only=False) if use_pretrain else None
    per = []
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(graphs, labels, groups))
    for seed in SEEDS:
        P.set_seed(seed)
        for fold, (tr, va) in enumerate(splits):
            cfg = {"model_family": "hgt_time", "runtime": RUNTIME, "loss": LOSS, "model": GMODEL}
            m = build_model("hgt_time", GMODEL, graphs[0])
            if pre is not None:
                m.load_state_dict(pre, strict=False)   # transfer input_projector/encoder/readout
            r = train_one_fold(model=m, train_loader=DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True),
                               val_loader=DataLoader([graphs[i] for i in va], batch_size=4, shuffle=False),
                               config=cfg, device=device, checkpoint_dir=ck, fold=fold, seed=seed)
            per.append(r["final_metrics"])
    def agg(k): v = [m[k] for m in per]; return float(np.mean(v)), float(np.std(v))
    return {k: agg(k) for k in ["macro_auroc", "macro_f1", "balanced_accuracy"]}


def main():
    print("label dist:", dict(Counter([int(torch.load(p, weights_only=False).y_graph[0].item())
          for p in sorted(glob.glob(str(GRAPHS / "*.pt")))])))
    for name, up in [("from scratch", False), ("GraphCL-pretrained", True)]:
        r = run(up)
        print(f"{name:20s} AUROC {r['macro_auroc'][0]:.3f}+-{r['macro_auroc'][1]:.2f}  "
              f"F1 {r['macro_f1'][0]:.3f}  BAcc {r['balanced_accuracy'][0]:.3f}")


if __name__ == "__main__":
    main()
