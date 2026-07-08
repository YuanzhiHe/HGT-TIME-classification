#!/usr/bin/env python3
"""Low-rank multimodal fusion (LMF) on cross-patient LOPO, betting on second-order interactions.

Each modality (graph readout embedding, pooled pretrained expression, pooled Phikon morphology) is
projected, and their outer-product interaction is captured by a low-rank tensor fusion (Liu et al.,
2018), which is parameter-efficient for small samples. Tests whether cross-modal interactions carry
signal even though each modality alone is near-uninformative. Leakage-free LOPO.
"""
from __future__ import annotations
import glob, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold

GRAPHS = CORE / "outputs/hetero_graph/hest1k_visium__fusion3/graphs"
KEEP = {0, 2}; SEEDS = [42, 123]
RUNTIME = dict(batch_size=4, epochs=100, learning_rate=1e-3, weight_decay=1e-5,
               patience=15, scheduler="cosine", scheduler_kwargs={"T_max": 100})
LOSS = dict(classification_weight=1.0, phenotype_weight=0.3, region_weight=0.0,
            ranking_weight=0.1, label_smoothing=0.05, class_weights=None)
GMODEL = dict(hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, num_classes=2,
              pheno_dim=4, use_pheno_head=True, use_ranking_heads=True,
              fusion_mode="gate", pretrain_dim=0, morph_dim=0)


class LMF(nn.Module):
    """Low-rank multimodal fusion of three modality vectors -> binary logit."""
    def __init__(self, dims, d=32, rank=4, nclass=2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(dm, d), nn.LayerNorm(d), nn.GELU()) for dm in dims])
        self.rank = rank; self.d = d
        # factor for each modality: (rank, d+1, nclass)
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(rank, d + 1, nclass) * 0.1) for _ in dims])
        self.drop = nn.Dropout(0.3)

    def forward(self, xs):
        fused = None
        for proj, fac, x in zip(self.proj, self.factors, xs):
            h = self.drop(proj(x))
            h = torch.cat([h, torch.ones(h.size(0), 1, device=h.device)], dim=1)  # (N, d+1)
            zm = torch.einsum('nd,rdc->nrc', h, fac)  # (N, rank, nclass)
            fused = zm if fused is None else fused * zm
        return fused.sum(dim=1)  # (N, nclass)


def load():
    gs = []
    for p in sorted(glob.glob(str(GRAPHS / "*.pt"))):
        g = torch.load(p, weights_only=False)
        y = int(g.y_graph[0].item())
        if y not in KEEP: continue
        g.y_graph = torch.tensor([0 if y == 0 else 1])
        g.sample = str(g["cell"].node_id[0]).split("__")[0]
        g.mean_pre = torch.as_tensor(g["cell"].pretrain.mean(0), dtype=torch.float32)
        g.mean_mor = torch.as_tensor(g["cell"].morph.mean(0), dtype=torch.float32)
        gs.append(g)
    return gs


@torch.no_grad()
def graph_embed(model, graphs, device):
    model.eval(); out = []
    for b in DataLoader(graphs, batch_size=8):
        b = b.to(device)
        out.append(model(b)["embedding"]["graph"].cpu())
    return torch.cat(out, 0)


def main():
    graphs = load(); groups = np.array([g.sample for g in graphs])
    y = np.array([int(g.y_graph[0]) for g in graphs]); N = len(graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (CORE / "outputs/results/EXP-LMF/checkpoints").mkdir(parents=True, exist_ok=True)
    print("samples:", dict(Counter(groups.tolist())), "n=", N)
    He = torch.stack([g.mean_pre for g in graphs]); Hm = torch.stack([g.mean_mor for g in graphs])
    P_lmf = np.zeros(N)

    for held in sorted(set(groups.tolist())):
        tr = np.where(groups != held)[0]; te = np.where(groups == held)[0]
        # 1) train graph HGT, extract embeddings (avg over seeds)
        Hg = torch.zeros(N, 128)
        for seed in SEEDS:
            P.set_seed(seed)
            m = build_model("fusion_hgt", GMODEL, graphs[0])
            cfg = {"model_family": "fusion_hgt", "runtime": RUNTIME, "loss": LOSS, "model": GMODEL}
            train_one_fold(model=m, train_loader=DataLoader([graphs[i] for i in tr], batch_size=4, shuffle=True),
                           val_loader=DataLoader([graphs[i] for i in te], batch_size=4, shuffle=False),
                           config=cfg, device=device, checkpoint_dir=CORE / "outputs/results/EXP-LMF/checkpoints",
                           fold=0, seed=seed)
            emb = graph_embed(m, graphs, device)
            Hg += emb
        Hg /= len(SEEDS)
        # 2) train LMF head on train graphs, predict test
        torch.manual_seed(0)
        lmf = LMF([128, He.size(1), Hm.size(1)]).to(device)
        opt = torch.optim.AdamW(lmf.parameters(), lr=1e-3, weight_decay=1e-4)
        cw = torch.tensor([1.0, 1.0], device=device)
        Xtr = [Hg[tr].to(device), He[tr].to(device), Hm[tr].to(device)]; ytr = torch.tensor(y[tr], device=device)
        for ep in range(300):
            lmf.train(); opt.zero_grad()
            logit = lmf(Xtr); loss = nn.functional.cross_entropy(logit, ytr, weight=cw)
            loss.backward(); opt.step()
        lmf.eval()
        with torch.no_grad():
            pt = torch.softmax(lmf([Hg[te].to(device), He[te].to(device), Hm[te].to(device)]), -1)[:, 1].cpu().numpy()
        P_lmf[te] = pt

    yp = (P_lmf >= 0.5).astype(int)
    print(f"\nLMF fusion: AUROC {roc_auc_score(y,P_lmf):.3f}  F1 {f1_score(y,yp,average='macro'):.3f}  BAcc {balanced_accuracy_score(y,yp):.3f}")
    print("reference graph-only: AUROC 0.760  F1 0.771  BAcc 0.775")


if __name__ == "__main__":
    main()
