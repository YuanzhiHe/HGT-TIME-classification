#!/usr/bin/env python3
"""GraphCL-style self-supervised pretraining of the HGT encoder on unlabeled graphs.

Data size is the proven bottleneck for the 36-region benchmark. This pretrains the HGT
encoder/readout on a large pool of unlabeled same-schema graphs (Section 2 + HEST-1k Visium +
Xenium) with a graph-contrastive objective (two augmented views per graph, NT-Xent), then the
weights are transferred and fine-tuned on the 36 labeled regions. Saves the pretrained state dict.
"""
from __future__ import annotations
import copy, glob, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(CORE / "scripts"))
import train_eval_pipeline as P
from train_eval_pipeline import build_model

POOLS = ["visium_breast_section2_regions__hetero_v1", "hest1k_visium__hetero_v1", "hest1k_xenium__hetero_v1"]
GMODEL = dict(hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, num_classes=3,
              pheno_dim=4, use_pheno_head=False, use_ranking_heads=False)
OUT = CORE / "outputs/pretrain/graphcl/graphcl_hgt.pt"


def load_pool():
    gs = []
    for d in POOLS:
        for p in sorted(glob.glob(str(CORE / f"outputs/hetero_graph/{d}/graphs/*.pt"))):
            gs.append(torch.load(p, weights_only=False))
    return gs


def augment(g):
    """Two-view augmentation: feature masking + spatial-edge dropout (returns a cloned view)."""
    v = copy.copy(g)
    # feature masking on cell nodes
    x = g["cell"].x.clone()
    m = (torch.rand(x.size(1)) > 0.25).float()
    v["cell"].x = x * m
    # spatial edge dropout
    et = ("cell", "spatial", "cell")
    for e in g.edge_types:
        if "spatial" in str(e).lower() or e == et:
            ei = g[e].edge_index
            keep = torch.rand(ei.size(1)) > 0.2
            v[e].edge_index = ei[:, keep]
    return v


def nt_xent(z1, z2, tau=0.2):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


class SSLWrap(nn.Module):
    def __init__(self, base, hidden=128):
        super().__init__()
        self.base = base
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def embed(self, data):
        x_dict = self.base.encode(data)
        num = int(data["cell"].batch.max().item()) + 1 if hasattr(data["cell"], "batch") else 1
        g, _ = self.base.readout(x_dict=x_dict, data=data, num_graphs=num)
        return self.proj(g)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P.set_seed(42)
    graphs = load_pool()
    print(f"SSL pool: {len(graphs)} unlabeled graphs")
    base = build_model("hgt_time", GMODEL, graphs[0])
    model = SSLWrap(base, hidden=GMODEL["hidden_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)

    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    for epoch in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(graphs)); tot = 0.0; nb = 0
        for i in range(0, len(graphs), 32):
            batch_g = [graphs[j] for j in idx[i:i + 32]]
            v1 = DataLoader([augment(g) for g in batch_g], batch_size=len(batch_g)).__iter__().__next__().to(device)
            v2 = DataLoader([augment(g) for g in batch_g], batch_size=len(batch_g)).__iter__().__next__().to(device)
            z1, z2 = model.embed(v1), model.embed(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if epoch % 10 == 0 or epoch == EPOCHS-1:
            print(f"epoch {epoch}: contrastive loss {tot/max(nb,1):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.base.state_dict(), OUT)
    print(f"saved pretrained HGT -> {OUT}")


if __name__ == "__main__":
    main()
