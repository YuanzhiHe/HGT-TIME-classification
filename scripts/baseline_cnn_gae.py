#!/usr/bin/env python3
"""Additional baselines on the internal breast benchmark to answer "why a heterogeneous graph":
a CNN on rasterized region expression (non-graph spatial), and a spatial graph autoencoder in the
style of GraphST/STAGATE (unsupervised spatial-GNN embedding + classifier). Same 36 regions,
5-fold x 3-seed leakage-controlled CV as the main table.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch_geometric.nn import GCNConv, global_mean_pool

CORE = Path(__file__).resolve().parents[1]
GRAPHS = CORE / "outputs/hetero_graph/visium_breast_regions__hetero_v1/graphs"
SEEDS = [42, 123, 2026]; G = 16  # raster grid


def load():
    gs = []
    for p in sorted(glob.glob(str(GRAPHS / "*.pt"))):
        g = torch.load(p, weights_only=False)
        if not bool(torch.as_tensor(g.label_mask).view(-1)[0]): continue
        x = g["cell"].x.numpy(); pos = g["cell"].pos.numpy()
        gs.append((x, pos, int(g.y_graph[0])))
    return gs


def rasterize(x, pos, C):
    p = (pos - pos.min(0)) / (pos.ptp(0) + 1e-6)
    grid = np.zeros((C, G, G), np.float32); cnt = np.zeros((G, G), np.float32)
    ij = np.clip((p * (G - 1)).astype(int), 0, G - 1)
    for k, (i, j) in enumerate(ij):
        grid[:, i, j] += x[k, :C]; cnt[i, j] += 1
    grid /= (cnt[None] + 1e-6)
    return grid


class CNN(nn.Module):
    def __init__(self, c, n=3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, n)

    def forward(self, x): return self.fc(self.net(x).flatten(1))


class SpatialGAE(nn.Module):
    """GraphST/STAGATE-style spatial GNN autoencoder + linear classifier head on pooled embedding."""
    def __init__(self, d, h=64, n=3):
        super().__init__()
        self.enc1 = GCNConv(d, h); self.enc2 = GCNConv(h, h); self.dec = GCNConv(h, d)
        self.head = nn.Linear(h, n)

    def forward(self, x, ei, batch):
        z = F.relu(self.enc1(x, ei)); z = self.enc2(z, ei)
        xhat = self.dec(z, ei)
        pooled = global_mean_pool(z, batch)
        return self.head(pooled), xhat


def metrics(y, prob):
    yp = prob.argmax(1)
    try: auroc = roc_auc_score(y, prob, multi_class="ovr", average="macro")
    except Exception: auroc = float("nan")
    return auroc, f1_score(y, yp, average="macro", zero_division=0), balanced_accuracy_score(y, yp)


def run_cnn(gs, C=50):
    y = np.array([g[2] for g in gs]); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(np.stack([rasterize(x, p, C) for x, p, _ in gs]))
    per = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = CNN(C).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
            Xtr = X[tr].to(dev); ytr = torch.tensor(y[tr]).to(dev)
            for _ in range(120):
                m.train(); opt.zero_grad(); loss = F.cross_entropy(m(Xtr), ytr); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): prob = F.softmax(m(X[va].to(dev)), 1).cpu().numpy()
            per.append(metrics(y[va], prob))
    return np.nanmean(per, 0), np.nanstd(per, 0)


def run_gae(gs):
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    y = np.array([g[2] for g in gs]); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = []
    for x, pos, yi in gs:
        from sklearn.neighbors import NearestNeighbors
        k = min(6, len(x) - 1); nn_ = NearestNeighbors(n_neighbors=k + 1).fit(pos)
        _, idx = nn_.kneighbors(pos); src = np.repeat(np.arange(len(x)), k); dst = idx[:, 1:].reshape(-1)
        ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        data.append(Data(x=torch.tensor(x), edge_index=ei, y=torch.tensor([yi])))
    per = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(data, y):
            m = SpatialGAE(data[0].x.shape[1]).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
            trl = DataLoader([data[i] for i in tr], batch_size=8, shuffle=True)
            for _ in range(120):
                m.train()
                for b in trl:
                    b = b.to(dev); opt.zero_grad(); logit, xhat = m(b.x, b.edge_index, b.batch)
                    loss = F.cross_entropy(logit, b.y) + 0.5 * F.mse_loss(xhat, b.x)
                    loss.backward(); opt.step()
            m.eval(); probs = []
            with torch.no_grad():
                for b in DataLoader([data[i] for i in va], batch_size=8):
                    b = b.to(dev); probs.append(F.softmax(m(b.x, b.edge_index, b.batch)[0], 1).cpu().numpy())
            per.append(metrics(y[va], np.concatenate(probs)))
    return np.nanmean(per, 0), np.nanstd(per, 0)


if __name__ == "__main__":
    gs = load(); print(f"loaded {len(gs)} breast regions")
    for name, fn in [("CNN (rasterized spatial)", run_cnn), ("Spatial GAE (GraphST/STAGATE-style)", run_gae)]:
        mu, sd = fn(gs)
        print(f"{name:38s} AUROC {mu[0]:.3f}+-{sd[0]:.2f}  F1 {mu[1]:.3f}  BAcc {mu[2]:.3f}", flush=True)
    print("reference HGT-TIME: AUROC 0.781  F1 0.342  BAcc 0.444")
