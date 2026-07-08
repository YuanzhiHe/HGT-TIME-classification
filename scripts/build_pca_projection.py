#!/usr/bin/env python3
"""Build a PCA-64 projected graph variant from the 306-dim combined_aug graphs (R3.4).

combined_aug cell.x = [50 base features | 256 pretrained expr embedding].
This replaces the 256-dim block with a PCA(64) projection (fit on the pooled cells of
the source graph set, applied to all graphs, L2-normalized), yielding a 114-dim variant
directly comparable to the random-projection combined_proj64. Writes to a NEW dir.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

CORE = Path(__file__).resolve().parents[1]


def build(src_dir: Path, out_dir: Path, fit_dir: Path | None = None, n_comp: int = 64, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    src_paths = sorted(src_dir.glob("*.pt"))
    fit_paths = sorted((fit_dir or src_dir).glob("*.pt"))

    # Fit PCA on the pooled 256-dim pretrained block of the fit set.
    pool = []
    for p in fit_paths:
        g = torch.load(p, weights_only=False)
        pool.append(g["cell"].x[:, 50:].numpy())
    X = np.concatenate(pool, axis=0)
    pca = PCA(n_components=n_comp, random_state=seed).fit(X)
    print(f"PCA fit on {X.shape[0]} cells x {X.shape[1]} dims; "
          f"explained var (64 comp) = {pca.explained_variance_ratio_.sum():.3f}")

    for p in src_paths:
        g = torch.load(p, weights_only=False)
        base = g["cell"].x[:, :50].numpy()
        pre = g["cell"].x[:, 50:].numpy()
        proj = pca.transform(pre)
        proj = normalize(proj, axis=1)  # L2, matching the random-projection variant
        newx = np.concatenate([base, proj], axis=1).astype(np.float32)
        g["cell"].x = torch.from_numpy(newx)
        torch.save(g, out_dir / p.name)
    print(f"Wrote {len(src_paths)} graphs -> {out_dir} (cell.x dim = {newx.shape[1]})")


if __name__ == "__main__":
    base = CORE / "outputs/hetero_graph"
    # Section 1
    build(base / "visium_breast_regions__combined_aug/graphs",
          base / "visium_breast_regions__combined_pca64/graphs",
          fit_dir=base / "visium_breast_regions__combined_aug/graphs")
    # Section 2 (fit PCA on Section 1 source, apply to Section 2 for fair transfer)
    s2 = base / "visium_breast_section2_regions__combined_aug/graphs"
    if s2.exists():
        build(s2, base / "visium_breast_section2_regions__combined_pca64/graphs",
              fit_dir=base / "visium_breast_regions__combined_aug/graphs")
