#!/usr/bin/env python3
"""Phase 1: attach H&E morphology embeddings (Phikon) as a third fusion stream.

For each fusion graph, parse cell node_id '<sample>__<barcode>', pull the 224x224 H&E patch
from the sample's HEST .h5, encode with the frozen Phikon histology foundation model
(independent of the expression data), and store as cell.morph (768-dim). Output: a new graph
set with cell.x (50) + cell.pretrain (256) + cell.morph (768).
"""
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

CORE = Path(__file__).resolve().parents[1]
PATCH_DIR = CORE / "data/hest1k_breast/patches"


def load_phikon(device):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon").to(device).eval()
    return m


@torch.no_grad()
def encode(model, patches_uint8, device, bs=64):
    # patches_uint8: (N,224,224,3) uint8 -> normalized (N,3,224,224)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    outs = []
    for i in range(0, len(patches_uint8), bs):
        b = torch.from_numpy(patches_uint8[i:i + bs]).float().permute(0, 3, 1, 2).to(device) / 255.0
        b = (b - mean) / std
        cls = model(pixel_values=b).last_hidden_state[:, 0]  # (B,768) CLS token
        outs.append(torch.nn.functional.normalize(cls, dim=-1).cpu())
    return torch.cat(outs, 0)


def build_patch_index():
    idx = {}
    for h5 in glob.glob(str(PATCH_DIR / "*.h5")):
        s = Path(h5).stem
        with h5py.File(h5, "r") as h:
            bc = {b[0].decode(): i for i, b in enumerate(h["barcode"][:])}
        idx[s] = (h5, bc)
    return idx


def main(src_dir, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_phikon(device)
    pindex = build_patch_index()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    graphs = sorted(glob.glob(os.path.join(src_dir, "*.pt")))
    done, skipped = 0, 0
    for gp in graphs:
        g = torch.load(gp, weights_only=False)
        nids = [str(x) for x in g["cell"].node_id]
        patches = np.zeros((len(nids), 224, 224, 3), dtype=np.uint8)
        ok = True
        # group by sample to open each h5 once; handle both node_id formats:
        #   Visium: '<sample>__<barcode>'   Xenium: '<sample>_<cellID>'
        by_sample = {}
        for i, nid in enumerate(nids):
            if "__" in nid:
                s, bc = nid.split("__", 1)
            else:
                s, bc = nid.rsplit("_", 1)
            by_sample.setdefault(s, []).append((i, bc))
        n_missing = 0
        for s, items in by_sample.items():
            if s not in pindex:
                n_missing += len(items); continue
            h5, bcmap = pindex[s]
            with h5py.File(h5, "r") as h:
                img = h["img"]
                for i, bc in items:
                    j = bcmap.get(bc)
                    if j is None:
                        n_missing += 1; continue
                    patches[i] = img[j]
        # accept graph if >=80% cells matched (missing patches stay zero)
        if n_missing > 0.2 * len(nids):
            skipped += 1; continue
        emb = encode(model, patches, device)
        g["cell"].morph = emb.contiguous()
        torch.save(g, out / Path(gp).name)
        done += 1
    print(f"{src_dir} -> {out_dir}: attached morph to {done} graphs, skipped {skipped}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
