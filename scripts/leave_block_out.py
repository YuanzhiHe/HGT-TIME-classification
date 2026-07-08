#!/usr/bin/env python3
"""Leave-spatial-block-out evaluation (reviewer R3.6).

Groups the 36 Section-1 regions into four contiguous spatial quadrant blocks of the
12x12 grid and holds out one whole block at a time (GroupKFold by block, not by region),
so entire contiguous tissue quadrants are unseen at test time -- a strictly harder spatial
split than the region-grouped CV. Reuses the pipeline's build_model / train_one_fold.
Writes to a NEW results dir; touches no published outputs.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

CORE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "scripts"))

import train_eval_pipeline as P
from train_eval_pipeline import build_model, train_one_fold


def block_of(graph_id: str) -> int:
    m = re.search(r"tile_r(\d+)_c(\d+)", graph_id)
    r, c = int(m.group(1)), int(m.group(2))
    return (0 if r < 6 else 2) + (0 if c < 6 else 1)  # 4 quadrant blocks: 0,1,2,3


def main(config_path: str, exp_id: str):
    import yaml
    with (PROJECT_ROOT / config_path).open() as f:
        config = yaml.safe_load(f)
    model_family = config.get("model_family", "hgt_time")
    graphs_dir = PROJECT_ROOT / config["input"]["graphs_dir"]
    graphs = [torch.load(p, weights_only=False) for p in sorted(graphs_dir.glob("*.pt"))]
    graphs = [g for g in graphs if bool(torch.as_tensor(getattr(g, "label_mask")).view(-1)[0].item())]
    gids = [g._global_store["graph_id"] for g in graphs]
    blocks = np.array([block_of(g) for g in gids])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get("runtime", {}).get("batch_size", 4)
    ckpt_dir = PROJECT_ROOT / "outputs/results" / exp_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in [42, 123, 2026]:
        P.set_seed(seed)
        for fold, held in enumerate(sorted(set(blocks.tolist()))):
            val_idx = np.where(blocks == held)[0]
            train_idx = np.where(blocks != held)[0]
            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
            model = build_model(model_family, config.get("model", {}), graphs[0])
            res = train_one_fold(model=model, train_loader=train_loader, val_loader=val_loader,
                                 config=config, device=device, checkpoint_dir=ckpt_dir,
                                 fold=fold, seed=seed)
            fm = res["final_metrics"]
            print(f"  seed {seed} block {held} (n_val={len(val_idx)}): "
                  f"AUROC {fm.get('macro_auroc',0):.3f} F1 {fm.get('macro_f1',0):.3f} BAcc {fm.get('balanced_accuracy',0):.3f}")
            all_results.append(res)

    def agg(key):
        v = [r["final_metrics"].get(key, 0.0) for r in all_results]
        return float(np.mean(v)), float(np.std(v))
    aggd = {f"{k}_mean": agg(k)[0] for k in ["macro_auroc", "macro_f1", "balanced_accuracy"]}
    aggd.update({f"{k}_std": agg(k)[1] for k in ["macro_auroc", "macro_f1", "balanced_accuracy"]})
    out = PROJECT_ROOT / "outputs/results" / exp_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.json").open("w") as f:
        json.dump({"experiment_id": exp_id, "aggregated": aggd, "per_fold": all_results}, f, indent=2, default=str)
    print(f"{exp_id}: AUROC {aggd['macro_auroc_mean']:.3f}+-{aggd['macro_auroc_std']:.3f} "
          f"F1 {aggd['macro_f1_mean']:.3f} BAcc {aggd['balanced_accuracy_mean']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
