#!/usr/bin/env python3
"""Held-out immune-gene ranking recovery (R3.4 / R4).

Loads the HGT checkpoints trained on graphs where a held-out subset of immune genes was
removed from the ranking supervision, ranks all gene nodes with the ranking head, and
reports whether the held-out genes still appear in the top-50 global ranking. If they do,
interpretability recovery is not merely reproducing the ranking prior.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

CORE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "scripts"))

from interpretability_analysis import load_model_from_checkpoint, find_checkpoints  # noqa


def main(config_path, exp_id, heldout_json):
    held = set(json.load(open(heldout_json)))
    with (PROJECT_ROOT / config_path).open() as f:
        config = yaml.safe_load(f)
    graphs_dir = PROJECT_ROOT / config["input"]["graphs_dir"]
    graphs = [torch.load(p, weights_only=False) for p in sorted(graphs_dir.glob("*.pt"))]
    exp_dir = PROJECT_ROOT / "outputs/results" / exp_id
    ckpts = find_checkpoints(exp_dir)
    loader = DataLoader(graphs, batch_size=4, shuffle=False)

    gene_scores = defaultdict(list)  # gene -> list of per-checkpoint mean scores
    for ckpt in ckpts:
        model = load_model_from_checkpoint(ckpt, config, graphs[0])
        model.eval()
        per_ck = defaultdict(list)
        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                gs = out.get("gene_score")
                if gs is None:
                    continue
                gs = gs.view(-1).cpu().numpy()
                ids = []
                # gene node_ids concatenated across batch in order
                for g in batch.to_data_list():
                    ids.extend(g["gene"].node_id)
                for gid, s in zip(ids, gs):
                    per_ck[gid].append(float(s))
        for gid, vals in per_ck.items():
            gene_scores[gid].append(float(np.mean(vals)))

    # global ranking: mean score across checkpoints
    ranking = sorted(((gid, float(np.mean(v))) for gid, v in gene_scores.items()),
                     key=lambda kv: kv[1], reverse=True)
    rank_of = {gid: i + 1 for i, (gid, _) in enumerate(ranking)}
    total = len(ranking)
    print(f"Ranked {total} genes across {len(ckpts)} held-out checkpoints. Top-15:")
    for gid, sc in ranking[:15]:
        tag = " <== HELD-OUT" if gid in held else ""
        print(f"  {rank_of[gid]:3d}. {gid:10s} {sc:.4f}{tag}")
    print("\nHeld-out gene recovery (removed from ranking supervision):")
    for gid in sorted(held):
        r = rank_of.get(gid)
        pct = f"top-{100*r/total:.0f}%" if r else "absent"
        intop = "YES" if (r and r <= 50) else "no"
        print(f"  {gid:10s} rank {r}/{total} ({pct})  in-top50={intop}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
