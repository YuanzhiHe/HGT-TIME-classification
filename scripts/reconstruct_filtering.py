#!/usr/bin/env python3
"""Reconstruct the 105 -> 36 candidate-region filtering breakdown (R1 / R3.5).

Safe: only calls summarize_graph_metadata (which runs derive_graph_labels over all
candidate regions). Does NOT invoke build_hetero_graphs.main() and therefore never
deletes any exported .pt graphs.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CORE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE / "scripts"))

import build_hetero_graphs as B  # noqa: E402


def resolve(p):
    return B.ensure_descendant(B.resolve_path(PROJECT_ROOT, p), PROJECT_ROOT)


def reconstruct(config_path: str):
    config = B.load_config(PROJECT_ROOT / config_path)
    scrna = resolve(config["input"]["scrna_manifest_path"])
    prior = resolve(config["input"]["prior_manifest_path"])
    spatial = resolve(config["input"]["spatial_manifest_path"])

    _, _ = B.load_reference_artifacts(scrna, PROJECT_ROOT)
    spatial_nodes, _, spatial_input_path = B.load_spatial_artifacts(spatial, PROJECT_ROOT)
    sxp = B.resolve_path(PROJECT_ROOT, config["input"].get("spatial_expression_path")) or spatial_input_path
    sxp = B.ensure_descendant(sxp, PROJECT_ROOT)
    bundle, _ = B.build_spatial_expression_lookup(
        expression_path=sxp,
        gene_symbol_column=config["input"].get("spatial_expression_gene_symbol_column"),
        assume_log1p_input=bool(config["input"].get("assume_log1p_input", True)),
    )
    obs = bundle["adata"].obs.copy()
    obs.index = obs.index.astype(str)
    nid = config["input"].get("spatial_node_id_column")
    obs["__node_id__"] = obs[nid].astype(str) if (nid and nid in obs.columns) else obs.index.astype(str)
    obs_lookup = obs.reset_index(drop=True).set_index("__node_id__", drop=False)

    gmf = B.summarize_graph_metadata(spatial_nodes=spatial_nodes, obs_lookup=obs_lookup, config=config)

    total = len(gmf)
    certain = gmf.loc[~gmf["is_uncertain"]]
    uncertain = gmf.loc[gmf["is_uncertain"]]

    print(f"Config: {config_path}")
    print(f"Total candidate regions (>=25 in-tissue spots): {total}")
    print(f"  Certain (exported, supervised): {len(certain)}")
    print(f"  Removed by uncertainty rule:    {len(uncertain)}")
    print()
    print("Preliminary time_label over ALL candidates:")
    for k, v in sorted(Counter(gmf["time_label"]).items()):
        print(f"  {k}: {v}")
    print()
    print("Final supervised label distribution (certain):")
    for k, v in sorted(Counter(certain["time_label"]).items()):
        print(f"  {k}: {v}")
    print()
    print("Removed-region label (preliminary) distribution:")
    for k, v in sorted(Counter(uncertain["time_label"]).items()):
        print(f"  {k}: {v}")
    if "spot_label_nunique" in gmf.columns:
        amb = gmf.loc[gmf["spot_label_nunique"] > 1]
        print(f"\nRegions with >1 spot-label class (ambiguous): {len(amb)}")
    out = PROJECT_ROOT / "Experiment/analysis/filtering_breakdown_candidate_table.tsv"
    keep_cols = [c for c in ["time_label", "is_uncertain", "spot_label_nunique",
                             "any_spot_uncertain", "n_spots",
                             "infiltration_score", "penetration_score",
                             "retention_score", "activation_score"] if c in gmf.columns]
    gmf.reset_index()[["graph_id" if "graph_id" in gmf.reset_index().columns else "index"] + keep_cols].to_csv(out, sep="\t", index=False)
    print(f"\nWrote candidate table -> {out}")


if __name__ == "__main__":
    reconstruct(sys.argv[1] if len(sys.argv) > 1 else "configs/hetero_graph.regions.yaml")
