#!/usr/bin/env python3
"""Subtype-level consensus ranking analysis.

Groups graph samples by their TIME subtype label (Hot=0, Excluded=1, Cold=2),
then computes per-subtype gene/pathway consensus rankings. This reveals
subtype-specific regulatory nodes and enables downstream comparison of
which genes are important for each microenvironment phenotype.

Usage:
    python subtype_consensus_ranking.py \\
        --config configs/hgt_time.default.yaml \\
        --experiment-id EXP-M01-HGT \\
        --output-dir outputs/results/EXP-M01-HGT/interpretability \\
        --topk 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

TIME_LABELS = {0: "Hot", 1: "Excluded", 2: "Cold"}


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def find_checkpoints(experiment_dir: Path) -> list[Path]:
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("best_fold*_seed*.pt"))


def load_model_from_checkpoint(
    ckpt_path: Path,
    config: dict[str, Any],
    sample_graph: Any,
) -> torch.nn.Module:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models import HGTTimeModel

    model_cfg = config.get("model", {})
    input_dims = {
        nt: int(sample_graph[nt].x.size(-1)) for nt in sample_graph.node_types
    }
    hgt_keys = {
        "hidden_dim", "num_layers", "num_heads", "dropout",
        "num_classes", "pheno_dim", "use_pheno_head",
        "use_cell_state_head", "cell_state_dim", "use_ranking_heads",
    }
    filtered = {k: v for k, v in model_cfg.items() if k in hgt_keys}
    model = HGTTimeModel(
        metadata=sample_graph.metadata(),
        input_dims=input_dims,
        **filtered,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_node_ids(store: Any, node_type: str) -> list[str]:
    """Extract node IDs from a node store."""
    node_ids_raw = getattr(store, "node_id", None)
    if node_ids_raw is not None:
        if isinstance(node_ids_raw, (list, tuple)):
            if node_ids_raw and isinstance(node_ids_raw[0], (list, tuple)):
                ids = []
                for group in node_ids_raw:
                    ids.extend(str(x) for x in group)
                return ids
            return [str(x) for x in node_ids_raw]
        if hasattr(node_ids_raw, "tolist"):
            return [str(x) for x in node_ids_raw.tolist()]
        return [str(node_ids_raw)]
    n = store.x.size(0)
    return [f"{node_type}_{i}" for i in range(n)]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Subtype-level consensus ranking")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    project_root = discover_project_root(Path(__file__))
    sys.path.insert(0, str(project_root / "Experiment" / "core_code"))

    config_path = project_root / args.config
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    experiment_id = args.experiment_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir:
        output_dir = project_root / args.output_dir
    else:
        output_root = project_root / config.get(
            "output_root", "Experiment/core_code/outputs/results"
        )
        output_dir = output_root / experiment_id / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graphs
    graphs_dir = project_root / config["input"]["graphs_dir"]
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    graphs = [torch.load(p, weights_only=False) for p in graph_paths]
    logger.info(f"Loaded {len(graphs)} graphs")

    # Group graphs by TIME label
    subtype_graphs: dict[int, list[Any]] = defaultdict(list)
    for g in graphs:
        if hasattr(g, "y_graph") and g.y_graph is not None and len(g.y_graph) > 0:
            label = int(g.y_graph[0].item())
        else:
            label = -1
        subtype_graphs[label].append(g)

    for label, glist in sorted(subtype_graphs.items()):
        name = TIME_LABELS.get(label, f"Unknown({label})")
        logger.info(f"  {name}: {len(glist)} graphs")

    # Find checkpoints
    exp_result_dir = (
        project_root
        / config.get("output_root", "Experiment/core_code/outputs/results")
        / experiment_id
    )
    checkpoints = find_checkpoints(exp_result_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {exp_result_dir}/checkpoints/")

    # Collect per-subtype rankings
    subtype_rankings: dict[str, dict[str, dict[str, list[float]]]] = {}
    # Structure: {subtype_name: {node_type: {node_id: [scores]}}}

    for label, glist in sorted(subtype_graphs.items()):
        if label < 0:
            continue
        subtype_name = TIME_LABELS.get(label, f"class_{label}")
        logger.info(f"Processing subtype: {subtype_name}")
        subtype_rankings[subtype_name] = {"gene": defaultdict(list), "pathway": defaultdict(list)}

        loader = DataLoader(glist, batch_size=4, shuffle=False)

        for ckpt_path in checkpoints:
            model = load_model_from_checkpoint(ckpt_path, config, graphs[0])
            model.to(device)
            model.eval()

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    outputs = model(batch)

                    for node_type in ("gene", "pathway"):
                        score_key = f"{node_type}_score"
                        scores = outputs.get(score_key)
                        if scores is None:
                            continue
                        batch_index = outputs["node_batch_index"][node_type].cpu()
                        scores_cpu = scores.detach().cpu()
                        node_ids = extract_node_ids(batch[node_type], node_type)

                        num_g = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
                        for gi in range(num_g):
                            mask = batch_index == gi
                            positions = mask.nonzero(as_tuple=False).view(-1)
                            for pos in positions.tolist():
                                nid = node_ids[pos]
                                s = float(scores_cpu[pos])
                                subtype_rankings[subtype_name][node_type][nid].append(s)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Build consensus ranking per subtype and compute cross-subtype coverage
    consensus_by_subtype: dict[str, dict[str, list[tuple[str, float]]]] = {}

    for subtype_name, nt_scores in subtype_rankings.items():
        consensus_by_subtype[subtype_name] = {}
        for node_type, score_dict in nt_scores.items():
            ranking = [
                (nid, float(np.mean(scores)))
                for nid, scores in score_dict.items()
            ]
            ranking.sort(key=lambda x: x[1], reverse=True)
            consensus_by_subtype[subtype_name][node_type] = ranking

    # Write per-subtype ranking TSVs
    for subtype_name, nt_rankings in consensus_by_subtype.items():
        for node_type, ranking in nt_rankings.items():
            out_path = output_dir / f"subtype_{subtype_name}_{node_type}_ranking.tsv"
            header = f"{node_type.capitalize()}_ID\tRank\tAvg_Score\tSubtype"
            lines = [header]
            for rank, (nid, score) in enumerate(ranking[:args.topk], start=1):
                lines.append(f"{nid}\t{rank}\t{score:.6f}\t{subtype_name}")
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info(f"  Written: {out_path}")

    # Cross-subtype consensus coverage matrix
    for node_type in ("gene", "pathway"):
        all_subtypes = sorted(consensus_by_subtype.keys())
        topk_sets = {}
        for st in all_subtypes:
            ranking = consensus_by_subtype[st].get(node_type, [])
            topk_sets[st] = set(nid for nid, _ in ranking[:args.topk])

        # Coverage: fraction of subtypes where each node appears in top-k
        all_nodes = set()
        for s in topk_sets.values():
            all_nodes |= s
        coverage: dict[str, dict[str, Any]] = {}
        for nid in all_nodes:
            present_in = [st for st in all_subtypes if nid in topk_sets[st]]
            coverage[nid] = {
                "n_subtypes": len(present_in),
                "fraction": len(present_in) / len(all_subtypes) if all_subtypes else 0,
                "subtypes": present_in,
            }

        # Write coverage TSV
        out_path = output_dir / f"subtype_consensus_coverage_{node_type}.tsv"
        header = f"{node_type.capitalize()}_ID\tN_Subtypes\tFraction\tPresent_In"
        lines = [header]
        for nid, cov in sorted(coverage.items(), key=lambda x: x[1]["fraction"], reverse=True):
            lines.append(
                f"{nid}\t{cov['n_subtypes']}\t{cov['fraction']:.4f}\t"
                f"{'; '.join(cov['subtypes'])}"
            )
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"  Coverage: {out_path}")

    # Summary JSON
    summary = {
        "subtypes_analyzed": list(consensus_by_subtype.keys()),
        "topk": args.topk,
        "n_checkpoints": len(checkpoints),
    }
    for node_type in ("gene", "pathway"):
        summary[f"{node_type}_top10_by_subtype"] = {}
        for st in sorted(consensus_by_subtype.keys()):
            ranking = consensus_by_subtype[st].get(node_type, [])
            summary[f"{node_type}_top10_by_subtype"][st] = [
                (nid, round(s, 6)) for nid, s in ranking[:10]
            ]

    summary_path = output_dir / "subtype_consensus_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Summary: {summary_path}")
    logger.info("=== Subtype consensus ranking complete ===")


if __name__ == "__main__":
    main()
