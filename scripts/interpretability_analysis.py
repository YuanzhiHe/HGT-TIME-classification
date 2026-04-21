#!/usr/bin/env python3
"""Interpretability analysis and candidate target ranking pipeline.

Loads trained HGT-TIME checkpoints, runs inference to collect gene/pathway
ranking scores, then evaluates ranking quality through three pillars:
  1. Ranking stability (bootstrap top-k Jaccard, Spearman rho across folds/seeds)
  2. Perturbation sensitivity (node masking → probability delta)
  3. Biological agreement (known immune target overlap, pathway enrichment)

Outputs structured TSV tables for gene-level and pathway-level candidate
target prioritisation (Tier 1 = high confidence, Tier 2 = exploratory).

Usage:
    python interpretability_analysis.py \\
        --config configs/hgt_time.default.yaml \\
        --experiment-id EXP-M01-HGT \\
        --output-dir outputs/results/EXP-M01-HGT/interpretability

    # With known target list and pathway gene sets:
    python interpretability_analysis.py \\
        --config configs/hgt_time.default.yaml \\
        --experiment-id EXP-M01-HGT \\
        --known-targets resources/known_immune_targets.txt \\
        --pathway-gmt resources/kegg_immune_pathways.gmt \\
        --topk 50
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy import stats as scipy_stats
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------

def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise SystemExit("Could not locate project root via repository structure")


# ---------------------------------------------------------------------------
# Known immune target loading
# ---------------------------------------------------------------------------

BUILTIN_IMMUNE_TARGETS = {
    # Immune checkpoints
    "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "PDCD1LG2",
    "VSIR", "CD80", "CD86", "IDO1", "SIGLEC15",
    # IFN-gamma pathway
    "IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1",
    # Antigen presentation
    "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB9",
    # T cell markers / activation
    "CD3D", "CD3E", "CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "ICOS",
    # Immune exclusion
    "TGFB1", "TGFB2", "VEGFA", "WNT5A", "CTNNB1",
    # Chemokines
    "CCL2", "CCL5", "CXCL12", "CXCL13",
    # Myeloid markers
    "CD68", "CD163", "CSF1R", "ARG1", "MRC1",
}


def load_known_targets(path: Path | None) -> set[str]:
    """Load known immune targets from file or use built-in set."""
    if path is None:
        logger.info(f"Using built-in immune target set ({len(BUILTIN_IMMUNE_TARGETS)} genes)")
        return BUILTIN_IMMUNE_TARGETS
    targets: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                targets.add(gene)
    logger.info(f"Loaded {len(targets)} known targets from {path}")
    return targets


# ---------------------------------------------------------------------------
# GMT pathway gene set loading
# ---------------------------------------------------------------------------

def load_gmt(path: Path | None) -> dict[str, set[str]]:
    """Load GMT format pathway gene sets. Returns {pathway_name: {gene_set}}."""
    if path is None:
        return {}
    gene_sets: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = set(parts[2:])
            gene_sets[name] = genes
    logger.info(f"Loaded {len(gene_sets)} pathway gene sets from {path}")
    return gene_sets


# ---------------------------------------------------------------------------
# Checkpoint and model loading
# ---------------------------------------------------------------------------

def find_checkpoints(experiment_dir: Path) -> list[Path]:
    """Find all checkpoint files in experiment directory."""
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("best_fold*_seed*.pt"))


def load_model_from_checkpoint(
    ckpt_path: Path,
    config: dict[str, Any],
    sample_graph: Any,
) -> torch.nn.Module:
    """Load a trained HGT model from a checkpoint."""
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


# ---------------------------------------------------------------------------
# Ranking collection across checkpoints
# ---------------------------------------------------------------------------

def collect_rankings_from_checkpoints(
    checkpoints: list[Path],
    config: dict[str, Any],
    graphs: list[Any],
    device: torch.device,
    topk: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    """Run inference on all graphs with each checkpoint and collect rankings.

    Returns:
        {
            "gene": [
                {  # per checkpoint
                    "ckpt": str,
                    "fold": int,
                    "seed": int,
                    "per_graph": [
                        {"graph_id": str, "scores": {node_id: score, ...}},
                        ...
                    ],
                    "global_ranking": [(node_id, avg_score), ...],  # sorted desc
                },
                ...
            ],
            "pathway": [ ... same structure ... ],
        }
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models import collect_topk_rankings

    all_rankings: dict[str, list[dict[str, Any]]] = {"gene": [], "pathway": []}
    loader = DataLoader(graphs, batch_size=4, shuffle=False)

    for ckpt_path in checkpoints:
        # Parse fold/seed from filename: best_fold{F}_seed{S}.pt
        stem = ckpt_path.stem
        fold = int(stem.split("fold")[1].split("_")[0])
        seed = int(stem.split("seed")[1])

        model = load_model_from_checkpoint(ckpt_path, config, graphs[0])
        model.to(device)
        model.eval()

        gene_scores_global: dict[str, list[float]] = defaultdict(list)
        pathway_scores_global: dict[str, list[float]] = defaultdict(list)
        per_graph_gene: list[dict[str, Any]] = []
        per_graph_pathway: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                outputs = model(batch)

                # Collect per-node scores for gene and pathway
                for node_type, score_key, global_dict, per_graph_list in [
                    ("gene", "gene_score", gene_scores_global, per_graph_gene),
                    ("pathway", "pathway_score", pathway_scores_global, per_graph_pathway),
                ]:
                    scores = outputs.get(score_key)
                    if scores is None:
                        continue
                    batch_index = outputs["node_batch_index"][node_type].cpu()
                    scores_cpu = scores.detach().cpu()

                    # Get node IDs
                    node_ids_raw = getattr(batch[node_type], "node_id", None)
                    if node_ids_raw is not None:
                        if isinstance(node_ids_raw, (list, tuple)):
                            if node_ids_raw and isinstance(node_ids_raw[0], (list, tuple)):
                                node_ids = []
                                for group in node_ids_raw:
                                    node_ids.extend(str(x) for x in group)
                            else:
                                node_ids = [str(x) for x in node_ids_raw]
                        elif hasattr(node_ids_raw, "tolist"):
                            node_ids = [str(x) for x in node_ids_raw.tolist()]
                        else:
                            node_ids = [str(node_ids_raw)]
                    else:
                        node_ids = [f"{node_type}_{i}" for i in range(scores_cpu.numel())]

                    # Get graph IDs
                    graph_ids_raw = getattr(batch, "graph_id", None)
                    if graph_ids_raw is not None:
                        if isinstance(graph_ids_raw, (list, tuple)):
                            graph_ids = [str(x) for x in graph_ids_raw]
                        else:
                            graph_ids = [str(graph_ids_raw)]
                    else:
                        num_g = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
                        graph_ids = [str(i) for i in range(num_g)]

                    for gi, gid in enumerate(graph_ids):
                        mask = batch_index == gi
                        positions = mask.nonzero(as_tuple=False).view(-1)
                        if positions.numel() == 0:
                            per_graph_list.append({"graph_id": gid, "scores": {}})
                            continue
                        g_scores = scores_cpu[positions]
                        g_node_ids = [node_ids[int(p)] for p in positions.tolist()]
                        score_dict = {}
                        for nid, s in zip(g_node_ids, g_scores.tolist()):
                            score_dict[nid] = s
                            global_dict[nid].append(s)
                        per_graph_list.append({"graph_id": gid, "scores": score_dict})

        # Build global ranking per checkpoint (average across graphs)
        for node_type, global_dict, per_graph_list, target_list in [
            ("gene", gene_scores_global, per_graph_gene, all_rankings["gene"]),
            ("pathway", pathway_scores_global, per_graph_pathway, all_rankings["pathway"]),
        ]:
            global_ranking = [
                (nid, float(np.mean(scores)))
                for nid, scores in global_dict.items()
            ]
            global_ranking.sort(key=lambda x: x[1], reverse=True)

            target_list.append({
                "ckpt": str(ckpt_path),
                "fold": fold,
                "seed": seed,
                "per_graph": per_graph_list,
                "global_ranking": global_ranking,
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_rankings


# ---------------------------------------------------------------------------
# Pillar 1: Ranking stability metrics
# ---------------------------------------------------------------------------

def compute_topk_jaccard(
    rankings: list[list[tuple[str, float]]],
    k: int,
) -> float:
    """Compute average pairwise Jaccard index of top-k lists across rankings."""
    if len(rankings) < 2:
        return 1.0
    topk_sets = [set(nid for nid, _ in r[:k]) for r in rankings]
    jaccards = []
    for i in range(len(topk_sets)):
        for j in range(i + 1, len(topk_sets)):
            intersection = len(topk_sets[i] & topk_sets[j])
            union = len(topk_sets[i] | topk_sets[j])
            jaccards.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(jaccards))


def compute_rank_correlation(
    rankings: list[list[tuple[str, float]]],
) -> float:
    """Compute average pairwise Spearman rho across rankings.

    Rankings are aligned by node_id. Nodes absent from a ranking
    receive the worst-case rank.
    """
    if len(rankings) < 2:
        return 1.0

    # Build union of all node IDs
    all_ids: set[str] = set()
    for r in rankings:
        all_ids.update(nid for nid, _ in r)
    sorted_ids = sorted(all_ids)
    n = len(sorted_ids)
    id_to_idx = {nid: i for i, nid in enumerate(sorted_ids)}

    # Build rank vectors (1-based, worst rank = n for missing nodes)
    rank_vectors = []
    for r in rankings:
        rv = np.full(n, n, dtype=float)  # default worst rank
        for rank, (nid, _) in enumerate(r, start=1):
            rv[id_to_idx[nid]] = rank
        rank_vectors.append(rv)

    # Average pairwise Spearman
    rhos = []
    for i in range(len(rank_vectors)):
        for j in range(i + 1, len(rank_vectors)):
            rho, _ = scipy_stats.spearmanr(rank_vectors[i], rank_vectors[j])
            if np.isnan(rho):
                rho = 0.0
            rhos.append(rho)
    return float(np.mean(rhos))


def compute_stability_metrics(
    all_rankings: dict[str, list[dict[str, Any]]],
    topk: int = 50,
) -> dict[str, dict[str, float]]:
    """Compute stability metrics for each node type across checkpoints."""
    stability: dict[str, dict[str, float]] = {}
    for node_type in ("gene", "pathway"):
        ckpt_rankings = all_rankings.get(node_type, [])
        if not ckpt_rankings:
            stability[node_type] = {
                "topk_jaccard": 0.0,
                "spearman_rho": 0.0,
                "n_checkpoints": 0,
            }
            continue

        global_rankings = [entry["global_ranking"] for entry in ckpt_rankings]
        stability[node_type] = {
            "topk_jaccard": compute_topk_jaccard(global_rankings, topk),
            "spearman_rho": compute_rank_correlation(global_rankings),
            "n_checkpoints": len(ckpt_rankings),
        }
    return stability


# ---------------------------------------------------------------------------
# Per-node stability (cross-checkpoint Jaccard membership)
# ---------------------------------------------------------------------------

def compute_per_node_stability(
    all_rankings: dict[str, list[dict[str, Any]]],
    topk: int = 50,
) -> dict[str, dict[str, float]]:
    """For each node, compute fraction of checkpoints where it appears in top-k.

    Returns: {node_type: {node_id: fraction_in_topk}}
    """
    result: dict[str, dict[str, float]] = {}
    for node_type in ("gene", "pathway"):
        ckpt_rankings = all_rankings.get(node_type, [])
        if not ckpt_rankings:
            result[node_type] = {}
            continue
        n_ckpts = len(ckpt_rankings)
        appearances: dict[str, int] = defaultdict(int)
        for entry in ckpt_rankings:
            topk_ids = set(nid for nid, _ in entry["global_ranking"][:topk])
            for nid in topk_ids:
                appearances[nid] += 1
        result[node_type] = {
            nid: count / n_ckpts for nid, count in appearances.items()
        }
    return result


# ---------------------------------------------------------------------------
# Pillar 2: Perturbation sensitivity
# ---------------------------------------------------------------------------

def compute_perturbation_delta(
    model: torch.nn.Module,
    graph: Any,
    target_nodes: list[str],
    node_type: str,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Compute probability delta when masking each target node.

    For each node in target_nodes:
      1. Run model on original graph → P(y|G)
      2. Zero out node features → P(y|G\\{v})
      3. Delta = P(y=predicted_class|G) - P(y=predicted_class|G\\{v})

    Returns: {node_id: {"delta_prob": float, "delta_pheno": float|None}}
    """
    model.eval()
    graph_orig = graph.clone().to(device)

    # Baseline prediction
    with torch.no_grad():
        out_orig = model(graph_orig)
    probs_orig = out_orig["graph_probs"][0].cpu().numpy()
    pred_class = int(np.argmax(probs_orig))
    pheno_orig = out_orig["pheno_pred"][0].cpu().numpy() if out_orig.get("pheno_pred") is not None else None

    # Get node ID mapping
    node_ids_raw = getattr(graph_orig[node_type], "node_id", None)
    if node_ids_raw is not None:
        if isinstance(node_ids_raw, (list, tuple)):
            if node_ids_raw and isinstance(node_ids_raw[0], (list, tuple)):
                node_ids = []
                for group in node_ids_raw:
                    node_ids.extend(str(x) for x in group)
            else:
                node_ids = [str(x) for x in node_ids_raw]
        elif hasattr(node_ids_raw, "tolist"):
            node_ids = [str(x) for x in node_ids_raw.tolist()]
        else:
            node_ids = [str(node_ids_raw)]
    else:
        n_nodes = graph_orig[node_type].x.size(0)
        node_ids = [f"{node_type}_{i}" for i in range(n_nodes)]

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    results: dict[str, dict[str, float]] = {}
    for nid in target_nodes:
        if nid not in id_to_idx:
            continue
        idx = id_to_idx[nid]

        # Create perturbed graph (zero out node features)
        graph_pert = graph.clone().to(device)
        graph_pert[node_type].x[idx] = 0.0

        with torch.no_grad():
            out_pert = model(graph_pert)
        probs_pert = out_pert["graph_probs"][0].cpu().numpy()

        delta_prob = float(probs_orig[pred_class] - probs_pert[pred_class])

        delta_pheno = None
        if pheno_orig is not None and out_pert.get("pheno_pred") is not None:
            pheno_pert = out_pert["pheno_pred"][0].cpu().numpy()
            delta_pheno = float(np.linalg.norm(pheno_orig - pheno_pert))

        results[nid] = {
            "delta_prob": delta_prob,
            "delta_pheno": delta_pheno,
            "baseline_prob": float(probs_orig[pred_class]),
        }

    return results


def run_perturbation_analysis(
    checkpoints: list[Path],
    config: dict[str, Any],
    graphs: list[Any],
    consensus_rankings: dict[str, list[tuple[str, float]]],
    device: torch.device,
    topk_perturb: int = 100,
    max_graphs: int = 10,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run perturbation sensitivity analysis using consensus top-k nodes.

    Uses a subset of graphs and averages deltas across checkpoints × graphs.

    Returns: {node_type: {node_id: {"delta_prob_mean": ..., "delta_prob_std": ..., ...}}}
    """
    # Select graphs for perturbation (limit for computational cost)
    selected_graphs = graphs[:max_graphs]

    # For each node type, take top-k from consensus ranking
    results: dict[str, dict[str, dict[str, float]]] = {}

    for node_type in ("gene", "pathway"):
        ranking = consensus_rankings.get(node_type, [])
        if not ranking:
            results[node_type] = {}
            continue

        target_nodes = [nid for nid, _ in ranking[:topk_perturb]]
        node_deltas: dict[str, list[dict[str, float]]] = defaultdict(list)

        # Use up to 3 checkpoints for perturbation (expensive operation)
        ckpts_to_use = checkpoints[:min(3, len(checkpoints))]

        for ckpt_path in ckpts_to_use:
            model = load_model_from_checkpoint(ckpt_path, config, graphs[0])
            model.to(device)
            model.eval()

            for graph in selected_graphs:
                deltas = compute_perturbation_delta(
                    model, graph, target_nodes, node_type, device,
                )
                for nid, delta_dict in deltas.items():
                    node_deltas[nid].append(delta_dict)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate across checkpoints × graphs
        aggregated: dict[str, dict[str, float]] = {}
        for nid, delta_list in node_deltas.items():
            prob_deltas = [d["delta_prob"] for d in delta_list]
            pheno_deltas = [d["delta_pheno"] for d in delta_list if d["delta_pheno"] is not None]
            aggregated[nid] = {
                "delta_prob_mean": float(np.mean(prob_deltas)),
                "delta_prob_std": float(np.std(prob_deltas)),
                "delta_pheno_mean": float(np.mean(pheno_deltas)) if pheno_deltas else 0.0,
                "delta_pheno_std": float(np.std(pheno_deltas)) if pheno_deltas else 0.0,
                "n_evaluations": len(delta_list),
            }
        results[node_type] = aggregated

    return results


# ---------------------------------------------------------------------------
# Pillar 3: Biological agreement
# ---------------------------------------------------------------------------

def compute_known_target_overlap(
    ranking: list[tuple[str, float]],
    known_targets: set[str],
    k: int,
) -> dict[str, Any]:
    """Compute overlap of top-k ranking with known immune targets."""
    topk_ids = [nid for nid, _ in ranking[:k]]
    overlap = set(topk_ids) & known_targets
    return {
        "k": k,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / k if k > 0 else 0.0,
        "overlapping_genes": sorted(overlap),
        "total_known_targets": len(known_targets),
    }


def compute_pathway_enrichment(
    ranking: list[tuple[str, float]],
    pathway_gene_sets: dict[str, set[str]],
    background_size: int,
    topk: int = 50,
) -> list[dict[str, Any]]:
    """Hypergeometric enrichment test for top-k ranked genes against pathway gene sets.

    Returns sorted list of enrichment results (ascending p-value).
    """
    if not pathway_gene_sets:
        return []

    topk_set = set(nid for nid, _ in ranking[:topk])
    K = len(topk_set)  # drawn
    N = background_size  # population

    enrichments = []
    for pathway_name, gene_set in pathway_gene_sets.items():
        M = len(gene_set)  # successes in population
        overlap = topk_set & gene_set
        x = len(overlap)  # successes in drawn
        if x == 0:
            continue

        # Hypergeometric test: P(X >= x)
        pval = float(scipy_stats.hypergeom.sf(x - 1, N, M, K))

        enrichments.append({
            "pathway": pathway_name,
            "overlap_count": x,
            "pathway_size": M,
            "topk_size": K,
            "background_size": N,
            "p_value": pval,
            "overlapping_genes": sorted(overlap),
        })

    # Sort by p-value
    enrichments.sort(key=lambda e: e["p_value"])

    # BH FDR correction
    n_tests = len(enrichments)
    for rank_i, enr in enumerate(enrichments, start=1):
        enr["fdr"] = min(enr["p_value"] * n_tests / rank_i, 1.0)

    return enrichments


# ---------------------------------------------------------------------------
# Consensus ranking builder
# ---------------------------------------------------------------------------

def build_consensus_ranking(
    all_rankings: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[str, float]]]:
    """Build consensus ranking by averaging scores across all checkpoints.

    Returns: {node_type: [(node_id, avg_score), ...]} sorted descending.
    """
    consensus: dict[str, list[tuple[str, float]]] = {}
    for node_type in ("gene", "pathway"):
        ckpt_rankings = all_rankings.get(node_type, [])
        if not ckpt_rankings:
            consensus[node_type] = []
            continue

        # Aggregate scores across checkpoints
        score_accumulator: dict[str, list[float]] = defaultdict(list)
        for entry in ckpt_rankings:
            for nid, score in entry["global_ranking"]:
                score_accumulator[nid].append(score)

        # Mean score
        avg_ranking = [
            (nid, float(np.mean(scores)))
            for nid, scores in score_accumulator.items()
        ]
        avg_ranking.sort(key=lambda x: x[1], reverse=True)
        consensus[node_type] = avg_ranking

    return consensus


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_tiers(
    consensus_ranking: list[tuple[str, float]],
    per_node_stability: dict[str, float],
    perturbation_results: dict[str, dict[str, float]],
    known_targets: set[str],
    topk: int = 50,
    stability_threshold: float = 0.6,
    delta_threshold: float = 0.01,
) -> list[dict[str, Any]]:
    """Assign Tier 1 (high confidence) or Tier 2 (exploratory) to top-k nodes.

    Tier 1 criteria (ALL must hold):
      - Appears in top-k in >= stability_threshold fraction of checkpoints
      - Perturbation delta_prob_mean >= delta_threshold
      - Known immune target match OR strong perturbation signal

    Tier 2: remaining top-k nodes with model score > 0.
    """
    tier_results: list[dict[str, Any]] = []

    for rank, (nid, score) in enumerate(consensus_ranking[:topk], start=1):
        stability = per_node_stability.get(nid, 0.0)
        pert = perturbation_results.get(nid, {})
        delta_prob = pert.get("delta_prob_mean", 0.0)
        delta_pheno = pert.get("delta_pheno_mean", 0.0)
        is_known = nid in known_targets

        is_stable = stability >= stability_threshold
        is_sensitive = delta_prob >= delta_threshold

        if is_stable and (is_sensitive or is_known):
            tier = "Tier 1"
        else:
            tier = "Tier 2"

        tier_results.append({
            "rank": rank,
            "node_id": nid,
            "avg_model_score": score,
            "cv_stability_jaccard": stability,
            "perturbation_delta_prob": delta_prob,
            "perturbation_delta_pheno": delta_pheno,
            "tier": tier,
            "known_target_match": is_known,
        })

    return tier_results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_gene_ranking_tsv(
    tier_results: list[dict[str, Any]],
    output_path: Path,
    enrichment_results: list[dict[str, Any]] | None = None,
) -> None:
    """Write gene-level ranking TSV."""
    # Build gene→pathway mapping from enrichment
    gene_pathways: dict[str, list[str]] = defaultdict(list)
    if enrichment_results:
        for enr in enrichment_results:
            if enr.get("fdr", 1.0) < 0.25:
                for gene in enr.get("overlapping_genes", []):
                    gene_pathways[gene].append(enr["pathway"])

    header = (
        "Gene_ID\tSymbol\tRank\tAvg_Model_Score\t"
        "CV_Stability_Jaccard\tPerturbation_Delta_Prob\t"
        "Perturbation_Delta_Pheno\tTier\tKnown_Target_Match\t"
        "Associated_Pathways"
    )
    lines = [header]
    for entry in tier_results:
        nid = entry["node_id"]
        pathways = "; ".join(gene_pathways.get(nid, []))
        line = (
            f"{nid}\t{nid}\t{entry['rank']}\t"
            f"{entry['avg_model_score']:.6f}\t"
            f"{entry['cv_stability_jaccard']:.4f}\t"
            f"{entry['perturbation_delta_prob']:.6f}\t"
            f"{entry['perturbation_delta_pheno']:.6f}\t"
            f"{entry['tier']}\t"
            f"{entry['known_target_match']}\t"
            f"{pathways}"
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Gene ranking written to {output_path}")


def write_pathway_ranking_tsv(
    tier_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write pathway-level ranking TSV."""
    header = (
        "Pathway_ID\tPathway_Name\tRank\tAvg_Model_Score\t"
        "CV_Stability\tPerturbation_Delta_Prob\t"
        "Perturbation_Delta_Pheno\tTier"
    )
    lines = [header]
    for entry in tier_results:
        line = (
            f"{entry['node_id']}\t{entry['node_id']}\t{entry['rank']}\t"
            f"{entry['avg_model_score']:.6f}\t"
            f"{entry['cv_stability_jaccard']:.4f}\t"
            f"{entry['perturbation_delta_prob']:.6f}\t"
            f"{entry['perturbation_delta_pheno']:.6f}\t"
            f"{entry['tier']}"
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Pathway ranking written to {output_path}")


def write_enrichment_tsv(
    enrichments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write pathway enrichment results TSV."""
    header = (
        "Pathway\tOverlap_Count\tPathway_Size\t"
        "TopK_Size\tBackground_Size\tP_Value\tFDR\t"
        "Overlapping_Genes"
    )
    lines = [header]
    for enr in enrichments:
        genes = "; ".join(enr.get("overlapping_genes", []))
        line = (
            f"{enr['pathway']}\t{enr['overlap_count']}\t"
            f"{enr['pathway_size']}\t{enr['topk_size']}\t"
            f"{enr['background_size']}\t{enr['p_value']:.6e}\t"
            f"{enr.get('fdr', 1.0):.6e}\t{genes}"
        )
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Enrichment results written to {output_path}")


def write_summary_json(
    stability_metrics: dict,
    consensus_rankings: dict,
    overlap_results: dict,
    enrichment_results: list,
    perturbation_summary: dict,
    tier_counts: dict,
    output_path: Path,
) -> None:
    """Write comprehensive JSON summary of all interpretability results."""
    summary = {
        "stability_metrics": stability_metrics,
        "biological_agreement": {
            "known_target_overlap": overlap_results,
            "pathway_enrichment_significant": len([
                e for e in enrichment_results if e.get("fdr", 1.0) < 0.05
            ]),
        },
        "perturbation_summary": {
            node_type: {
                "n_nodes_evaluated": len(pert_dict),
                "mean_delta_prob": float(np.mean([
                    v.get("delta_prob_mean", 0.0) for v in pert_dict.values()
                ])) if pert_dict else 0.0,
            }
            for node_type, pert_dict in perturbation_summary.items()
        },
        "tier_counts": tier_counts,
        "consensus_ranking_top10": {
            node_type: [(nid, round(s, 6)) for nid, s in ranking[:10]]
            for node_type, ranking in consensus_rankings.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Summary JSON written to {output_path}")


# ---------------------------------------------------------------------------
# Type weights & node attention summary
# ---------------------------------------------------------------------------

def collect_type_weights(
    checkpoints: list[Path],
    config: dict[str, Any],
    graphs: list[Any],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Collect average type gating weights across checkpoints and graphs.

    Returns: {"mean": {node_type: avg_weight}, "std": {node_type: std_weight}}
    """
    type_weight_accum: dict[str, list[float]] = defaultdict(list)

    for ckpt_path in checkpoints:
        model = load_model_from_checkpoint(ckpt_path, config, graphs[0])
        model.to(device)
        model.eval()

        loader = DataLoader(graphs, batch_size=4, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                outputs = model(batch)
                tw = outputs.get("readout", {}).get("type_weights", {})
                for nt, weights in tw.items():
                    type_weight_accum[nt].extend(weights.cpu().tolist())

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "mean": {nt: float(np.mean(vals)) for nt, vals in type_weight_accum.items()},
        "std": {nt: float(np.std(vals)) for nt, vals in type_weight_accum.items()},
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Interpretability analysis pipeline")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--experiment-id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--topk", type=int, default=50, help="Top-k for ranking analysis")
    parser.add_argument("--topk-perturb", type=int, default=100,
                        help="Top-k nodes to evaluate with perturbation")
    parser.add_argument("--max-graphs-perturb", type=int, default=10,
                        help="Max graphs for perturbation analysis")
    parser.add_argument("--known-targets", type=str, default=None,
                        help="Path to known immune targets file (one gene per line)")
    parser.add_argument("--pathway-gmt", type=str, default=None,
                        help="Path to GMT file with pathway gene sets")
    parser.add_argument("--skip-perturbation", action="store_true",
                        help="Skip perturbation analysis (expensive)")
    parser.add_argument("--stability-threshold", type=float, default=0.6,
                        help="Min fraction of checkpoints for Tier 1")
    parser.add_argument("--delta-threshold", type=float, default=0.01,
                        help="Min perturbation delta_prob for Tier 1")
    args = parser.parse_args()

    project_root = discover_project_root(Path(__file__))
    sys.path.insert(0, str(project_root / "Experiment" / "core_code"))

    # Load config
    config_path = project_root / args.config
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    experiment_id = args.experiment_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine output directory
    if args.output_dir:
        output_dir = project_root / args.output_dir
    else:
        output_root = project_root / config.get(
            "output_root", "outputs/results"
        )
        output_dir = output_root / experiment_id / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graphs
    graphs_dir = project_root / config["input"]["graphs_dir"]
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    if not graph_paths:
        raise SystemExit(f"No graph files found under {graphs_dir}")
    graphs = [torch.load(p, weights_only=False) for p in graph_paths]
    logger.info(f"Loaded {len(graphs)} graphs from {graphs_dir}")

    # Apply ablation transforms if configured
    ablation_cfg = config.get("ablation", {})
    transform_names = ablation_cfg.get("transforms", [])
    if transform_names:
        from models.graph_transforms import apply_ablation_transforms
        graphs = [apply_ablation_transforms(g, transform_names) for g in graphs]
        logger.info(f"Applied ablation transforms: {transform_names}")

    # Find checkpoints
    exp_result_dir = (
        project_root
        / config.get("output_root", "outputs/results")
        / experiment_id
    )
    checkpoints = find_checkpoints(exp_result_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {exp_result_dir}/checkpoints/")
    logger.info(f"Found {len(checkpoints)} checkpoints")

    # Load external resources
    known_targets = load_known_targets(
        Path(args.known_targets) if args.known_targets else None
    )
    pathway_gene_sets = load_gmt(
        Path(args.pathway_gmt) if args.pathway_gmt else None
    )

    # ========================================================================
    # Phase 1: Collect rankings from all checkpoints
    # ========================================================================
    logger.info("=== Phase 1: Collecting rankings from checkpoints ===")
    all_rankings = collect_rankings_from_checkpoints(
        checkpoints, config, graphs, device, topk=args.topk,
    )

    # Build consensus ranking
    consensus_rankings = build_consensus_ranking(all_rankings)

    # ========================================================================
    # Phase 2: Ranking stability metrics
    # ========================================================================
    logger.info("=== Phase 2: Computing ranking stability metrics ===")
    stability_metrics = compute_stability_metrics(all_rankings, topk=args.topk)
    per_node_stability = compute_per_node_stability(all_rankings, topk=args.topk)

    for nt, metrics in stability_metrics.items():
        logger.info(
            f"  {nt}: Jaccard@{args.topk}={metrics['topk_jaccard']:.4f}, "
            f"Spearman={metrics['spearman_rho']:.4f}, "
            f"checkpoints={metrics['n_checkpoints']}"
        )

    # ========================================================================
    # Phase 3: Type weights analysis
    # ========================================================================
    logger.info("=== Phase 3: Collecting type gating weights ===")
    type_weights = collect_type_weights(checkpoints, config, graphs, device)
    for nt, mean_w in type_weights["mean"].items():
        logger.info(f"  {nt}: mean_weight={mean_w:.4f} ± {type_weights['std'].get(nt, 0):.4f}")

    # ========================================================================
    # Phase 4: Perturbation sensitivity (optional)
    # ========================================================================
    perturbation_results: dict[str, dict[str, dict[str, float]]] = {"gene": {}, "pathway": {}}
    if not args.skip_perturbation:
        logger.info("=== Phase 4: Perturbation sensitivity analysis ===")
        perturbation_results = run_perturbation_analysis(
            checkpoints=checkpoints,
            config=config,
            graphs=graphs,
            consensus_rankings=consensus_rankings,
            device=device,
            topk_perturb=args.topk_perturb,
            max_graphs=args.max_graphs_perturb,
        )
        for nt, pert_dict in perturbation_results.items():
            if pert_dict:
                mean_delta = np.mean([v["delta_prob_mean"] for v in pert_dict.values()])
                logger.info(f"  {nt}: {len(pert_dict)} nodes evaluated, mean delta_prob={mean_delta:.6f}")
    else:
        logger.info("=== Phase 4: Perturbation analysis SKIPPED ===")

    # ========================================================================
    # Phase 5: Biological agreement
    # ========================================================================
    logger.info("=== Phase 5: Biological agreement analysis ===")

    # Known target overlap (gene-level only)
    gene_overlap = compute_known_target_overlap(
        consensus_rankings.get("gene", []),
        known_targets,
        k=args.topk,
    )
    logger.info(
        f"  Known target overlap@{args.topk}: "
        f"{gene_overlap['overlap_count']}/{args.topk} "
        f"({gene_overlap['overlap_fraction']:.2%})"
    )
    if gene_overlap["overlapping_genes"]:
        logger.info(f"  Overlapping genes: {', '.join(gene_overlap['overlapping_genes'][:10])}")

    # Pathway enrichment (gene-level)
    all_gene_ids = set(nid for nid, _ in consensus_rankings.get("gene", []))
    enrichment_results = compute_pathway_enrichment(
        consensus_rankings.get("gene", []),
        pathway_gene_sets,
        background_size=max(len(all_gene_ids), 1),
        topk=args.topk,
    )
    sig_enrichments = [e for e in enrichment_results if e.get("fdr", 1.0) < 0.05]
    logger.info(f"  Significant pathway enrichments (FDR<0.05): {len(sig_enrichments)}")

    # ========================================================================
    # Phase 6: Tier assignment and output
    # ========================================================================
    logger.info("=== Phase 6: Tier assignment and output generation ===")

    tier_counts = {}
    for node_type in ("gene", "pathway"):
        tier_results = assign_tiers(
            consensus_ranking=consensus_rankings.get(node_type, []),
            per_node_stability=per_node_stability.get(node_type, {}),
            perturbation_results=perturbation_results.get(node_type, {}),
            known_targets=known_targets if node_type == "gene" else set(),
            topk=args.topk,
            stability_threshold=args.stability_threshold,
            delta_threshold=args.delta_threshold,
        )

        n_tier1 = sum(1 for t in tier_results if t["tier"] == "Tier 1")
        n_tier2 = sum(1 for t in tier_results if t["tier"] == "Tier 2")
        tier_counts[node_type] = {"tier1": n_tier1, "tier2": n_tier2}
        logger.info(f"  {node_type}: Tier 1 = {n_tier1}, Tier 2 = {n_tier2}")

        # Write TSV
        if node_type == "gene":
            write_gene_ranking_tsv(
                tier_results,
                output_dir / "gene_ranking.tsv",
                enrichment_results=enrichment_results,
            )
        else:
            write_pathway_ranking_tsv(
                tier_results,
                output_dir / "pathway_ranking.tsv",
            )

    # Write enrichment results
    if enrichment_results:
        write_enrichment_tsv(enrichment_results, output_dir / "pathway_enrichment.tsv")

    # Write comprehensive summary
    write_summary_json(
        stability_metrics=stability_metrics,
        consensus_rankings=consensus_rankings,
        overlap_results=gene_overlap,
        enrichment_results=enrichment_results,
        perturbation_summary=perturbation_results,
        tier_counts=tier_counts,
        output_path=output_dir / "interpretability_summary.json",
    )

    # Write type weights
    type_weights_path = output_dir / "type_weights.json"
    with type_weights_path.open("w", encoding="utf-8") as f:
        json.dump(type_weights, f, indent=2)
    logger.info(f"Type weights written to {type_weights_path}")

    logger.info("=== Interpretability analysis complete ===")


if __name__ == "__main__":
    main()
