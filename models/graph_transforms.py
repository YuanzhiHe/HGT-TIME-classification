"""Graph-level ablation transforms for HeteroData objects.

Each transform takes a PyG HeteroData graph and returns a modified copy
suitable for a specific ablation experiment. Transforms are composable
and applied at data-load time (before batching), so the model architecture
itself remains unchanged.

Usage in config YAML:
    ablation:
      transforms:
        - drop_spatial_edges
        - drop_pathway_nodes

Usage in code:
    from models.graph_transforms import apply_ablation_transforms
    graphs = [apply_ablation_transforms(g, transform_names) for g in graphs]
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


# ============================================================
# H1: Spatial information ablations
# ============================================================

def drop_spatial_edges(data: Any) -> Any:
    """Remove all cell-cell spatial edges.

    Tests H1: whether spatial adjacency is necessary for TIME classification,
    especially Hot vs. Excluded distinction.
    """
    data = copy.copy(data)
    to_remove = []
    for et in list(data.edge_types):
        if et[0] == "cell" and et[2] == "cell":
            to_remove.append(et)
    for et in to_remove:
        del data[et]
    return data


def permute_spatial_edges(data: Any) -> Any:
    """Randomly permute cell-cell spatial edge endpoints while preserving count.

    Tests H1: whether the benefit comes from *true* local spatial topology
    rather than just graph density / connectivity.
    """
    data = copy.copy(data)
    for et in list(data.edge_types):
        if et[0] == "cell" and et[2] == "cell":
            edge_store = data[et]
            ei = edge_store.edge_index
            num_nodes = data["cell"].x.size(0)
            num_edges = ei.size(1)
            # Permute: random source and target from the cell population
            perm_src = torch.randint(0, num_nodes, (num_edges,), device=ei.device)
            perm_dst = torch.randint(0, num_nodes, (num_edges,), device=ei.device)
            edge_store.edge_index = torch.stack([perm_src, perm_dst], dim=0)
    return data


# ============================================================
# H2: Heterogeneous schema ablations
# ============================================================

def drop_pathway_nodes(data: Any) -> Any:
    """Remove all pathway nodes and gene-pathway edges.

    Tests H2: whether pathway-level aggregation provides independent value
    beyond gene-level features and PPI edges.
    """
    data = copy.copy(data)
    # Remove pathway node store
    if "pathway" in data.node_types:
        del data["pathway"]
    # Remove edges involving pathway
    to_remove = []
    for et in list(data.edge_types):
        if "pathway" in et:
            to_remove.append(et)
    for et in to_remove:
        del data[et]
    return data


def drop_prior_edges(data: Any) -> Any:
    """Remove all gene-gene PPI (STRING) interaction edges.

    Tests H2: whether PPI prior edges regularize gene embeddings and
    stabilize target ranking, independent of pathway structure.
    """
    data = copy.copy(data)
    to_remove = []
    for et in list(data.edge_types):
        if et[0] == "gene" and et[2] == "gene":
            to_remove.append(et)
    for et in to_remove:
        del data[et]
    return data


def drop_gene_nodes(data: Any) -> Any:
    """Remove all gene nodes and all edges involving genes.

    Extreme ablation: tests whether cell-pathway direct edges plus
    spatial structure alone carry the signal.
    """
    data = copy.copy(data)
    if "gene" in data.node_types:
        del data["gene"]
    to_remove = []
    for et in list(data.edge_types):
        if "gene" in et:
            to_remove.append(et)
    for et in to_remove:
        del data[et]
    return data


def homogeneous_collapse(data: Any) -> Any:
    """Collapse heterogeneous graph into a homogeneous cell-only graph.

    Removes gene and pathway nodes; retains only cell nodes and
    cell-cell spatial edges. This tests whether the full heterogeneous
    schema adds value vs. a standard spatial GNN.

    Note: This transform is used with the HomogeneousGraphClassifier
    baseline, NOT with HGTTimeModel (which requires >=1 edge type).
    """
    data = copy.copy(data)
    # Remove non-cell node types
    for nt in list(data.node_types):
        if nt != "cell":
            del data[nt]
    # Remove non cell-cell edges
    to_remove = []
    for et in list(data.edge_types):
        if et[0] != "cell" or et[2] != "cell":
            to_remove.append(et)
    for et in to_remove:
        del data[et]
    return data


# ============================================================
# Robustness ablations
# ============================================================

def drop_ranking_targets(data: Any) -> Any:
    """Remove ranking supervision targets (gene/pathway positive masks).

    Tests whether the ranking loss auxiliary objective helps or hurts
    the main classification performance.
    """
    data = copy.copy(data)
    for nt in ("gene", "pathway"):
        if nt in data.node_types:
            store = data[nt]
            for attr in ("target_pos_mask", "target_weight"):
                if hasattr(store, attr):
                    delattr(store, attr)
    return data


def subsample_cells(data: Any, keep_ratio: float = 0.5) -> Any:
    """Randomly subsample a fraction of cell nodes and their edges.

    Tests robustness to cell count variation (e.g. tissue section depth).
    """
    data = copy.copy(data)
    if "cell" not in data.node_types:
        return data

    n_cells = data["cell"].x.size(0)
    n_keep = max(1, int(n_cells * keep_ratio))
    perm = torch.randperm(n_cells)[:n_keep].sort().values

    # Remap cell features
    data["cell"].x = data["cell"].x[perm]

    # Copy other cell attributes
    for attr_name in list(data["cell"].keys()):
        if attr_name == "x":
            continue
        attr_val = data["cell"][attr_name]
        if isinstance(attr_val, Tensor) and attr_val.size(0) == n_cells:
            data["cell"][attr_name] = attr_val[perm]

    # Remap edges involving cells
    old_to_new = torch.full((n_cells,), -1, dtype=torch.long)
    old_to_new[perm] = torch.arange(n_keep, dtype=torch.long)

    for et in list(data.edge_types):
        edge_store = data[et]
        ei = edge_store.edge_index
        src_is_cell = (et[0] == "cell")
        dst_is_cell = (et[2] == "cell")

        if src_is_cell:
            new_src = old_to_new[ei[0]]
            valid_src = new_src >= 0
        else:
            new_src = ei[0]
            valid_src = torch.ones(ei.size(1), dtype=torch.bool)

        if dst_is_cell:
            new_dst = old_to_new[ei[1]]
            valid_dst = new_dst >= 0
        else:
            new_dst = ei[1]
            valid_dst = torch.ones(ei.size(1), dtype=torch.bool)

        valid = valid_src & valid_dst
        if valid.any():
            edge_store.edge_index = torch.stack([
                new_src[valid] if src_is_cell else ei[0][valid],
                new_dst[valid] if dst_is_cell else ei[1][valid],
            ], dim=0)
        else:
            edge_store.edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data


# ============================================================
# Transform registry and application
# ============================================================

TRANSFORM_REGISTRY: dict[str, Callable] = {
    "drop_spatial_edges": drop_spatial_edges,
    "permute_spatial_edges": permute_spatial_edges,
    "drop_pathway_nodes": drop_pathway_nodes,
    "drop_prior_edges": drop_prior_edges,
    "drop_gene_nodes": drop_gene_nodes,
    "homogeneous_collapse": homogeneous_collapse,
    "drop_ranking_targets": drop_ranking_targets,
    "subsample_cells_50": lambda data: subsample_cells(data, keep_ratio=0.5),
    "subsample_cells_25": lambda data: subsample_cells(data, keep_ratio=0.25),
}


def apply_ablation_transforms(data: Any, transform_names: Sequence[str]) -> Any:
    """Apply a sequence of named ablation transforms to a graph.

    Args:
        data: A PyG HeteroData or Data object.
        transform_names: List of transform names from TRANSFORM_REGISTRY.

    Returns:
        Transformed graph (a shallow copy; original is not modified).
    """
    for name in transform_names:
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown ablation transform: '{name}'. "
                f"Available: {sorted(TRANSFORM_REGISTRY.keys())}"
            )
        data = TRANSFORM_REGISTRY[name](data)
    return data


def get_available_transforms() -> list[str]:
    """Return list of available transform names."""
    return sorted(TRANSFORM_REGISTRY.keys())
