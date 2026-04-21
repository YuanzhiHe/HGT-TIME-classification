#!/usr/bin/env python3
"""Inject pretrained expression encoder features into HeteroData graphs.

Unlike inject_pretrained_features.py (which uses the image encoder and needs
H&E patches), this script uses the **expression encoder** branch of the
pretrained CLIP model.  For each cell node it reconstructs a gene-expression
vector aligned with the pretraining vocabulary from the graph's
(cell, expresses, gene) edges, then passes it through the frozen expression
encoder to obtain a 256-dim (or whatever embed_dim) L2-normalised embedding.

Those embeddings are concatenated (or replace, etc.) onto cell.x so that
downstream HGT-TIME auto-detects the larger input dimension.

Usage:
    python inject_expr_features.py \
        --checkpoint outputs/pretrain/stbank/best_pretrain.pt \
        --graphs-dir  outputs/hetero_graph/visium_breast_regions__hetero_v1/graphs \
        --output-dir  outputs/hetero_graph/visium_breast_regions__pretrain_aug/graphs \
        --stbank-dir  data/stbank \
        --mode concat
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.multimodal_pretrain import MultimodalPretrainModel, STBankDataset

logger = logging.getLogger(__name__)


def load_pretrained_expr_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load expression encoder + config from pretrained checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    model = MultimodalPretrainModel(
        image_backbone=model_cfg.get("image_backbone", "vit_small_patch16_224"),
        expr_input_dim=config.get("data", {}).get("n_genes", 2000),
        embed_dim=model_cfg.get("embed_dim", 256),
        expr_hidden_dim=model_cfg.get("expr_hidden_dim", 256),
        expr_n_layers=model_cfg.get("expr_n_layers", 3),
        expr_dropout=model_cfg.get("expr_dropout", 0.1),
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    encoder = model.expr_encoder
    encoder.eval()
    encoder.to(device)
    return encoder, config


def build_gene_vocab(stbank_dir: str, vocab_size: int = 2000) -> dict[str, int]:
    """Rebuild the same gene vocabulary that was used during pretraining."""
    ds = STBankDataset(
        data_dir=stbank_dir,
        vocab_size=vocab_size,
        transform=None,
        mode="gene_sentence",
    )
    return ds.gene_vocab  # {gene_name: index}


def reconstruct_expression_vector(
    graph,
    gene_vocab: dict[str, int],
    vocab_size: int = 2000,
) -> torch.Tensor:
    """Reconstruct a (n_cells, vocab_size) expression matrix for cell nodes.

    Uses the (cell, expresses, gene) edges.  For each cell-gene pair the edge
    weight is placed at the gene's position in the pretraining vocabulary.
    Genes not in the vocabulary are skipped.
    """
    n_cells = graph["cell"].x.size(0)
    expr_mat = torch.zeros(n_cells, vocab_size)

    gene_names = graph["gene"].node_id  # list of gene name strings
    edge_index = graph["cell", "expresses", "gene"].edge_index  # (2, E)
    edge_weight = graph["cell", "expresses", "gene"].edge_weight  # (E,)

    # Build graph-local gene index -> vocab index mapping
    local_to_vocab = {}
    mapped = 0
    for local_idx, gname in enumerate(gene_names):
        if gname in gene_vocab:
            local_to_vocab[local_idx] = gene_vocab[gname]
            mapped += 1

    # Fill expression matrix
    for e in range(edge_index.size(1)):
        cell_idx = edge_index[0, e].item()
        gene_local_idx = edge_index[1, e].item()
        if gene_local_idx in local_to_vocab:
            vocab_idx = local_to_vocab[gene_local_idx]
            expr_mat[cell_idx, vocab_idx] = edge_weight[e].item()

    return expr_mat, mapped, len(gene_names)


@torch.no_grad()
def encode_expressions(
    encoder: torch.nn.Module,
    expr_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode expression matrix through pretrained expression encoder."""
    n = expr_matrix.size(0)
    embeddings = []
    for start in range(0, n, batch_size):
        batch = expr_matrix[start : start + batch_size].to(device)
        emb = encoder(batch)  # (B, embed_dim), L2-normalized
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def project_features(features: torch.Tensor, target_dim: int, seed: int = 42) -> torch.Tensor:
    """Reduce pretrained feature dimensionality via a fixed random projection.

    Uses a seeded random linear projection followed by L2 normalization.
    This is deterministic given the same seed and input shape.
    """
    src_dim = features.size(-1)
    if src_dim == target_dim:
        return features
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(src_dim, target_dim, generator=gen) / (src_dim ** 0.5)
    projected = features @ W
    return F.normalize(projected, dim=-1)


def augment_graph(graph, pretrained_features: torch.Tensor, mode: str = "concat"):
    """Augment cell node features in-place."""
    original_x = graph["cell"].x
    if mode == "concat":
        graph["cell"].x = torch.cat([original_x, pretrained_features], dim=-1)
    elif mode == "replace":
        graph["cell"].x = pretrained_features
    elif mode == "add_attr":
        graph["cell"].pretrained_emb = pretrained_features
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Inject pretrained expression embeddings into HeteroData graphs"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graphs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stbank-dir", required=True,
                        help="ST-bank data dir (to rebuild gene vocab)")
    parser.add_argument("--mode", default="concat",
                        choices=["concat", "replace", "add_attr"])
    parser.add_argument("--project-dim", type=int, default=0,
                        help="Project pretrained features to this dim before concat (0=no projection)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    graphs_dir = Path(args.graphs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained expression encoder
    logger.info(f"Loading pretrained expression encoder from {args.checkpoint}")
    encoder, config = load_pretrained_expr_encoder(args.checkpoint, device)
    vocab_size = config.get("data", {}).get("n_genes", 2000)

    # Rebuild gene vocabulary (same one used during pretraining)
    logger.info(f"Rebuilding gene vocabulary from {args.stbank_dir} (size={vocab_size})")
    gene_vocab = build_gene_vocab(args.stbank_dir, vocab_size)
    logger.info(f"Gene vocabulary: {len(gene_vocab)} genes")

    # Process graphs
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    logger.info(f"Found {len(graph_paths)} graph files")

    total_mapped = 0
    total_genes = 0
    augmented = 0

    for gpath in graph_paths:
        graph = torch.load(gpath, weights_only=False, map_location="cpu")

        if "cell" not in graph.node_types or graph["cell"].x.size(0) == 0:
            logger.warning(f"Skipping {gpath.name}: no cell nodes")
            torch.save(graph, output_dir / gpath.name)
            continue

        # Reconstruct expression vectors
        expr_mat, mapped, n_genes = reconstruct_expression_vector(
            graph, gene_vocab, vocab_size
        )
        total_mapped += mapped
        total_genes += n_genes

        # Encode through pretrained expression encoder
        pretrained_features = encode_expressions(encoder, expr_mat, device)

        # Optionally project to lower dimension
        if args.project_dim > 0:
            pretrained_features = project_features(pretrained_features, args.project_dim)

        # Augment graph
        old_dim = graph["cell"].x.size(-1)
        augment_graph(graph, pretrained_features, mode=args.mode)
        new_dim = graph["cell"].x.size(-1)

        torch.save(graph, output_dir / gpath.name)
        augmented += 1

        if augmented <= 3 or augmented % 10 == 0:
            logger.info(
                f"  {gpath.name}: cell.x {old_dim}->{new_dim}, "
                f"genes mapped {mapped}/{n_genes}"
            )

    logger.info(f"Done. Augmented {augmented} graphs.")
    logger.info(
        f"Average gene coverage: {total_mapped/max(augmented,1):.0f}/"
        f"{total_genes/max(augmented,1):.0f} "
        f"({total_mapped/max(total_genes,1)*100:.1f}%)"
    )
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
