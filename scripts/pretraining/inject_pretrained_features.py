#!/usr/bin/env python3
"""Inject pretrained multimodal encoder features into existing HeteroData graphs.

This script bridges Phase 1 (multimodal pretraining) and Phase 2 (HGT-TIME):

1. Loads a pretrained MultimodalPretrainModel checkpoint.
2. For each HeteroData graph .pt file, extracts the H&E patch images
   associated with each cell/spot node.
3. Encodes the patches through the pretrained image encoder.
4. Augments the cell node features by concatenating the pretrained embeddings
   (or replacing, depending on --mode).
5. Writes the augmented graph .pt files to a new directory.

The augmented graphs can be directly loaded by train_eval_pipeline.py — the
HGTTimeModel auto-detects input_dims from the graph's node feature dimensions.

Usage:
    python inject_pretrained_features.py \\
        --checkpoint outputs/pretrain/best_model.pt \\
        --graphs-dir outputs/hetero_graph/visium_breast__hetero_v1/graphs \\
        --patches-dir data/visium_patches \\
        --output-dir outputs/hetero_graph/visium_breast__pretrain_aug/graphs \\
        --mode concat
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


def build_image_transform() -> transforms.Compose:
    """Standard image preprocessing for ViT-based models."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_pretrained_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load the pretrained image encoder from a MultimodalPretrainModel checkpoint.

    Returns the image encoder submodule in eval mode.
    """
    from models.multimodal_pretrain import MultimodalPretrainModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model config from checkpoint
    config = ckpt.get("config", {})
    model = MultimodalPretrainModel(
        image_backbone=config.get("image_backbone", "vit_base_patch16_224"),
        expr_input_dim=config.get("expr_input_dim", 2000),
        embed_dim=config.get("embed_dim", 512),
    )

    # Load weights — handle both full model and state_dict-only checkpoints
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Assume the checkpoint IS the state dict
        model.load_state_dict(ckpt)

    encoder = model.image_encoder
    encoder.eval()
    encoder.to(device)
    return encoder


def extract_patches_for_graph(
    graph,
    patches_dir: Path,
    graph_path: Path,
) -> Optional[list[Path]]:
    """Find H&E patch image files corresponding to each cell node in a graph.

    The mapping depends on how graphs were constructed. We support several
    naming conventions:
        1. graph has `cell.patch_path` attribute (list of paths)
        2. graph has `cell.spot_id` or `cell.node_id` attributes; patches
           are named <spot_id>.png under patches_dir
        3. graph has `graph_id`; patches are in patches_dir/<graph_id>/
    """
    n_cells = graph["cell"].x.size(0) if "cell" in graph.node_types else 0
    if n_cells == 0:
        return None

    # Method 1: explicit patch_path attribute
    if hasattr(graph["cell"], "patch_path"):
        paths = graph["cell"].patch_path
        if isinstance(paths, (list, tuple)):
            return [Path(p) for p in paths]

    # Method 2: spot_id-based lookup
    node_ids = None
    for attr in ("spot_id", "node_id", "cell_id"):
        if hasattr(graph["cell"], attr):
            node_ids = getattr(graph["cell"], attr)
            break

    if node_ids is not None:
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.tolist()
        elif isinstance(node_ids, str):
            node_ids = [node_ids]

        patch_paths = []
        for nid in node_ids:
            p = patches_dir / f"{nid}.png"
            if not p.exists():
                p = patches_dir / f"{nid}.jpg"
            patch_paths.append(p)
        return patch_paths

    # Method 3: graph_id-based directory
    graph_id = getattr(graph, "graph_id", graph_path.stem)
    if isinstance(graph_id, (list, tuple)):
        graph_id = graph_id[0] if graph_id else graph_path.stem
    graph_patch_dir = patches_dir / str(graph_id)
    if graph_patch_dir.is_dir():
        patch_files = sorted(graph_patch_dir.glob("*.png")) + sorted(graph_patch_dir.glob("*.jpg"))
        return patch_files[:n_cells]

    return None


@torch.no_grad()
def encode_patches(
    encoder: torch.nn.Module,
    patch_paths: list[Path],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode a list of patch images through the pretrained encoder.

    Returns (N, embed_dim) tensor on CPU.
    Missing patches get zero embeddings.
    """
    from PIL import Image

    n = len(patch_paths)
    embeddings = []

    for start in range(0, n, batch_size):
        batch_paths = patch_paths[start : start + batch_size]
        images = []
        valid_mask = []
        for p in batch_paths:
            if p.exists():
                img = Image.open(p).convert("RGB")
                images.append(transform(img))
                valid_mask.append(True)
            else:
                # Placeholder — will be zeroed out
                images.append(torch.zeros(3, 224, 224))
                valid_mask.append(False)

        batch_tensor = torch.stack(images).to(device)
        emb = encoder(batch_tensor)  # (B, embed_dim), L2-normalized

        # Zero out missing patches
        for i, valid in enumerate(valid_mask):
            if not valid:
                emb[i] = 0.0

        embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)


def augment_graph(
    graph,
    pretrained_features: torch.Tensor,
    mode: str = "concat",
) -> None:
    """Augment cell node features in-place with pretrained embeddings.

    Args:
        graph: PyG HeteroData object
        pretrained_features: (n_cells, embed_dim) from pretrained encoder
        mode: "concat" appends to existing features; "replace" uses only
              pretrained features; "add_attr" stores as separate attribute
    """
    cell_store = graph["cell"]
    original_x = cell_store.x  # (n_cells, original_dim)

    if mode == "concat":
        cell_store.x = torch.cat([original_x, pretrained_features], dim=-1)
    elif mode == "replace":
        cell_store.x = pretrained_features
    elif mode == "add_attr":
        cell_store.pretrained_emb = pretrained_features
    else:
        raise ValueError(f"Unknown mode: {mode}. Use concat, replace, or add_attr.")


def main():
    parser = argparse.ArgumentParser(description="Inject pretrained features into HeteroData graphs")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--graphs-dir", required=True, help="Directory with .pt graph files")
    parser.add_argument("--patches-dir", required=True, help="Directory with H&E patch images")
    parser.add_argument("--output-dir", required=True, help="Output directory for augmented graphs")
    parser.add_argument("--mode", default="concat", choices=["concat", "replace", "add_attr"],
                        help="How to combine pretrained features with existing cell features")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding patches")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    graphs_dir = Path(args.graphs_dir)
    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder
    logger.info(f"Loading pretrained encoder from {args.checkpoint}")
    encoder = load_pretrained_encoder(args.checkpoint, device)
    transform = build_image_transform()

    # Process each graph
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    logger.info(f"Found {len(graph_paths)} graph files in {graphs_dir}")

    augmented_count = 0
    skipped_count = 0

    for gpath in graph_paths:
        graph = torch.load(gpath, weights_only=False)

        patch_paths = extract_patches_for_graph(graph, patches_dir, gpath)
        if patch_paths is None:
            # No patches available — copy graph as-is
            logger.warning(f"No patches found for {gpath.name}, copying without augmentation")
            torch.save(graph, output_dir / gpath.name)
            skipped_count += 1
            continue

        n_cells = graph["cell"].x.size(0)
        if len(patch_paths) < n_cells:
            # Pad with missing paths
            patch_paths.extend([Path("/dev/null")] * (n_cells - len(patch_paths)))
        elif len(patch_paths) > n_cells:
            patch_paths = patch_paths[:n_cells]

        # Encode patches
        pretrained_features = encode_patches(
            encoder, patch_paths, transform, device, batch_size=args.batch_size
        )

        # Augment graph
        augment_graph(graph, pretrained_features, mode=args.mode)

        # Save
        torch.save(graph, output_dir / gpath.name)
        new_dim = graph["cell"].x.size(-1)
        augmented_count += 1
        logger.info(f"  {gpath.name}: cell.x {n_cells}x{new_dim} (mode={args.mode})")

    logger.info(f"Done. Augmented: {augmented_count}, Skipped: {skipped_count}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
