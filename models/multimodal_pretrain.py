"""Multimodal contrastive pretraining: H&E image patches <-> spatial transcriptomics.

This module implements a CLIP-style contrastive pretraining framework that aligns
H&E histology patch representations with spatial transcriptomics gene expression
profiles. The pretrained encoder provides transferable features for downstream
TIME typing on heterogeneous graphs.

Architecture:
    - Image encoder: ViT-based (UNI / CTransPath / CONCH / vanilla ViT-B/16)
    - Expression encoder: gene-expression MLP or gene-sentence transformer
    - Projection heads: linear layers mapping to shared embedding space
    - Training objective: symmetric InfoNCE (CLIP loss)

Data sources:
    - ST-bank (~2.18M patches, 32 organs, OmiCLIP/Loki)
    - HEST-1k (~1,276 samples, 26 organs, multi-platform)

References:
    - OmiCLIP (Loki): Nature Methods 2025
    - HEST-1k: NeurIPS 2024 Spotlight
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temperature-scaled InfoNCE (CLIP) loss
# ---------------------------------------------------------------------------
class CLIPLoss(nn.Module):
    """Symmetric contrastive loss with learnable temperature."""

    def __init__(self, init_temp: float = 0.07, learnable: bool = True):
        super().__init__()
        self.log_temp = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temp)),
            requires_grad=learnable,
        )

    def forward(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            image_embeds: (B, D) L2-normalized image embeddings
            text_embeds: (B, D) L2-normalized text/expression embeddings

        Returns:
            Scalar loss value.
        """
        temperature = self.log_temp.exp().clamp(max=100.0)
        logits = image_embeds @ text_embeds.t() * temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2.0


# ---------------------------------------------------------------------------
# Expression encoder: MLP operating on raw gene expression vectors
# ---------------------------------------------------------------------------
class ExpressionEncoder(nn.Module):
    """Encode per-spot gene expression into a fixed-dimensional embedding.

    Supports two modes:
        1. "vector": raw expression vector -> MLP -> embedding
        2. "gene_sentence": top-k gene names -> tokenizer -> transformer

    For pretraining with ST-bank, mode="gene_sentence" mirrors OmiCLIP.
    For pretraining with HEST-1k raw counts, mode="vector" is more flexible.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dim: int = 512,
        embed_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
        mode: str = "vector",
    ):
        super().__init__()
        self.mode = mode
        if mode == "vector":
            layers = []
            dim_in = input_dim
            for i in range(n_layers - 1):
                layers.extend([
                    nn.Linear(dim_in, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                dim_in = hidden_dim
            layers.append(nn.Linear(dim_in, embed_dim))
            self.encoder = nn.Sequential(*layers)
        else:
            raise NotImplementedError(
                "gene_sentence mode requires a text tokenizer; "
                "use Loki's built-in CoCa tokenizer for that path."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode expression to embedding.

        Args:
            x: (B, input_dim) gene expression vector (log-normalized or raw)

        Returns:
            (B, embed_dim) L2-normalized embedding
        """
        h = self.encoder(x)
        return F.normalize(h, dim=-1)


# ---------------------------------------------------------------------------
# Image encoder wrapper
# ---------------------------------------------------------------------------
class ImageEncoderWrapper(nn.Module):
    """Wrap a pretrained vision encoder (ViT) for patch-level features.

    Supports loading from:
        - timm (e.g., vit_base_patch16_224)
        - UNI (https://huggingface.co/MahmoodLab/UNI)
        - CTransPath
        - CONCH

    The encoder output is projected to a shared embedding space.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        embed_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone.startswith("uni"):
            self._init_uni(embed_dim)
        else:
            self._init_timm(backbone, embed_dim, pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_timm(self, backbone: str, embed_dim: int, pretrained: bool) -> None:
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.projector = nn.Linear(feat_dim, embed_dim)

    def _init_uni(self, embed_dim: int) -> None:
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.projector = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image patch to embedding.

        Args:
            x: (B, 3, 224, 224) image tensor

        Returns:
            (B, embed_dim) L2-normalized embedding
        """
        h = self.backbone(x)
        h = self.projector(h)
        return F.normalize(h, dim=-1)


# ---------------------------------------------------------------------------
# Full multimodal pretraining model
# ---------------------------------------------------------------------------
class MultimodalPretrainModel(nn.Module):
    """CLIP-style multimodal pretraining: H&E patches <-> gene expression.

    This model learns aligned representations between histology image patches
    and spatial transcriptomics gene expression profiles. The learned image
    encoder can then provide morphology-informed node features for downstream
    heterogeneous graph TIME typing.

    Training loop (external):
        1. Sample a batch of (image_patch, expression_vector) pairs
        2. Forward through both encoders
        3. Compute CLIPLoss
        4. Backpropagate
    """

    def __init__(
        self,
        image_backbone: str = "vit_base_patch16_224",
        expr_input_dim: int = 2000,
        embed_dim: int = 512,
        expr_hidden_dim: int = 512,
        expr_n_layers: int = 3,
        expr_dropout: float = 0.1,
        init_temp: float = 0.07,
        freeze_image_backbone: bool = False,
    ):
        super().__init__()
        self.image_encoder = ImageEncoderWrapper(
            backbone=image_backbone,
            embed_dim=embed_dim,
            pretrained=True,
            freeze_backbone=freeze_image_backbone,
        )
        self.expr_encoder = ExpressionEncoder(
            input_dim=expr_input_dim,
            hidden_dim=expr_hidden_dim,
            embed_dim=embed_dim,
            n_layers=expr_n_layers,
            dropout=expr_dropout,
            mode="vector",
        )
        self.loss_fn = CLIPLoss(init_temp=init_temp, learnable=True)

    def forward(
        self,
        images: torch.Tensor,
        expressions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for contrastive pretraining.

        Args:
            images: (B, 3, 224, 224)
            expressions: (B, expr_input_dim) log-normalized gene expression

        Returns:
            dict with keys: loss, image_embeds, expr_embeds, temperature
        """
        img_emb = self.image_encoder(images)
        expr_emb = self.expr_encoder(expressions)
        loss = self.loss_fn(img_emb, expr_emb)
        return {
            "loss": loss,
            "image_embeds": img_emb,
            "expr_embeds": expr_emb,
            "temperature": self.loss_fn.log_temp.exp().item(),
        }

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images only (for downstream feature extraction)."""
        return self.image_encoder(images)

    def encode_expressions(self, expressions: torch.Tensor) -> torch.Tensor:
        """Encode expressions only (for downstream feature extraction)."""
        return self.expr_encoder(expressions)


# ---------------------------------------------------------------------------
# Pretraining dataset
# ---------------------------------------------------------------------------
class STBankDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for ST-bank image-expression pairs.

    Reads the downloaded ST-bank data:
        - images/ directory with PNG patches
        - text.csv with gene sentences

    For "vector" mode pretraining, gene sentences are converted back to
    expression vectors using a gene vocabulary.
    """

    def __init__(
        self,
        data_dir: str,
        gene_vocab: Optional[dict[str, int]] = None,
        vocab_size: int = 2000,
        transform=None,
        mode: str = "gene_sentence",
    ):
        import csv
        from pathlib import Path

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        self.gene_vocab = gene_vocab
        self.vocab_size = vocab_size

        # Detect image directory (may be images/ or images/image/)
        img_base = self.data_dir / "images"
        if (img_base / "image").is_dir():
            self.img_dir = img_base / "image"
        else:
            self.img_dir = img_base

        # Load text.csv — columns: (sample_id, Gene Sentence, img_idx, img_path)
        text_path = self.data_dir / "text.csv"
        self.pairs: list[tuple[str, str, str]] = []  # (patch_id, gene_sentence, img_path)
        with open(text_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    patch_id, gene_sentence, img_idx, img_path = row[0], row[1], row[2], row[3]
                    self.pairs.append((patch_id, gene_sentence, img_path))
                elif len(row) >= 2:
                    self.pairs.append((row[0], row[1], ""))

        # Auto-build gene vocab if not provided
        if self.gene_vocab is None:
            print("Building gene vocabulary from ST-bank...")
            from collections import Counter
            gene_counter: Counter = Counter()
            for _, gene_sentence, _ in self.pairs:
                for gene in gene_sentence.strip().split():
                    gene_counter[gene] += 1
            top_genes = [g for g, _ in gene_counter.most_common(self.vocab_size)]
            self.gene_vocab = {g: i for i, g in enumerate(top_genes)}
            print(f"  Built vocab with {len(self.gene_vocab)} genes from {len(gene_counter)} total unique genes")

        print(f"Loaded {len(self.pairs):,} image-expression pairs from ST-bank")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from PIL import Image

        patch_id, gene_sentence, img_filename = self.pairs[idx]

        # Load image — try img_path column first, then fallback patterns
        img_path = self.img_dir / f"{img_filename}.png" if img_filename else None
        if img_path is None or not img_path.exists():
            img_path = self.img_dir / f"{patch_id}_hires.png"
        if not img_path.exists():
            img_path = self.img_dir / f"{patch_id}.png"

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image = default_transform(image)

        # Convert gene sentence to expression vector
        if self.mode == "gene_sentence" and self.gene_vocab is not None:
            genes = gene_sentence.strip().split()
            expr_vec = torch.zeros(self.vocab_size)
            for rank, gene in enumerate(genes):
                if gene in self.gene_vocab:
                    # Assign decreasing weight by rank
                    expr_vec[self.gene_vocab[gene]] = len(genes) - rank
            # L2-normalize
            norm = expr_vec.norm()
            if norm > 0:
                expr_vec = expr_vec / norm
        else:
            expr_vec = torch.zeros(self.vocab_size)

        return {"image": image, "expression": expr_vec, "sample_id": patch_id}


class HEST1kDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for HEST-1k image-expression pairs.

    Reads downloaded HEST-1k data:
        - patches/*.h5 with keys: img (N,224,224,3), barcode (N,1), coords (N,2)
        - st/*.h5ad (AnnData with expression + spatial coords)

    Barcodes are used to align patches with expression profiles. Expression
    vectors are pre-computed per sample during __init__ using a shared gene
    vocabulary (built from top HVGs across samples), so __getitem__ only needs
    to load the image patch from h5.
    """

    def __init__(
        self,
        data_dir: str,
        gene_vocab: Optional[dict[str, int]] = None,
        vocab_size: int = 2000,
        sample_ids: Optional[list[str]] = None,
        transform=None,
    ):
        from pathlib import Path
        import h5py
        import anndata
        import numpy as np
        from collections import Counter

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.vocab_size = vocab_size

        st_dir = self.data_dir / "st"
        patches_dir = self.data_dir / "patches"

        if sample_ids:
            h5ad_files = [st_dir / f"{sid}.h5ad" for sid in sample_ids
                          if (st_dir / f"{sid}.h5ad").exists()]
        else:
            h5ad_files = sorted(st_dir.glob("*.h5ad"))

        # --- Pass 1: build shared gene vocabulary if not provided ---
        if gene_vocab is None:
            print("Building gene vocabulary from HEST-1k samples...")
            gene_counter: Counter = Counter()
            for h5ad_path in h5ad_files:
                adata = anndata.read_h5ad(h5ad_path, backed="r")
                for g in adata.var_names:
                    gene_counter[g] += 1
                adata.file.close()
            top_genes = [g for g, _ in gene_counter.most_common(vocab_size)]
            gene_vocab = {g: i for i, g in enumerate(top_genes)}
            print(f"  Built vocab with {len(gene_vocab)} genes from "
                  f"{len(gene_counter)} unique across {len(h5ad_files)} samples")
        self.gene_vocab = gene_vocab

        # --- Pass 2: precompute expression vectors and build index ---
        # entries: list of (sample_id, local_patch_idx) for __getitem__
        # expr_cache: {sample_id: (n_matched, vocab_size) tensor}
        # patch_files: {sample_id: path to .h5}
        self.entries: list[tuple[str, int]] = []
        self.expr_cache: dict[str, torch.Tensor] = {}
        self.patch_files: dict[str, str] = {}

        skipped_samples = 0
        total_matched = 0

        for h5ad_path in h5ad_files:
            sid = h5ad_path.stem
            patch_path = patches_dir / f"{sid}.h5"
            if not patch_path.exists():
                skipped_samples += 1
                continue

            # Read h5ad obs barcodes and expression
            adata = anndata.read_h5ad(h5ad_path)
            obs_barcodes = list(adata.obs.index)

            # Read patch barcodes from h5
            with h5py.File(patch_path, "r") as hf:
                n_patches = hf["img"].shape[0]
                raw_bc = hf["barcode"][:, 0]
                patch_barcodes = [
                    b.decode("utf-8") if isinstance(b, bytes) else str(b)
                    for b in raw_bc
                ]

            # Build barcode -> obs row mapping
            obs_bc_map = {bc: i for i, bc in enumerate(obs_barcodes)}

            # Match: for each patch, find expression row
            matched_patch_idxs = []
            matched_obs_idxs = []
            for pi, pbc in enumerate(patch_barcodes):
                if pbc in obs_bc_map:
                    matched_patch_idxs.append(pi)
                    matched_obs_idxs.append(obs_bc_map[pbc])

            if not matched_patch_idxs:
                skipped_samples += 1
                continue

            # Extract expression matrix for matched spots
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)

            # Map to shared vocabulary
            var_to_vocab = {}
            for vi, gname in enumerate(adata.var_names):
                if gname in self.gene_vocab:
                    var_to_vocab[vi] = self.gene_vocab[gname]

            expr_mat = np.zeros((len(matched_obs_idxs), vocab_size), dtype=np.float32)
            for row_i, obs_i in enumerate(matched_obs_idxs):
                for var_i, vocab_i in var_to_vocab.items():
                    expr_mat[row_i, vocab_i] = X[obs_i, var_i]

            # Log-normalize and L2-normalize
            expr_mat = np.log1p(expr_mat)
            norms = np.linalg.norm(expr_mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            expr_mat = expr_mat / norms

            self.expr_cache[sid] = torch.from_numpy(expr_mat)
            self.patch_files[sid] = str(patch_path)

            # Store (sample_id, index_within_matched) and save the patch indices
            # We need to map local index -> actual h5 patch index
            if not hasattr(self, '_patch_idx_map'):
                self._patch_idx_map: dict[str, list[int]] = {}
            self._patch_idx_map[sid] = matched_patch_idxs

            for local_i in range(len(matched_patch_idxs)):
                self.entries.append((sid, local_i))

            total_matched += len(matched_patch_idxs)

        n_samples = len(self.expr_cache)
        print(f"Loaded {total_matched:,} matched pairs from {n_samples} HEST-1k samples "
              f"(skipped {skipped_samples})")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import h5py
        from PIL import Image

        sid, local_i = self.entries[idx]
        h5_patch_idx = self._patch_idx_map[sid][local_i]

        # Load patch from h5
        with h5py.File(self.patch_files[sid], "r") as hf:
            patch = hf["img"][h5_patch_idx]  # (224, 224, 3) uint8

        image = Image.fromarray(patch)
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image = default_transform(image)

        expr_vec = self.expr_cache[sid][local_i]

        return {"image": image, "expression": expr_vec, "sample_id": sid}
