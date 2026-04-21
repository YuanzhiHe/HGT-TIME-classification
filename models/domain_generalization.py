"""Patient-level and platform-level domain generalization for TIME typing.

This module implements domain-invariant learning strategies that enable the
HGT-TIME model to generalize across patients, tissue sections, and spatial
transcriptomics platforms (Visium, Visium HD, Xenium).

Three complementary strategies:

1. **Domain-Adversarial Training (DANN)**
   - A gradient-reversal layer forces the HGT encoder to learn domain-invariant
     representations by training a domain discriminator that the encoder must fool.
   - Domain labels: patient_id, platform_type, or section_id.

2. **Multi-Domain Batch Normalization (MDBN)**
   - Platform-specific normalization parameters absorb domain-specific statistics,
     while shared graph convolution weights capture domain-invariant patterns.

3. **Domain-Invariant Contrastive Regularization (DICR)**
   - Pulls same-class embeddings from different domains closer together;
     pushes different-class embeddings apart, regardless of domain.
   - Operates in the HGT embedding space after graph readout.

The "pretrain -> finetune -> OOD" closed loop:
    Phase 1: Multimodal contrastive pretraining on ST-bank/HEST-1k (multimodal_pretrain.py)
    Phase 2: Fine-tune HGT-TIME with domain generalization on labeled spatial cohorts
    Phase 3: Evaluate on held-out platforms/patients (OOD test)

References:
    - Ganin et al. "Domain-Adversarial Training of Neural Networks" JMLR 2016
    - Li et al. "Revisiting Batch Normalization for Practical Domain Adaptation" ICLR 2017 Workshop
    - Motiian et al. "Unified Deep Supervised Domain Adaptation and Generalization" ICCV 2017
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for DANN)
# ---------------------------------------------------------------------------
class GradientReversalFunction(Function):
    """Gradient reversal layer from Ganin et al. (2016)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Module wrapper for gradient reversal."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# Domain Discriminator
# ---------------------------------------------------------------------------
class DomainDiscriminator(nn.Module):
    """MLP domain classifier with gradient reversal.

    Predicts domain labels (patient, platform, or section) from graph embeddings.
    The gradient reversal layer ensures the upstream encoder learns domain-invariant
    features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_domains: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.grl = GradientReversal(alpha=alpha)

        layers = []
        dim_in = input_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, n_domains))
        self.classifier = nn.Sequential(*layers)

    def set_alpha(self, alpha: float) -> None:
        """Update GRL strength (schedule during training)."""
        self.grl.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict domain from graph embedding.

        Args:
            x: (B, input_dim) graph-level embedding

        Returns:
            (B, n_domains) domain logits
        """
        x = self.grl(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Multi-Domain Batch Normalization
# ---------------------------------------------------------------------------
class MultiDomainBatchNorm1d(nn.Module):
    """Platform-specific BatchNorm that shares affine parameters but uses
    domain-specific running statistics.

    During training, each domain's samples are normalized using their own
    running mean/variance. During evaluation, domain_id selects the appropriate
    statistics.
    """

    def __init__(self, num_features: int, n_domains: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.n_domains = n_domains
        self.num_features = num_features

        # Shared affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Per-domain running statistics
        self.register_buffer(
            "running_mean", torch.zeros(n_domains, num_features)
        )
        self.register_buffer(
            "running_var", torch.ones(n_domains, num_features)
        )
        self.register_buffer("num_batches_tracked", torch.zeros(n_domains, dtype=torch.long))

        self.eps = eps
        self.momentum = momentum

    def forward(self, x: torch.Tensor, domain_id: int = 0) -> torch.Tensor:
        """Apply domain-specific batch normalization.

        Args:
            x: (B, num_features)
            domain_id: integer identifying the platform/domain

        Returns:
            Normalized tensor of same shape.
        """
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean[domain_id] = (
                (1 - self.momentum) * self.running_mean[domain_id] + self.momentum * mean
            )
            self.running_var[domain_id] = (
                (1 - self.momentum) * self.running_var[domain_id] + self.momentum * var
            )
            self.num_batches_tracked[domain_id] += 1
        else:
            mean = self.running_mean[domain_id]
            var = self.running_var[domain_id]

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Domain-Invariant Contrastive Regularization (DICR)
# ---------------------------------------------------------------------------
class DomainInvariantContrastiveLoss(nn.Module):
    """Contrastive regularization that enforces domain invariance.

    For each anchor sample, positives are same-class samples from DIFFERENT
    domains; negatives are different-class samples regardless of domain.
    This encourages the embedding space to cluster by TIME class rather than
    by patient/platform.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        class_labels: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain-invariant contrastive loss.

        Args:
            embeddings: (B, D) L2-normalized graph embeddings
            class_labels: (B,) TIME class labels (0=Hot, 1=Excluded, 2=Cold)
            domain_labels: (B,) domain labels (patient_id, platform, section)

        Returns:
            Scalar loss.
        """
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Similarity matrix
        sim = embeddings @ embeddings.t() / self.temperature  # (B, B)

        # Masks
        class_match = class_labels.unsqueeze(0) == class_labels.unsqueeze(1)  # (B, B)
        domain_match = domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)  # (B, B)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)

        # Positives: same class, different domain
        pos_mask = class_match & ~domain_match & eye_mask
        # Negatives: different class (any domain)
        neg_mask = ~class_match & eye_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # For each anchor, compute InfoNCE over its positives vs all negatives
        loss = torch.tensor(0.0, device=embeddings.device)
        n_valid = 0
        for i in range(B):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_sim = sim[i, pos_indices]  # (n_pos,)
            neg_sim = sim[i, neg_indices]  # (n_neg,)

            # InfoNCE-style: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            for p in pos_sim:
                denom = p.exp() + neg_sim.exp().sum()
                loss = loss - (p - denom.log())
                n_valid += 1

        if n_valid > 0:
            loss = loss / n_valid
        return loss


# ---------------------------------------------------------------------------
# Domain-Aware GRL Schedule
# ---------------------------------------------------------------------------
def dann_alpha_schedule(epoch: int, max_epochs: int) -> float:
    """Progressive GRL strength schedule from Ganin et al.

    alpha = 2 / (1 + exp(-10 * p)) - 1,  where p = epoch / max_epochs
    """
    import math
    p = epoch / max(max_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ---------------------------------------------------------------------------
# Domain Generalization Wrapper for HGT-TIME
# ---------------------------------------------------------------------------
class DomainGeneralizedHGTTIME(nn.Module):
    """Wraps the base HGT-TIME model with domain generalization components.

    Adds:
        - Platform-level domain adversarial discriminator
        - Patient-level domain adversarial discriminator (optional)
        - Domain-invariant contrastive regularization
        - Multi-domain batch normalization (optional, applied to readout)

    The wrapper does NOT modify the base HGT encoder; it adds parallel
    branches that influence training through gradient reversal and
    auxiliary losses.
    """

    def __init__(
        self,
        base_model: nn.Module,
        graph_embed_dim: int = 128,
        n_platforms: int = 3,
        n_patients: int = 10,
        use_platform_dann: bool = True,
        use_patient_dann: bool = False,
        use_dicr: bool = True,
        use_mdbn: bool = False,
        dann_hidden_dim: int = 64,
        dann_n_layers: int = 2,
        dann_dropout: float = 0.1,
        dann_alpha: float = 1.0,
        dicr_temperature: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model

        # Platform-level domain adversarial
        self.use_platform_dann = use_platform_dann
        if use_platform_dann:
            self.platform_discriminator = DomainDiscriminator(
                input_dim=graph_embed_dim,
                hidden_dim=dann_hidden_dim,
                n_domains=n_platforms,
                n_layers=dann_n_layers,
                dropout=dann_dropout,
                alpha=dann_alpha,
            )

        # Patient-level domain adversarial
        self.use_patient_dann = use_patient_dann
        if use_patient_dann:
            self.patient_discriminator = DomainDiscriminator(
                input_dim=graph_embed_dim,
                hidden_dim=dann_hidden_dim,
                n_domains=n_patients,
                n_layers=dann_n_layers,
                dropout=dann_dropout,
                alpha=dann_alpha,
            )

        # Domain-invariant contrastive regularization
        self.use_dicr = use_dicr
        if use_dicr:
            self.dicr_loss = DomainInvariantContrastiveLoss(temperature=dicr_temperature)

        # Multi-domain batch normalization
        self.use_mdbn = use_mdbn
        if use_mdbn:
            self.mdbn = MultiDomainBatchNorm1d(graph_embed_dim, n_domains=n_platforms)

    def set_dann_alpha(self, alpha: float) -> None:
        """Update GRL strength for all discriminators."""
        if self.use_platform_dann:
            self.platform_discriminator.set_alpha(alpha)
        if self.use_patient_dann:
            self.patient_discriminator.set_alpha(alpha)

    def forward(
        self,
        batch,
        platform_labels: Optional[torch.Tensor] = None,
        patient_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with domain generalization.

        Args:
            batch: HeteroData batch for the base HGT-TIME model
            platform_labels: (B,) platform domain labels
            patient_labels: (B,) patient domain labels

        Returns:
            dict with base model outputs plus domain losses.
        """
        # Base model forward
        out = self.base_model(batch)

        # Extract graph-level embedding from HGT-TIME output dict.
        # HGTTimeModel stores it at out["embedding"]["graph"] (shape (B, hidden_dim)).
        embedding_dict = out.get("embedding", {})
        graph_embed = embedding_dict.get("graph", None)
        if graph_embed is None:
            # Fallback: use the graph logits as a proxy (should not happen
            # with a properly constructed HGTTimeModel).
            graph_embed = out.get("graph_logits", torch.zeros(1))

        # Apply MDBN if enabled
        if self.use_mdbn and platform_labels is not None:
            # Apply per-platform normalization
            unique_platforms = platform_labels.unique()
            normalized = torch.zeros_like(graph_embed)
            for pid in unique_platforms:
                mask = platform_labels == pid
                normalized[mask] = self.mdbn(graph_embed[mask], domain_id=pid.item())
            graph_embed = normalized

        # Platform DANN loss
        if self.use_platform_dann and platform_labels is not None:
            platform_logits = self.platform_discriminator(graph_embed)
            out["platform_dann_loss"] = F.cross_entropy(platform_logits, platform_labels)
            out["platform_dann_acc"] = (
                platform_logits.argmax(dim=-1) == platform_labels
            ).float().mean()

        # Patient DANN loss
        if self.use_patient_dann and patient_labels is not None:
            patient_logits = self.patient_discriminator(graph_embed)
            out["patient_dann_loss"] = F.cross_entropy(patient_logits, patient_labels)

        # Domain-invariant contrastive regularization
        if self.use_dicr and platform_labels is not None:
            class_labels = out["graph_logits"].argmax(dim=-1)
            out["dicr_loss"] = self.dicr_loss(
                F.normalize(graph_embed, dim=-1),
                class_labels,
                platform_labels,
            )

        return out


# ---------------------------------------------------------------------------
# Combined loss for domain generalization
# ---------------------------------------------------------------------------
class DomainGeneralizationLoss(nn.Module):
    """Combines base HGT-TIME loss with domain generalization losses.

    Total loss = base_loss
               + lambda_platform * platform_dann_loss
               + lambda_patient * patient_dann_loss
               + lambda_dicr * dicr_loss
    """

    def __init__(
        self,
        lambda_platform: float = 0.1,
        lambda_patient: float = 0.05,
        lambda_dicr: float = 0.1,
    ):
        super().__init__()
        self.lambda_platform = lambda_platform
        self.lambda_patient = lambda_patient
        self.lambda_dicr = lambda_dicr

    def forward(self, model_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            model_out: dict from DomainGeneralizedHGTTIME.forward()

        Returns:
            dict with total_loss and component losses.
        """
        total = model_out.get("loss", torch.tensor(0.0))

        components = {"base_loss": total.item() if isinstance(total, torch.Tensor) else total}

        if "platform_dann_loss" in model_out:
            dann_loss = self.lambda_platform * model_out["platform_dann_loss"]
            total = total + dann_loss
            components["platform_dann_loss"] = model_out["platform_dann_loss"].item()

        if "patient_dann_loss" in model_out:
            patient_loss = self.lambda_patient * model_out["patient_dann_loss"]
            total = total + patient_loss
            components["patient_dann_loss"] = model_out["patient_dann_loss"].item()

        if "dicr_loss" in model_out:
            dicr_loss = self.lambda_dicr * model_out["dicr_loss"]
            total = total + dicr_loss
            components["dicr_loss"] = model_out["dicr_loss"].item()

        return {"total_loss": total, "components": components}
