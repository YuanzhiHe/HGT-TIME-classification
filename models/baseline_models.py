from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


def _extract_pseudo_bulk(batch: Any) -> tuple[Tensor, Tensor, int]:
    """Mean-pool cell node features per graph to form pseudo-bulk vectors.

    Returns (x_pooled, batch_index, num_graphs).
    """
    if isinstance(batch, HeteroData):
        x = batch["cell"].x
        batch_idx = batch["cell"].batch
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
    else:
        x = batch.x
        batch_idx = batch.batch
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1

    x_pool = torch.zeros(num_graphs, x.size(-1), device=x.device)
    x_pool.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, x.size(-1)), x)
    sizes = torch.bincount(batch_idx, minlength=num_graphs).unsqueeze(-1).float().clamp(min=1)
    x_pool = x_pool / sizes
    return x_pool, batch_idx, num_graphs


def _extract_homo_data(batch: Any) -> Data:
    """Extract a homogeneous Data object (cell nodes + spatial edges) from HeteroData."""
    if isinstance(batch, HeteroData):
        x = batch["cell"].x
        batch_idx = batch["cell"].batch
        # Try multiple edge type names for spatial adjacency
        edge_index = None
        for et in batch.edge_types:
            if et[0] == "cell" and et[2] == "cell":
                edge_index = batch[et].edge_index
                break
        if edge_index is None:
            # Fallback: isolated nodes
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        return Data(x=x, edge_index=edge_index, batch=batch_idx)
    return batch


def _get_graph_labels(batch: Any) -> Tensor:
    """Retrieve graph-level labels from a batch."""
    if hasattr(batch, "y_graph") and batch.y_graph is not None:
        return batch.y_graph.view(-1).long()
    if isinstance(batch, HeteroData) and "cell" in batch.node_types:
        cell_store = batch["cell"]
        if hasattr(cell_store, "y") and cell_store.y is not None:
            return cell_store.y.view(-1).long()
    return torch.zeros(1, dtype=torch.long)


def _wrap_logits(logits: Tensor) -> dict[str, Any]:
    """Wrap raw logits into the standard output dict expected by the training pipeline."""
    return {
        "graph_logits": logits,
        "graph_probs": F.softmax(logits, dim=-1),
    }


class FoundationModelBaseline(nn.Module):
    """Foundation model baseline: pretrained expression embeddings + MLP.

    Expects augmented graphs where cell.x = [original_feats, pretrained_256d].
    Uses the **full** cell feature vector (original + pretrained) with mean-pool
    per graph, then classifies through a 2-layer MLP.  No graph structure is used.

    This tests whether adding pretrained CLIP embeddings to the feature set
    improves a non-graph MLP baseline — isolating the value of pretraining
    from the value of graph structure.
    """

    def __init__(
        self,
        input_dim: int = 306,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Any) -> dict[str, Any]:
        x_pool, _, _ = _extract_pseudo_bulk(batch)
        logits = self.classifier(x_pool)
        return _wrap_logits(logits)


class LinearDeconvolutionBaseline(nn.Module):
    """Linear deconvolution baseline (CIBERSORTx-like).

    Single linear layer from pseudo-bulk expression profile to TIME class logits.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, num_classes)

    def forward(self, batch: Any) -> dict[str, Any]:
        x_pool, _, _ = _extract_pseudo_bulk(batch)
        logits = self.network(x_pool)
        return _wrap_logits(logits)


class NonGraphBaseline(nn.Module):
    """Non-graph non-linear baseline (Scaden-like MLP)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, batch: Any) -> dict[str, Any]:
        x_pool, _, _ = _extract_pseudo_bulk(batch)
        logits = self.network(x_pool)
        return _wrap_logits(logits)


class GraphLevelMLP(nn.Module):
    """Simple MLP on pre-computed graph-level features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Any) -> dict[str, Any]:
        x_pool, _, _ = _extract_pseudo_bulk(batch)
        logits = self.network(x_pool)
        return _wrap_logits(logits)


class HomogeneousGraphClassifier(nn.Module):
    """Graph classifier using GCN or GAT on homogeneous cell-cell spatial graph."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: Literal["gcn", "gat"] = "gcn",
        heads: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gat":
                self.convs.append(
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=max(hidden_dim // heads, 1),
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Any) -> dict[str, Any]:
        data = _extract_homo_data(batch)
        x = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index)
            x = norm(residual + self.dropout(x))
            x = torch.relu(x)
        pooled = global_mean_pool(x, data.batch)
        logits = self.classifier(pooled)
        return _wrap_logits(logits)
