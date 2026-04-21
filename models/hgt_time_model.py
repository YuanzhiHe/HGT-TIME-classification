from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import HGTConv
from torch_geometric.utils import scatter, softmax


RANKABLE_NODE_TYPES = ("gene", "pathway")


def _as_graph_id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _flatten_node_ids(node_ids: Any) -> list[str]:
    if node_ids is None:
        return []
    if isinstance(node_ids, (list, tuple)):
        if node_ids and isinstance(node_ids[0], (list, tuple)):
            flattened: list[str] = []
            for group in node_ids:
                flattened.extend(str(item) for item in group)
            return flattened
        return [str(item) for item in node_ids]
    return [str(node_ids)]


def _infer_num_graphs(data: Any) -> int:
    if hasattr(data, "num_graphs"):
        return int(data.num_graphs)
    graph_ids = _as_graph_id_list(getattr(data, "graph_id", None))
    if graph_ids:
        return len(graph_ids)
    for node_type in data.node_types:
        store = data[node_type]
        if hasattr(store, "batch") and store.batch.numel() > 0:
            return int(store.batch.max().item()) + 1
    return 1


def _node_batch_index(data: Any, node_type: str, device: torch.device) -> Tensor:
    store = data[node_type]
    if hasattr(store, "batch") and store.batch is not None:
        return store.batch.to(device=device, dtype=torch.long)
    x = store.x
    return torch.zeros(x.size(0), dtype=torch.long, device=device)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        if apply_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualHGTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_types = metadata[0]
        self.conv = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types})
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types})
        self.ffn = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
                for node_type in self.node_types
            }
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        delta_dict = self.conv(x_dict, edge_index_dict)
        updated: dict[str, Tensor] = {}
        for node_type in self.node_types:
            residual = x_dict[node_type]
            delta = delta_dict.get(node_type)
            if delta is None:
                delta = torch.zeros_like(residual)
            x = self.norm1[node_type](residual + self.dropout(delta))
            updated[node_type] = self.norm2[node_type](x + self.ffn[node_type](x))
        return updated


class TypeAwareGraphReadout(nn.Module):
    def __init__(self, node_types: list[str], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_types = node_types
        self.node_gate = nn.ModuleDict(
            {node_type: MLP(hidden_dim, hidden_dim, 1, dropout=dropout) for node_type in node_types}
        )
        self.type_gate = nn.ModuleDict(
            {node_type: MLP(hidden_dim, hidden_dim, 1, dropout=dropout) for node_type in node_types}
        )
        self.type_embedding = nn.ParameterDict(
            {node_type: nn.Parameter(torch.randn(1, hidden_dim) * 0.02) for node_type in node_types}
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        data: Any,
        num_graphs: int,
    ) -> tuple[Tensor, dict[str, Any]]:
        pooled_by_type: dict[str, Tensor] = {}
        node_attention: dict[str, Tensor] = {}
        type_logits: list[Tensor] = []
        type_masks: list[Tensor] = []

        reference_device = next(iter(x_dict.values())).device
        for node_type in self.node_types:
            x = x_dict[node_type]
            if x.size(0) == 0:
                pooled = x.new_zeros((num_graphs, x.size(-1)))
                attention = x.new_zeros((0,))
                mask = torch.zeros(num_graphs, dtype=torch.bool, device=reference_device)
            else:
                batch_index = _node_batch_index(data, node_type, x.device)
                gate_logits = self.node_gate[node_type](x).squeeze(-1)
                attention = softmax(gate_logits, batch_index)
                pooled = scatter(
                    attention.unsqueeze(-1) * x,
                    batch_index,
                    dim=0,
                    dim_size=num_graphs,
                    reduce="sum",
                )
                counts = scatter(
                    torch.ones_like(attention),
                    batch_index,
                    dim=0,
                    dim_size=num_graphs,
                    reduce="sum",
                )
                mask = counts > 0
            pooled_by_type[node_type] = pooled
            node_attention[node_type] = attention
            type_logits.append(
                self.type_gate[node_type](pooled + self.type_embedding[node_type]).squeeze(-1)
            )
            type_masks.append(mask)

        type_logit_tensor = torch.stack(type_logits, dim=1)
        type_mask_tensor = torch.stack(type_masks, dim=1)
        type_logit_tensor = type_logit_tensor.masked_fill(~type_mask_tensor, float("-inf"))
        empty_graphs = ~type_mask_tensor.any(dim=1)
        if empty_graphs.any():
            type_logit_tensor[empty_graphs] = 0.0
        type_weights = torch.softmax(type_logit_tensor, dim=1)
        type_weights = torch.where(type_mask_tensor, type_weights, torch.zeros_like(type_weights))
        denom = type_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        type_weights = type_weights / denom

        stacked_types = torch.stack([pooled_by_type[node_type] for node_type in self.node_types], dim=1)
        graph_embedding = (stacked_types * type_weights.unsqueeze(-1)).sum(dim=1)
        return graph_embedding, {
            "pooled_by_type": pooled_by_type,
            "node_attention": node_attention,
            "type_weights": {
                node_type: type_weights[:, index]
                for index, node_type in enumerate(self.node_types)
            },
        }


class GraphConditionedNodeScorer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = MLP(hidden_dim * 2, hidden_dim, 1, dropout=dropout)

    def forward(self, node_embedding: Tensor, graph_embedding: Tensor, batch_index: Tensor) -> Tensor:
        if node_embedding.size(0) == 0:
            return node_embedding.new_zeros((0,))
        graph_context = graph_embedding[batch_index]
        score = self.scorer(torch.cat([node_embedding, graph_context], dim=-1))
        return score.squeeze(-1)


class HGTTimeModel(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 3,
        pheno_dim: int = 4,
        use_pheno_head: bool = True,
        use_cell_state_head: bool = False,
        cell_state_dim: int = 4,
        use_ranking_heads: bool = True,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.hidden_dim = hidden_dim
        self.use_pheno_head = use_pheno_head
        self.use_cell_state_head = use_cell_state_head
        self.use_ranking_heads = use_ranking_heads

        self.input_projector = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(input_dims[node_type], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for node_type in self.node_types
            }
        )
        self.encoder = nn.ModuleList(
            [
                ResidualHGTBlock(
                    hidden_dim=hidden_dim,
                    metadata=metadata,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = TypeAwareGraphReadout(self.node_types, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = MLP(hidden_dim, hidden_dim, num_classes, dropout=dropout)
        self.pheno_head = MLP(hidden_dim, hidden_dim, pheno_dim, dropout=dropout) if use_pheno_head else None
        self.cell_state_head = (
            MLP(hidden_dim, hidden_dim, cell_state_dim, dropout=dropout)
            if use_cell_state_head
            else None
        )
        self.gene_rank_head = GraphConditionedNodeScorer(hidden_dim, dropout) if use_ranking_heads else None
        self.pathway_rank_head = GraphConditionedNodeScorer(hidden_dim, dropout) if use_ranking_heads else None

    def encode(self, data: Any) -> dict[str, Tensor]:
        x_dict = {
            node_type: self.input_projector[node_type](data[node_type].x)
            for node_type in self.node_types
        }
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in data.edge_types
        }
        for block in self.encoder:
            x_dict = block(x_dict, edge_index_dict)
        return x_dict

    def forward(self, data: Any) -> dict[str, Any]:
        x_dict = self.encode(data)
        num_graphs = _infer_num_graphs(data)
        graph_embedding, readout_payload = self.readout(x_dict=x_dict, data=data, num_graphs=num_graphs)
        graph_logits = self.classifier(graph_embedding)

        output: dict[str, Any] = {
            "graph_logits": graph_logits,
            "graph_probs": graph_logits.softmax(dim=-1),
            "pheno_pred": self.pheno_head(graph_embedding) if self.pheno_head is not None else None,
            "cell_state_pred": self.cell_state_head(x_dict["cell"]) if self.cell_state_head is not None else None,
            "gene_score": None,
            "pathway_score": None,
            "embedding": {
                "graph": graph_embedding,
                **x_dict,
            },
            "readout": readout_payload,
            "node_batch_index": {},
        }

        for node_type in self.node_types:
            output["node_batch_index"][node_type] = _node_batch_index(
                data,
                node_type=node_type,
                device=x_dict[node_type].device,
            )

        if self.use_ranking_heads and self.gene_rank_head is not None and "gene" in x_dict:
            output["gene_score"] = self.gene_rank_head(
                x_dict["gene"],
                graph_embedding,
                output["node_batch_index"]["gene"],
            )
        if self.use_ranking_heads and self.pathway_rank_head is not None and "pathway" in x_dict:
            output["pathway_score"] = self.pathway_rank_head(
                x_dict["pathway"],
                graph_embedding,
                output["node_batch_index"]["pathway"],
            )
        return output


def collect_topk_rankings(
    outputs: dict[str, Any],
    data: Any,
    topk: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    graph_ids = _as_graph_id_list(getattr(data, "graph_id", None))
    if not graph_ids:
        graph_ids = [str(index) for index in range(_infer_num_graphs(data))]

    rankings: dict[str, list[dict[str, Any]]] = {}
    for node_type in RANKABLE_NODE_TYPES:
        score_key = f"{node_type}_score"
        scores = outputs.get(score_key)
        if scores is None:
            continue
        batch_index = outputs["node_batch_index"][node_type]
        node_ids = _flatten_node_ids(getattr(data[node_type], "node_id", None))
        if len(node_ids) != scores.numel():
            node_ids = [f"{node_type}_{index}" for index in range(scores.numel())]
        per_graph: list[dict[str, Any]] = []
        detached_scores = scores.detach().cpu()
        detached_batch = batch_index.detach().cpu()
        for graph_index, graph_id in enumerate(graph_ids):
            node_mask = detached_batch == graph_index
            node_positions = node_mask.nonzero(as_tuple=False).view(-1)
            if node_positions.numel() == 0:
                per_graph.append({"graph_id": graph_id, "topk": []})
                continue
            node_scores = detached_scores[node_positions]
            keep = min(topk, int(node_scores.numel()))
            top_values, top_indices = torch.topk(node_scores, k=keep)
            top_nodes = []
            for score_value, relative_index in zip(top_values.tolist(), top_indices.tolist()):
                absolute_index = int(node_positions[int(relative_index)].item())
                top_nodes.append(
                    {
                        "node_id": node_ids[absolute_index],
                        "score": float(score_value),
                    }
                )
            per_graph.append({"graph_id": graph_id, "topk": top_nodes})
        rankings[node_type] = per_graph
    return rankings
