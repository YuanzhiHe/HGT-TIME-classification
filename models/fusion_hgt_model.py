"""Fusion HGT-TIME: a genuine multimodal fusion operator for the cell nodes.

Instead of early-concatenating the frozen pretrained expression embedding into the cell
feature vector (which lets a single linear input projection blend the two sources), this
model keeps the graph-derived features and the pretrained embedding as two separate
streams and fuses them with a learned operator before message passing:

  z_g = proj_graph(x_graph)          # 50-dim graph features -> d
  z_p = proj_pretrain(x_pretrain)    # 256-dim pretrained embedding -> d
  fusion_mode == 'gate':   h = g * z_g + (1-g) * z_p,  g = sigmoid(W[z_g; z_p])
  fusion_mode == 'xattn':  h = token0 of MHA over the 2-token sequence [z_g, z_p]

Everything downstream (gene/pathway projections, HGT encoder, heads) is identical to
HGTTimeModel, so the comparison isolates the fusion operator.

Requires graphs whose cell store carries both `x` (graph features) and `pretrain`
(the pretrained embedding); build with scripts/build_fusion_graphs.py.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models.hgt_time_model import (
    MLP,
    GraphConditionedNodeScorer,
    ResidualHGTBlock,
    TypeAwareGraphReadout,
    _infer_num_graphs,
    _node_batch_index,
)


def _proj_block(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class FusionHGTTimeModel(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        pretrain_dim: int = 256,
        morph_dim: int = 0,
        fusion_mode: str = "gate",
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
        self.fusion_mode = fusion_mode
        self.morph_dim = morph_dim
        self.use_pheno_head = use_pheno_head
        self.use_cell_state_head = use_cell_state_head
        self.use_ranking_heads = use_ranking_heads

        # cell fusion streams: graph (always) + expression + optional morphology
        self.cell_graph_proj = _proj_block(input_dims["cell"], hidden_dim, dropout)
        self.cell_pretrain_proj = _proj_block(pretrain_dim, hidden_dim, dropout) if pretrain_dim > 0 else None
        self.cell_morph_proj = _proj_block(morph_dim, hidden_dim, dropout) if morph_dim > 0 else None
        self.n_streams = 1 + (pretrain_dim > 0) + (morph_dim > 0)
        if fusion_mode == "gate":
            # per-dimension softmax weights over streams
            self.gate = nn.Linear(self.n_streams * hidden_dim, self.n_streams * hidden_dim)
            self.fuse_norm = nn.LayerNorm(hidden_dim)
        elif fusion_mode == "xattn":
            self.mha = nn.MultiheadAttention(hidden_dim, num_heads=num_heads,
                                             dropout=dropout, batch_first=True)
            self.modality_emb = nn.Parameter(torch.randn(self.n_streams, hidden_dim) * 0.02)
            self.fuse_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # non-cell node projections (standard)
        self.input_projector = nn.ModuleDict(
            {
                nt: _proj_block(input_dims[nt], hidden_dim, dropout)
                for nt in self.node_types if nt != "cell"
            }
        )
        self.encoder = nn.ModuleList(
            [ResidualHGTBlock(hidden_dim=hidden_dim, metadata=metadata,
                              num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = TypeAwareGraphReadout(self.node_types, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = MLP(hidden_dim, hidden_dim, num_classes, dropout=dropout)
        self.pheno_head = MLP(hidden_dim, hidden_dim, pheno_dim, dropout=dropout) if use_pheno_head else None
        self.cell_state_head = (
            MLP(hidden_dim, hidden_dim, cell_state_dim, dropout=dropout) if use_cell_state_head else None
        )
        self.gene_rank_head = GraphConditionedNodeScorer(hidden_dim, dropout) if use_ranking_heads else None
        self.pathway_rank_head = GraphConditionedNodeScorer(hidden_dim, dropout) if use_ranking_heads else None

    def _fuse_cell(self, cell) -> Tensor:
        streams = [self.cell_graph_proj(cell.x)]
        if self.cell_pretrain_proj is not None:
            streams.append(self.cell_pretrain_proj(cell.pretrain))
        if self.cell_morph_proj is not None:
            streams.append(self.cell_morph_proj(cell.morph))
        d = self.hidden_dim
        if self.fusion_mode == "gate":
            cat = torch.cat(streams, dim=-1)                       # (N, K*d)
            w = self.gate(cat).view(-1, self.n_streams, d)          # (N, K, d)
            w = torch.softmax(w, dim=1)                             # per-dim softmax over streams
            stacked = torch.stack(streams, dim=1)                  # (N, K, d)
            h = (w * stacked).sum(dim=1)                            # (N, d)
            return self.fuse_norm(h)
        # cross-modal attention over a K-token sequence per cell
        seq = torch.stack(streams, dim=1) + self.modality_emb.unsqueeze(0)  # (N,K,d)
        attn, _ = self.mha(seq, seq, seq, need_weights=False)
        h = attn[:, 0, :] + streams[0]  # graph-token output, residual on graph stream
        return self.fuse_norm(h)

    def encode(self, data: Any) -> dict[str, Tensor]:
        x_dict = {}
        for nt in self.node_types:
            if nt == "cell":
                x_dict["cell"] = self._fuse_cell(data["cell"])
            else:
                x_dict[nt] = self.input_projector[nt](data[nt].x)
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
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
            "embedding": {"graph": graph_embedding, **x_dict},
            "readout": readout_payload,
            "node_batch_index": {},
        }
        for nt in self.node_types:
            output["node_batch_index"][nt] = _node_batch_index(data, node_type=nt, device=x_dict[nt].device)
        if self.use_ranking_heads and self.gene_rank_head is not None and "gene" in x_dict:
            output["gene_score"] = self.gene_rank_head(
                x_dict["gene"], graph_embedding, output["node_batch_index"]["gene"])
        if self.use_ranking_heads and self.pathway_rank_head is not None and "pathway" in x_dict:
            output["pathway_score"] = self.pathway_rank_head(
                x_dict["pathway"], graph_embedding, output["node_batch_index"]["pathway"])
        return output
