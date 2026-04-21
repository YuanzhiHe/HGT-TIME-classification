from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _reshape_graph_level_tensor(tensor: Tensor, width: int) -> Tensor:
    if tensor.dim() == 1:
        return tensor.reshape(-1, width)
    return tensor


def _pairwise_ranking_loss(
    scores: Tensor,
    batch_index: Tensor,
    positive_mask: Tensor,
    target_weight: Tensor | None = None,
    max_pairs_per_graph: int | None = None,
) -> Tensor:
    losses: list[Tensor] = []
    unique_graphs = batch_index.unique(sorted=True)
    for graph_id in unique_graphs.tolist():
        node_mask = batch_index == int(graph_id)
        pos_scores = scores[node_mask][positive_mask[node_mask]]
        neg_scores = scores[node_mask][~positive_mask[node_mask]]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        pairwise = F.softplus(-(pos_scores[:, None] - neg_scores[None, :]))
        if target_weight is not None:
            pos_weight = target_weight[node_mask][positive_mask[node_mask]].clamp_min(1.0)
            pairwise = pairwise * pos_weight[:, None]
        if max_pairs_per_graph is not None and pairwise.numel() > max_pairs_per_graph:
            pairwise = pairwise.reshape(-1)
            stride = max(pairwise.numel() // max_pairs_per_graph, 1)
            pairwise = pairwise[::stride][:max_pairs_per_graph]
        losses.append(pairwise.mean())
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


class HGTTimeLoss(nn.Module):
    def __init__(
        self,
        classification_weight: float = 1.0,
        phenotype_weight: float = 0.3,
        region_weight: float = 0.0,
        ranking_weight: float = 0.0,
        label_smoothing: float = 0.0,
        huber_delta: float = 1.0,
        max_ranking_pairs_per_graph: int | None = 256,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.classification_weight = classification_weight
        self.phenotype_weight = phenotype_weight
        self.region_weight = region_weight
        self.ranking_weight = ranking_weight
        self.label_smoothing = label_smoothing
        self.huber_delta = huber_delta
        self.max_ranking_pairs_per_graph = max_ranking_pairs_per_graph
        if class_weights is None:
            self.register_buffer("class_weights", None, persistent=False)
        else:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, outputs: dict[str, Any], data: Any) -> dict[str, Any]:
        graph_logits = outputs["graph_logits"]
        device = graph_logits.device

        label_mask = getattr(data, "label_mask", None)
        if label_mask is None:
            label_mask = torch.ones(graph_logits.size(0), dtype=torch.bool, device=device)
        else:
            label_mask = label_mask.to(device=device, dtype=torch.bool).view(-1)
        y_graph = getattr(data, "y_graph").to(device=device, dtype=torch.long).view(-1)
        if label_mask.any():
            loss_cls = F.cross_entropy(
                graph_logits[label_mask],
                y_graph[label_mask],
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss_cls = graph_logits.new_zeros(())

        loss_pheno = graph_logits.new_zeros(())
        pheno_pred = outputs.get("pheno_pred")
        if pheno_pred is not None and hasattr(data, "y_pheno") and hasattr(data, "pheno_mask"):
            pheno_target = _reshape_graph_level_tensor(
                getattr(data, "y_pheno").to(device=device, dtype=torch.float32),
                pheno_pred.size(-1),
            )
            pheno_mask = _reshape_graph_level_tensor(
                getattr(data, "pheno_mask").to(device=device, dtype=torch.bool),
                pheno_pred.size(-1),
            )
            pheno_mask = pheno_mask & label_mask.unsqueeze(-1)
            if pheno_mask.any():
                pheno_error = F.huber_loss(
                    pheno_pred,
                    pheno_target,
                    reduction="none",
                    delta=self.huber_delta,
                )
                loss_pheno = pheno_error[pheno_mask].mean()

        loss_region = graph_logits.new_zeros(())
        cell_state_pred = outputs.get("cell_state_pred")
        if cell_state_pred is not None and "cell" in data.node_types:
            cell_store = data["cell"]
            if hasattr(cell_store, "y_region"):
                region_target = getattr(cell_store, "y_region").to(device=device, dtype=torch.float32)
                if region_target.dim() == 1:
                    region_target = region_target.unsqueeze(-1)
                region_mask = getattr(cell_store, "region_mask", None)
                if region_mask is None:
                    region_mask = torch.ones_like(region_target, dtype=torch.bool, device=device)
                else:
                    region_mask = region_mask.to(device=device, dtype=torch.bool)
                    if region_mask.dim() == 1:
                        region_mask = region_mask.unsqueeze(-1).expand_as(region_target)
                region_error = F.huber_loss(
                    cell_state_pred,
                    region_target,
                    reduction="none",
                    delta=self.huber_delta,
                )
                if region_mask.any():
                    loss_region = region_error[region_mask].mean()

        loss_ranking = graph_logits.new_zeros(())
        ranking_terms = 0
        for node_type in ("gene", "pathway"):
            score_key = f"{node_type}_score"
            node_score = outputs.get(score_key)
            if node_score is None or node_type not in data.node_types:
                continue
            node_store = data[node_type]
            if not hasattr(node_store, "target_pos_mask"):
                continue
            positive_mask = getattr(node_store, "target_pos_mask").to(device=device, dtype=torch.bool).view(-1)
            if positive_mask.numel() != node_score.numel():
                continue
            target_weight = None
            if hasattr(node_store, "target_weight"):
                target_weight = getattr(node_store, "target_weight").to(device=device, dtype=torch.float32).view(-1)
            batch_index = outputs["node_batch_index"][node_type].to(device=device, dtype=torch.long)
            node_loss = _pairwise_ranking_loss(
                scores=node_score,
                batch_index=batch_index,
                positive_mask=positive_mask,
                target_weight=target_weight,
                max_pairs_per_graph=self.max_ranking_pairs_per_graph,
            )
            if node_loss.requires_grad or node_loss.item() > 0:
                loss_ranking = loss_ranking + node_loss
                ranking_terms += 1
        if ranking_terms > 1:
            loss_ranking = loss_ranking / float(ranking_terms)

        total_loss = (
            self.classification_weight * loss_cls
            + self.phenotype_weight * loss_pheno
            + self.region_weight * loss_region
            + self.ranking_weight * loss_ranking
        )
        return {
            "loss": total_loss,
            "loss_cls": loss_cls,
            "loss_pheno": loss_pheno,
            "loss_region": loss_region,
            "loss_ranking": loss_ranking,
            "supervised_graphs": int(label_mask.sum().item()),
            "ranking_terms": ranking_terms,
        }
