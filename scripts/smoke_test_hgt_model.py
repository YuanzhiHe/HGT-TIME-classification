#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the HGT TIME model on hetero_v1 graphs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hgt_time.mock.yaml"),
        help="Path to the HGT model YAML config",
    )
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def build_default_config() -> dict[str, Any]:
    return {
        "input": {
            "graphs_dir": "outputs/hetero_graph/visium_mock__hetero_v1/graphs",
        },
        "runtime": {
            "batch_size": 2,
            "shuffle": False,
            "topk": 3,
            "inject_mock_targets": False,
        },
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "num_classes": 3,
            "pheno_dim": 4,
            "use_pheno_head": True,
            "use_cell_state_head": False,
            "cell_state_dim": 4,
            "use_ranking_heads": True,
        },
        "loss": {
            "classification_weight": 1.0,
            "phenotype_weight": 0.3,
            "region_weight": 0.0,
            "ranking_weight": 0.2,
            "label_smoothing": 0.0,
            "huber_delta": 1.0,
            "max_ranking_pairs_per_graph": 64,
            "class_weights": None,
        },
        "targets": {
            "gene_positive_ids": [],
            "pathway_positive_ids": [],
        },
    }


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def ensure_descendant(path: Path, project_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"Path escapes project root: {resolved}") from exc
    return resolved


def flatten_node_ids(node_ids: Any) -> list[str]:
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


def inject_mock_targets(batch: Any, config: dict[str, Any]) -> None:
    target_spec = {
        "gene": set(config["targets"].get("gene_positive_ids", [])),
        "pathway": set(config["targets"].get("pathway_positive_ids", [])),
    }
    for node_type, positives in target_spec.items():
        if node_type not in batch.node_types or not positives:
            continue
        flat_ids = flatten_node_ids(getattr(batch[node_type], "node_id", None))
        if not flat_ids:
            continue
        pos_mask = torch.tensor([node_id in positives for node_id in flat_ids], dtype=torch.bool)
        batch[node_type].target_pos_mask = pos_mask
        batch[node_type].target_weight = pos_mask.to(dtype=torch.float32)


def main() -> None:
    args = parse_args()
    project_root = discover_project_root(args.config)
    with args.config.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = deep_update(copy.deepcopy(build_default_config()), user_config)

    core_code_root = ensure_descendant(project_root / "Experiment/core_code", project_root)
    sys.path.insert(0, str(core_code_root))
    from models import HGTTimeLoss, HGTTimeModel, collect_topk_rankings

    random.seed(0)
    torch.manual_seed(0)

    graphs_dir = ensure_descendant(resolve_path(project_root, config["input"]["graphs_dir"]), project_root)
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    if not graph_paths:
        raise SystemExit(f"No graph files found under {graphs_dir}")

    graphs = [torch.load(path) for path in graph_paths]
    sample = graphs[0]
    model = HGTTimeModel(
        metadata=sample.metadata(),
        input_dims={node_type: int(sample[node_type].x.size(-1)) for node_type in sample.node_types},
        **config["model"],
    )
    criterion = HGTTimeLoss(**config["loss"])

    loader = DataLoader(
        graphs,
        batch_size=int(config["runtime"]["batch_size"]),
        shuffle=bool(config["runtime"]["shuffle"]),
    )
    batch = next(iter(loader))
    if bool(config["runtime"].get("inject_mock_targets", False)):
        inject_mock_targets(batch, config)

    model.train()
    outputs = model(batch)
    loss_payload = criterion(outputs, batch)
    loss_payload["loss"].backward()
    rankings = collect_topk_rankings(
        outputs,
        batch,
        topk=int(config["runtime"].get("topk", 3)),
    )

    pheno_pred = outputs.get("pheno_pred")
    summary = {
        "graphs_in_batch": len(getattr(batch, "graph_id", [])),
        "graph_ids": list(getattr(batch, "graph_id", [])),
        "graph_logits_shape": list(outputs["graph_logits"].shape),
        "pheno_pred_shape": list(pheno_pred.shape) if pheno_pred is not None else None,
        "gene_score_shape": list(outputs["gene_score"].shape) if outputs.get("gene_score") is not None else None,
        "pathway_score_shape": list(outputs["pathway_score"].shape) if outputs.get("pathway_score") is not None else None,
        "loss": float(loss_payload["loss"].detach().cpu().item()),
        "loss_cls": float(loss_payload["loss_cls"].detach().cpu().item()),
        "loss_pheno": float(loss_payload["loss_pheno"].detach().cpu().item()),
        "loss_ranking": float(loss_payload["loss_ranking"].detach().cpu().item()),
        "supervised_graphs": int(loss_payload["supervised_graphs"]),
        "ranking_terms": int(loss_payload["ranking_terms"]),
        "type_weights": {
            node_type: outputs["readout"]["type_weights"][node_type].detach().cpu().tolist()
            for node_type in outputs["readout"]["type_weights"]
        },
        "top_rankings": rankings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
