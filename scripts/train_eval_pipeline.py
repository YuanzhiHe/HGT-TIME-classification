#!/usr/bin/env python3
"""Unified training protocol and evaluation pipeline for TIME classification.

Supports all model families (linear_deconvolution, non_graph, homogeneous_graph,
hgt_time) with a consistent interface:
  - Patient-level stratified grouped k-fold cross-validation
  - Multi-seed repetition for reproducibility
  - Cosine / step LR scheduling
  - Early stopping on validation macro-AUROC
  - Per-fold model checkpointing
  - Unified metric computation and result logging (JSON + TSV)

Usage:
    python train_eval_pipeline.py --config configs/baseline_homo_gcn.yaml
    python train_eval_pipeline.py --config configs/hgt_time.default.yaml --seeds 42 123 2026
    python train_eval_pipeline.py --run-all-registry --registry configs/experiment_registry.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn, optim
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 3) -> dict[str, float]:
    """Compute all evaluation metrics from true labels and predicted probabilities."""
    y_pred = np.argmax(y_prob, axis=1)
    metrics: dict[str, float] = {}

    # Primary metrics
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUROC (macro, one-vs-rest)
    try:
        if num_classes == 2:
            metrics["macro_auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["macro_auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["macro_auroc"] = 0.5

    # AUPRC (macro, one-vs-rest)
    try:
        from sklearn.metrics import average_precision_score

        if num_classes == 2:
            metrics["macro_auprc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            auprc_per_class = []
            for c in range(num_classes):
                y_c = (y_true == c).astype(int)
                if y_c.sum() > 0:
                    auprc_per_class.append(float(average_precision_score(y_c, y_prob[:, c])))
            metrics["macro_auprc"] = float(np.mean(auprc_per_class)) if auprc_per_class else 0.0
    except Exception:
        metrics["macro_auprc"] = 0.0

    # Brier score (multiclass decomposition)
    try:
        brier_scores = []
        for c in range(num_classes):
            y_c = (y_true == c).astype(int)
            brier_scores.append(brier_score_loss(y_c, y_prob[:, c]))
        metrics["brier_score"] = float(np.mean(brier_scores))
    except Exception:
        metrics["brier_score"] = float("nan")

    # ECE (Expected Calibration Error, 10 bins)
    try:
        confidences = np.max(y_prob, axis=1)
        correctness = (y_pred == y_true).astype(float)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                avg_conf = confidences[mask].mean()
                avg_acc = correctness[mask].mean()
                ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
        metrics["ece"] = float(ece)
    except Exception:
        metrics["ece"] = float("nan")

    # Confusion-based diagnostics: Excluded (class 1) -> Hot (class 0)
    try:
        labels = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        denom = cm[1, :].sum()
        metrics["excluded_to_hot_confusion"] = float(cm[1, 0] / denom) if denom > 0 else 0.0
    except Exception:
        metrics["excluded_to_hot_confusion"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _infer_cell_input_dim(sample_graph: Any) -> int:
    """Infer input dimension from cell node features in a sample graph."""
    from torch_geometric.data import HeteroData

    if isinstance(sample_graph, HeteroData) and "cell" in sample_graph.node_types:
        return int(sample_graph["cell"].x.size(-1))
    if hasattr(sample_graph, "x"):
        return int(sample_graph.x.size(-1))
    raise ValueError("Cannot infer input_dim: no cell node features found")


def build_model(
    model_family: str,
    model_kwargs: dict[str, Any],
    sample_graph: Any,
) -> nn.Module:
    """Instantiate a model given the family name and config kwargs.

    For non-HGT models, input_dim is auto-inferred from the sample graph
    if not explicitly set in the config.
    """
    from models import (
        HGTTimeModel,
        HomogeneousGraphClassifier,
        LinearDeconvolutionBaseline,
        NonGraphBaseline,
    )
    from models.baseline_models import FoundationModelBaseline

    # Auto-infer input_dim for non-HGT models
    if model_family != "hgt_time" and "input_dim" not in model_kwargs:
        model_kwargs = {**model_kwargs, "input_dim": _infer_cell_input_dim(sample_graph)}

    if model_family == "foundation_model":
        return FoundationModelBaseline(
            input_dim=model_kwargs["input_dim"],
            hidden_dim=model_kwargs.get("hidden_dim", 128),
            num_classes=model_kwargs["num_classes"],
            dropout=model_kwargs.get("dropout", 0.1),
        )
    elif model_family == "linear_deconvolution":
        return LinearDeconvolutionBaseline(
            input_dim=model_kwargs["input_dim"],
            num_classes=model_kwargs["num_classes"],
        )
    elif model_family == "non_graph":
        return NonGraphBaseline(
            input_dim=model_kwargs["input_dim"],
            hidden_dim=model_kwargs["hidden_dim"],
            num_classes=model_kwargs["num_classes"],
            num_layers=model_kwargs.get("num_layers", 3),
            dropout=model_kwargs.get("dropout", 0.1),
        )
    elif model_family == "homogeneous_graph":
        return HomogeneousGraphClassifier(
            input_dim=model_kwargs["input_dim"],
            hidden_dim=model_kwargs["hidden_dim"],
            num_classes=model_kwargs["num_classes"],
            num_layers=model_kwargs.get("num_layers", 2),
            dropout=model_kwargs.get("dropout", 0.1),
            conv_type=model_kwargs.get("conv_type", "gcn"),
            heads=model_kwargs.get("heads", 2),
        )
    elif model_family in ("hgt_time", "domain_generalized"):
        input_dims = {
            nt: int(sample_graph[nt].x.size(-1)) for nt in sample_graph.node_types
        }
        hgt_keys = {
            "hidden_dim", "num_layers", "num_heads", "dropout",
            "num_classes", "pheno_dim", "use_pheno_head",
            "use_cell_state_head", "cell_state_dim", "use_ranking_heads",
        }
        filtered = {k: v for k, v in model_kwargs.items() if k in hgt_keys}
        base_model = HGTTimeModel(
            metadata=sample_graph.metadata(),
            input_dims=input_dims,
            **filtered,
        )
        if model_family == "domain_generalized":
            from models.domain_generalization import DomainGeneralizedHGTTIME
            dg_cfg = model_kwargs.get("domain_generalization", {})
            return DomainGeneralizedHGTTIME(
                base_model=base_model,
                graph_embed_dim=filtered.get("hidden_dim", 128),
                n_platforms=dg_cfg.get("n_platforms", 3),
                n_patients=dg_cfg.get("n_patients", 10),
                use_platform_dann=dg_cfg.get("use_platform_dann", True),
                use_patient_dann=dg_cfg.get("use_patient_dann", False),
                use_dicr=dg_cfg.get("use_dicr", True),
                use_mdbn=dg_cfg.get("use_mdbn", False),
                dann_hidden_dim=dg_cfg.get("dann_hidden_dim", 64),
                dann_n_layers=dg_cfg.get("dann_n_layers", 2),
                dann_dropout=dg_cfg.get("dann_dropout", 0.1),
                dann_alpha=dg_cfg.get("dann_alpha", 1.0),
                dicr_temperature=dg_cfg.get("dicr_temperature", 0.1),
            )
        return base_model
    elif model_family == "fusion_hgt":
        from models.fusion_hgt_model import FusionHGTTimeModel
        input_dims = {nt: int(sample_graph[nt].x.size(-1)) for nt in sample_graph.node_types}
        fusion_keys = {
            "hidden_dim", "num_layers", "num_heads", "dropout", "num_classes",
            "pheno_dim", "use_pheno_head", "use_cell_state_head", "cell_state_dim",
            "use_ranking_heads", "pretrain_dim", "morph_dim", "fusion_mode",
        }
        filtered = {k: v for k, v in model_kwargs.items() if k in fusion_keys}
        return FusionHGTTimeModel(
            metadata=sample_graph.metadata(),
            input_dims=input_dims,
            **filtered,
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    scheduler_kwargs: dict[str, Any],
) -> Any:
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    elif scheduler_name == "none":
        return None
    else:
        logger.warning(f"Unknown scheduler '{scheduler_name}', skipping.")
        return None


# ---------------------------------------------------------------------------
# Training loop for one fold
# ---------------------------------------------------------------------------

def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path | None = None,
    fold: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a model for one fold and return metrics + training history."""
    from models import HGTTimeLoss

    model_family = config.get("model_family", "homogeneous_graph")
    runtime = config.get("runtime", {})
    loss_cfg = config.get("loss", {})

    epochs = runtime.get("epochs", 100)
    lr = float(runtime.get("learning_rate", 1e-3))
    wd = float(runtime.get("weight_decay", 1e-5))
    patience = runtime.get("patience", 15)
    eval_every = runtime.get("eval_every", 1)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = build_scheduler(
        optimizer,
        runtime.get("scheduler", "cosine"),
        runtime.get("scheduler_kwargs", {"T_max": epochs}),
    )

    # Loss function setup
    hgt_loss_fn = None
    if model_family in ("hgt_time", "fusion_hgt"):
        hgt_loss_fn = HGTTimeLoss(
            classification_weight=loss_cfg.get("classification_weight", 1.0),
            phenotype_weight=loss_cfg.get("phenotype_weight", 0.0),
            region_weight=loss_cfg.get("region_weight", 0.0),
            ranking_weight=loss_cfg.get("ranking_weight", 0.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
            class_weights=loss_cfg.get("class_weights"),
        ).to(device)

    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    class_weights = loss_cfg.get("class_weights")
    ce_weight = None
    if class_weights and model_family not in ("hgt_time", "fusion_hgt"):
        ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    best_val_auroc = -1.0
    best_state = None
    patience_counter = 0
    history: list[dict[str, Any]] = []
    avg_train_loss = 0.0

    # Domain generalization setup
    is_domain_gen = model_family == "domain_generalized"
    dg_loss_fn = None
    if is_domain_gen:
        from models.domain_generalization import DomainGeneralizationLoss, dann_alpha_schedule
        dg_cfg = config.get("model", {}).get("domain_generalization", {})
        dg_loss_fn = DomainGeneralizationLoss(
            lambda_platform=dg_cfg.get("lambda_platform", 0.1),
            lambda_patient=dg_cfg.get("lambda_patient", 0.05),
            lambda_dicr=dg_cfg.get("lambda_dicr", 0.1),
        )

    model.to(device)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        # Update DANN alpha schedule
        if is_domain_gen:
            alpha = dann_alpha_schedule(epoch, epochs)
            model.set_dann_alpha(alpha)

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Domain-generalized forward pass
            if is_domain_gen:
                # Extract domain labels from the batch metadata
                platform_labels = _extract_domain_labels(batch, "platform_id", device)
                patient_labels = _extract_domain_labels(batch, "patient_id", device)
                outputs = model(batch, platform_labels=platform_labels, patient_labels=patient_labels)
            else:
                outputs = model(batch)
            logits = outputs["graph_logits"]

            if model_family in ("hgt_time", "domain_generalized", "fusion_hgt") and hgt_loss_fn is not None:
                loss_dict = hgt_loss_fn(outputs, batch)
                loss = loss_dict["loss"]
                # Add domain generalization losses
                if is_domain_gen and dg_loss_fn is not None:
                    dg_result = dg_loss_fn({"loss": loss, **outputs})
                    loss = dg_result["total_loss"]
            else:
                y = _get_labels(batch, logits.size(0), device)
                loss = F.cross_entropy(logits, y, weight=ce_weight, label_smoothing=label_smoothing)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * logits.size(0)
            train_samples += logits.size(0)

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss_sum / max(train_samples, 1)

        # ---- Evaluate ----
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_metrics, avg_val_loss = evaluate_model(model, val_loader, config, device)
            val_auroc = val_metrics.get("macro_auroc", 0.0)

            history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                if checkpoint_dir is not None:
                    ckpt_path = checkpoint_dir / f"best_fold{fold}_seed{seed}.pt"
                    torch.save(best_state, ckpt_path)
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation with best model
    final_metrics, final_val_loss = evaluate_model(model, val_loader, config, device)

    return {
        "fold": fold,
        "seed": seed,
        "best_val_auroc": best_val_auroc,
        "final_metrics": final_metrics,
        "train_loss_final": avg_train_loss,
        "val_loss_final": final_val_loss,
        "epochs_trained": epoch + 1,
        "history": history,
    }


def _get_labels(batch: Any, num_graphs: int, device: torch.device) -> torch.Tensor:
    """Extract graph-level labels from a batch."""
    if hasattr(batch, "y_graph") and batch.y_graph is not None:
        return batch.y_graph.to(device).view(-1).long()
    return torch.zeros(num_graphs, dtype=torch.long, device=device)


def _extract_domain_labels(
    batch: Any, attr_name: str, device: torch.device,
) -> torch.Tensor | None:
    """Extract domain labels from batch metadata for domain generalization.

    Supports both numeric tensors and string-valued attributes.
    String values are mapped to integer indices (deterministically sorted).
    Returns None if the attribute is not present on the batch.
    """
    raw = getattr(batch, attr_name, None)
    if raw is None:
        return None

    if isinstance(raw, torch.Tensor):
        return raw.to(device).view(-1).long()

    # Handle string-valued domain labels (e.g., patient_id, platform_name)
    if isinstance(raw, (list, tuple)):
        unique_sorted = sorted(set(str(v) for v in raw))
        mapping = {v: i for i, v in enumerate(unique_sorted)}
        indices = [mapping[str(v)] for v in raw]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # Single value
    return torch.zeros(1, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """Evaluate model on a DataLoader, returning (metrics_dict, avg_loss)."""
    from models import HGTTimeLoss

    model_family = config.get("model_family", "homogeneous_graph")
    loss_cfg = config.get("loss", {})
    num_classes = config.get("model", {}).get("num_classes", 3)

    hgt_loss_fn = None
    if model_family in ("hgt_time", "domain_generalized", "fusion_hgt"):
        hgt_loss_fn = HGTTimeLoss(
            classification_weight=loss_cfg.get("classification_weight", 1.0),
            phenotype_weight=loss_cfg.get("phenotype_weight", 0.0),
            region_weight=loss_cfg.get("region_weight", 0.0),
            ranking_weight=loss_cfg.get("ranking_weight", 0.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
            class_weights=loss_cfg.get("class_weights"),
        ).to(device)

    label_smoothing = loss_cfg.get("label_smoothing", 0.0)
    class_weights = loss_cfg.get("class_weights")
    ce_weight = None
    if class_weights and model_family not in ("hgt_time", "domain_generalized", "fusion_hgt"):
        ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    is_domain_gen = model_family == "domain_generalized"

    model.eval()
    all_y_true: list[int] = []
    all_y_prob: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # For domain_generalized: forward without domain labels during eval
            # (the base model forward runs normally; DANN/DICR losses are just skipped)
            if is_domain_gen:
                outputs = model(batch)
            else:
                outputs = model(batch)
            logits = outputs["graph_logits"]
            probs = F.softmax(logits, dim=-1)

            y = _get_labels(batch, logits.size(0), device)

            if model_family in ("hgt_time", "domain_generalized", "fusion_hgt") and hgt_loss_fn is not None:
                loss_dict = hgt_loss_fn(outputs, batch)
                loss = loss_dict["loss"]
            else:
                loss = F.cross_entropy(logits, y, weight=ce_weight, label_smoothing=label_smoothing)

            total_loss += loss.item() * logits.size(0)
            total_samples += logits.size(0)

            all_y_true.extend(y.cpu().numpy().tolist())
            all_y_prob.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_true_arr = np.array(all_y_true)
    y_prob_arr = np.array(all_y_prob)
    metrics = compute_metrics(y_true_arr, y_prob_arr, num_classes=num_classes)
    return metrics, avg_loss


# ---------------------------------------------------------------------------
# Main runner for a single config
# ---------------------------------------------------------------------------

def run_experiment(config: dict[str, Any], project_root: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    """Run full cross-validation experiment for a single config across multiple seeds."""
    experiment_id = config.get("experiment_id", "EXP-UNKNOWN")
    model_family = config.get("model_family", "homogeneous_graph")
    runtime = config.get("runtime", {})
    split_cfg = config.get("split", {})

    if seeds is None:
        seeds = [runtime.get("seed", 42)]

    n_folds = split_cfg.get("n_folds", 5)
    split_unit = split_cfg.get("unit", "patient_id")
    batch_size = runtime.get("batch_size", 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running {experiment_id} ({model_family}) on {device} with seeds={seeds}")

    # Load graphs
    graphs_dir = project_root / config["input"]["graphs_dir"]
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    if not graph_paths:
        raise SystemExit(f"No graph files found under {graphs_dir}")

    graphs = [torch.load(path, weights_only=False) for path in graph_paths]
    logger.info(f"  Loaded {len(graphs)} graphs from {graphs_dir}")

    filtered_graphs = []
    dropped_uncertain = 0
    for graph in graphs:
        label_mask = getattr(graph, "label_mask", None)
        if label_mask is None or bool(torch.as_tensor(label_mask).view(-1)[0].item()):
            filtered_graphs.append(graph)
        else:
            dropped_uncertain += 1
    if dropped_uncertain:
        logger.info(f"  Dropped {dropped_uncertain} graphs with label_mask=false before splitting")
    graphs = filtered_graphs
    if not graphs:
        raise SystemExit(f"{experiment_id}: no supervised graphs remain after filtering label_mask=false")

    # Filter classes if configured (e.g., drop Excluded when too few samples)
    keep_classes = config.get("split", {}).get("keep_classes", None)
    if keep_classes is not None:
        keep_set = set(keep_classes)
        before = len(graphs)
        graphs = [g for g in graphs if int(g.y_graph[0].item()) in keep_set]
        logger.info(
            f"  keep_classes={keep_classes}: kept {len(graphs)}/{before} graphs"
        )
        # Remap labels to contiguous 0..K-1
        remap = {old: new for new, old in enumerate(sorted(keep_set))}
        for g in graphs:
            old_label = int(g.y_graph[0].item())
            g.y_graph[0] = remap[old_label]
        logger.info(f"  Remapped labels: {remap}")

    # Apply ablation transforms if configured
    ablation_cfg = config.get("ablation", {})
    transform_names = ablation_cfg.get("transforms", [])
    if transform_names:
        from models.graph_transforms import apply_ablation_transforms

        graphs = [apply_ablation_transforms(g, transform_names) for g in graphs]
        logger.info(f"  Applied ablation transforms: {transform_names}")

    # Extract patient IDs and labels for stratified split
    groups: list[str] = []
    labels: list[int] = []
    for g in graphs:
        group_value = getattr(g, split_unit, None)
        if group_value is None:
            group_value = getattr(g, "graph_id", "unknown_0")
        groups.append(str(group_value))
        if hasattr(g, "y_graph") and g.y_graph is not None and len(g.y_graph) > 0:
            labels.append(int(g.y_graph[0].item()))
        else:
            labels.append(0)

    unique_groups = {group for group in groups}
    label_counts = Counter(labels)
    label_group_counts = {
        label: len({group for group, group_label in zip(groups, labels) if group_label == label})
        for label in sorted(label_counts)
    }
    if len(graphs) < n_folds:
        raise SystemExit(
            f"{experiment_id}: cannot run {n_folds}-fold CV with only {len(graphs)} graphs "
            f"under {graphs_dir}. Rebuild/export more real graphs or lower split.n_folds."
        )
    if len(unique_groups) < n_folds:
        raise SystemExit(
            f"{experiment_id}: cannot run {n_folds}-fold grouped CV with only "
            f"{len(unique_groups)} unique {split_unit} groups under {graphs_dir}. "
            f"Found {len(graphs)} graphs total."
        )
    insufficient_label_groups = {
        label: count for label, count in label_group_counts.items() if count < n_folds
    }
    if insufficient_label_groups:
        raise SystemExit(
            f"{experiment_id}: stratified {n_folds}-fold CV is not feasible because some labels "
            f"have fewer than {n_folds} unique {split_unit} groups: {insufficient_label_groups}. "
            f"Label counts by graph: {dict(label_counts)}. Graph source: {graphs_dir}"
        )

    # Output directory
    output_root = project_root / config.get("output_root", "outputs/results")
    exp_dir = output_root / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        logger.info(f"  --- Seed {seed} ---")

        sgkf = StratifiedGroupKFold(n_splits=n_folds)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(graphs, labels, groups)):
            logger.info(f"    Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")

            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]

            train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

            model = build_model(model_family, config.get("model", {}), graphs[0])
            fold_result = train_one_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                checkpoint_dir=checkpoint_dir,
                fold=fold,
                seed=seed,
            )
            fold_result["experiment_id"] = experiment_id
            fold_result["model_family"] = model_family
            fold_result["split_protocol"] = f"CV_{n_folds}fold"
            all_results.append(fold_result)

            fm = fold_result["final_metrics"]
            logger.info(
                f"    Fold {fold} | AUROC={fm.get('macro_auroc', 0):.4f} "
                f"F1={fm.get('macro_f1', 0):.4f} "
                f"BAcc={fm.get('balanced_accuracy', 0):.4f}"
            )

    # Aggregate across all seeds and folds
    aggregated = aggregate_results(all_results)
    aggregated["experiment_id"] = experiment_id
    aggregated["model_family"] = model_family
    aggregated["seeds"] = seeds
    aggregated["n_folds"] = n_folds

    # Save results
    results_path = exp_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"experiment_id": experiment_id, "aggregated": aggregated, "per_fold": all_results},
            f,
            indent=2,
            default=str,
        )
    logger.info(f"  Results saved to {results_path}")

    # Append to global summary TSV
    append_to_summary_tsv(all_results, output_root / "results_summary.tsv")

    return aggregated


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-fold results into mean +/- std."""
    metric_keys = [
        "macro_auroc", "macro_f1", "balanced_accuracy", "macro_auprc",
        "brier_score", "ece", "excluded_to_hot_confusion",
    ]
    agg: dict[str, Any] = {}
    for key in metric_keys:
        values = [
            r["final_metrics"].get(key, 0.0)
            for r in results
            if not np.isnan(r["final_metrics"].get(key, 0.0))
        ]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"] = float("nan")
    return agg


# ---------------------------------------------------------------------------
# TSV summary writer
# ---------------------------------------------------------------------------

TSV_HEADER = (
    "experiment_id\tmodel_family\tsplit_protocol\tseed\tfold\t"
    "macro_auroc\tmacro_f1\tbalanced_accuracy\tmacro_auprc\t"
    "brier_score\tece\texcluded_to_hot_confusion\t"
    "train_loss_final\tval_loss_final\tnotes"
)


def append_to_summary_tsv(results: list[dict[str, Any]], tsv_path: Path) -> None:
    """Append per-fold results to the global summary TSV."""
    write_header = not tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(TSV_HEADER + "\n")
        for r in results:
            fm = r["final_metrics"]
            line = (
                f"{r.get('experiment_id', 'N/A')}\t"
                f"{r.get('model_family', 'N/A')}\t"
                f"{r.get('split_protocol', 'N/A')}\t"
                f"{r.get('seed', 'N/A')}\t"
                f"{r.get('fold', 'N/A')}\t"
                f"{fm.get('macro_auroc', 0):.4f}\t"
                f"{fm.get('macro_f1', 0):.4f}\t"
                f"{fm.get('balanced_accuracy', 0):.4f}\t"
                f"{fm.get('macro_auprc', 0):.4f}\t"
                f"{fm.get('brier_score', 0):.4f}\t"
                f"{fm.get('ece', 0):.4f}\t"
                f"{fm.get('excluded_to_hot_confusion', 0):.4f}\t"
                f"{r.get('train_loss_final', 0):.4f}\t"
                f"{r.get('val_loss_final', 0):.4f}\t"
                f"epochs={r.get('epochs_trained', 'N/A')}"
            )
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Run all experiments from registry
# ---------------------------------------------------------------------------

def run_all_from_registry(registry_path: Path, project_root: Path, seeds: list[int] | None = None) -> None:
    """Run all experiments defined in the experiment registry."""
    with registry_path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    defaults = registry.get("defaults", {})
    if seeds is None:
        seeds = defaults.get("seeds", [42])

    experiments = registry.get("experiments", [])
    logger.info(f"Registry contains {len(experiments)} experiments")

    for exp in experiments:
        exp_config_path = project_root / exp["config"]
        if not exp_config_path.exists():
            logger.warning(f"Config not found: {exp_config_path}, skipping {exp['id']}")
            continue

        with exp_config_path.open("r", encoding="utf-8") as f:
            loaded_config = yaml.safe_load(f) or {}

        registry_config = {
            "input": {},
            "split": {},
            "output_root": defaults.get("output_root", "outputs/results"),
        }
        if defaults.get("graphs_dir") is not None:
            registry_config["input"]["graphs_dir"] = defaults["graphs_dir"]
        if defaults.get("n_folds") is not None:
            registry_config["split"]["n_folds"] = defaults["n_folds"]
        if defaults.get("inner_folds") is not None:
            registry_config["split"]["inner_folds"] = defaults["inner_folds"]
        if defaults.get("split_unit") is not None:
            registry_config["split"]["unit"] = defaults["split_unit"]

        config = deep_merge_dicts(registry_config, loaded_config)
        config["experiment_id"] = config.get("experiment_id") or exp["id"]

        run_experiment(config, project_root, seeds=seeds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified training & evaluation pipeline for TIME classification"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a single model YAML config",
    )
    parser.add_argument(
        "--run-all-registry",
        action="store_true",
        help="Run all experiments from the registry",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/experiment_registry.yaml"),
        help="Path to experiment registry YAML",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to use (overrides config/registry)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    ref_path = args.config or args.registry
    if ref_path is None:
        ref_path = Path(".")
    project_root = discover_project_root(ref_path)

    sys.path.insert(0, str(project_root / "Experiment" / "core_code"))

    if args.run_all_registry:
        registry_path = project_root / args.registry if not args.registry.is_absolute() else args.registry
        run_all_from_registry(registry_path, project_root, seeds=args.seeds)
    elif args.config is not None:
        config_path = project_root / args.config if not args.config.is_absolute() else args.config
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        run_experiment(config, project_root, seeds=args.seeds)
    else:
        logger.error("Provide --config or --run-all-registry")
        sys.exit(1)


if __name__ == "__main__":
    main()
