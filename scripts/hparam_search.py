#!/usr/bin/env python3
"""Optuna-based hyperparameter search for TIME classification models.

Reads search spaces from hyperparameter_search_strategy.yaml and performs
Bayesian optimization (TPE) with median pruning. Each trial runs inner
k-fold CV and reports macro-AUROC as the objective.

Usage:
    # Search for HGT-TIME
    python hparam_search.py \
        --config configs/hgt_time.default.yaml \
        --search-config configs/hyperparameter_search_strategy.yaml \
        --n-trials 30

    # Search for a baseline
    python hparam_search.py \
        --config configs/baseline_homo_gcn.yaml \
        --search-config configs/hyperparameter_search_strategy.yaml \
        --n-trials 20

    # Resume a previous study
    python hparam_search.py \
        --config configs/hgt_time.default.yaml \
        --search-config configs/hyperparameter_search_strategy.yaml \
        --study-name hgt_time_search \
        --storage sqlite:///Experiment/core_code/outputs/results/optuna.db
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def discover_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "instance.json").exists():
            return candidate
    raise SystemExit("Could not locate project root via instance.json")


def sample_params(trial: Any, space: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Sample hyperparameters from a search space definition."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        full_name = f"{prefix}{name}" if prefix else name
        if isinstance(spec, dict) and "type" in spec:
            ptype = spec["type"]
            if ptype == "loguniform":
                params[name] = trial.suggest_float(full_name, spec["low"], spec["high"], log=True)
            elif ptype == "uniform":
                params[name] = trial.suggest_float(full_name, spec["low"], spec["high"])
            elif ptype == "int":
                params[name] = trial.suggest_int(full_name, spec["low"], spec["high"])
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(full_name, spec["choices"])
            else:
                logger.warning(f"Unknown param type: {ptype} for {full_name}")
        elif isinstance(spec, dict):
            # Nested dict (e.g. loss_weights)
            params[name] = sample_params(trial, spec, prefix=f"{full_name}.")
        else:
            params[name] = spec
    return params


def sample_params_random(space: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Sample hyperparameters without Optuna for dependency-light fallback."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, dict) and "type" in spec:
            ptype = spec["type"]
            if ptype == "loguniform":
                low = float(spec["low"])
                high = float(spec["high"])
                params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            elif ptype == "uniform":
                params[name] = float(rng.uniform(float(spec["low"]), float(spec["high"])))
            elif ptype == "int":
                params[name] = int(rng.randint(int(spec["low"]), int(spec["high"])))
            elif ptype == "categorical":
                params[name] = rng.choice(list(spec["choices"]))
            else:
                logger.warning(f"Unknown param type: {ptype} for {name}")
        elif isinstance(spec, dict):
            params[name] = sample_params_random(spec, rng)
        else:
            params[name] = spec
    return params


def apply_params_to_config(config: dict[str, Any], common_params: dict, model_params: dict) -> dict[str, Any]:
    """Apply sampled hyperparameters to a config dict."""
    cfg = copy.deepcopy(config)

    # Common runtime params
    runtime = cfg.setdefault("runtime", {})
    for key in ("learning_rate", "weight_decay", "batch_size"):
        if key in common_params:
            runtime[key] = common_params[key]

    # Model params
    model = cfg.setdefault("model", {})
    loss = cfg.setdefault("loss", {})
    loss_param_names = {
        "classification_weight",
        "phenotype_weight",
        "region_weight",
        "ranking_weight",
        "label_smoothing",
        "huber_delta",
        "max_ranking_pairs_per_graph",
        "class_weights",
    }
    for key, val in model_params.items():
        if key == "loss_weights":
            for lk, lv in val.items():
                loss[lk] = lv
        elif key in loss_param_names:
            loss[key] = val
        else:
            model[key] = val

    return cfg


def objective(
    trial: Any,
    base_config: dict[str, Any],
    search_config: dict[str, Any],
    project_root: Path,
    inner_folds: int = 3,
) -> float:
    """Optuna objective function: returns mean macro-AUROC over inner folds."""
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from torch_geometric.loader import DataLoader

    # Lazy import to keep top-level lean
    sys.path.insert(0, str(project_root / "Experiment" / "core_code"))
    from scripts.train_eval_pipeline import (
        build_model,
        evaluate_model,
        set_seed,
        train_one_fold,
    )

    model_family = base_config.get("model_family", "homogeneous_graph")
    spaces = search_config.get("spaces", {})

    # Sample parameters
    common_params = sample_params(trial, spaces.get("common_runtime", {}))
    model_space_key = model_family if model_family in spaces else "non_graph"
    model_params = sample_params(trial, spaces.get(model_space_key, {}))

    config = apply_params_to_config(base_config, common_params, model_params)

    # Reduce epochs for search efficiency
    config.setdefault("runtime", {})["epochs"] = min(config.get("runtime", {}).get("epochs", 100), 50)
    config["runtime"]["patience"] = 10

    seed = 42
    set_seed(seed)

    # Load graphs
    graphs_dir = project_root / config["input"]["graphs_dir"]
    graph_paths = sorted(graphs_dir.glob("*.pt"))
    if not graph_paths:
        raise RuntimeError(f"No graphs found under {graphs_dir}")
    graphs = [torch.load(p, weights_only=False) for p in graph_paths]

    # Extract groups + labels
    groups: list[str] = []
    labels: list[int] = []
    for g in graphs:
        gid = getattr(g, "graph_id", "unknown_0")
        pid = gid.split("|")[0] if isinstance(gid, str) else str(gid)
        groups.append(pid)
        if hasattr(g, "y_graph") and g.y_graph is not None and len(g.y_graph) > 0:
            labels.append(int(g.y_graph[0].item()))
        else:
            labels.append(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.get("runtime", {}).get("batch_size", 4)

    # Auto-infer input_dim from data if not in config
    if model_family != "hgt_time":
        cell_dim = int(graphs[0]["cell"].x.size(-1)) if hasattr(graphs[0], "node_types") else graphs[0].x.size(-1)
        config.setdefault("model", {})["input_dim"] = cell_dim

    # Inner CV
    actual_folds = min(inner_folds, len(set(groups)))
    sgkf = StratifiedGroupKFold(n_splits=actual_folds)
    fold_aurocs: list[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(graphs, labels, groups)):
        train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=batch_size, shuffle=False)

        model = build_model(model_family, config.get("model", {}), graphs[0])
        result = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=None,
            fold=fold_i,
            seed=seed,
        )
        auroc = result["final_metrics"].get("macro_auroc", 0.0)
        fold_aurocs.append(auroc)

        # Report intermediate for pruning
        if trial is not None:
            trial.report(np.mean(fold_aurocs), fold_i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    return float(np.mean(fold_aurocs))


def run_search(args: argparse.Namespace) -> None:
    """Run the hyperparameter search."""
    try:
        import optuna
    except ModuleNotFoundError:
        logger.warning("Optuna is not installed; falling back to built-in random search.")
        run_random_search(args)
        return

    project_root = discover_project_root(args.config)
    sys.path.insert(0, str(project_root / "Experiment" / "core_code"))

    config_path = project_root / args.config if not args.config.is_absolute() else args.config
    with config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    search_config_path = project_root / args.search_config if not args.search_config.is_absolute() else args.search_config
    with search_config_path.open("r", encoding="utf-8") as f:
        search_config = yaml.safe_load(f) or {}

    strategy = search_config.get("search_strategy", {})
    n_trials = args.n_trials or strategy.get("n_trials", 30)
    inner_folds = strategy.get("cv_folds", 3)
    metric = strategy.get("metric", "macro_auroc")
    mode = strategy.get("mode", "maximize")

    model_family = base_config.get("model_family", "unknown")
    study_name = args.study_name or f"{model_family}_search"

    # Create or load study
    storage = args.storage
    if storage is None:
        db_path = project_root / "Experiment" / "core_code" / "outputs" / "results" / "optuna.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    pruner = optuna.pruners.MedianPruner() if strategy.get("pruner") == "median" else optuna.pruners.NopPruner()
    direction = "maximize" if mode == "maximize" else "minimize"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info(f"Starting search: study={study_name}, trials={n_trials}, metric={metric}, direction={direction}")

    study.optimize(
        lambda trial: objective(trial, base_config, search_config, project_root, inner_folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Report results
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {metric}: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Save best params to file
    output_dir = project_root / "Experiment" / "core_code" / "outputs" / "results" / "hparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_result = {
        "study_name": study_name,
        "model_family": model_family,
        "best_trial": study.best_trial.number,
        f"best_{metric}": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials_completed": len(study.trials),
    }
    result_path = output_dir / f"{study_name}_best.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, default=str)
    logger.info(f"Best params saved to {result_path}")

    # Generate best config YAML
    common_params = {}
    model_params = {}
    for pname, pval in study.best_trial.params.items():
        if pname in ("learning_rate", "weight_decay", "batch_size"):
            common_params[pname] = pval
        else:
            # Strip nested prefixes
            clean_name = pname.split(".")[-1] if "." in pname else pname
            model_params[clean_name] = pval

    best_config = apply_params_to_config(base_config, common_params, model_params)
    best_config_path = output_dir / f"{study_name}_best_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Best config saved to {best_config_path}")

    # Save all trials summary
    trials_summary = []
    for t in study.trials:
        trials_summary.append({
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
        })
    trials_path = output_dir / f"{study_name}_all_trials.json"
    with trials_path.open("w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, default=str)
    logger.info(f"All trials saved to {trials_path}")


def _split_param_groups(flat_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    common_params: dict[str, Any] = {}
    model_params: dict[str, Any] = {}
    for pname, pval in flat_params.items():
        if pname in ("learning_rate", "weight_decay", "batch_size"):
            common_params[pname] = pval
        else:
            model_params[pname] = pval
    return common_params, model_params


def run_random_search(args: argparse.Namespace) -> None:
    """Dependency-light fallback when Optuna is unavailable."""
    project_root = discover_project_root(args.config)

    config_path = project_root / args.config if not args.config.is_absolute() else args.config
    with config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    search_config_path = project_root / args.search_config if not args.search_config.is_absolute() else args.search_config
    with search_config_path.open("r", encoding="utf-8") as f:
        search_config = yaml.safe_load(f) or {}

    strategy = search_config.get("search_strategy", {})
    n_trials = args.n_trials or strategy.get("n_trials", 30)
    inner_folds = int(strategy.get("cv_folds", 3))
    metric = strategy.get("metric", "macro_auroc")
    mode = strategy.get("mode", "maximize")
    maximize = mode == "maximize"

    model_family = base_config.get("model_family", "unknown")
    study_name = args.study_name or f"{model_family}_search"
    spaces = search_config.get("spaces", {})
    model_space_key = model_family if model_family in spaces else "non_graph"

    rng = random.Random(args.random_seed)
    output_dir = project_root / "Experiment" / "core_code" / "outputs" / "results" / "hparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_summary: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None

    logger.info(
        "Starting random search: study=%s trials=%d metric=%s mode=%s seed=%d",
        study_name,
        n_trials,
        metric,
        mode,
        args.random_seed,
    )

    for trial_idx in range(n_trials):
        common_params = sample_params_random(spaces.get("common_runtime", {}), rng)
        model_params = sample_params_random(spaces.get(model_space_key, {}), rng)
        flat_params = {**common_params}
        for key, value in model_params.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flat_params[nested_key] = nested_value
            else:
                flat_params[key] = value

        trial_config = apply_params_to_config(base_config, common_params, model_params)
        value = objective(
            trial=None,
            base_config=trial_config,
            search_config={"spaces": {}},
            project_root=project_root,
            inner_folds=inner_folds,
        )

        trial_record = {
            "number": trial_idx,
            "value": value,
            "state": "COMPLETE",
            "params": flat_params,
        }
        trials_summary.append(trial_record)
        logger.info("Trial %d/%d | %s=%.4f | params=%s", trial_idx + 1, n_trials, metric, value, flat_params)

        if best_trial is None:
            best_trial = trial_record
        else:
            if maximize and value > float(best_trial["value"]):
                best_trial = trial_record
            elif not maximize and value < float(best_trial["value"]):
                best_trial = trial_record

    if best_trial is None:
        raise RuntimeError("Random search completed without any trials.")

    result_path = output_dir / f"{study_name}_best.json"
    best_result = {
        "study_name": study_name,
        "model_family": model_family,
        "search_backend": "random_fallback",
        "best_trial": int(best_trial["number"]),
        f"best_{metric}": float(best_trial["value"]),
        "best_params": best_trial["params"],
        "n_trials_completed": len(trials_summary),
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, default=str)
    logger.info("Best params saved to %s", result_path)

    common_params, model_params = _split_param_groups(best_trial["params"])
    best_config = apply_params_to_config(base_config, common_params, model_params)
    best_config_path = output_dir / f"{study_name}_best_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Best config saved to %s", best_config_path)

    trials_path = output_dir / f"{study_name}_all_trials.json"
    with trials_path.open("w", encoding="utf-8") as f:
        json.dump(trials_summary, f, indent=2, default=str)
    logger.info("All trials saved to %s", trials_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for TIME models")
    parser.add_argument("--config", type=Path, required=True, help="Base model config YAML")
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path("Experiment/core_code/configs/hyperparameter_search_strategy.yaml"),
        help="Search space config YAML",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Number of trials (overrides search config)")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///path.db)")
    parser.add_argument("--random-seed", type=int, default=20260408, help="Fallback random-search seed")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
