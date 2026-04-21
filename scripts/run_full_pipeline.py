#!/usr/bin/env python3
"""Full pretrain → fine-tune → OOD pipeline orchestrator.

Executes the three-phase closed loop:
    Phase 1: Multimodal contrastive pretraining (ST-bank + HEST-1k)
    Phase 2: Fine-tune HGT-TIME on labeled spatial cohort graphs
    Phase 3: Domain generalization training + OOD evaluation

Each phase can be run independently or as a full pipeline.

Usage:
    # Full pipeline
    python run_full_pipeline.py --phases 1 2 3 --config configs/full_pipeline.yaml

    # Skip pretraining (use existing checkpoint), run fine-tune + OOD
    python run_full_pipeline.py --phases 2 3 \\
        --pretrain-checkpoint outputs/pretrain/best_model.pt \\
        --config configs/full_pipeline.yaml

    # Only OOD evaluation on existing checkpoints
    python run_full_pipeline.py --phases 3 \\
        --checkpoints-dir outputs/results/EXP-DG01/checkpoints \\
        --config configs/full_pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and log output."""
    logger.info(f"[{description}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"[{description}] FAILED with return code {result.returncode}")
    else:
        logger.info(f"[{description}] Completed successfully")
    return result.returncode


def phase1_pretrain(config: dict[str, Any], project_root: Path) -> Path | None:
    """Phase 1: Multimodal contrastive pretraining.

    Returns path to the best checkpoint, or None on failure.
    """
    pretrain_cfg = config.get("phase1_pretrain", {})
    pretrain_config_path = project_root / pretrain_cfg.get(
        "config", "configs/pretraining/pretrain_combined.yaml"
    )
    output_dir = project_root / pretrain_cfg.get(
        "output_dir", "outputs/pretrain"
    )

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "pretraining" / "run_pretrain.py"),
        "--config", str(pretrain_config_path),
        "--output-dir", str(output_dir),
    ]

    if pretrain_cfg.get("epochs"):
        cmd.extend(["--epochs", str(pretrain_cfg["epochs"])])
    if pretrain_cfg.get("batch_size"):
        cmd.extend(["--batch-size", str(pretrain_cfg["batch_size"])])

    rc = run_command(cmd, "Phase 1: Pretraining")
    if rc != 0:
        return None

    # Find the best checkpoint
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        return best_ckpt

    # Fallback: latest checkpoint
    ckpts = sorted(output_dir.glob("checkpoint_epoch*.pt"))
    return ckpts[-1] if ckpts else None


def phase1_inject_features(
    pretrain_checkpoint: Path,
    config: dict[str, Any],
    project_root: Path,
) -> Path | None:
    """Inject pretrained features into HeteroData graphs.

    Returns path to the augmented graphs directory.
    """
    inject_cfg = config.get("phase1_inject", {})
    graphs_dir = project_root / inject_cfg.get(
        "graphs_dir", "outputs/hetero_graph/visium_breast__hetero_v1/graphs"
    )
    patches_dir = project_root / inject_cfg.get("patches_dir", "data/visium_patches")
    output_dir = project_root / inject_cfg.get(
        "output_dir", "outputs/hetero_graph/visium_breast__pretrain_aug/graphs"
    )
    mode = inject_cfg.get("mode", "concat")

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "pretraining" / "inject_pretrained_features.py"),
        "--checkpoint", str(pretrain_checkpoint),
        "--graphs-dir", str(graphs_dir),
        "--patches-dir", str(patches_dir),
        "--output-dir", str(output_dir),
        "--mode", mode,
    ]

    rc = run_command(cmd, "Phase 1: Feature injection")
    if rc != 0:
        return None
    return output_dir


def phase2_finetune(
    config: dict[str, Any],
    project_root: Path,
    graphs_dir: Path | None = None,
) -> Path | None:
    """Phase 2: Fine-tune HGT-TIME on labeled graphs.

    Returns path to the experiment output directory.
    """
    finetune_cfg = config.get("phase2_finetune", {})
    finetune_config_path = project_root / finetune_cfg.get(
        "config", "configs/hgt_time.default.yaml"
    )

    # If augmented graphs are available, override the input path
    with open(finetune_config_path) as f:
        ft_config = yaml.safe_load(f)

    if graphs_dir is not None:
        ft_config["input"]["graphs_dir"] = str(graphs_dir.relative_to(project_root))
        # Write a temporary config with the overridden path
        tmp_config_path = finetune_config_path.parent / "hgt_time_augmented_tmp.yaml"
        with open(tmp_config_path, "w") as f:
            yaml.dump(ft_config, f, default_flow_style=False)
        finetune_config_path = tmp_config_path

    seeds = finetune_cfg.get("seeds", [42, 123, 2026])

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_eval_pipeline.py"),
        "--config", str(finetune_config_path),
        "--seeds", *[str(s) for s in seeds],
    ]

    rc = run_command(cmd, "Phase 2: Fine-tuning")
    if rc != 0:
        return None

    exp_id = ft_config.get("experiment_id", "EXP-UNKNOWN")
    output_root = project_root / ft_config.get("output_root", "outputs/results")
    return output_root / exp_id


def phase3_domain_gen(
    config: dict[str, Any],
    project_root: Path,
    graphs_dir: Path | None = None,
) -> Path | None:
    """Phase 3: Domain generalization training.

    Returns path to the experiment output directory.
    """
    dg_cfg = config.get("phase3_domain_gen", {})
    dg_config_path = project_root / dg_cfg.get(
        "config", "configs/hgt_time_domain_gen.yaml"
    )

    with open(dg_config_path) as f:
        dg_config = yaml.safe_load(f)

    if graphs_dir is not None:
        dg_config["input"]["graphs_dir"] = str(graphs_dir.relative_to(project_root))
        tmp_config_path = dg_config_path.parent / "hgt_time_domain_gen_tmp.yaml"
        with open(tmp_config_path, "w") as f:
            yaml.dump(dg_config, f, default_flow_style=False)
        dg_config_path = tmp_config_path

    seeds = dg_cfg.get("seeds", [42, 123, 2026])

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_eval_pipeline.py"),
        "--config", str(dg_config_path),
        "--seeds", *[str(s) for s in seeds],
    ]

    rc = run_command(cmd, "Phase 3: Domain generalization training")
    if rc != 0:
        return None

    exp_id = dg_config.get("experiment_id", "EXP-DG01-DOMAIN-GEN")
    output_root = project_root / dg_config.get("output_root", "outputs/results")
    return output_root / exp_id


def phase3_ood_eval(
    config: dict[str, Any],
    project_root: Path,
    checkpoints_dir: Path | None = None,
) -> Path | None:
    """Phase 3: OOD evaluation (cross-section and/or LOPO).

    Returns path to the OOD results JSON.
    """
    ood_cfg = config.get("phase3_ood_eval", {})

    # Cross-section evaluation
    cs_cfg = ood_cfg.get("cross_section", {})
    if cs_cfg.get("enabled", True):
        train_graphs = project_root / cs_cfg.get(
            "train_graphs", "outputs/hetero_graph/visium_breast__hetero_v1/graphs"
        )
        test_graphs = project_root / cs_cfg.get(
            "test_graphs", "outputs/hetero_graph/section2__hetero_v1/graphs"
        )
        ckpt_dir = checkpoints_dir or (project_root / cs_cfg.get("checkpoints", ""))
        model_config = project_root / cs_cfg.get(
            "config", "configs/hgt_time_domain_gen.yaml"
        )
        output_path = project_root / cs_cfg.get(
            "output", "outputs/results/ood_cross_section.json"
        )

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "ood_evaluation.py"),
            "--protocol", "cross_section",
            "--config", str(model_config),
            "--train-graphs", str(train_graphs),
            "--test-graphs", str(test_graphs),
            "--checkpoints", str(ckpt_dir / "checkpoints") if ckpt_dir else "",
            "--output", str(output_path),
        ]

        rc = run_command(cmd, "Phase 3: Cross-section OOD eval")
        if rc == 0:
            logger.info(f"Cross-section results: {output_path}")

    # LOPO evaluation
    lopo_cfg = ood_cfg.get("lopo", {})
    if lopo_cfg.get("enabled", False):
        graphs_dir = project_root / lopo_cfg.get("graphs_dir", "")
        model_config = project_root / lopo_cfg.get(
            "config", "configs/hgt_time_domain_gen.yaml"
        )
        output_path = project_root / lopo_cfg.get(
            "output", "outputs/results/ood_lopo.json"
        )

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "ood_evaluation.py"),
            "--protocol", "lopo",
            "--config", str(model_config),
            "--graphs-dir", str(graphs_dir),
            "--output", str(output_path),
        ]

        rc = run_command(cmd, "Phase 3: LOPO OOD eval")
        if rc == 0:
            logger.info(f"LOPO results: {output_path}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Full pretrain-finetune-OOD pipeline")
    parser.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3],
                        help="Which phases to run (1=pretrain, 2=finetune, 3=domain_gen+OOD)")
    parser.add_argument("--config", required=True, help="Full pipeline config YAML")
    parser.add_argument("--pretrain-checkpoint", default=None,
                        help="Skip Phase 1, use this pretrained checkpoint")
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Skip Phase 2, use these checkpoints for OOD eval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    project_root = Path(args.config).parent
    for candidate in [project_root, *project_root.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "models").is_dir():
            project_root = candidate
            break

    logger.info(f"Project root: {project_root}")
    logger.info(f"Phases to run: {args.phases}")

    pretrain_checkpoint = Path(args.pretrain_checkpoint) if args.pretrain_checkpoint else None
    augmented_graphs_dir = None
    experiment_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else None

    # Phase 1: Pretraining
    if 1 in args.phases:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Multimodal contrastive pretraining")
        logger.info("=" * 60)

        pretrain_checkpoint = phase1_pretrain(config, project_root)
        if pretrain_checkpoint is None:
            logger.error("Phase 1 pretraining failed. Continuing without pretrained features.")
        else:
            logger.info(f"Phase 1 checkpoint: {pretrain_checkpoint}")
            # Inject pretrained features into graphs
            augmented_graphs_dir = phase1_inject_features(pretrain_checkpoint, config, project_root)
            if augmented_graphs_dir:
                logger.info(f"Augmented graphs at: {augmented_graphs_dir}")

    # Phase 2: Fine-tuning
    if 2 in args.phases:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: HGT-TIME fine-tuning")
        logger.info("=" * 60)

        experiment_dir = phase2_finetune(config, project_root, graphs_dir=augmented_graphs_dir)
        if experiment_dir:
            logger.info(f"Phase 2 output: {experiment_dir}")

    # Phase 3: Domain generalization + OOD
    if 3 in args.phases:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Domain generalization + OOD evaluation")
        logger.info("=" * 60)

        dg_dir = phase3_domain_gen(config, project_root, graphs_dir=augmented_graphs_dir)
        if dg_dir:
            logger.info(f"Phase 3 DG output: {dg_dir}")

        phase3_ood_eval(config, project_root, checkpoints_dir=dg_dir or experiment_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
