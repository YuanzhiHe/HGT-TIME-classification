"""Run multimodal contrastive pretraining on ST-bank and/or HEST-1k data.

Usage:
    # Pretrain on ST-bank
    python run_pretrain.py --config configs/pretraining/pretrain_stbank.yaml

    # Pretrain on HEST-1k
    python run_pretrain.py --config configs/pretraining/pretrain_hest1k.yaml

    # Pretrain on combined data
    python run_pretrain.py --config configs/pretraining/pretrain_combined.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import yaml

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.multimodal_pretrain import (
    MultimodalPretrainModel,
    STBankDataset,
    HEST1kDataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_transforms(img_size: int = 224):
    """Build image transforms for pretraining with augmentation."""
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform


def build_dataset(cfg: dict, transform):
    """Build training dataset from config.

    When both ST-bank and HEST-1k are specified, the ST-bank gene vocabulary
    is built first and shared with HEST-1k so both datasets use a common
    gene index.
    """
    datasets = []
    shared_vocab = None
    vocab_size = cfg.get("n_genes", 2000)

    # Build ST-bank first (it defines the gene vocabulary)
    if cfg.get("stbank_dir"):
        logger.info(f"Loading ST-bank from {cfg['stbank_dir']}")
        ds = STBankDataset(
            data_dir=cfg["stbank_dir"],
            vocab_size=vocab_size,
            transform=transform,
            mode="gene_sentence",
        )
        shared_vocab = ds.gene_vocab
        datasets.append(ds)

    if cfg.get("hest1k_dir"):
        logger.info(f"Loading HEST-1k from {cfg['hest1k_dir']}")
        ds = HEST1kDataset(
            data_dir=cfg["hest1k_dir"],
            gene_vocab=shared_vocab,
            vocab_size=vocab_size,
            sample_ids=cfg.get("hest1k_sample_ids"),
            transform=transform,
        )
        datasets.append(ds)

    if not datasets:
        raise ValueError("No data sources specified in config")

    if len(datasets) > 1:
        combined = ConcatDataset(datasets)
        logger.info(f"Combined dataset: {len(combined):,} total pairs")
        return combined
    return datasets[0]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    max_steps: int | None = None,
    save_dir: Path | None = None,
    save_every_steps: int = 2000,
    global_step: int = 0,
) -> dict:
    """Train for one epoch (or up to max_steps)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        expressions = batch["expression"].to(device)

        out = model(images, expressions)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            temp = out["temperature"]
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={avg_loss:.4f} temp={temp:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # Periodic save within epoch
        if save_dir and global_step % save_every_steps == 0:
            torch.save(
                {"global_step": global_step, "model_state_dict": model.state_dict(),
                 "loss": total_loss / n_batches},
                save_dir / f"checkpoint_step{global_step}.pt",
            )
            logger.info(f"  Saved step checkpoint at step {global_step}")

        if max_steps and global_step >= max_steps:
            logger.info(f"Reached max_steps={max_steps}, stopping.")
            break

    if scheduler:
        scheduler.step()

    return {"loss": total_loss / max(n_batches, 1), "global_step": global_step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many gradient steps (overrides epochs)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    out_dir = Path(cfg.get("output_dir", "outputs/pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = args.max_steps or cfg.get("max_steps")

    # Build dataset
    transform = build_transforms(cfg.get("img_size", 224))
    dataset = build_dataset(cfg.get("data", {}), transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset):,} pairs, {len(loader)} batches per epoch")

    # Build model
    model_cfg = cfg.get("model", {})
    model = MultimodalPretrainModel(
        image_backbone=model_cfg.get("image_backbone", "vit_base_patch16_224"),
        expr_input_dim=cfg.get("data", {}).get("n_genes", 2000),
        embed_dim=model_cfg.get("embed_dim", 512),
        expr_hidden_dim=model_cfg.get("expr_hidden_dim", 512),
        expr_n_layers=model_cfg.get("expr_n_layers", 3),
        expr_dropout=model_cfg.get("expr_dropout", 0.1),
        init_temp=model_cfg.get("init_temp", 0.07),
        freeze_image_backbone=model_cfg.get("freeze_image_backbone", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {n_params:,} params ({n_trainable:,} trainable)")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("epochs", 50),
        eta_min=cfg.get("min_lr", 1e-6),
    )

    # Training loop
    best_loss = float("inf")
    epochs = cfg.get("epochs", 50)
    training_log = []
    global_step = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            model, loader, optimizer, scheduler, device, epoch,
            max_steps=max_steps,
            save_dir=out_dir,
            save_every_steps=cfg.get("save_every_steps", 2000),
            global_step=global_step,
        )
        elapsed = time.time() - t0
        global_step = metrics["global_step"]

        logger.info(f"Epoch {epoch}/{epochs} done in {elapsed:.1f}s — loss={metrics['loss']:.4f}")
        training_log.append({"epoch": epoch, **metrics, "time_s": elapsed})

        # Save checkpoint
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": cfg,
                },
                out_dir / "best_pretrain.pt",
            )
            logger.info(f"  Saved best checkpoint (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % cfg.get("save_every", 10) == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                out_dir / f"checkpoint_epoch{epoch}.pt",
            )

        if max_steps and global_step >= max_steps:
            stopped_early = True
            break

    # Save training log
    with open(out_dir / "pretrain_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), out_dir / "final_pretrain.pt")
    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}, steps: {global_step}")
    logger.info(f"Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
