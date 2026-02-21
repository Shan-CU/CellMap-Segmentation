"""
Ablation experiment runner for 3D CellMap segmentation.

Runs multiple ablation configs sequentially on the same GPU allocation.
Each config runs for 100 epochs (~3.5h on 2×H100, ~6h on 2×L40S).

Usage:
    # Run all masking ablation configs
    torchrun --nproc_per_node=2 train_ablation.py --axis masking

    # Run specific config by name
    torchrun --nproc_per_node=2 train_ablation.py --axis masking --config bbox_tight_bg020

    # Run weighting ablation
    torchrun --nproc_per_node=2 train_ablation.py --axis weighting

    # Run tversky tuning
    torchrun --nproc_per_node=2 train_ablation.py --axis tversky

    # Run configs by index range (for splitting across jobs)
    torchrun --nproc_per_node=2 train_ablation.py --axis masking --start 0 --end 4

    # List all configs without running
    python train_ablation.py --axis masking --list
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure the experiment directory is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "configs"))


def parse_args():
    parser = argparse.ArgumentParser(description="3D Ablation Experiment Runner")
    parser.add_argument(
        "--axis", type=str, required=True,
        choices=["masking", "weighting", "tversky"],
        help="Ablation axis to run",
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="Run a specific config by name (e.g., 'bbox_tight_bg020')",
    )
    parser.add_argument(
        "--start-idx", type=int, default=None,
        help="Start index in priority order (inclusive). "
             "Also reads ABL_START env var.",
    )
    parser.add_argument(
        "--end-idx", type=int, default=None,
        help="End index in priority order (exclusive, -1 = all). "
             "Also reads ABL_END env var.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all configs and exit (no training)",
    )
    parser.add_argument(
        "--skip-completed", action="store_true", default=True,
        help="Skip configs that already have a best_model.pth checkpoint",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from last checkpoint if available",
    )
    return parser.parse_args()


def load_axis_module(axis: str):
    """Load the config module for the given ablation axis."""
    module_names = {
        "masking": "cfg_ablation_masking",
        "weighting": "cfg_ablation_weighting",
        "tversky": "cfg_ablation_tversky",
    }
    return importlib.import_module(module_names[axis])


def is_completed(output_dir: str) -> bool:
    """Check if a config has been fully trained (has ablation_summary.json)."""
    return os.path.exists(os.path.join(output_dir, "ablation_summary.json"))


def run_single_config(cfg, resume: bool = True):
    """Run training for a single ablation config.

    This function re-uses the existing train.py infrastructure by
    importing and calling main() with the config directly.
    """
    from train import (
        setup_ddp, cleanup_ddp, is_main_process, set_seed,
        train_one_epoch, validate, build_optimizer, build_scheduler,
        save_checkpoint, load_checkpoint, reduce_tensor,
    )
    from data.ds_cellmap import CellMapDataset, load_datalist, flat_collate_fn, batch_to_device
    from models.mdl_cellmap import Net
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    # --- DDP setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Seed ---
    set_seed(getattr(cfg, "seed", 42), rank=global_rank)

    # --- Data ---
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"ABLATION: {cfg.name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        print(f"Loading datalist from: {cfg.datalist}")

    train_files, val_files = load_datalist(cfg)
    if is_main_process():
        print(f"Train: {len(train_files)} volumes, Val: {len(val_files)} volumes")

    # Build datasets — volume cache is populated here (before workers fork)
    train_dataset = CellMapDataset(train_files, cfg, mode="train")
    val_dataset = CellMapDataset(val_files, cfg, mode="val")

    # Barrier to ensure all ranks have finished caching before proceeding
    if world_size > 1:
        torch.distributed.barrier()

    # --- Samplers ---
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    # --- DataLoaders ---
    batch_size = getattr(cfg, "batch_size", 2)
    num_workers = getattr(cfg, "num_workers", 4)

    pin_memory = getattr(cfg, "pin_memory", False)
    prefetch = getattr(cfg, "prefetch_factor", 4) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=flat_collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=flat_collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
    )

    # --- Model ---
    model = Net(cfg).to(device)
    if getattr(cfg, "syncbn", False) and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) // getattr(cfg, "grad_accumulation", 1)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=getattr(cfg, "mixed_precision", False))

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_metric = 0.0

    if resume:
        last_ckpt = os.path.join(output_dir, "checkpoint_last.pth")
        if os.path.exists(last_ckpt):
            if is_main_process():
                print(f"Resuming from: {last_ckpt}")
            meta = load_checkpoint(model, optimizer, scheduler, last_ckpt, device)
            start_epoch = meta["epoch"] + 1
            global_step = meta["step"]
            best_metric = meta["best_metric"]

    # --- Training ---
    epochs = getattr(cfg, "epochs", 100)
    eval_epochs = getattr(cfg, "eval_epochs", 5)

    if is_main_process():
        print(f"Config: {cfg.name}")
        print(f"  Loss: {cfg.loss_type}")
        print(f"  Tversky α={cfg.tversky_alpha}, β={cfg.tversky_beta}")
        print(f"  τ={getattr(cfg, 'tau', 'N/A')}")
        print(f"  bbox_bg_weight={cfg.bbox_bg_weight}, bbox_pad_fraction={cfg.bbox_pad_fraction}")
        print(f"  masksup_ratio={getattr(cfg, 'masksup_ratio', 0.0)}")
        print(f"  Epochs: {start_epoch}→{epochs}, Eval every {eval_epochs}")
        print(f"  Batch: {batch_size} × {world_size} GPUs, Patch: {cfg.roi_size}")

    # TensorBoard
    writer = None
    if is_main_process():
        tb_dir = os.path.join(output_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, global_step, device, world_size,
        )
        epoch_time = time.time() - t0

        if is_main_process():
            print(f"[{cfg.name}] Epoch {epoch}/{epochs-1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
            if writer:
                writer.add_scalar("train/loss", avg_loss, epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Validation
        if (epoch + 1) % eval_epochs == 0 or epoch == epochs - 1:
            mean_dice, dice_dict = validate(model, val_loader, cfg, device, world_size)
            is_best = mean_dice > best_metric
            if is_best:
                best_metric = mean_dice

            if is_main_process():
                print(f"  Val Dice: {mean_dice:.4f} (best: {best_metric:.4f})")
                if writer:
                    writer.add_scalar("val/mean_dice", mean_dice, epoch)
                    writer.add_scalar("val/best_dice", best_metric, epoch)
                    for name, val in dice_dict.items():
                        writer.add_scalar(f"val_dice/{name}", val, epoch)
                    writer.flush()

            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch, step=global_step, best_metric=best_metric,
                output_dir=output_dir, is_best=is_best,
                save_weights_only=False,
                save_every_n_epochs=getattr(cfg, "save_every_n_epochs", 25),
            )
        elif getattr(cfg, "save_checkpoint", True):
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch, step=global_step, best_metric=best_metric,
                output_dir=output_dir, is_best=False,
                save_weights_only=False,
                save_every_n_epochs=getattr(cfg, "save_every_n_epochs", 25),
            )

    if is_main_process():
        print(f"\n[{cfg.name}] Training complete. Best Dice: {best_metric:.4f}")

        # Save summary JSON
        summary = {
            "name": cfg.name,
            "best_dice": best_metric,
            "epochs": epochs,
            "loss_type": cfg.loss_type,
            "tversky_alpha": cfg.tversky_alpha,
            "tversky_beta": cfg.tversky_beta,
            "tau": getattr(cfg, "tau", None),
            "bbox_bg_weight": cfg.bbox_bg_weight,
            "bbox_pad_fraction": cfg.bbox_pad_fraction,
            "masksup_ratio": getattr(cfg, "masksup_ratio", 0.0),
        }
        with open(os.path.join(output_dir, "ablation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to: {output_dir}/ablation_summary.json")

        if writer:
            writer.close()

    # Clean up model to free VRAM before next config
    del model, optimizer, scheduler, scaler
    del train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()

    return best_metric


def main():
    args = parse_args()

    # Load axis module
    axis_mod = load_axis_module(args.axis)

    if args.list:
        # List all configs and exit
        all_configs = axis_mod.get_all_configs()
        print(f"\n{'='*60}")
        print(f"Ablation axis: {args.axis}")
        print(f"Total configs: {len(all_configs)}")
        print(f"{'='*60}")
        for i, (name, cfg) in enumerate(all_configs):
            completed = "✓" if is_completed(cfg.output_dir) else " "
            print(f"  [{completed}] {i:2d}. {name}")
            print(f"        loss={cfg.loss_type}, α={cfg.tversky_alpha}, β={cfg.tversky_beta}")
            print(f"        τ={getattr(cfg, 'tau', 'N/A')}, "
                  f"bbox_bg={cfg.bbox_bg_weight}, bbox_pad={cfg.bbox_pad_fraction}, "
                  f"masksup={getattr(cfg, 'masksup_ratio', 0.0)}")
        return

    # DDP must be initialized before training
    from train import setup_ddp, cleanup_ddp, is_main_process

    local_rank, world_size, global_rank = setup_ddp()

    # Determine which configs to run
    if args.config:
        # Run single specific config
        cfg = axis_mod.get_config(args.config)
        configs_to_run = [(args.config, cfg)]
    else:
        # Run range from priority list
        all_configs = axis_mod.get_all_configs()
        # Read start/end from CLI args or env vars
        start = args.start_idx if args.start_idx is not None else int(os.environ.get("ABL_START", 0))
        end_raw = args.end_idx if args.end_idx is not None else int(os.environ.get("ABL_END", -1))
        end = end_raw if end_raw > 0 else len(all_configs)
        configs_to_run = all_configs[start:end]

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"3D ABLATION EXPERIMENT: {args.axis}")
        print(f"Configs to run: {len(configs_to_run)}")
        print(f"  {[name for name, _ in configs_to_run]}")
        print(f"{'='*60}")

    results = {}
    for i, (name, cfg) in enumerate(configs_to_run):
        if args.skip_completed and is_completed(cfg.output_dir):
            if is_main_process():
                print(f"\n[{i+1}/{len(configs_to_run)}] SKIPPING {name} (already completed)")
            continue

        if is_main_process():
            print(f"\n[{i+1}/{len(configs_to_run)}] Running: {name}")
            t0 = time.time()

        best_dice = run_single_config(cfg, resume=args.resume)
        results[name] = best_dice

        if is_main_process():
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(configs_to_run)}] {name} done in {elapsed/3600:.1f}h, "
                  f"best Dice: {best_dice:.4f}")

    # Print summary
    if is_main_process() and results:
        print(f"\n{'='*60}")
        print(f"ABLATION RESULTS: {args.axis}")
        print(f"{'='*60}")
        for name, dice in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {name:30s}  Dice: {dice:.4f}")

        # Save overall summary
        summary_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/ablations"
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"summary_{args.axis}.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {summary_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
