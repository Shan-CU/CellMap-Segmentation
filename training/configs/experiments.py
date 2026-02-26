"""
Ablation experiment configurations.

Defines all ablation experiments as named configurations that can be run
via the training script. Each experiment tests one variable while holding
everything else constant.

Phase 1: Quick ablations (50 epochs, 500 iters/epoch) on ResNet 2D
  - A: Loss function sweep (including Focal Tversky, Unified Focal, Boundary)
  - B: Tversky α/β sweep (including high-α precision configs from R2)
  - C: Class weighting (τ) sweep (0, 0.5, 1.0, 1.5, 2.0)
  - D: Masking strategy sweep
  - E: Training technique sweep (EMA, deep supervision, sampler ablation)

Phase 1b: Quick ablations on SegResNet 3D (same sweeps, 3D data)

Phase 2: Architecture comparison with winning loss/masking (100 epochs)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""
    experiment_name: str
    model: str
    loss: str
    loss_kwargs: Dict = field(default_factory=dict)
    model_kwargs: Dict = field(default_factory=dict)
    use_foreground_mask: bool = True

    # Data
    batch_size: int = 8
    input_shape: List[int] = field(default_factory=lambda: [1, 256, 256])
    input_scale: List[float] = field(default_factory=lambda: [8, 8, 8])

    # Training
    epochs: int = 50
    iterations_per_epoch: int = 500
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 5

    # Validation
    validation_time_limit: int = 120
    val_every_n_epochs: int = 5

    # AMP
    amp: bool = True

    # EMA
    ema: bool = False
    ema_decay: float = 0.999

    # Deep supervision
    deep_supervision: bool = False
    ds_weights: Optional[List[float]] = None

    # Sampler
    weighted_sampler: bool = True

    # Data loading improvements
    intensity_aug: bool = False
    class_aware_sampling: bool = False

    def to_cli_args(self, run_dir: str = "runs/ablation") -> str:
        """Convert to CLI argument string for training/train.py."""
        args = [
            f"--experiment_name {self.experiment_name}",
            f"--run_dir {run_dir}",
            f"--model {self.model}",
            f"--loss {self.loss}",
            f"--batch_size {self.batch_size}",
            f"--input_shape {' '.join(map(str, self.input_shape))}",
            f"--input_scale {' '.join(map(str, self.input_scale))}",
            f"--epochs {self.epochs}",
            f"--iterations_per_epoch {self.iterations_per_epoch}",
            f"--learning_rate {self.learning_rate}",
            f"--max_grad_norm {self.max_grad_norm}",
            f"--scheduler {self.scheduler}",
            f"--warmup_epochs {self.warmup_epochs}",
            f"--validation_time_limit {self.validation_time_limit}",
            f"--val_every_n_epochs {self.val_every_n_epochs}",
        ]
        if self.loss_kwargs:
            args.append(f"--loss_kwargs '{json.dumps(self.loss_kwargs)}'")
        if self.model_kwargs:
            args.append(f"--model_kwargs '{json.dumps(self.model_kwargs)}'")
        if not self.use_foreground_mask:
            args.append("--no_foreground_mask")
        if self.amp:
            args.append("--amp")
        else:
            args.append("--no_amp")
        if self.ema:
            args.append(f"--ema --ema_decay {self.ema_decay}")
        if self.deep_supervision:
            args.append("--deep_supervision")
            if self.ds_weights:
                args.append(f"--ds_weights {' '.join(map(str, self.ds_weights))}")
        if not self.weighted_sampler:
            args.append("--no_weighted_sampler")
        if self.intensity_aug:
            args.append("--intensity_aug")
        if self.class_aware_sampling:
            args.append("--class_aware_sampling")
        return " \\\n    ".join(args)


# ============================================================================
# PHASE 1A: Loss function sweep (2D, ResNet baseline)
# ============================================================================

LOSS_SWEEP_2D = [
    ExperimentConfig(
        experiment_name="loss_2d_bce",
        model="resnet_2d", loss="bce",
    ),
    ExperimentConfig(
        experiment_name="loss_2d_focal",
        model="resnet_2d", loss="focal",
    ),
    ExperimentConfig(
        experiment_name="loss_2d_dice_bce",
        model="resnet_2d", loss="dice_bce",
    ),
    ExperimentConfig(
        experiment_name="loss_2d_tversky",
        model="resnet_2d", loss="tversky",
    ),
    ExperimentConfig(
        experiment_name="loss_2d_balanced_softmax_tversky",
        model="resnet_2d", loss="balanced_softmax_tversky",
    ),
    # --- NEW: Focal Tversky (Abraham & Khan 2019) ---
    ExperimentConfig(
        experiment_name="loss_2d_focal_tversky",
        model="resnet_2d", loss="focal_tversky",
    ),
    # --- NEW: Asymmetric Unified Focal (Yeung et al. MedIA 2022) ---
    ExperimentConfig(
        experiment_name="loss_2d_unified_focal",
        model="resnet_2d", loss="unified_focal",
    ),
    # --- NEW: Boundary-weighted Tversky ---
    ExperimentConfig(
        experiment_name="loss_2d_boundary_tversky",
        model="resnet_2d", loss="boundary_tversky",
    ),
]

# ============================================================================
# PHASE 1B: Tversky α/β sweep (2D, ResNet)
# ============================================================================

TVERSKY_SWEEP_2D = [
    ExperimentConfig(
        experiment_name="tversky_2d_balanced",
        model="resnet_2d", loss="tversky_balanced",
    ),
    ExperimentConfig(
        experiment_name="tversky_2d_precision_06_04",
        model="resnet_2d", loss="tversky",
        loss_kwargs={"alpha": 0.6, "beta": 0.4},
    ),
    ExperimentConfig(
        experiment_name="tversky_2d_precision_07_03",
        model="resnet_2d", loss="tversky_precision",
    ),
    ExperimentConfig(
        experiment_name="tversky_2d_recall",
        model="resnet_2d", loss="tversky_recall",
    ),
    # --- NEW: High-α configs (from R2 precision-boosting findings) ---
    ExperimentConfig(
        experiment_name="tversky_2d_a08_b04",
        model="resnet_2d", loss="tversky_a08_b04",
    ),
    ExperimentConfig(
        experiment_name="tversky_2d_a08_b06",
        model="resnet_2d", loss="tversky_a08_b06",
    ),
]

# ============================================================================
# PHASE 1C: Class weighting (τ) sweep (2D, ResNet)
# ============================================================================

WEIGHTING_SWEEP_2D = [
    ExperimentConfig(
        experiment_name="tau_2d_0",
        model="resnet_2d", loss="bst_tau0",
    ),
    ExperimentConfig(
        experiment_name="tau_2d_05",
        model="resnet_2d", loss="bst_tau05",
    ),
    ExperimentConfig(
        experiment_name="tau_2d_10",
        model="resnet_2d", loss="balanced_softmax_tversky",  # τ=1.0 default
    ),
    ExperimentConfig(
        experiment_name="tau_2d_15",
        model="resnet_2d", loss="bst_tau15",
    ),
    # --- NEW: τ=2.0 (tested in R2, strong logit adjustment) ---
    ExperimentConfig(
        experiment_name="tau_2d_20",
        model="resnet_2d", loss="bst_tau20",
    ),
]

# ============================================================================
# PHASE 1D: Masking strategy sweep (2D, ResNet)
# ============================================================================

MASKING_SWEEP_2D = [
    ExperimentConfig(
        experiment_name="mask_2d_none",
        model="resnet_2d", loss="bst_no_bbox",
        use_foreground_mask=False,
    ),
    ExperimentConfig(
        experiment_name="mask_2d_fg_only",
        model="resnet_2d", loss="bst_no_bbox",
        use_foreground_mask=True,
    ),
    ExperimentConfig(
        experiment_name="mask_2d_bbox_only",
        model="resnet_2d", loss="balanced_softmax_tversky",
        use_foreground_mask=False,
    ),
    ExperimentConfig(
        experiment_name="mask_2d_bbox_fg",
        model="resnet_2d", loss="balanced_softmax_tversky",
        use_foreground_mask=True,
    ),
    ExperimentConfig(
        experiment_name="mask_2d_bbox_loose",
        model="resnet_2d", loss="bst_bbox_loose",
        use_foreground_mask=True,
    ),
    ExperimentConfig(
        experiment_name="mask_2d_masksup03",
        model="resnet_2d", loss="bst_masksup03",
        use_foreground_mask=True,
    ),
    ExperimentConfig(
        experiment_name="mask_2d_masksup03_no_bbox",
        model="resnet_2d", loss="bst_masksup03_no_bbox",
        use_foreground_mask=True,
    ),
]

# ============================================================================
# PHASE 1E: Training technique sweep (2D, ResNet)
# ============================================================================

TECHNIQUE_SWEEP_2D = [
    # EMA — exponential moving average (standard in nnU-Net v2, Auto3DSeg)
    ExperimentConfig(
        experiment_name="tech_2d_ema",
        model="resnet_2d", loss="balanced_softmax_tversky",
        ema=True, ema_decay=0.999,
    ),
    # Weighted sampler ablation — does data-level reweighting help on top of τ?
    ExperimentConfig(
        experiment_name="tech_2d_no_weighted_sampler",
        model="resnet_2d", loss="balanced_softmax_tversky",
        weighted_sampler=False,
    ),
    # Focal Tversky with mild focal (γ=0.5) — ablation of focal strength
    ExperimentConfig(
        experiment_name="tech_2d_focal_tversky_mild",
        model="resnet_2d", loss="focal_tversky_g05",
    ),
]

# ============================================================================
# PHASE 1E-bis: Re-validate sweep E techniques with dice_bce base loss
# The original sweep E used BST — need to confirm EMA / sampler effects
# transfer to the winning loss (dice_bce) before Phase 2.
# Baseline = loss_2d_dice_bce from sweep A (already ran).
# ============================================================================

TECHNIQUE_DICEBCE_SWEEP_2D = [
    # EMA with dice_bce — does EMA help the winning loss?
    ExperimentConfig(
        experiment_name="tech_2d_dicebce_ema",
        model="resnet_2d", loss="dice_bce",
        use_foreground_mask=True,
        ema=True, ema_decay=0.999,
    ),
    # No weighted sampler with dice_bce — confirm sampler is still critical
    ExperimentConfig(
        experiment_name="tech_2d_dicebce_no_sampler",
        model="resnet_2d", loss="dice_bce",
        use_foreground_mask=True,
        weighted_sampler=False,
    ),
]

# ============================================================================
# PHASE 1 (3D versions): Same sweeps on SegResNet 3D
# ============================================================================

def _make_3d_variant(cfg: ExperimentConfig) -> ExperimentConfig:
    """Create 3D variant of a 2D experiment config."""
    name = cfg.experiment_name.replace("_2d_", "_3d_")
    return ExperimentConfig(
        experiment_name=name,
        model="segresnet_3d",
        loss=cfg.loss,
        loss_kwargs=cfg.loss_kwargs,
        use_foreground_mask=cfg.use_foreground_mask,
        batch_size=2,  # smaller batch for 3D
        input_shape=[128, 128, 128],
        input_scale=[8, 8, 8],
        epochs=50,
        iterations_per_epoch=250,  # fewer iters for 3D (slower)
        learning_rate=1e-4,
        validation_time_limit=180,
        val_every_n_epochs=5,
    )


LOSS_SWEEP_3D = [_make_3d_variant(c) for c in LOSS_SWEEP_2D]
TVERSKY_SWEEP_3D = [_make_3d_variant(c) for c in TVERSKY_SWEEP_2D]
WEIGHTING_SWEEP_3D = [_make_3d_variant(c) for c in WEIGHTING_SWEEP_2D]
MASKING_SWEEP_3D = [_make_3d_variant(c) for c in MASKING_SWEEP_2D]

# Technique sweep for 3D — includes deep supervision (only works with SegResNet)
_TECHNIQUE_3D_BASE = [_make_3d_variant(c) for c in TECHNIQUE_SWEEP_2D]
TECHNIQUE_SWEEP_3D = _TECHNIQUE_3D_BASE + [
    # Deep supervision — SegResNetDS with dsdepth=4 (multi-scale gradients)
    ExperimentConfig(
        experiment_name="tech_3d_deep_supervision",
        model="segresnet_3d",
        loss="balanced_softmax_tversky",
        deep_supervision=True,
        model_kwargs={"dsdepth": 4},
        batch_size=2,
        input_shape=[128, 128, 128],
        input_scale=[8, 8, 8],
        epochs=50,
        iterations_per_epoch=250,
        learning_rate=1e-4,
        validation_time_limit=180,
        val_every_n_epochs=5,
    ),
]


# ============================================================================
# VALIDATION: Data loading improvements (intensity aug + class-aware sampling)
# Base config: dice_bce + EMA + fg_mask + weighted_sampler (best from Phase 1)
# These experiments validate cherry-picked features from OrganelleSeg.
# ============================================================================

_VAL_BASE = dict(
    model="resnet_2d",
    loss="dice_bce",
    use_foreground_mask=True,
    ema=True,
    ema_decay=0.999,
    epochs=50,
    iterations_per_epoch=500,
    val_every_n_epochs=5,
)

VALIDATION_DATA_LOADING = [
    # Intensity augmentation only (brightness ±0.1, contrast 0.8-1.2, noise σ 0.01-0.05)
    ExperimentConfig(
        experiment_name="val_intensity_aug",
        intensity_aug=True,
        class_aware_sampling=False,
        **_VAL_BASE,
    ),
    # Class-aware crop weighting only (inverse-sqrt, 70/30 blend)
    ExperimentConfig(
        experiment_name="val_crop_weights",
        intensity_aug=False,
        class_aware_sampling=True,
        **_VAL_BASE,
    ),
    # Combined: both improvements
    ExperimentConfig(
        experiment_name="val_combined",
        intensity_aug=True,
        class_aware_sampling=True,
        **_VAL_BASE,
    ),
]


# ============================================================================
# PHASE 2: Architecture comparison (after picking best loss/masking)
# ============================================================================

def make_arch_comparison_2d(
    loss: str = "balanced_softmax_tversky",
    loss_kwargs: dict = None,
    use_foreground_mask: bool = True,
    epochs: int = 100,
) -> list[ExperimentConfig]:
    """Generate architecture comparison configs for 2D models."""
    loss_kwargs = loss_kwargs or {}
    models = ["resnet_2d", "unet_2d", "swin_2d", "vit_2d"]
    return [
        ExperimentConfig(
            experiment_name=f"arch_2d_{m.replace('_2d', '')}",
            model=m, loss=loss, loss_kwargs=loss_kwargs,
            use_foreground_mask=use_foreground_mask,
            epochs=epochs,
            iterations_per_epoch=1000,
            val_every_n_epochs=5,
        )
        for m in models
    ]


def make_arch_comparison_3d(
    loss: str = "balanced_softmax_tversky",
    loss_kwargs: dict = None,
    use_foreground_mask: bool = True,
    epochs: int = 100,
) -> list[ExperimentConfig]:
    """Generate architecture comparison configs for 3D models."""
    loss_kwargs = loss_kwargs or {}
    models = ["segresnet_3d", "swinunetr_3d", "unet_3d", "resnet_3d"]
    return [
        ExperimentConfig(
            experiment_name=f"arch_3d_{m.replace('_3d', '')}",
            model=m, loss=loss, loss_kwargs=loss_kwargs,
            use_foreground_mask=use_foreground_mask,
            batch_size=2,
            input_shape=[128, 128, 128],
            input_scale=[8, 8, 8],
            epochs=epochs,
            iterations_per_epoch=500,
            val_every_n_epochs=5,
            validation_time_limit=180,
        )
        for m in models
    ]


# ============================================================================
# ALL EXPERIMENTS
# ============================================================================

ALL_PHASE1_2D = LOSS_SWEEP_2D + TVERSKY_SWEEP_2D + WEIGHTING_SWEEP_2D + MASKING_SWEEP_2D + TECHNIQUE_SWEEP_2D
ALL_PHASE1_3D = LOSS_SWEEP_3D + TVERSKY_SWEEP_3D + WEIGHTING_SWEEP_3D + MASKING_SWEEP_3D + TECHNIQUE_SWEEP_3D
ALL_PHASE1 = ALL_PHASE1_2D + ALL_PHASE1_3D


def print_experiment_summary():
    """Print a summary of all defined experiments."""
    print("=" * 80)
    print("PHASE 1: Quick Ablations (50 epochs)")
    print("=" * 80)

    groups = [
        ("Loss Sweep 2D", LOSS_SWEEP_2D),
        ("Tversky α/β Sweep 2D", TVERSKY_SWEEP_2D),
        ("Weighting (τ) Sweep 2D", WEIGHTING_SWEEP_2D),
        ("Masking Strategy Sweep 2D", MASKING_SWEEP_2D),
        ("Training Technique Sweep 2D", TECHNIQUE_SWEEP_2D),
        ("Loss Sweep 3D", LOSS_SWEEP_3D),
        ("Tversky α/β Sweep 3D", TVERSKY_SWEEP_3D),
        ("Weighting (τ) Sweep 3D", WEIGHTING_SWEEP_3D),
        ("Masking Strategy Sweep 3D", MASKING_SWEEP_3D),
        ("Training Technique Sweep 3D", TECHNIQUE_SWEEP_3D),
    ]

    total = 0
    for group_name, experiments in groups:
        print(f"\n  {group_name} ({len(experiments)} experiments):")
        for exp in experiments:
            dim = "3D" if "3d" in exp.model else "2D"
            print(f"    {exp.experiment_name:<40} [{exp.model}] loss={exp.loss}")
        total += len(experiments)

    print(f"\n  Total Phase 1 experiments: {total}")
    print(f"  Estimated GPU hours (2D): ~{len(ALL_PHASE1_2D) * 1.5:.0f}h on L40S")
    print(f"  Estimated GPU hours (3D): ~{len(ALL_PHASE1_3D) * 3:.0f}h on L40S")


if __name__ == "__main__":
    print_experiment_summary()
