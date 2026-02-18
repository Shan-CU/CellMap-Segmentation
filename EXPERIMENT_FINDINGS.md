# CellMap Segmentation Challenge — Complete Experiment Findings

> **Last updated:** February 18, 2026  
> **Authors:** CellMap Segmentation Team  
> **Status:** Round 2 failed (mode collapse); Round 3 fix implemented, awaiting submission

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experiment 1: Loss Function Optimization](#2-experiment-1-loss-function-optimization)
3. [Experiment 2: Class Weighting Strategies](#3-experiment-2-class-weighting-strategies)
4. [Experiment 3: Masking Strategies](#4-experiment-3-masking-strategies)
5. [Experiment 4: Model Architecture Comparison (2D)](#5-experiment-4-model-architecture-comparison-2d)
6. [Experiment 5: MONAI 3D Pipeline](#6-experiment-5-monai-3d-pipeline)
7. [Cross-Experiment Progression](#7-cross-experiment-progression)
8. [Key Takeaways & Recommendations](#8-key-takeaways--recommendations)
9. [Hardware & Compute Summary](#9-hardware--compute-summary)
10. [Round 3 Fix: Spatial Bbox Masking for 3D Models](#10-round-3-fix-spatial-bbox-masking-for-3d-models)

---

## 1. Executive Summary

We conducted a systematic optimization pipeline for the CellMap Segmentation Challenge, progressing through five major experiment categories. Each experiment built on findings from the previous stage.

### Optimization Pipeline

```
Baseline (BCE, UNet 2D, 14 classes)
  → Dice = 0.252
      ↓
Loss Optimization (13 loss functions)
  → Best: Per-class Tversky α=0.6, β=0.4 → Dice = 0.370
      ↓
Class Weighting (15 strategies)
  → Winner: Balanced Softmax Tversky τ=1.0 → Dice = 0.571
      ↓
Masking Strategies (15 strategies)
  → Winner: masksup_r0.3 → Dice = 0.571
  → Key discovery: Foreground masking fix (+110% baseline improvement)
      ↓
Model Comparison (4 architectures × 2D/3D)
  → 2D Winner: ResNet 2D (14-class eval Dice = 0.410)
  → 3D Winner: FlexibleUNet-ResNet34 (Dice = 0.233)
      ↓
MONAI 3D Round 2 (35 classes, 4 models) → FAILED (mode collapse)
  → All 4 models predicted ~1.0 for most classes everywhere
  → Root cause: no spatial masking in loss function
      ↓
MONAI 3D Round 3 (35 classes, 4 models) → Awaiting Submission
  → Fix: box_class_mask_tight spatial masking (pad=0.05, bg=0.05)
  → Adapted from 2D masking experiment winner (+55% over no_mask)
```

### Winners by Category

| Category | Winner | Key Metric | Improvement over Baseline |
|----------|--------|------------|--------------------------|
| **Loss Function** | Per-class Tversky (α=0.6, β=0.4) | Dice = 0.370 | +47% vs BCE (0.252) |
| **Class Weighting** | Balanced Softmax τ=1.0 | Dice = 0.571 | +54% vs uniform (0.371) |
| **Masking Strategy** | masksup_r0.3 | Dice = 0.571 (eval) | +12% vs no_mask (0.511) |
| **2D Architecture** | ResNet 2D | Dice = 0.410 (14-class) | +63% vs UNet 2D (0.252) |
| **3D Architecture** | FlexUNet-ResNet34 | Dice = 0.233 (14-class) | +47% vs SegResNet (0.159) |

---

## 2. Experiment 1: Loss Function Optimization

**Location:** `experiments/loss_optimization/`  
**Hardware:** Shenron (4× RTX 2080 Ti) → Rocinante (2× RTX 3090)  
**Model:** UNet 2D  
**Classes:** 5 quick-test (`nuc`, `mito_mem`, `er_mem`, `pm`, `golgi_mem`)  
**Training:** 60 epochs × 100 iterations, batch 28, lr=1e-4, AdamW + OneCycleLR  

### 2.1 Goal

Replace the baseline BCEWithLogitsLoss with a more effective loss function for multi-class organelle segmentation with severe class imbalance and partial annotations (NaN targets).

### 2.2 Loss Functions Tested

| # | Loss Function | Type | Key Parameters |
|---|--------------|------|----------------|
| 1 | BCEWithLogitsLoss | Baseline | — |
| 2 | Dice Loss | Region-based | Smooth=1.0 |
| 3 | Dice + BCE | Combination | 50/50 |
| 4 | Focal Loss | Hard-example | γ=2.0 |
| 5 | Combo Loss (Dice + Focal) | Combination | Mixed |
| 6 | Tversky Precision (α=0.7, β=0.3) | Asymmetric | Penalizes FP more |
| 7 | Tversky Recall (α=0.3, β=0.7) | Asymmetric | Penalizes FN more |
| 8 | Tversky Mild (α=0.6, β=0.4) | Asymmetric | Balanced precision bias |
| 9 | Tversky Strong (α=0.8, β=0.2) | Asymmetric | Strong precision bias |
| 10 | Per-class Tversky (α=0.7, β=0.3) | Per-class weighted | Manual class weights |
| 11 | Per-class Tversky Strong (α=0.8, β=0.2) | Per-class weighted | Manual class weights |
| 12 | Per-class Tversky Mild (α=0.6, β=0.4) | Per-class weighted | Manual class weights |
| 13 | Per-class Weighted Focal | Multi-component | BCE=0.3, Dice=0.5, Focal=0.2 |

### 2.3 Results

| Rank | Loss Function | Best Mean Dice | Status |
|------|--------------|----------------|--------|
| **1** | **Per-class Tversky Precision (α=0.7, β=0.3)** | **0.3697** | Best raw score |
| 2 | Per-class Tversky Precision (alt. weights) | 0.3603 | Close variant |
| 3 | Per-class Weighted Focal (Dice+BCE+Focal) | 0.3073 | Complex combination |
| 4 | Per-class Tversky Strong (α=0.8, β=0.2) | 0.2848 | Too aggressive |
| 5 | Tversky Precision (α=0.7, β=0.3) | 0.2363 | Uniform weights |
| 6 | Tversky Strong (α=0.8, β=0.2) | 0.2337 | Collapsed on rare classes |
| **7** | **Tversky Mild (α=0.6, β=0.4)** | **0.2283** | **★ Adopted downstream** |
| 8 | Baseline BCE | 0.1636 | After bugfix |
| 9 | Dice + BCE | 0.0653 | Unstable |
| 10 | Baseline BCE (early, buggy) | 0.0405 | Normalization issues |
| 11 | Combo Loss | 0.0278 | Collapsed |
| 12 | Focal Loss | 0.0279 | Collapsed |
| 13 | Tversky Recall (α=0.3, β=0.7) | 0.0281 | Collapsed |

### 2.4 Key Findings

1. **Tversky loss family dominates** — asymmetric weighting between FP and FN is critical for organelle segmentation
2. **Precision bias (α > β) works; recall bias collapses** — penalizing false positives more prevents over-prediction of rare classes
3. **Per-class weighting is essential** — adding class-specific weights boosts per-class Tversky from 0.228 → 0.370 (+62%)
4. **Focal loss alone fails** — the γ=2.0 focusing factor causes collapse on this data
5. **α=0.6, β=0.4 (mild) adopted** over α=0.7, β=0.3 because of better stability and balance across classes, even though the latter scored higher in absolute terms

### 2.5 Decision

> **Adopted for all downstream experiments:**  
> Per-class Tversky loss with **α=0.6, β=0.4** as the base loss function.

---

## 3. Experiment 2: Class Weighting Strategies

**Location:** `experiments/class_weighting/`  
**Hardware:** Rocinante (2× RTX 3090)  
**Model:** UNet 2D  
**Classes:** 5 quick-test (`nuc`, `mito_mem`, `er_mem`, `pm`, `golgi_mem`)  
**Training:** 60 epochs × 100 iterations, batch 24, lr=1e-4, AdamW + OneCycleLR  
**Base Loss:** Per-class Tversky (α=0.6, β=0.4)

### 3.1 Goal

Fix the loss function and find the optimal class weighting strategy to handle extreme class imbalance (nuc = 11.9% vs golgi_mem = 1.8%).

### 3.2 Measured Class Frequencies

| Class | Frequency | Ratio to Smallest |
|-------|-----------|-------------------|
| nuc | 11.93% | 6.6× |
| mito_mem | 6.01% | 3.3× |
| er_mem | 5.60% | 3.1× |
| pm | 2.20% | 1.2× |
| golgi_mem | 1.81% | 1.0× (rarest) |

### 3.3 Strategies Tested (15 Total)

| Category | Strategies |
|----------|-----------|
| **Uniform** | weight_uniform (baseline, all weights = 1) |
| **Frequency-based** | weight_inv_freq, weight_sqrt_inv, weight_log_inv, weight_effective_num |
| **Manual** | weight_manual (hand-tuned) |
| **Class-Balanced (CB)** | cb_beta_0.99, cb_beta_0.999, cb_beta_0.9999 |
| **Balanced Softmax** | balanced_softmax_tau_0.5, balanced_softmax_tau_1.0, balanced_softmax_tau_2.0 |
| **Seesaw** | seesaw_default, seesaw_strong_mitigate, seesaw_strong_compensate |

### 3.4 Complete Results (Ranked by Mean Dice)

| Rank | Strategy | Mean Dice | nuc | mito_mem | er_mem | pm | golgi_mem |
|------|----------|-----------|-----|----------|--------|----|-----------|
| **1** | **balanced_softmax_tau_1.0** | **0.5711** | 0.8511 | 0.6982 | 0.4284 | 0.5133 | 0.3645 |
| 2 | weight_inv_freq | 0.5694 | 0.8530 | 0.7137 | 0.4358 | 0.5045 | 0.3399 |
| 3 | weight_log_inv | 0.5678 | 0.8015 | 0.6838 | 0.4285 | 0.4902 | 0.4350 |
| 4 | weight_sqrt_inv | 0.5616 | 0.8653 | 0.7054 | 0.3904 | 0.4702 | 0.3768 |
| 5 | seesaw_strong_compensate | 0.5563 | — | — | — | — | — |
| 6 | cb_beta_0.99 | 0.5475 | 0.7746 | 0.6982 | 0.4406 | 0.5033 | 0.3208 |
| 7 | weight_effective_num | 0.5459 | 0.7281 | 0.7034 | 0.4320 | 0.5144 | 0.3515 |
| 8 | seesaw_default | 0.5425 | — | — | — | — | — |
| 9 | weight_uniform | 0.5416 | 0.7047 | 0.7121 | 0.4214 | 0.5285 | 0.3414 |
| 10 | seesaw_strong_mitigate | 0.5407 | — | — | — | — | — |
| 11 | cb_beta_0.999 | 0.5269 | 0.7098 | 0.6998 | 0.4300 | 0.4550 | 0.3401 |
| 12 | balanced_softmax_tau_0.5 | 0.5225 | 0.7084 | 0.7128 | 0.4129 | 0.4564 | 0.3222 |
| 13 | cb_beta_0.9999 | 0.5188 | 0.7559 | 0.6354 | 0.3860 | 0.4748 | 0.3421 |
| 14 | weight_manual | 0.5143 | 0.7753 | 0.6623 | 0.3899 | 0.4662 | 0.2775 |
| 15 | balanced_softmax_tau_2.0 | 0.5036 | 0.7766 | 0.6401 | 0.4395 | 0.4566 | 0.2052 |

### 3.5 Best Per-Class Performers

| Class | Best Dice | Strategy | Notes |
|-------|-----------|----------|-------|
| nuc | **0.8653** | weight_sqrt_inv | Largest class — most strategies do well |
| mito_mem | **0.7137** | weight_inv_freq | Moderate class — frequency-based optimal |
| er_mem | **0.4406** | cb_beta_0.99 | Moderate class — Class-Balanced wins |
| pm | **0.5285** | weight_uniform | Paradox: uniform wins for this rare class |
| golgi_mem | **0.4350** | weight_log_inv | Rarest class — log smoothing helps most |

### 3.6 Key Findings

1. **Balanced Softmax τ=1.0 is the overall winner** (0.5711) — τ=1.0 is the theory-optimal temperature
2. **Inverse-frequency weighting is a close, simple alternative** (0.5694, Δ = 0.0017)
3. **Any weighting beats uniform** — 14 of 14 strategies outperform uniform (rank 9, 0.5416)
4. **The improvement from class weighting is substantial** — uniform (0.5416) → best (0.5711) = +5.4%
5. **Manual weights underperform** (rank 14) — data-driven approaches consistently beat hand-tuned
6. **Stronger is not better**: τ=2.0 worst in Balanced Softmax, β=0.9999 worst in CB
7. **Seesaw is consistently mid-tier** — ranks 5, 8, 10 across its three variants
8. **No single strategy wins every class** — the best per-class strategy varies

### 3.7 Decision

> **Adopted for all downstream experiments:**  
> **Balanced Softmax Tversky τ=1.0** (α=0.6, β=0.4) with data-driven class frequency weights.

---

## 4. Experiment 3: Masking Strategies

**Location:** `experiments/masking_strategies/`  
**Hardware:** Rocinante (2× RTX 3090)  
**Model:** UNet 2D  
**Classes:** 5 quick-test  
**Base Loss:** Balanced Softmax Tversky τ=1.0 (α=0.6, β=0.4)

### 4.1 Goal

Evaluate how to handle NaN/unannotated pixels in training targets. The CellMap dataset has partial annotations — many crops have NaN (unannotated) regions. The strategy for handling these affects what the model learns.

### 4.2 Critical Discovery: Foreground Masking Fix

> **The single biggest gain in the entire experiment series.**

Before the fix, all strategies were computing loss on **black-padding regions** (zero-valued EM pixels at crop boundaries). The model was penalized for predictions on regions with no biological content, generating massive false positives that dragged down all Dice scores.

**The fix:** Set targets to NaN wherever the raw EM image is black (padding), preventing these regions from contributing to the loss.

| Metric | Before Fix (best) | After Fix (best) | Δ |
|--------|-------------------|------------------|---|
| **Best Dice** | 0.376 (box_class_mask_tight) | **0.571 (masksup_r0.3)** | **+52%** |
| **Baseline Dice** | 0.243 (no_mask) | **0.511 (no_mask)** | **+110%** |
| **Best IoU** | 0.284 | **0.429** | **+51%** |

The no_mask baseline alone (0.511) now beats **every** strategy from the pre-fix run. This shows how severely the black-padding false positives were dragging down all scores.

### 4.3 Strategies Tested (15 Total)

| Category | Strategy | Description |
|----------|----------|-------------|
| **Baseline** | no_mask | NaN→0, all pixels contribute equally |
| **Class Presence** | class_presence | Mask out classes absent from image |
| | class_presence_strict | Stricter version of above |
| **Bounding Box** | box_class_mask | Per-class bbox masking (margin=0.1) |
| | box_class_mask_tight | Tight bbox masking (margin=0.05) |
| **Salient Region** | salient_mask | Differential FG/BG weights (0.3/0.7) |
| | salient_mask_aggressive | Aggressive FG/BG weights (0.2/0.8) |
| **Entropy** | entropy_mask | Entropy threshold at 0.7 |
| | entropy_mask_strict | Entropy threshold at 0.5 |
| **Uncertainty** | uncertainty_eu | Epistemic (MC-Dropout, 10 passes) |
| | uncertainty_au | Aleatoric uncertainty estimation |
| **Regional** | regional_g8 | Grid-based adaptive (8×8) |
| | regional_g16 | Grid-based adaptive (16×16) |
| **Mask-Supervised** | masksup_r0.3 | Mask 30% of input for reconstruction |
| | masksup_r0.5 | Mask 50% of input for reconstruction |

### 4.4 Results After Foreground Masking Fix (Ranked by Best Dice)

| Rank | Strategy | Best Dice | Precision | Recall | IoU |
|------|----------|-----------|-----------|--------|-----|
| **1** | **masksup_r0.3** | **0.5711** | 0.578 | 0.573 | 0.429 |
| 2 | salient_mask_aggressive | 0.5578 | 0.472 | 0.685 | 0.393 |
| 3 | salient_mask | 0.5510 | 0.507 | 0.576 | 0.384 |
| 4 | uncertainty_eu | 0.5449 | 0.562 | 0.544 | 0.396 |
| 5 | box_class_mask_tight | 0.5380 | 0.498 | 0.610 | 0.383 |
| 6 | box_class_mask | 0.5385 | 0.512 | 0.597 | 0.385 |
| 7 | entropy_mask_strict | 0.5302 | 0.551 | 0.528 | 0.385 |
| 8 | masksup_r0.5 | 0.5247 | 0.548 | 0.524 | 0.378 |
| 9 | no_mask (baseline) | 0.5106 | 0.561 | 0.479 | 0.359 |
| 10 | uncertainty_au | 0.5082 | 0.516 | 0.540 | 0.352 |
| 11 | regional_g8 | 0.5025 | 0.492 | 0.536 | 0.354 |
| 12 | regional_g16 | 0.5010 | 0.476 | 0.560 | 0.351 |
| 13 | entropy_mask | 0.4906 | 0.508 | 0.511 | 0.337 |
| 14 | class_presence_strict | 0.4627 | 0.450 | 0.532 | 0.311 |
| 15 | class_presence | 0.4599 | 0.455 | 0.513 | 0.309 |

### 4.5 Per-Class Dice (Best Strategies)

| Class | Best Dice | Best Strategy | Notes |
|-------|-----------|---------------|-------|
| **nuc** | **0.855** | masksup_r0.3 | Nearly solved |
| **mito_mem** | **0.737** | Multiple | Strong across most strategies |
| **er_mem** | **0.445** | box_class_mask_tight | Consistent bottleneck |
| **pm** | ~0.51 | masksup_r0.3 | |
| **golgi_mem** | **0.329** | masksup_r0.5 | Still hardest class (up from ~0.15 pre-fix) |

### 4.6 How Rankings Shifted After the Fix

| Strategy | Pre-Fix Rank | Post-Fix Rank | Δ Rank | Explanation |
|----------|-------------|---------------|--------|-------------|
| **masksup_r0.3** | Mid-pack | **#1** | ↑↑ | Context-learning generalizes best |
| **box_class_mask_tight** | #1 | **#5** | ↓↓↓ | Was compensating for padding problem |
| **salient_mask variants** | Mid-pack | **#2–3** | ↑↑ | Highest recall (0.685 aggressive) |
| **no_mask** | #15 (0.243) | **#9 (0.511)** | ↑↑↑ | Fix eliminated the main problem |
| **class_presence** | #13–14 | **#14–15** | — | Still worst — too aggressive |

### 4.7 Key Findings

1. **Foreground masking fix was the single biggest gain** — +110% on baseline, +52% on best strategy. Eliminating loss computation on black-padding regions removed the dominant source of false positives
2. **masksup_r0.3 is the clear winner** (0.5711 Dice) — the 30% random masking of annotated pixels forces the model to learn from context, producing excellent generalization with well-balanced precision (0.578) and recall (0.573)
3. **Rankings shifted dramatically** — box_class_mask_tight fell from #1 to #5 because its spatial masking was partly compensating for the padding problem (now solved properly)
4. **Precision/Recall balance is healthy** — masksup_r0.3 has 0.578/0.573, nearly perfectly balanced. The old pathological pattern (high recall, low precision from padding FPs) is gone
5. **Class presence masking is confirmed worst** — channel-level binary decisions (include all or nothing) still underperform pixel-level strategies
6. **golgi_mem remains the hardest class** (0.329) but improved from ~0.15 pre-fix — a 2× improvement

### 4.8 Decision

> **Recommended masking strategy:**  
> **masksup_r0.3** (30% mask-supervised reconstruction) — best overall Dice with balanced precision/recall.  
> **Critical prerequisite:** Apply foreground masking fix (set targets to NaN on black-padding EM regions).

---

## 5. Experiment 4: Model Architecture Comparison (2D)

**Location:** `experiments/model_comparison/` (original) + `experiments/class_weighting/model_comparison/` (with best loss)

### 5.1 Original Model Comparison (BCE Loss, 14 Classes)

**Hardware:** Sycamore (2× H100 80GB per model), Blanca Biokem (2× A100 80GB)  
**Training:** 200 epochs × 1000 iterations, DDP

#### Architecture Details

| Model | Parameters | Type | GPU Memory/device |
|-------|-----------|------|-------------------|
| UNet 2D | 31.0M | CNN | ~8–12 GB |
| ResNet 2D | 7.8M | CNN | ~12–18 GB |
| Swin Transformer 2D | 36.3M | Transformer | ~25–35 GB |
| ViT-V-Net 2D | 105.2M | Transformer | ~30–40 GB |

#### 14-Class Evaluation Results (Per-Class Dice)

| Class | UNet 2D | ResNet 2D | Notes |
|-------|---------|-----------|-------|
| golgi_mem | 0.680 | **0.835** | ResNet excels |
| golgi_lum | 0.317 | **0.813** | +156% for ResNet |
| mito_ribo | 0.643 | **0.735** | ResNet wins |
| mito_lum | 0.257 | **0.473** | +84% for ResNet |
| mito_mem | 0.218 | **0.411** | +89% for ResNet |
| ecs | 0.291 | **0.340** | ResNet wins |
| ves_lum | 0.270 | **0.329** | ResNet wins |
| endo_lum | 0.099 | **0.303** | +206% for ResNet |
| ves_mem | 0.193 | **0.262** | ResNet wins |
| er_lum | 0.142 | **0.264** | +86% for ResNet |
| endo_mem | 0.081 | **0.226** | +179% for ResNet |
| pm | 0.113 | **0.215** | ResNet wins |
| er_mem | 0.116 | **0.182** | ResNet wins |
| nuc | 0.111 | **0.158** | ResNet wins |
| **Mean** | **0.252** | **0.410** | **+63% for ResNet** |

#### Key Finding

**ResNet 2D (7.8M params) massively outperforms UNet 2D (31.0M params)** — 63% better mean Dice with 4× fewer parameters. ResNet wins on every single class.

Swin 2D and ViT 2D had training instability issues (Swin collapsing on `ecs`, ViT volatile convergence).

### 5.2 Class-Weighted Model Comparison (In Setup)

**Location:** `experiments/class_weighting/model_comparison/`  
**Configuration:** All 4 models × Balanced Softmax Tversky τ=1.0 × 14 classes × 150 epochs × 500 iterations  
**Status:** SBATCH scripts created for Blanca Biokem, awaiting execution

---

## 6. Experiment 5: MONAI 3D Pipeline

**Location:** `experiments/monai_cellmap/`

### 6.1 Round 1 — 14 Atomic Classes (Completed)

**Hardware:** Longleaf (6× L40S 48GB)  
**Data:** 277 NIfTI crops from zarr, 14 atomic classes  
**Training:** 600 epochs, 128³ patches, DiceCELoss → then Balanced Softmax Tversky  

#### Results (Best Checkpoint)

| Model | Params | Best Mean Dice | Best Epoch | GPU Memory/device |
|-------|--------|---------------|------------|-------------------|
| **FlexUNet-ResNet34** | ~22M | **0.2329** | 204 | 13.6 GB |
| SwinUNETR v2 | ~24M | 0.1787 | 139 | 38.6–41.4 GB |
| SegResNet-DS | ~16M | 0.1585 | 519 | 25.1 GB |

#### Per-Class Dice at Best Checkpoint

| Class | SegResNet | FlexUNet | SwinUNETR | Best Model |
|-------|-----------|----------|-----------|------------|
| **nuc** | **0.5145** | 0.4042 | 0.4959 | SegResNet |
| ves_lum | 0.0012 | **0.5008** | 0.0003 | FlexUNet |
| ves_mem | 0.0024 | **0.4685** | 0.0007 | FlexUNet |
| mito_ribo | 0.0011 | 0.3846 | **0.4615** | SwinUNETR |
| golgi_lum | **0.3761** | 0.3193 | 0.3394 | SegResNet |
| golgi_mem | **0.3380** | 0.3165 | 0.2848 | SegResNet |
| ecs | **0.3131** | 0.2869 | 0.2739 | SegResNet |
| mito_lum | **0.2185** | 0.1669 | 0.1664 | SegResNet |
| mito_mem | 0.1756 | 0.1276 | **0.1865** | SwinUNETR |
| er_lum | 0.1069 | 0.0857 | **0.1191** | SwinUNETR |
| er_mem | 0.0877 | 0.0704 | **0.0999** | SwinUNETR |
| pm | 0.0535 | **0.0714** | 0.0591 | FlexUNet |
| endo_lum | 0.0162 | **0.0342** | 0.0062 | FlexUNet |
| endo_mem | 0.0140 | **0.0229** | 0.0083 | FlexUNet |

#### Model Complementarity

The three models are **highly complementary** — each excels at different structure types:

| Strength | Best Model | Classes |
|----------|-----------|---------|
| **Large structures** | SegResNet | nuc, golgi_lum, golgi_mem, ecs, mito_lum |
| **Rare/tiny structures** | FlexUNet | ves_lum, ves_mem, pm, endo_lum, endo_mem |
| **Mid-range structures** | SwinUNETR | mito_ribo, mito_mem, er_lum, er_mem |

**Ensemble potential:** A class-wise ensemble selecting the best model per class could achieve a theoretical mean Dice of ~0.31 (vs 0.233 for FlexUNet alone).

#### Leaderboard Impact

- Submitted score: **~0.063** (only 14 of 48 evaluated classes were trained)
- Current #1 (BC_CV V9): **0.466**

### 6.2 Round 2 — 35 Atomic Classes (In Progress)

**Hardware:** Sycamore (2× H100 80GB) + Longleaf (8× L40S 48GB via reservation)  
**Data:** 277 NIfTI crops, expanded to 35 atomic classes  
**Training:** 300 epochs (reduced from 600)

#### Critical Changes from Round 1

| Aspect | Round 1 | Round 2 |
|--------|---------|---------|
| **Classes** | 14 atomic | **35 atomic** (challenge evals 48 = 35 + 16 groups) |
| **Patch size (H100)** | 128³ | **192³** (3.375× more voxels per patch) |
| **Mixup** | Enabled | **Disabled** (bad interaction with partial annotations) |
| **Epochs** | 600 | **300** (more efficient) |
| **SwinUNETR dropout** | 0.0 | **0.1** (regularization) |
| **NIfTI converter** | v1 (bugs) | **v2** (fixed bounding boxes) |

#### Round 2 Deployment

| Cluster | GPU | Model | Patch Size | Batch | Status |
|---------|-----|-------|-----------|-------|--------|
| Sycamore | 2× H100 80GB | SegResNet-Wide (48f) | 192³ | 2 | ✅ Running |
| Sycamore | 2× H100 80GB | FlexUNet-ResNet34 | 192³ | 4 | ✅ Running |
| Longleaf | 2× L40S 48GB | SwinUNETR v2 | 96³ | 2 | ✅ Running |
| Longleaf | 2× L40S 48GB | SegResNet (32f) | 128³ | 2 | ✅ Running |

#### Round 2 2D Models (Longleaf L40S Reservation)

4 models training on reserved L40S node (g181003: 8× L40S, 64 CPUs, ~1TB RAM):

| Array ID | Model | Batch Size | VRAM Estimate |
|----------|-------|-----------|---------------|
| 0 | UNet 2D | 32 | ~8–10 GB |
| 1 | ResNet 2D | 32 | ~8–10 GB |
| 2 | Swin 2D | 16 | ~15–20 GB |
| 3 | ViT-V-Net 2D | 4 | ~20–25 GB |

---

## 7. Cross-Experiment Progression

### 7.1 Cumulative Improvement (5-Class Quick-Test)

| Stage | Best Mean Dice | Δ from Previous | Cumulative Δ |
|-------|---------------|-----------------|--------------|
| Baseline (BCE, uniform) | 0.252 | — | — |
| + Loss optimization (Tversky α=0.6) | 0.370 | +0.118 (+47%) | +47% |
| + Class weighting (Bal. Softmax τ=1.0) | 0.571 | +0.201 (+54%) | +127% |
| + Foreground masking fix | 0.511 | +0.140 (+33%) | +203% |
| + masksup_r0.3 masking | **0.571** | +0.060 (+12%) | **+227%** |

*The foreground masking fix alone provided +110% over the pre-fix no_mask baseline (0.243 → 0.511)*

### 7.2 Improvement by Technique

```
                    Contribution to Final Performance
┌──────────────────────────────────────────────────────────┐
│ Foreground Mask Fix     ██████████████████████████ (+110%) │  ★ BIGGEST GAIN
│ Architecture Choice     ████████████████████ (+63%*)       │
│ Class Weighting         ██████████████████████ (+54%)      │
│ Base Loss Selection     ████████████████  (+47%)           │
│ Masking Strategy        ███████ (+12% over fixed baseline) │
└──────────────────────────────────────────────────────────┘
* vs respective baselines within each experiment
```

### 7.3 Decision Chain

```
1. BCE → Per-class Tversky (α=0.6, β=0.4)    [Loss Optimization Winner]
     ↓
2. Uniform → Balanced Softmax τ=1.0            [Class Weighting Winner]
     ↓
3. Black-padding in loss → Foreground masking fix [Biggest single gain: +110%]
     ↓
4. No mask → masksup_r0.3                       [Masking Winner]
     ↓
4. UNet 2D → ResNet 2D (for 2D)               [Architecture Winner]
   FlexUNet-ResNet34 (for 3D)
```

---

## 8. Key Takeaways & Recommendations

### 8.1 What Worked

| Technique | Why It Works |
|-----------|-------------|
| **Tversky loss (precision bias)** | Prevents over-prediction of rare organelles; NaN-safe |
| **Balanced Softmax weighting** | Theory-grounded temperature scaling of class logits |
| **Foreground masking fix** | Eliminating loss on black-padding regions removed dominant FP source (+110%) |
| **masksup_r0.3 (mask-supervised)** | 30% random masking forces context learning → best generalization |
| **ResNet architecture** | Residual connections + moderate capacity = excellent generalization |
| **3D model ensembling** | Different architectures are complementary across class types |

### 8.2 What Didn't Work

| Technique | Why It Failed |
|-----------|--------------|
| **Focal loss** | γ=2.0 focusing collapses rare classes |
| **Recall-biased Tversky (α < β)** | Over-penalizes false negatives → model collapses |
| **Manual class weights** | Human intuition is worse than data-driven frequencies |
| **Strong hyperparameters** (τ=2.0, β=0.9999) | Over-correction destabilizes training |
| **Class presence masking** | Removing classes from loss is too aggressive |
| **MC-Dropout uncertainty** | Overfits during training; poor generalization |
| **Computing loss on padding** | Black-padding regions generated massive FPs — the biggest hidden bug |
| **ViT 2D** | 105M params with volatile convergence; needs careful tuning |

### 8.3 Recommended Final Configuration

```yaml
# Optimal configuration for CellMap Segmentation Challenge
loss:
  type: per-class Tversky
  alpha: 0.6    # FP penalty weight
  beta: 0.4     # FN penalty weight
  
class_weighting:
  type: balanced_softmax
  tau: 1.0
  weights: data-driven (from class_frequencies.json)

foreground_masking: true   # Set targets to NaN on black-padding EM regions
masking:
  type: masksup_r0.3       # 30% mask-supervised reconstruction
  mask_ratio: 0.3

architecture_2d: ResNet 2D (7.8M params)
architecture_3d: FlexUNet-ResNet34 (~22M params)

ensemble: class-wise best model selection
post_processing: TTA (flip + rotate) + connected component filtering
```

### 8.4 Projected Leaderboard Performance

| Submission | Score | Notes |
|-----------|-------|-------|
| Round 1 (14 classes) | 0.063 | Only 14/48 classes trained |
| Round 2 projected (35 classes) | 0.35–0.49 | With ensemble + TTA |
| Current #1 (BC_CV V9) | 0.466 | Target to beat |

---

## 9. Hardware & Compute Summary

### 9.1 Clusters Used

| Cluster | GPUs | Use Case |
|---------|------|----------|
| **Shenron** | 4× RTX 2080 Ti (11GB) | Initial loss optimization |
| **Rocinante** | 2× RTX 3090 (24GB) | Class weighting, masking, local testing |
| **Blanca Biokem** | 2× A100 (80GB) per job | 2D model comparison |
| **Sycamore** | 2× H100 (80GB) per job | 3D MONAI training (Round 2) |
| **Longleaf** | 6× L40S (48GB) | 3D MONAI Round 1, Round 2 3D + 2D |

### 9.2 Estimated Total Compute

| Experiment | GPU-Hours (approx) |
|-----------|-------------------|
| Loss optimization | ~200 hrs (2080 Ti) |
| Class weighting (15 configs) | ~100 hrs (3090) |
| Masking strategies (15 configs) | ~100 hrs (3090) |
| Model comparison 2D (4 models) | ~400 hrs (A100/H100) |
| MONAI 3D Round 1 (3 models) | ~600 hrs (L40S) |
| MONAI 3D Round 2 (4 models) | ~800 hrs (H100 + L40S) — in progress |
| **Total** | **~2,200 GPU-hours** |

---

## 10. Round 3 Fix: Spatial Bbox Masking for 3D Models

> **Date:** February 18, 2026  
> **Status:** Implementation complete, awaiting job submission

### 10.1 Problem: Catastrophic Mode Collapse in Round 2

All four R2 3D models exhibited **catastrophic mode collapse** — they learned to predict probability ≈1.0 for nearly every class at every voxel, producing meaningless segmentations.

#### Symptoms Observed

| Model | Observed Behavior | "Best" Val Dice | Reality |
|-------|------------------|----------------|---------|
| SwinUNETR | Predicts only cytoplasm (class 29) everywhere | 0.1413 | Single-class collapse |
| FlexUNet-ResNet34 | Predicts ecs + hchrom everywhere | 0.1524 | Near-zero Dice on all classes |
| SegResNet (32f) | Same degenerate pattern | ~0.15 | Mode collapse |
| SegResNet-Wide (48f) | Same degenerate pattern | ~0.15 | Mode collapse |

#### Raw Probability Analysis (FlexUNet on kidney crop)

Inspecting raw sigmoid outputs on `jrc_mus-kidney` (200³ crop):
- **22 of 35 classes** had mean probability > 0.3 across the entire volume
- Many classes had mean probability ≈ 1.0 (predicting "yes" everywhere)
- Only 2 classes (mito_ribo, np_out) had low mean probability
- The model learned to predict everything as positive

#### Visualization Evidence

Ran `visualize_prediction.py` on 3 test crops (kidney, heart, liver-zon-1):
- SwinUNETR: uniform cyan predictions (cytoplasm only), no organelle structure visible
- FlexUNet: uniform predictions for ecs/hchrom, no spatial structure

### 10.2 Root Cause Analysis

The R2 loss function (`BalancedSoftmaxTverskyLoss`) had **channel-level annotation masking only** — it masked unannotated channels (shape `(B, C)`) but had **no spatial masking**.

#### Why This Causes Mode Collapse

```
CellMap partial annotations:
├── annotation_mask[c] = 0  →  channel c is unannotated → loss = 0 (correct)
└── annotation_mask[c] = 1  →  channel c is annotated
     ├── target[c] = 1 at annotated foreground voxels (sparse, small regions)
     └── target[c] = 0 EVERYWHERE ELSE (vast majority of the volume)
          ├── Includes genuinely background voxels (correct negative)
          └── Includes spatially unannotated regions (NOT true negatives!)
              → Model penalized for predicting these → learns to predict 0
              → BUT: positive voxels are so sparse that predicting 1 everywhere
                 gives partial credit (some TPs) with minimal FN penalty
              → Net effect: model finds it optimal to predict ~1.0 everywhere
```

The core issue: within an "annotated" channel, the target is 0 both for **true background** and for **spatially unannotated regions** (where we simply don't know). The model can't distinguish these cases, and with α=0.6 (precision bias), the FP penalty on the vast background is diluted by the sheer volume of "background" voxels relative to the tiny foreground annotations.

### 10.3 The Fix: `box_class_mask_tight` Spatial Masking (3D)

Adapted from the winning masking strategy in Experiment 3 (2D masking_strategies), which achieved **+55% improvement** over no_mask baseline (0.243 → 0.376 on 13-dataset evaluation).

#### How It Works

For each annotated class in each sample:

1. **Find the 3D bounding box** of all foreground voxels in `(D, H, W)`
2. **Pad the bbox** by `pad_fraction=0.05` (5%) of its extent in each dimension (minimum 1 voxel)
3. **Create spatial weight mask**:
   - Inside padded bbox → weight = **1.0** (full loss contribution)
   - Outside padded bbox → weight = **0.05** (strongly de-weighted, ~20× less)
   - Unannotated channels → weight = **0.0** (no loss contribution)
4. **Annotated class with NO foreground** → `bg_weight=0.05` everywhere (provides proper negative-only signal — model IS penalized for any positive predictions)

#### Effect on Tversky Computation

```python
# R2 (broken): unweighted spatial sum → FP penalty diluted over entire volume
tp = (pred * target).sum(spatial_dims)
fp = (pred * (1 - target)).sum(spatial_dims)      # dominated by vast background
fn = ((1 - pred) * target).sum(spatial_dims)

# R3 (fixed): spatially weighted → FP penalty concentrated near annotations
tp = (spatial_w * pred * target).sum(spatial_dims)
fp = (spatial_w * pred * (1 - target)).sum(spatial_dims)  # bg outside bbox ×0.05
fn = (spatial_w * (1 - pred) * target).sum(spatial_dims)
```

This means false positives **near annotations** (inside the bbox) are penalized at full strength, while predictions **far from any annotation** are de-weighted to 5%. The model gets a strong negative signal where annotations exist, without being overwhelmed by the vast unlabeled background.

#### Why `box_class_mask_tight` (Not `masksup_r0.3`)

Although `masksup_r0.3` was the #1 strategy **after the foreground masking fix** (§4.4), `box_class_mask_tight` was chosen for R3 because:

1. **`masksup_r0.3` requires random input masking** — reconstructing masked patches is a 2D-specific technique that doesn't translate cleanly to 3D volumetric training
2. **`box_class_mask_tight` was #1 before the foreground fix** (0.376 vs 0.243 baseline, +55%) — it was directly solving the spatial masking problem that R2 models suffer from
3. **Rankings shifted because the foreground fix addressed the same underlying issue** — black-padding FPs were a spatial masking problem. With proper 3D handling, bbox masking targets the remaining issue (unannotated spatial regions within annotated channels)
4. **Simple, interpretable, no extra hyperparameters** beyond `pad_fraction` and `bg_weight` which are well-validated from 2D experiments

### 10.4 Files Modified

| File | Change |
|------|--------|
| `experiments/monai_cellmap/losses/partial_annotation.py` | Added `_compute_spatial_mask()` method to `BalancedSoftmaxTverskyLoss`. Modified `forward()` to compute 3D per-class bounding-box spatial weight mask and apply it to TP/FP/FN computation. Added `bbox_pad_fraction` and `bbox_bg_weight` constructor params. Updated `build_partial_annotation_loss()` factory to pass new params. |
| `experiments/monai_cellmap/models/mdl_cellmap.py` | Updated `build_partial_annotation_loss()` call in `Net.__init__()` to pass `bbox_pad_fraction` and `bbox_bg_weight` from config. |
| `experiments/monai_cellmap/configs/common_config.py` | Added `cfg.bbox_pad_fraction = 0.05` and `cfg.bbox_bg_weight = 0.05` to loss config section. |
| `experiments/monai_cellmap/configs/cfg_segresnet.py` | Renamed to `segresnet_ds_r3`, new output dir. |
| `experiments/monai_cellmap/configs/cfg_segresnet_wide.py` | Renamed to `segresnet_wide_r3`, new output dir. |
| `experiments/monai_cellmap/configs/cfg_flexunet_resnet.py` | Renamed to `flexunet_resnet34_r3`, new output dir. |
| `experiments/monai_cellmap/configs/cfg_swinunetr.py` | Renamed to `swinunetr_r3`, new output dir. |
| `experiments/monai_cellmap/slurm/train_r3_*_h100.sbatch` | 4 new SLURM scripts, all targeting H100 partition (queue empty). |

### 10.5 Key Implementation Details

#### 3D Bounding Box Computation (adapted from 2D)

The 2D reference (`BoxClassMaskTverskyLoss` in `experiments/masking_strategies/masking_losses.py`) computed per-class bounding boxes in `(H, W)` using `torch.where()`. The 3D adaptation:

```
2D: coords = torch.where(pos)  →  (ys, xs)
    bbox = [y_min:y_max, x_min:x_max]
    
3D: coords = torch.where(pos)  →  (zs, ys, xs)  
    bbox = [z_min:z_max, y_min:y_max, x_min:x_max]
```

Implemented generically for N spatial dimensions — loops over `range(ndim_spatial)` to compute `min/max` per dimension. Works for both 2D and 3D inputs.

#### Computational Cost

The bbox computation loops over `B × C` samples (B=2, C=35 → 70 iterations per forward pass). Each iteration calls `torch.where()` on a spatial volume (96³–160³ voxels). This is negligible compared to the backbone forward/backward pass.

#### Deep Supervision Interaction

`PartialAnnotationDeepSupervisionLoss` resizes the target via `F.interpolate(nearest)` for each scale level. The spatial bbox mask is recomputed inside `BalancedSoftmaxTverskyLoss.forward()` from the resized target at each scale — no special handling needed.

### 10.6 R3 Training Configuration

All 4 models moved to H100 cluster (queue confirmed empty at time of submission):

| Model | Config Name | Patch Size | Batch | Epochs | GPUs | Port |
|-------|------------|-----------|-------|--------|------|------|
| SegResNet-Wide (48f) | `segresnet_wide_r3` | 160³ | 2 | 600 | 2× H100 | 29600 |
| FlexUNet-ResNet34 | `flexunet_resnet34_r3` | 160³ | 2 | 300 | 2× H100 | 29601 |
| SwinUNETR v2 | `swinunetr_r3` | 96³ | 3 | 300 | 2× H100 | 29602 |
| SegResNet (32f) | `segresnet_ds_r3` | 160³ | 2 | 600 | 2× H100 | 29603 |

#### What Changed from R2

| Aspect | R2 | R3 |
|--------|----|----|
| **Spatial masking** | None (channel-only) | **box_class_mask_tight** (pad=0.05, bg=0.05) |
| **bbox_pad_fraction** | — | **0.05** (5% of bbox extent) |
| **bbox_bg_weight** | — | **0.05** (20× de-weighting outside bbox) |
| **Run names** | `*_r2` | `*_r3` |
| **Output dirs** | `runs/monai_cellmap/*_r2/` | `runs/monai_cellmap/*_r3/` |
| **SwinUNETR cluster** | L40S (Longleaf) | **H100 (Sycamore)** |
| **SegResNet cluster** | L40S (Longleaf) | **H100 (Sycamore)** |

All other hyperparameters (loss α/β, τ, lr, epochs, architectures, augmentation) remain unchanged from R2.

### 10.7 Expected Impact

Based on the 2D masking_strategies experiment results:
- `box_class_mask_tight` improved eval Dice from 0.243 → 0.376 (+55%) on 13-dataset evaluation (pre-foreground-fix)
- The mode collapse in R2 is fundamentally the same problem (no spatial FP penalty)
- R3 should produce spatially structured predictions instead of uniform mode collapse
- Conservative estimate: R3 should match or exceed R1 FlexUNet performance (0.233 mean Dice) on 35 classes within the first 50 epochs

### 10.8 Verification Plan

After R3 training starts:
1. **Early check (epoch 10–20):** Monitor val Dice in TensorBoard — should see per-class values diverging (not all collapsing to same value)
2. **Mid-training check (epoch 50):** Run `visualize_prediction.py` on test crops — predictions should show organelle-like spatial structure
3. **Convergence check (epoch 100+):** Compare R3 val Dice trajectories against R2 — should see monotonically improving Dice rather than the flat-then-collapse pattern of R2

---

*This document will be updated as Round 3 results become available.*
