#!/usr/bin/env python3
"""
Deep diagnostic: understand WHY the model predicts all-positive.
Check training data, loss gradients, and Dice computation carefully.
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['SINGLE_GPU_MODE'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.dirname(__file__))

import torch
torch.set_num_threads(4)

import logging
logging.disable(logging.INFO)  # Suppress cellmap_data spam

from config_shenron import QUICK_TEST_CLASSES, CHECKPOINT_DIR
from train_local import create_dataloaders, create_model

classes = QUICK_TEST_CLASSES

print("="*70)
print("DIAGNOSTIC 1: Training data analysis")
print("="*70)

train_loader, val_loader = create_dataloaders(
    classes=classes, batch_size=4, iterations_per_epoch=10, input_shape=(1, 256, 256)
)

# Check multiple DIFFERENT training batches
print("\n--- Training data (multiple batches) ---")
total_pos = {c: 0 for c in classes}
total_valid = {c: 0 for c in classes}
total_nan = {c: 0 for c in classes}
total_pixels = {c: 0 for c in classes}

for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 5:
        break
    targets = batch['output']
    inputs = batch['input']
    
    print(f"\n  Train batch {batch_idx}: input shape={inputs.shape}, "
          f"input range=[{inputs.min():.3f}, {inputs.max():.3f}]")
    
    for i, c in enumerate(classes):
        t = targets[:, i]
        v = ~t.isnan()
        n_valid = v.sum().item()
        n_total = t.numel()
        n_nan = t.isnan().sum().item()
        t_clean = t[v]
        n_pos = (t_clean > 0.5).sum().item() if t_clean.numel() > 0 else 0
        
        total_pos[c] += n_pos
        total_valid[c] += n_valid
        total_nan[c] += n_nan
        total_pixels[c] += n_total
        
        if batch_idx == 0:
            print(f"    {c:12s}: valid={n_valid}/{n_total} ({n_valid/n_total*100:.1f}%), "
                  f"pos={n_pos} ({n_pos/max(n_valid,1)*100:.2f}%)")

print(f"\n--- Training data summary (5 batches) ---")
for c in classes:
    v = total_valid[c]
    p = total_pos[c]
    n = total_nan[c]
    t = total_pixels[c]
    print(f"  {c:12s}: valid={v}/{t} ({v/t*100:.1f}%), pos={p}/{v} ({p/max(v,1)*100:.2f}%), nan={n/t*100:.1f}%")

print("\n" + "="*70)
print("DIAGNOSTIC 2: Is val_loader returning same data every iteration?")
print("="*70)

# Get data from val_loader multiple times
val_hashes = []
for i in range(3):
    batch = next(iter(val_loader))
    inp = batch['input']
    tgt = batch['output']
    # Hash by summing elements
    h = (inp.sum().item(), tgt.nan_to_num(0).sum().item())
    val_hashes.append(h)
    print(f"  Iter {i}: input_sum={h[0]:.4f}, target_sum={h[1]:.4f}")

if all(h == val_hashes[0] for h in val_hashes):
    print("  >>> CONFIRMED: Val loader returns IDENTICAL data every iteration!")
else:
    print("  Val loader returns different data each time (OK)")

print("\n" + "="*70)
print("DIAGNOSTIC 3: What does 'baseline_bce' loss actually see?")
print("="*70)

# The key question: if the loss masks NaN pixels, BUT 85% of pixels are NaN,
# the model only trains on 15% of pixels. Of those, ~96-100% are NEGATIVE.
# So the optimal BCE strategy is predict-all-negative. But the model does the opposite!
# WHY?

# Let's compute what the loss gradient looks like
batch = next(iter(train_loader))
targets = batch['output']
valid_mask = ~targets.isnan()
targets_clean = targets.nan_to_num(0)

# For valid pixels only, what's the class balance?
print("\n  Class balance within VALID pixels (training batch):")
for i, c in enumerate(classes):
    v = valid_mask[:, i]
    t = targets_clean[:, i][v]
    n_pos = (t > 0.5).sum().item()
    n_neg = (t <= 0.5).sum().item()
    total = n_pos + n_neg
    pos_frac = n_pos / max(total, 1)
    print(f"    {c:12s}: pos={n_pos:>6} neg={n_neg:>6} pos_frac={pos_frac:.4f}")

print("\n" + "="*70)
print("DIAGNOSTIC 4: Dice computation bug hunt")
print("="*70)

# The Tversky loss uses valid_mask multiplication, but let's check if the
# pred_flat multiplication with valid_mask is doing what we think.
# 
# In TverskyLoss: pred_flat = (pred_sigmoid * valid_mask).flatten(2)
# This means: for NaN pixels, pred_flat = 0 (because valid_mask = 0)
# And: target_flat = 0 (because valid_mask = 0)
#
# But then: fp = pred_flat * (1 - target_flat)
# For NaN pixels: fp = 0 * (1 - 0) = 0 * 1 = 0 ✓ (NaN pixels don't contribute to FP)
# For valid positive: fp = pred * (1 - 1) = 0 ✓
# For valid negative: fp = pred * (1 - 0) = pred  ✓
#
# Wait, but for NaN pixels: (1 - target_flat) = 1, and pred_flat = 0.
# So NaN pixels appear as "negative targets with zero prediction" = they don't add to FP or FN.
# This means the loss effectively ignores them. Good.
#
# BUT: (1 - pred_flat) for NaN pixels = 1. And target_flat = 0.
# fn = (1 - pred_flat) * target_flat = 1 * 0 = 0.  ✓ OK no FN contribution.
#
# So the masking IS correct in the loss. The issue must be elsewhere.

# Let's check: does the model have a bias initialization issue?
print("\n  Checking model output bias...")
model = create_model(len(classes), input_channels=1)
model.eval()

# Check if final conv layer has bias
final_layer = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        final_layer = (name, module)

if final_layer:
    name, layer = final_layer
    print(f"  Final conv layer: {name}")
    print(f"    weight shape: {layer.weight.shape}")
    if layer.bias is not None:
        print(f"    bias: {layer.bias.data}")
        print(f"    bias → sigmoid: {torch.sigmoid(layer.bias.data)}")
    else:
        print(f"    NO BIAS")

print("\n" + "="*70)
print("DIAGNOSTIC 5: Check trained model more carefully")
print("="*70)

ckpts = sorted(CHECKPOINT_DIR.glob("*baseline_bce*best.pth"))
if ckpts:
    ckpt = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    model = create_model(len(classes), input_channels=1)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.cuda()
    
    # Get a TRAINING batch (not val!) and check predictions
    batch = next(iter(train_loader))
    inputs = batch['input'].cuda()
    targets = batch['output'].cuda()
    
    with torch.no_grad():
        outputs = model(inputs)
    
    pred_sigmoid = torch.sigmoid(outputs)
    
    print("\n  Trained model on TRAINING data:")
    print(f"    Logits: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
    print(f"    Sigmoid: min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}")
    
    valid_mask = ~targets.isnan()
    target_clean = targets.nan_to_num(0)
    
    print("\n  Per-class on training data (valid pixels only):")
    for i, c in enumerate(classes):
        v = valid_mask[:, i]
        p = pred_sigmoid[:, i][v]
        t = target_clean[:, i][v]
        
        pred_pos = (p > 0.5).sum().item()
        target_pos = (t > 0.5).sum().item()
        n_valid = v.sum().item()
        
        tp = ((p > 0.5) & (t > 0.5)).sum().item()
        fp = ((p > 0.5) & (t <= 0.5)).sum().item()
        fn = ((p <= 0.5) & (t > 0.5)).sum().item()
        tn = ((p <= 0.5) & (t <= 0.5)).sum().item()
        dice = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        
        print(f"    {c:12s}: pred_pos={pred_pos}/{n_valid} ({pred_pos/max(n_valid,1)*100:.1f}%), "
              f"target_pos={target_pos}/{n_valid} ({target_pos/max(n_valid,1)*100:.1f}%)")
        print(f"               TP={tp:>7} FP={fp:>7} FN={fn:>7} TN={tn:>7} Dice={dice:.4f}")
    
    # Check model on ALL pixels including NaN regions
    print("\n  Sigmoid distribution on ALL pixels (including NaN target regions):")
    for i, c in enumerate(classes):
        p = pred_sigmoid[:, i].flatten()
        v = valid_mask[:, i].flatten()
        p_valid = p[v.bool()]
        p_nan = p[~v.bool()]
        print(f"    {c:12s}: valid_region sigmoid mean={p_valid.mean():.4f}, "
              f"NaN_region sigmoid mean={p_nan.mean():.4f}")

print("\n" + "="*70)
print("DIAGNOSTIC 6: Check NaN distribution - is it spatial or random?")
print("="*70)

batch = next(iter(val_loader))
targets = batch['output']
# NaN pattern: is it the same across all classes?
print("  NaN mask comparison across classes (batch 0, sample 0):")
nan_masks = []
for i, c in enumerate(classes):
    m = targets[0, i].isnan()
    nan_masks.append(m)
    print(f"    {c:12s}: {m.sum().item()}/{m.numel()} NaN ({m.float().mean()*100:.1f}%)")

# Are they all the SAME mask?
all_same = all(torch.equal(nan_masks[0], m) for m in nan_masks[1:])
print(f"\n  All classes have SAME NaN mask: {all_same}")

if not all_same:
    # Check pairwise
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            same = torch.equal(nan_masks[i], nan_masks[j])
            if not same:
                diff = (nan_masks[i] != nan_masks[j]).sum().item()
                print(f"    {classes[i]} vs {classes[j]}: {diff} pixels differ")

# What does the NaN mask look like spatially?
m = nan_masks[0]
print(f"\n  NaN mask spatial structure (class {classes[0]}):")
print(f"    Shape: {m.shape}")
# Check if it's a contiguous block
valid = ~m
rows_with_valid = valid.any(dim=1).sum().item()
cols_with_valid = valid.any(dim=0).sum().item()
print(f"    Rows with valid data: {rows_with_valid}/{m.shape[0]}")
print(f"    Cols with valid data: {cols_with_valid}/{m.shape[1]}")
# Is it a rectangular crop within the image?
if rows_with_valid < m.shape[0]:
    first_valid_row = valid.any(dim=1).nonzero()[0].item() if valid.any() else -1
    last_valid_row = valid.any(dim=1).nonzero()[-1].item() if valid.any() else -1
    first_valid_col = valid.any(dim=0).nonzero()[0].item() if valid.any() else -1
    last_valid_col = valid.any(dim=0).nonzero()[-1].item() if valid.any() else -1
    print(f"    Valid region: rows [{first_valid_row}:{last_valid_row}], cols [{first_valid_col}:{last_valid_col}]")

print("\nDone!")
