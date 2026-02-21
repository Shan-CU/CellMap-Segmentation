#!/usr/bin/env python3
"""Quick diagnostic: inspect val data and model predictions to find root cause of all-positive predictions."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.dirname(__file__))
os.environ['SINGLE_GPU_MODE'] = '1'

import torch
torch.set_num_threads(4)

from config_shenron import QUICK_TEST_CLASSES, CHECKPOINT_DIR
from train_local import create_dataloaders, create_model

classes = QUICK_TEST_CLASSES
print(f"Classes: {classes}")

# ---- 1. Inspect validation data ----
print("\n" + "="*60)
print("STEP 1: Loading validation data...")
print("="*60)

_, val_loader = create_dataloaders(
    classes=classes, batch_size=4, iterations_per_epoch=10, input_shape=(1, 256, 256)
)

# Check multiple val batches
for batch_idx in range(min(3, len(val_loader))):
    batch = next(iter(val_loader))
    inputs = batch['input']
    targets = batch['output']
    
    print(f"\n--- Val batch {batch_idx} ---")
    print(f"  Input:  shape={inputs.shape}, min={inputs.min():.4f}, max={inputs.max():.4f}, mean={inputs.mean():.4f}")
    print(f"  Target: shape={targets.shape}, min={targets.nan_to_num(-99).min():.4f}, max={targets.nan_to_num(-99).max():.4f}")
    
    nan_frac = targets.isnan().float().mean().item()
    print(f"  Target NaN fraction: {nan_frac:.4f} ({nan_frac*100:.1f}%)")
    
    valid = ~targets.isnan()
    if valid.any():
        t_valid = targets[valid]
        print(f"  Valid target values: min={t_valid.min():.4f}, max={t_valid.max():.4f}, mean={t_valid.mean():.4f}")
        print(f"  Unique values (first 10): {torch.unique(t_valid).tolist()[:10]}")
    
    for i, c in enumerate(classes):
        t = targets[:, i]
        v = ~t.isnan()
        n_valid = v.sum().item()
        n_total = t.numel()
        t_clean = t[v]
        n_pos = (t_clean > 0.5).sum().item() if t_clean.numel() > 0 else 0
        n_neg = (t_clean <= 0.5).sum().item() if t_clean.numel() > 0 else 0
        print(f"    {c:12s}: valid={n_valid}/{n_total} ({n_valid/n_total*100:.1f}%), "
              f"pos={n_pos} ({n_pos/max(n_valid,1)*100:.2f}%), neg={n_neg}")

# ---- 2. Load trained model & check predictions ----
print("\n" + "="*60)
print("STEP 2: Loading best checkpoint and checking predictions...")
print("="*60)

ckpts = sorted(CHECKPOINT_DIR.glob("*baseline_bce*best.pth"))
if not ckpts:
    ckpts = sorted(CHECKPOINT_DIR.glob("*best.pth"))

if ckpts:
    ckpt_path = ckpts[0]
    print(f"Loading: {ckpt_path.name}")
    
    model = create_model(len(classes), input_channels=1)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.cuda()
    
    batch = next(iter(val_loader))
    inputs = batch['input'].cuda()
    targets = batch['output'].cuda()
    
    with torch.no_grad():
        outputs = model(inputs)
    
    pred_sigmoid = torch.sigmoid(outputs)
    
    print(f"\n  Raw logits:  min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
    print(f"  Sigmoid:     min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}")
    
    pred_binary = (pred_sigmoid > 0.5).float()
    valid_mask = ~targets.isnan()
    target_clean = targets.nan_to_num(0)
    
    print(f"\n  Per-class analysis:")
    for i, c in enumerate(classes):
        p = pred_sigmoid[:, i]
        t = targets[:, i]
        v = ~t.isnan()
        t_c = t.nan_to_num(0)
        
        pred_pos = (p > 0.5).sum().item()
        pred_total = p.numel()
        target_pos = (t_c[v] > 0.5).sum().item() if v.any() else 0
        target_total = v.sum().item()
        
        tp = ((p > 0.5) & (t_c > 0.5) & v).sum().item()
        fp = ((p > 0.5) & (t_c <= 0.5) & v).sum().item()
        fn = ((p <= 0.5) & (t_c > 0.5) & v).sum().item()
        tn = ((p <= 0.5) & (t_c <= 0.5) & v).sum().item()
        
        dice = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
        
        print(f"    {c:12s}: pred_pos={pred_pos}/{pred_total} ({pred_pos/pred_total*100:.1f}%), "
              f"target_pos={target_pos}/{target_total} ({target_pos/max(target_total,1)*100:.1f}%)")
        print(f"               sigmoid: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}")
        print(f"               TP={tp:>8} FP={fp:>8} FN={fn:>8} TN={tn:>8} Dice={dice:.4f}")
else:
    print("No checkpoints found!")

print("\n" + "="*60)
print("STEP 3: Check untrained model (fresh random weights)")
print("="*60)

fresh_model = create_model(len(classes), input_channels=1)
fresh_model.eval()
fresh_model.cuda()

batch = next(iter(val_loader))
inputs = batch['input'].cuda()
targets = batch['output'].cuda()

with torch.no_grad():
    outputs = fresh_model(inputs)

pred_sigmoid = torch.sigmoid(outputs)
print(f"  Fresh model logits: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
print(f"  Fresh model sigmoid: min={pred_sigmoid.min():.4f}, max={pred_sigmoid.max():.4f}, mean={pred_sigmoid.mean():.4f}")
print(f"  Fresh model pred positive fraction: {(pred_sigmoid > 0.5).float().mean():.4f}")

print("\nDone!")
