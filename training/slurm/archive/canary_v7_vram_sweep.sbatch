#!/bin/bash
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=canary_v7_vram
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Canary v7: VRAM sweep — find max batch size at 96^3 on L40S
# ============================================================================
# Runs a standalone PyTorch script that:
#   1. Builds the exact same SegResNetDS model as train.py
#   2. Does real forward + backward + optimizer step with AMP
#   3. Tests batch sizes 2,4,6,8,10,12,16 — records peak VRAM
#   4. Catches OOM gracefully and reports the ceiling
# No cellmap-data needed — uses random tensors. Pure GPU test.
# ============================================================================

set -euo pipefail

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

echo "============================================"
echo "Canary v7: VRAM sweep at 96^3"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

python3 << 'PYTHON_EOF'
import torch
import gc
import sys

def test_batch_size(batch_size, shape=(96, 96, 96), num_classes=48, in_channels=1, num_steps=3):
    """Run real fwd+bwd+optim at given batch size, return peak VRAM in MB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    from monai.networks.nets import SegResNetDS

    model = SegResNetDS(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dsdepth=1,
    ).cuda()

    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = torch.nn.CrossEntropyLoss()

    for step in range(num_steps):
        # Random input — same dtype/shape as real training
        x = torch.randn(batch_size, in_channels, *shape, device='cuda', dtype=torch.float32)
        # Random target — class indices
        y = torch.randint(0, num_classes, (batch_size, *shape), device='cuda')

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits = model(x)
            # logits shape: (B, C, D, H, W)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Cleanup
    del model, optimizer, scaler, loss_fn, x, y, logits, loss
    torch.cuda.empty_cache()
    gc.collect()

    return peak_mb


gpu_name = torch.cuda.get_device_name(0)
gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
print(f"\nGPU: {gpu_name} ({gpu_total_mb:.0f} MB)")
print(f"Testing SegResNetDS @ 96^3, 48 classes, AMP, 3 steps each\n")
print(f"{'Batch':>6}  {'Peak VRAM':>12}  {'% GPU':>7}  {'Status'}")
print("-" * 50)

max_safe_batch = 0
for bs in [2, 4, 6, 8, 10, 12, 16]:
    try:
        peak = test_batch_size(bs)
        pct = 100 * peak / gpu_total_mb
        status = "✓ safe" if pct < 85 else "⚠ tight"
        print(f"{bs:>6}  {peak:>10.0f} MB  {pct:>6.1f}%  {status}")
        max_safe_batch = bs
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>6}  {'OOM':>10}  {'---':>6}   ✗ OOM")
        torch.cuda.empty_cache()
        gc.collect()
        break
    except Exception as e:
        print(f"{bs:>6}  ERROR: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        break

print(f"\n{'='*50}")
print(f"Max safe batch size: {max_safe_batch}")
print(f"Recommended production batch: {min(max_safe_batch, 12)}")
print(f"  (leave headroom for cuDNN autotuning, validation, etc.)")
PYTHON_EOF

echo ""
echo "=== Final GPU state ==="
nvidia-smi
echo "Canary v7 complete"
