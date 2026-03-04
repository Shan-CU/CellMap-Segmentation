#!/usr/bin/env python3
"""Quick GPU memory profiler for SegResNetDS at different ROI/batch configs.
Run on a GPU node to get real peak memory numbers.
"""
import torch
import gc
import sys

def profile_config(roi, batch, num_classes=48, amp=True):
    """Profile peak GPU memory for a single forward+backward pass."""
    from monai.networks.nets import SegResNetDS
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    model = SegResNetDS(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        init_filters=32,
        blocks_down=(1, 2, 2, 4, 4),
        blocks_up=(1, 1, 1, 1),
        dsdepth=4,
        norm="INSTANCE",
    ).cuda()
    
    # Simulate AMP training (fp16 forward, fp32 master weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    
    # EMA adds ~1x model params
    ema_params = {n: p.clone() for n, p in model.named_parameters()}
    
    x = torch.randn(batch, 1, roi, roi, roi, device="cuda")
    target = torch.randint(0, 2, (batch, num_classes, roi, roi, roi), device="cuda", dtype=torch.float16)
    
    baseline_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    try:
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            # Deep supervision returns a list
            if isinstance(logits, (list, tuple)):
                # Compute loss on main output + DS outputs (simplified)
                loss = logits[0].sum()
                for ds_out in logits[1:]:
                    loss = loss + ds_out.sum() * 0.1
            else:
                loss = logits.sum()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        
        # Cleanup
        del x, target, logits, loss, ema_params, optimizer, scaler, model
        torch.cuda.empty_cache()
        gc.collect()
        
        return peak_mem, baseline_mem, True
    
    except torch.cuda.OutOfMemoryError:
        del x, target, ema_params, optimizer, scaler, model
        torch.cuda.empty_cache()
        gc.collect()
        return 0, baseline_mem, False


def main():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"{'ROI':>5} {'Batch':>5} {'Peak':>8} {'Baseline':>10} {'Status':>8}")
    print("=" * 45)
    
    configs = [
        # Already measured: (96,4)=5.8, (128,1)=4.4, (128,2)=7.3, (128,4)=13.1
        # (160,1)=7.2, (160,2)=12.9, (192,1)=11.3, (224,1)=17.1
        # Now test larger batch with big ROIs
        (192, 2),
        (192, 4),
        (224, 2),
        (224, 4),
        (160, 4),
    ]
    
    for roi, batch in configs:
        peak, baseline, success = profile_config(roi, batch)
        if success:
            headroom = gpu_mem - peak
            status = "✅" if headroom > 5 else ("⚠️" if headroom > 0 else "❌")
            print(f"{roi:>5} {batch:>5} {peak:>7.1f}GB {baseline:>9.1f}GB {status:>8} (headroom: {headroom:.1f}GB)")
        else:
            print(f"{roi:>5} {batch:>5}      OOM {baseline:>9.1f}GB       ❌")


if __name__ == "__main__":
    main()
