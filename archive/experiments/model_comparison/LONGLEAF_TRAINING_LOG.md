
### UNet 2D Training Progress (from previous Sycamore run)

Analysis from TensorBoard logs at step ~128,000:

| Metric | Current Value | Best Value | Best Step |
|--------|---------------|------------|-----------|
| Train Loss | ~0.44 | 0.050 | 12,573 |
| Val Loss | 0.502 | 0.396 | 12,000 |
| **Val Dice** | 0.483 | **0.515** | **121,000** |

#### Assessment
- **Model Status**: NOT collapsed, still learning
- **Overfitting**: Yes, validation loss degraded from 0.396 → 0.502
- **Best Checkpoint**: Step 121,000 (Dice = 0.515)
- **Recommendation**: Use checkpoint from step 121,000 for inference

#### Training Curves Summary
- Training loss was lowest early (step ~12k) suggesting good initial fit
- Validation Dice continued improving until step 121k
- After step 121k, model started to plateau/slightly overfit
- Gradient norms stable (~0.15-0.25), no exploding gradients
- Learning rate at 1.6e-5 (likely using cosine annealing schedule)

---

### References
- [Longleaf Getting Started](https://help.rc.unc.edu/getting-started-on-longleaf/)
- [Longleaf SLURM Examples](https://help.rc.unc.edu/longleaf-slurm-examples/)
- [Modules Documentation](https://help.rc.unc.edu/modules)
