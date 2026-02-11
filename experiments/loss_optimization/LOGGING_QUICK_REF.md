# Data Logging Quick Reference Card

## 📁 Log File Location
```
experiments/loss_optimization/results/<run_name>_data_debug.log
```

## 🚀 Quick Commands

### Run Training (Logging is Automatic)
```bash
python train_local.py --mode quick_test
./run.sh compare-losses unet_2d
```

### Analyze Logs
```bash
# Summary of one run
python analyze_logs.py results/<run_name>_data_debug.log

# Compare all runs
python analyze_logs.py results/*_data_debug.log
```

### Search Logs
```bash
# Check normalization
grep "INPUT:" results/*_data_debug.log | grep "Range"

# Check class balance
grep "Positive ratio" results/*_data_debug.log

# Find NaN issues
grep "Has NaN: True" results/*_data_debug.log

# Check if binary
grep "Is binary" results/*_data_debug.log

# See validation results
grep "Validation Results" results/*_data_debug.log -A 10
```

## 📊 What's Logged

| Category | Details |
|----------|---------|
| **Input** | Shape, dtype, value range [0,1], NaN/Inf |
| **Target** | Per-class: NaN pixels, binary check, positive ratio |
| **Transforms** | Spatial (rotate/flip), value (normalize), shape (squeeze) |
| **Dataloader** | Batch size, workers, shuffle, dataset type |
| **Model** | Output shape, logit/sigmoid ranges per class |
| **Loss** | Loss function type, value, NaN warnings |
| **Validation** | TP/FP/FN counts, Dice per class |

## 🎯 Key Checks

### ✅ Expected Values
- Input range: [0.0, 1.0] (normalized)
- Ground truth: Binary (0 or 1)
- Sigmoid outputs: [0.0, 1.0]
- Some NaN in targets is OK (unlabeled regions)

### ⚠️ Warning Signs
- Input max > 1.5 → Not normalized!
- "Is binary: False" → Ground truth not 0/1
- Loss = NaN → Gradient explosion
- All classes have very low positive ratios → Severe class imbalance

## 📈 When to Check Logs

1. **Before training** - Verify dataloader config
2. **First epoch** - Check first batch details
3. **Poor performance** - Analyze class balance, NaN locations
4. **NaN loss** - Check input/target ranges
5. **Low Dice** - Check positive ratios, model outputs

## 🔍 Debugging Workflows

### Input Not Normalized?
```bash
grep "INPUT:" results/*_data_debug.log | grep "Max:"
# If > 1.5, check raw_value_transforms in create_dataloaders()
```

### Class Not Learning?
```bash
grep "endo_mem" results/*_data_debug.log | grep "Positive ratio"
# Very low ratio → rare class, may need more weight
```

### Too Many NaN Pixels?
```bash
grep "Has NaN: True" results/*_data_debug.log
# Shows which classes have unlabeled regions
```

### Model Not Predicting?
```bash
grep "Sigmoid stats" results/*_data_debug.log
# Check if predictions are all near 0 or 1 (overconfident)
```

## 📚 Documentation Files

- **[LOGGING_SUMMARY.md](LOGGING_SUMMARY.md)** - This summary of what was added
- **[DATA_LOGGING_GUIDE.md](DATA_LOGGING_GUIDE.md)** - Complete detailed guide
- **[analyze_logs.py](analyze_logs.py)** - Log analysis tool
- **[test_logging.py](test_logging.py)** - Test the logging system

## 💡 Pro Tips

1. **First batch is gold** - Most issues show up in first batch logging
2. **Use the analyzer** - Don't grep manually, use `analyze_logs.py`
3. **Compare runs** - Analyze multiple logs to see patterns
4. **Check class balance** - Low positive ratios explain low Dice
5. **Validation sigmoid** - Should span [0, 1] after a few epochs

## 🧪 Test It Works
```bash
python test_logging.py
# Should print: ✅ Test passed!
```

---

**Need help?** Check [DATA_LOGGING_GUIDE.md](DATA_LOGGING_GUIDE.md) for detailed examples.
