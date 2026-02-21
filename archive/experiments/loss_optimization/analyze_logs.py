#!/usr/bin/env python3
"""
Analyze and summarize data debug logs.

Usage:
    python analyze_logs.py results/<run_name>_data_debug.log
    python analyze_logs.py results/*_data_debug.log  # Analyze all logs
"""

import sys
import re
from pathlib import Path
from collections import defaultdict


def extract_key_info(log_file):
    """Extract key information from a data debug log."""
    info = {
        'run_name': Path(log_file).stem.replace('_data_debug', ''),
        'classes': [],
        'batch_size': None,
        'input_shape': None,
        'train_batches': 0,
        'val_batches': 0,
        'input_range': {},
        'target_binary': {},
        'target_nan_counts': {},
        'positive_ratios': {},
        'validation_dice': {},
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract classes
    classes_match = re.search(r'Classes: \[(.*?)\]', content)
    if classes_match:
        classes_str = classes_match.group(1)
        info['classes'] = [c.strip().strip("'\"") for c in classes_str.split(',')]
    
    # Extract batch size
    batch_match = re.search(r'Batch size: (\d+)', content)
    if batch_match:
        info['batch_size'] = int(batch_match.group(1))
    
    # Extract input shape
    shape_match = re.search(r'Input shape: \(([\d, ]+)\)', content)
    if shape_match:
        info['input_shape'] = shape_match.group(1)
    
    # Count batches
    info['train_batches'] = len(re.findall(r'TRAIN BATCH', content))
    info['val_batches'] = len(re.findall(r'VAL BATCH', content))
    
    # Extract input range from first batch
    input_min = re.search(r'INPUT:.*?Min: ([\d.]+)', content, re.DOTALL)
    input_max = re.search(r'INPUT:.*?Max: ([\d.]+)', content, re.DOTALL)
    if input_min and input_max:
        info['input_range'] = {
            'min': float(input_min.group(1)),
            'max': float(input_max.group(1))
        }
    
    # Extract per-class ground truth info
    for class_name in info['classes']:
        # Is binary?
        binary_match = re.search(
            rf'{class_name}:.*?Is binary \(0/1\): (True|False)',
            content, re.DOTALL
        )
        if binary_match:
            info['target_binary'][class_name] = binary_match.group(1) == 'True'
        
        # NaN count
        nan_match = re.search(
            rf'{class_name}:.*?Has NaN: (True|False) \((\d+) / (\d+) pixels\)',
            content, re.DOTALL
        )
        if nan_match:
            info['target_nan_counts'][class_name] = {
                'has_nan': nan_match.group(1) == 'True',
                'nan_pixels': int(nan_match.group(2)),
                'total_pixels': int(nan_match.group(3))
            }
        
        # Positive ratio
        ratio_match = re.search(
            rf'{class_name}:.*?Positive ratio \(>0\.5\): ([\d.]+)',
            content, re.DOTALL
        )
        if ratio_match:
            info['positive_ratios'][class_name] = float(ratio_match.group(1))
        
        # Final validation Dice
        dice_match = re.search(
            rf'{class_name}: Dice=([\d.]+)',
            content
        )
        if dice_match:
            # Get last occurrence
            all_dice = re.findall(rf'{class_name}: Dice=([\d.]+)', content)
            if all_dice:
                info['validation_dice'][class_name] = float(all_dice[-1])
    
    return info


def print_summary(info):
    """Print a formatted summary of the log analysis."""
    print("\n" + "=" * 80)
    print("DATA PIPELINE ANALYSIS: {}".format(info['run_name']))
    print("=" * 80)
    
    print("\n📊 CONFIGURATION:")
    print("  Classes: {}".format(', '.join(info['classes'])))
    print("  Batch size: {}".format(info['batch_size']))
    print("  Input shape: {}".format(info['input_shape']))
    print("  Train batches logged: {}".format(info['train_batches']))
    print("  Val batches logged: {}".format(info['val_batches']))
    
    if info['input_range']:
        print("\n📥 INPUT DATA:")
        print("  Range: [{:.6f}, {:.6f}]".format(
            info['input_range']['min'], info['input_range']['max']
        ))
        if info['input_range']['max'] > 1.5:
            print("  ⚠️  WARNING: Input not normalized to [0, 1]!")
        else:
            print("  ✅ Input properly normalized")
    
    if info['target_binary']:
        print("\n🎯 GROUND TRUTH:")
        all_binary = all(info['target_binary'].values())
        if all_binary:
            print("  ✅ All classes are binary (0/1)")
        else:
            print("  ⚠️  Some classes are not binary:")
            for cls, is_bin in info['target_binary'].items():
                if not is_bin:
                    print("    - {}".format(cls))
    
    if info['target_nan_counts']:
        print("\n🚫 NaN PIXELS (unlabeled regions):")
        for cls, nan_info in info['target_nan_counts'].items():
            if nan_info['has_nan']:
                pct = 100 * nan_info['nan_pixels'] / nan_info['total_pixels']
                print("  {}: {:.1f}% ({}/{} pixels)".format(
                    cls, pct, nan_info['nan_pixels'], nan_info['total_pixels']
                ))
    
    if info['positive_ratios']:
        print("\n➕ CLASS BALANCE (positive pixel ratios):")
        sorted_ratios = sorted(
            info['positive_ratios'].items(), key=lambda x: x[1], reverse=True
        )
        for cls, ratio in sorted_ratios:
            status = "🟢" if ratio > 0.1 else ("🟡" if ratio > 0.01 else "🔴")
            print("  {} {}: {:.4f} ({:.2f}%)".format(
                status, cls, ratio, ratio * 100
            ))
    
    if info['validation_dice']:
        print("\n🎲 FINAL VALIDATION DICE:")
        sorted_dice = sorted(
            info['validation_dice'].items(), key=lambda x: x[1], reverse=True
        )
        mean_dice = sum(info['validation_dice'].values()) / len(info['validation_dice'])
        for cls, dice in sorted_dice:
            status = "🟢" if dice > 0.5 else ("🟡" if dice > 0.3 else "🔴")
            print("  {} {}: {:.4f}".format(status, cls, dice))
        print("  Mean: {:.4f}".format(mean_dice))
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file> [<log_file2> ...]")
        print("\nExample:")
        print("  python analyze_logs.py results/*_data_debug.log")
        sys.exit(1)
    
    log_files = []
    for pattern in sys.argv[1:]:
        # Handle wildcards
        path = Path(pattern)
        if '*' in pattern:
            log_files.extend(path.parent.glob(path.name))
        else:
            log_files.append(path)
    
    if not log_files:
        print("No log files found!")
        sys.exit(1)
    
    print("\n📁 Found {} log file(s)".format(len(log_files)))
    
    for log_file in log_files:
        if not log_file.exists():
            print("⚠️  File not found: {}".format(log_file))
            continue
        
        try:
            info = extract_key_info(log_file)
            print_summary(info)
        except Exception as e:
            print("❌ Error analyzing {}: {}".format(log_file, e))
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
