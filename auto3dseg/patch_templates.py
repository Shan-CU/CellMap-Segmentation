#!/usr/bin/env python3
"""
Patch Auto3DSeg algorithm templates for partial annotation support.

After BundleGen generates algorithm bundles, run this script to inject
the partial annotation loss and training loop modifications into each
template's scripts.

This patches TWO types of templates:
1. segresnet/segresnet2d: Modifies segmenter.py (train_epoch method)
2. swinunetr/dints: Modifies train.py (run function)

Usage:
    python patch_templates.py --work_dir auto3dseg/work_dir

The script:
1. Copies partial_annotation.py into each algorithm bundle
2. Patches the loss construction to use PartialAnnotationLossV2
3. Patches the training loop to extract annotation masks from batch_data
   and set them on the loss function before each loss computation
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def find_algorithm_dirs(work_dir: str) -> list[Path]:
    """Find all generated algorithm bundle directories."""
    algo_dirs = []
    for d in Path(work_dir).iterdir():
        if d.is_dir() and (d / "scripts").is_dir():
            algo_dirs.append(d)
    return sorted(algo_dirs)


def detect_template_type(algo_dir: Path) -> str:
    """Detect whether this is a segresnet-style or swinunetr-style template."""
    scripts_dir = algo_dir / "scripts"
    if (scripts_dir / "segmenter.py").exists():
        return "segresnet"
    elif (scripts_dir / "train.py").exists():
        return "swinunetr"
    else:
        return "unknown"


def copy_partial_annotation_module(algo_dir: Path, source_dir: Path) -> None:
    """Copy partial_annotation.py into the algorithm's scripts directory."""
    src = source_dir / "partial_annotation.py"
    dst = algo_dir / "scripts" / "partial_annotation.py"
    shutil.copy2(str(src), str(dst))
    print(f"    Copied partial_annotation.py → {dst}")


def patch_segresnet(algo_dir: Path) -> bool:
    """
    Patch segresnet/segresnet2d segmenter.py for partial annotation support.

    Changes:
    1. Add import for partial_annotation module
    2. Replace loss construction in __init__ to use PartialAnnotationLossV2
    3. Patch train_epoch to extract annotation mask and set it on loss_function
    """
    segmenter_path = algo_dir / "scripts" / "segmenter.py"
    if not segmenter_path.exists():
        print(f"    ERROR: {segmenter_path} not found")
        return False

    text = segmenter_path.read_text()
    original = text

    # --- 1. Add import ---
    import_line = (
        "\n# === CellMap partial annotation support ===\n"
        "try:\n"
        "    from partial_annotation import (\n"
        "        PartialAnnotationLossV2,\n"
        "        PartialAnnotationDeepSupervisionLoss,\n"
        "        parse_annotation_mask_from_batch,\n"
        "    )\n"
        "    HAS_PARTIAL_ANNOTATION = True\n"
        "except ImportError:\n"
        "    HAS_PARTIAL_ANNOTATION = False\n"
        "# === end CellMap ===\n"
    )

    if "partial_annotation" not in text:
        # Insert after the existing imports (after the LabelEmbedClassIndex class)
        # Find a good insertion point — after the last top-level import block
        match = re.search(
            r"^(class LabelEmbedClassIndex.*?)(?=\n(?:def |class ))",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            insert_pos = match.end()
            text = text[:insert_pos] + "\n" + import_line + text[insert_pos:]
        else:
            # Fallback: insert after all imports
            lines = text.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith(("import ", "from ")) or line.strip().startswith(("import ", "from ")):
                    insert_idx = i + 1
            text = "\n".join(lines[:insert_idx]) + import_line + "\n".join(lines[insert_idx:])

    # --- 2. Replace loss construction in Segmenter.__init__ ---
    # Find: loss_function = ConfigParser(config["loss"]).get_parsed_content(...)
    #        self.loss_function = DeepSupervisionLoss(loss_function)
    # Replace with: partial annotation loss

    old_loss_pattern = (
        r'(loss_function = ConfigParser\(config\["loss"\]\)\.get_parsed_content\(.*?\))\s*\n'
        r'(\s*)(self\.loss_function = DeepSupervisionLoss\(loss_function\))'
    )
    new_loss = (
        r'\1\n'
        r'\2# === CellMap: partial annotation loss ===\n'
        r'\2if HAS_PARTIAL_ANNOTATION and config.get("sigmoid", False):\n'
        r'\2    _num_classes = config.get("output_classes", 14)\n'
        r'\2    _pa_loss = PartialAnnotationLossV2(\n'
        r'\2        sigmoid=True, squared_pred=True,\n'
        r'\2        smooth_nr=1e-5, smooth_dr=1e-5,\n'
        r'\2        num_classes=_num_classes,\n'
        r'\2    )\n'
        r'\2    self.loss_function = PartialAnnotationDeepSupervisionLoss(_pa_loss)\n'
        r'\2    if self.global_rank == 0:\n'
        r'\2        print(f"Using PartialAnnotationLoss with {_num_classes} classes")\n'
        r'\2else:\n'
        r'\2    \3\n'
        r'\2# === end CellMap ===\n'
    )

    text_new = re.sub(old_loss_pattern, new_loss, text, count=1)
    if text_new == text:
        print("    WARNING: Could not patch loss construction (pattern not found)")
        print("    Will attempt manual patching...")
    else:
        text = text_new

    # --- 3. Patch train_epoch to set annotation mask ---
    # Find the line: data = batch_data["image"]...
    # Add annotation mask extraction before the loss call

    # Pattern: In train_epoch, after extracting data and target from batch_data,
    # but before the loss computation
    train_epoch_patch = (
        '\n            # === CellMap: set annotation mask ===\n'
        '            if HAS_PARTIAL_ANNOTATION and hasattr(loss_function, "set_annotation_mask"):\n'
        '                _ann_mask = parse_annotation_mask_from_batch(batch_data, num_classes=target.shape[1])\n'
        '                if _ann_mask is not None:\n'
        '                    loss_function.set_annotation_mask(_ann_mask)\n'
        '            # === end CellMap ===\n'
    )

    # Find the pattern in train_epoch where data and target are extracted
    # then loss is computed. Insert mask setting between them.
    pattern = (
        r'(target = batch_data\["label"\]\.as_subclass\(torch\.Tensor\)\.to\(.*?\))'
    )
    match = re.search(pattern, text)
    if match:
        insert_pos = match.end()
        text = text[:insert_pos] + train_epoch_patch + text[insert_pos:]
    else:
        print("    WARNING: Could not find train_epoch batch_data pattern")

    if text != original:
        segmenter_path.write_text(text)
        print(f"    Patched {segmenter_path.name}")
        return True
    else:
        print(f"    No changes made to {segmenter_path.name}")
        return False


def patch_swinunetr(algo_dir: Path) -> bool:
    """
    Patch swinunetr/dints train.py for partial annotation support.

    Changes:
    1. Add import for partial_annotation module inside if __name__ block
    2. Replace loss_function creation to use PartialAnnotationLossV2
    3. Patch the training loop to extract annotation mask and set it
       (inside both amp/autocast and non-amp branches)
    """
    train_path = algo_dir / "scripts" / "train.py"
    if not train_path.exists():
        print(f"    ERROR: {train_path} not found")
        return False

    text = train_path.read_text()
    original = text

    # --- 1. Add import inside if __name__ == "__main__" block ---
    # The import MUST be indented under the if __name__ block, otherwise
    # the try/except at module level will swallow the subsequent fire.Fire()
    # into the except branch.
    import_block = (
        '    # === CellMap partial annotation support ===\n'
        '    try:\n'
        '        from partial_annotation import (\n'
        '            PartialAnnotationLossV2,\n'
        '            parse_annotation_mask_from_batch,\n'
        '        )\n'
        '        HAS_PARTIAL_ANNOTATION = True\n'
        '    except ImportError:\n'
        '        HAS_PARTIAL_ANNOTATION = False\n'
        '    # === end CellMap ===\n'
    )

    if "partial_annotation" not in text:
        # Find the if __name__ == "__main__": block and insert after its first line
        main_match = re.search(r'(if __name__ == "__main__":\n\s+from monai\.utils import optional_import\n)', text)
        if main_match:
            insert_pos = main_match.end()
            text = text[:insert_pos] + import_block + text[insert_pos:]
        else:
            print("    WARNING: Could not find if __name__ block for import insertion")

    # --- 2. Replace loss_function after it's created ---
    # swinunetr uses: loss_function = parser.get_parsed_content("loss")
    # dints uses:     loss_function = parser.get_parsed_content("training#loss")
    loss_replacement = (
        '\n'
        '    # === CellMap: partial annotation loss ===\n'
        '    if HAS_PARTIAL_ANNOTATION and not softmax:\n'
        '        _num_classes = output_classes\n'
        '        loss_function = PartialAnnotationLossV2(\n'
        '            sigmoid=True, squared_pred=True,\n'
        '            smooth_nr=1e-5, smooth_dr=1e-5,\n'
        '            num_classes=_num_classes,\n'
        '        )\n'
        '        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:\n'
        '            logger.debug(f"Using PartialAnnotationLoss with {_num_classes} classes")\n'
        '    # === end CellMap ==='
    )

    # Match either "loss" or "training#loss" key
    loss_pattern = r'(loss_function = parser\.get_parsed_content\("(?:training#)?loss"\))'
    if re.search(loss_pattern, text) and "CellMap: partial annotation" not in text:
        text = re.sub(loss_pattern, r'\1' + loss_replacement, text, count=1)

    # --- 3. Patch training loop to set annotation mask before loss ---
    # The mask extraction must be INSIDE the `with autocast` block (amp branch)
    # and also in the `else` (non-amp) branch.
    #
    # Original structure:
    #   if amp:
    #       with autocast("cuda"):
    #           outputs = model(inputs)
    #           loss = loss_function(outputs.float(), labels)  # 32-space indent
    #       scaler.scale(loss).backward()
    #   else:
    #       outputs = model(inputs)
    #       loss = loss_function(outputs.float(), labels)      # 28-space indent
    #
    # We insert mask extraction before each loss = ... line.

    mask_amp = (
        '                                # === CellMap: set annotation mask ===\n'
        '                                if HAS_PARTIAL_ANNOTATION and hasattr(loss_function, "set_annotation_mask"):\n'
        '                                    _ann_mask = parse_annotation_mask_from_batch(batch_data, num_classes=labels.shape[1])\n'
        '                                    if _ann_mask is not None:\n'
        '                                        loss_function.set_annotation_mask(_ann_mask)\n'
        '                                # === end CellMap ===\n'
    )

    mask_noamp = (
        '                            # === CellMap: set annotation mask ===\n'
        '                            if HAS_PARTIAL_ANNOTATION and hasattr(loss_function, "set_annotation_mask"):\n'
        '                                _ann_mask = parse_annotation_mask_from_batch(batch_data, num_classes=labels.shape[1])\n'
        '                                if _ann_mask is not None:\n'
        '                                    loss_function.set_annotation_mask(_ann_mask)\n'
        '                            # === end CellMap ===\n'
    )

    if "CellMap: set annotation mask" not in text:
        # amp branch: loss at 32-space indent (inside with autocast)
        amp_loss = '                                loss = loss_function(outputs.float(), labels)'
        # non-amp branch: loss at 28-space indent
        noamp_loss = '                            loss = loss_function(outputs.float(), labels)'

        # Replace amp branch first (deeper indent = more specific match)
        if amp_loss in text:
            text = text.replace(amp_loss, mask_amp + amp_loss, 1)

        # Replace non-amp branch
        if noamp_loss in text:
            text = text.replace(noamp_loss, mask_noamp + noamp_loss, 1)

    if text != original:
        train_path.write_text(text)
        print(f"    Patched {train_path.name}")
        return True
    else:
        print(f"    No changes made to {train_path.name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Patch Auto3DSeg templates for partial annotation support"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="auto3dseg/work_dir",
        help="Auto3DSeg work directory containing generated algorithm bundles",
    )
    args = parser.parse_args()

    # Find the source partial_annotation.py
    script_dir = Path(__file__).parent
    source_pa = script_dir / "partial_annotation.py"
    if not source_pa.exists():
        print(f"ERROR: {source_pa} not found")
        sys.exit(1)

    # Find algorithm directories
    algo_dirs = find_algorithm_dirs(args.work_dir)
    if not algo_dirs:
        print(f"No algorithm bundles found in {args.work_dir}")
        print("Run BundleGen first to generate algorithm templates.")
        sys.exit(1)

    print(f"Found {len(algo_dirs)} algorithm bundles in {args.work_dir}")
    print()

    patched = 0
    for algo_dir in algo_dirs:
        template_type = detect_template_type(algo_dir)
        print(f"  [{template_type}] {algo_dir.name}")

        # Copy module
        copy_partial_annotation_module(algo_dir, script_dir)

        # Patch based on template type
        if template_type == "segresnet":
            if patch_segresnet(algo_dir):
                patched += 1
        elif template_type == "swinunetr":
            if patch_swinunetr(algo_dir):
                patched += 1
        else:
            print(f"    SKIP: unknown template type")

    print(f"\nDone. Patched {patched}/{len(algo_dirs)} algorithm bundles.")


if __name__ == "__main__":
    main()
