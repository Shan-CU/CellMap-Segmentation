#!/usr/bin/env python3
"""
Generate a publication-quality class color legend as a PNG image.

Uses matplotlib with LaTeX-style rendering to produce a clean, paper-ready
legend matching the 48-class color palette used in validation visualizations.

Usage:
    python -m training.make_legend
    python -m training.make_legend --output runs/ablation/class_legend.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellmap_segmentation_challenge.utils import get_tested_classes

# 48 perceptually distinct colors (3 HSV rings × 16 hues, interleaved)
# Matches train.py and regen_val_images.py exactly
CLASS_COLORS = [
    (0.9, 0.0, 0.0),      #  0 ecs
    (0.35, 1.0, 0.35),    #  1 pm
    (0.0, 0.0, 0.5),      #  2 mito_mem
    (0.9, 0.338, 0.0),    #  3 mito_lum
    (0.35, 1.0, 0.594),   #  4 mito_ribo
    (0.188, 0.0, 0.5),    #  5 golgi_mem
    (0.9, 0.675, 0.0),    #  6 golgi_lum
    (0.35, 1.0, 0.838),   #  7 ves_mem
    (0.375, 0.0, 0.5),    #  8 ves_lum
    (0.787, 0.9, 0.0),    #  9 endo_mem
    (0.35, 0.919, 1.0),   # 10 endo_lum
    (0.5, 0.0, 0.438),    # 11 lyso_mem
    (0.45, 0.9, 0.0),     # 12 lyso_lum
    (0.35, 0.675, 1.0),   # 13 ld_mem
    (0.5, 0.0, 0.25),     # 14 ld_lum
    (0.113, 0.9, 0.0),    # 15 er_mem
    (0.35, 0.431, 1.0),   # 16 er_lum
    (0.5, 0.0, 0.062),    # 17 eres_mem
    (0.0, 0.9, 0.225),    # 18 eres_lum
    (0.512, 0.35, 1.0),   # 19 ne_mem
    (0.5, 0.125, 0.0),    # 20 ne_lum
    (0.0, 0.9, 0.562),    # 21 np_out
    (0.756, 0.35, 1.0),   # 22 np_in
    (0.5, 0.312, 0.0),    # 23 hchrom
    (0.0, 0.9, 0.9),      # 24 echrom
    (1.0, 0.35, 1.0),     # 25 nucpl
    (0.5, 0.5, 0.0),      # 26 mt_out
    (0.0, 0.562, 0.9),    # 27 cyto
    (1.0, 0.35, 0.756),   # 28 mt_in
    (0.312, 0.5, 0.0),    # 29 nuc
    (0.0, 0.225, 0.9),    # 30 golgi
    (1.0, 0.35, 0.512),   # 31 ves
    (0.125, 0.5, 0.0),    # 32 endo
    (0.113, 0.0, 0.9),    # 33 lyso
    (1.0, 0.431, 0.35),   # 34 ld
    (0.0, 0.5, 0.062),    # 35 eres
    (0.45, 0.0, 0.9),     # 36 perox_mem
    (1.0, 0.675, 0.35),   # 37 perox_lum
    (0.0, 0.5, 0.25),     # 38 perox
    (0.787, 0.0, 0.9),    # 39 mito
    (1.0, 0.919, 0.35),   # 40 er
    (0.0, 0.5, 0.438),    # 41 ne
    (0.9, 0.0, 0.675),    # 42 np
    (0.838, 1.0, 0.35),   # 43 chrom
    (0.0, 0.375, 0.5),    # 44 mt
    (0.9, 0.0, 0.338),    # 45 cell
    (0.594, 1.0, 0.35),   # 46 er_mem_all
    (0.0, 0.188, 0.5),    # 47 ne_mem_all
]

# Human-readable display names for the 48 classes
DISPLAY_NAMES = {
    "ecs": "ECS (extracellular space)",
    "pm": "Plasma membrane",
    "mito_mem": "Mito. membrane",
    "mito_lum": "Mito. lumen",
    "mito_ribo": "Mito. ribosome",
    "golgi_mem": "Golgi membrane",
    "golgi_lum": "Golgi lumen",
    "ves_mem": "Vesicle membrane",
    "ves_lum": "Vesicle lumen",
    "endo_mem": "Endosome membrane",
    "endo_lum": "Endosome lumen",
    "lyso_mem": "Lysosome membrane",
    "lyso_lum": "Lysosome lumen",
    "ld_mem": "Lipid droplet membrane",
    "ld_lum": "Lipid droplet lumen",
    "er_mem": "ER membrane",
    "er_lum": "ER lumen",
    "eres_mem": "ER exit site membrane",
    "eres_lum": "ER exit site lumen",
    "ne_mem": "Nuclear env. membrane",
    "ne_lum": "Nuclear env. lumen",
    "np_out": "Nuclear pore (outer)",
    "np_in": "Nuclear pore (inner)",
    "hchrom": "Heterochromatin",
    "echrom": "Euchromatin",
    "nucpl": "Nucleoplasm",
    "mt_out": "Microtubule (outer)",
    "cyto": "Cytoplasm",
    "mt_in": "Microtubule (inner)",
    "nuc": "Nucleus",
    "golgi": "Golgi",
    "ves": "Vesicle",
    "endo": "Endosome",
    "lyso": "Lysosome",
    "ld": "Lipid droplet",
    "eres": "ER exit site",
    "perox_mem": "Peroxisome membrane",
    "perox_lum": "Peroxisome lumen",
    "perox": "Peroxisome",
    "mito": "Mitochondria",
    "er": "ER",
    "ne": "Nuclear envelope",
    "np": "Nuclear pore",
    "chrom": "Chromatin",
    "mt": "Microtubule",
    "cell": "Cell",
    "er_mem_all": "ER membrane (all)",
    "ne_mem_all": "Nuclear env. mem. (all)",
}


def make_legend(output_path: str, ncols: int = 4, dpi: int = 300):
    """Generate a publication-quality class legend PNG."""
    classes = get_tested_classes()
    n = len(classes)
    nrows = int(np.ceil(n / ncols))

    # Use serif font for LaTeX-like appearance
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "text.usetex": False,  # Don't require actual LaTeX installation
    })

    fig_width = ncols * 2.8
    fig_height = nrows * 0.32 + 0.7  # tight layout
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.axis("off")
    ax.invert_yaxis()

    # Title
    fig.suptitle(
        "CellMap Segmentation — Class Color Legend (48 classes)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
        fontfamily="serif",
    )

    swatch_w = 0.22
    swatch_h = 0.6
    text_x_offset = 0.30
    y_pad = 0.15

    for idx, cname in enumerate(classes):
        col = idx % ncols
        row = idx // ncols

        color = CLASS_COLORS[idx]
        display = DISPLAY_NAMES.get(cname, cname)

        # Color swatch with dark border
        rect = mpatches.FancyBboxPatch(
            (col + 0.05, row + y_pad),
            swatch_w, swatch_h,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=(0.2, 0.2, 0.2),
            linewidth=0.8,
        )
        ax.add_patch(rect)

        # Class label: "short_name — Full Name"
        # Use monospace for the short name, serif for full name
        label = f"{cname}"
        ax.text(
            col + text_x_offset, row + 0.45,
            label,
            fontsize=7.5,
            fontfamily="monospace",
            fontweight="bold",
            va="center",
            color=(0.1, 0.1, 0.1),
        )
        # Full name on second line if space allows
        if display != cname:
            ax.text(
                col + text_x_offset, row + 0.72,
                display,
                fontsize=6.0,
                fontfamily="serif",
                fontstyle="italic",
                va="center",
                color=(0.35, 0.35, 0.35),
            )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Legend saved to {output_path} ({dpi} DPI)")


def main():
    parser = argparse.ArgumentParser(description="Generate class color legend PNG")
    parser.add_argument("--output", type=str,
                        default="runs/ablation/class_legend.png",
                        help="Output path for the legend image")
    parser.add_argument("--ncols", type=int, default=4,
                        help="Number of columns in the legend grid")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (300 for print, 150 for screen)")
    args = parser.parse_args()
    make_legend(args.output, ncols=args.ncols, dpi=args.dpi)


if __name__ == "__main__":
    main()
