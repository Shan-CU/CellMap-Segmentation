#!/usr/bin/env python3
"""Generate a visual pipeline diagram using matplotlib for full control over layout."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "pipeline_visual.png")

# ── Colors ──
GOLD    = "#CFB87C"
BLUE    = "#156082"
LTBLUE  = "#D0DEE8"
GREEN   = "#196B24"
LTGREEN = "#D4E8D4"
DKGRAY  = "#333333"
GRAY    = "#999999"
WHITE   = "#FFFFFF"
CREAM   = "#F5F0E6"
LTGRAY  = "#F2F2F2"

fig, ax = plt.subplots(figsize=(13.33, 5.8), dpi=300)
ax.set_xlim(0, 13.33)
ax.set_ylim(0, 5.8)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")


def rounded_box(x, y, w, h, color, label, sublabel=None, fontsize=14,
                sublabel_size=10, text_color="white"):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=color,
                          linewidth=0, zorder=2)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                fontfamily="sans-serif", zorder=3)
        ax.text(cx, cy - 0.17, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=text_color, fontfamily="sans-serif",
                zorder=3, alpha=0.85)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color,
                fontfamily="sans-serif", zorder=3)


def arrow(x1, y1, x2, y2, color=GOLD, lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=18),
                zorder=4)


def section_bg(x, y, w, h, color, label):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=GRAY,
                          linewidth=0.8, linestyle="--",
                          alpha=0.6, zorder=1)
    ax.add_patch(box)
    ax.text(x + 0.15, y + h - 0.15, label, fontsize=10,
            color=GRAY, fontfamily="sans-serif", va="top", zorder=2)


# ════════════════════════════════════════════════════════════
# ROW 1 (middle): DATA → MODEL → LOSS → OUTPUT
# ════════════════════════════════════════════════════════════
row1_y = 2.6
box_h = 1.0
bw = 1.35

# Section backgrounds
section_bg(0.2, row1_y - 0.28, 5.2, box_h + 0.65, LTGRAY, "Data Layer")
section_bg(5.7, row1_y - 0.28, 2.3, box_h + 0.65, LTBLUE, "Model")
section_bg(8.3, row1_y - 0.28, 4.8, box_h + 0.65, CREAM, "Loss Pipeline")

# Data boxes
rounded_box(0.45, row1_y, bw, box_h, DKGRAY, "22 Zarr", "Volumes", fontsize=12)
rounded_box(2.05, row1_y, bw, box_h, DKGRAY, "289 Crops", "48 Classes", fontsize=12)
rounded_box(3.65, row1_y, bw, box_h, DKGRAY, "Weighted", "Sampler", fontsize=12)
arrow(0.45 + bw, row1_y + box_h / 2, 2.05, row1_y + box_h / 2)
arrow(2.05 + bw, row1_y + box_h / 2, 3.65, row1_y + box_h / 2)

# Model
arrow(3.65 + bw, row1_y + box_h / 2, 5.9, row1_y + box_h / 2)
rounded_box(5.9, row1_y, 1.9, box_h, BLUE, "FlexUNet", "ResNet-34", fontsize=13)

# Loss pipeline
lbw = 1.2
arrow(5.9 + 1.9, row1_y + box_h / 2, 8.55, row1_y + box_h / 2)
rounded_box(8.55, row1_y, lbw, box_h, GOLD, "NaN", "Mask", fontsize=12, text_color=WHITE)
rounded_box(10.0, row1_y, lbw, box_h, GOLD, "BST", "αD + βDS", fontsize=12, text_color=WHITE)
rounded_box(11.45, row1_y, lbw, box_h, GOLD, "Logit", "Adj. τ", fontsize=12, text_color=WHITE)
arrow(8.55 + lbw, row1_y + box_h / 2, 10.0, row1_y + box_h / 2)
arrow(10.0 + lbw, row1_y + box_h / 2, 11.45, row1_y + box_h / 2)


# ════════════════════════════════════════════════════════════
# ROW 2 (bottom): ABLATION SWEEPS
# ════════════════════════════════════════════════════════════
row2_y = 0.45
sweep_h = 0.8

section_bg(0.2, row2_y - 0.2, 12.9, sweep_h + 0.45, "#E8E0D0",
           "Ablation Sweeps — each isolates one variable")

sweeps = [
    ("Sweep A", "Loss Function", "8 configs"),
    ("Sweep B", "Tversky α/β", "6 configs"),
    ("Sweep C", "Class Weight τ", "5 configs"),
    ("Sweep D", "Masking", "7 configs"),
    ("Sweep E", "Training Tricks", "3+ configs"),
]

sw = 2.2
gap = 0.28
start_x = 0.55
for i, (name, desc, count) in enumerate(sweeps):
    x = start_x + i * (sw + gap)
    rounded_box(x, row2_y, sw, sweep_h, BLUE, name,
                f"{desc}  ({count})", fontsize=11, sublabel_size=9, text_color=WHITE)

# Dashed arrows connecting sweeps to the pipeline components they vary
connections = [
    # (sweep_idx, target_cx, target_bottom_y)
    (0, 10.0 + lbw / 2, row1_y),       # A → BST loss
    (1, 10.0 + lbw / 2, row1_y),       # B → BST (α/β)
    (2, 11.45 + lbw / 2, row1_y),      # C → Logit adj
    (3, 8.55 + lbw / 2, row1_y),       # D → NaN mask
    (4, 5.9 + 1.9 / 2, row1_y),        # E → Model
]

for sweep_i, target_x, target_y in connections:
    sx = start_x + sweep_i * (sw + gap) + sw / 2
    sy = row2_y + sweep_h
    ax.annotate("", xy=(target_x, target_y),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0,
                                connectionstyle="arc3,rad=0.15",
                                linestyle="dashed"),
                zorder=3)


# ════════════════════════════════════════════════════════════
# TOP: OUTPUT + COMPUTE + KEY MESSAGE
# ════════════════════════════════════════════════════════════

# Output
arrow(11.45 + lbw / 2, row1_y + box_h, 11.45 + lbw / 2, 4.65)
rounded_box(10.85, 4.7, 1.8, 0.7, GREEN, "48-Class", "Segmentation",
            fontsize=12, text_color=WHITE)

# Compute badges
for i, (label, detail) in enumerate([("2D: ~3h  L40S 48 GB", "384 GB RAM"),
                                      ("3D: ~15h  H100 80 GB", "512 GB RAM")]):
    cx = 0.3 + i * 3.6
    cy = 4.85
    box = FancyBboxPatch((cx, cy), 3.2, 0.48,
                          boxstyle="round,pad=0.06",
                          facecolor=LTGRAY, edgecolor=GRAY,
                          linewidth=0.5, zorder=2)
    ax.add_patch(box)
    ax.text(cx + 0.12, cy + 0.24, label, fontsize=9.5,
            fontweight="bold", color=DKGRAY, va="center",
            fontfamily="sans-serif", zorder=3)
    ax.text(cx + 3.08, cy + 0.24, detail, fontsize=8.5,
            color=GRAY, va="center", ha="right",
            fontfamily="sans-serif", zorder=3)

# Key message
ax.text(8.0, 5.09, "BST loss held constant → each sweep isolates one variable",
        fontsize=10.5, color=BLUE, fontweight="bold", fontfamily="sans-serif",
        va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=LTBLUE, edgecolor=BLUE,
                  linewidth=1.2, alpha=0.7),
        zorder=5)

plt.tight_layout(pad=0.1)
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
plt.close()

from PIL import Image
w, h = Image.open(OUT).size
print(f"✅ pipeline_visual.png: {w}×{h}")
