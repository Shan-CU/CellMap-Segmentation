#!/bin/bash
#SBATCH --job-name=dl_blank_crops
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/download_blank_crops_%j.log

# =============================================================================
# Re-download 28 blank-image crops with --fetch-all-em-resolutions
#
# These 28 crops had blank (all-zero) raw EM images because:
#   - 25 crops: GT labels at 8nm resolution, so default csc fetch-data skipped
#     s0 (2nm) and s1 (4nm) raw EM as "sampled more densely than groundtruth"
#   - 3 crops (197, 379, 380): download crashed (JSONDecodeError) before EM fetch
#
# Using --fetch-all-em-resolutions forces download of ALL EM pyramid levels
# including s0, which our NIfTI converter needs.
#
# Excluded: crop257 (jrc_cos7-1a) — no raw EM data at any scale, irrecoverable.
# =============================================================================

echo "============================================================"
echo "Re-downloading 28 blank crops with --fetch-all-em-resolutions"
echo "Start time: $(date)"
echo "============================================================"

source ~/.bashrc
micromamba activate csc
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# 28 crop IDs to re-download:
# jrc_cos7-1b:          238
# jrc_mus-heart-1:      423
# jrc_mus-kidney:        179, 184, 230, 231
# jrc_mus-liver:         177
# jrc_mus-liver-3:       473
# jrc_mus-liver-zon-1:   282, 289, 324, 337, 346, 349, 351, 386, 410, 412, 413
# jrc_mus-liver-zon-2:   355, 356, 357, 358, 367, 387
# jrc_ut21-1413-003:     197
# jrc_zf-cardiac-1:      379, 380

CROP_IDS="238,423,179,184,230,231,177,473,282,289,324,337,346,349,351,386,410,412,413,355,356,357,358,367,387,197,379,380"

echo "Crop IDs: $CROP_IDS"
echo ""

csc fetch-data \
    --crops "$CROP_IDS" \
    -d data/ \
    --fetch-all-em-resolutions \
    --num-workers 64 \
    --batch-size 512

DOWNLOAD_EXIT=$?

echo ""
echo "============================================================"
echo "Download exit code: $DOWNLOAD_EXIT"
echo "End time: $(date)"
echo "============================================================"

# Verify s0 raw EM is now present for each crop's dataset
echo ""
echo "=== Verification: checking for s0 raw EM data ==="

DATASETS=(
    "jrc_cos7-1b"
    "jrc_mus-heart-1"
    "jrc_mus-kidney"
    "jrc_mus-liver"
    "jrc_mus-liver-3"
    "jrc_mus-liver-zon-1"
    "jrc_mus-liver-zon-2"
    "jrc_ut21-1413-003"
    "jrc_zf-cardiac-1"
)

for ds in "${DATASETS[@]}"; do
    s0_path="data/${ds}/${ds}.zarr/recon-1/em/fibsem-uint8/s0"
    if [[ -d "$s0_path" ]]; then
        # Count chunks
        n_chunks=$(find "$s0_path" -type f | wc -l)
        echo "  ✓ $ds: s0 exists ($n_chunks chunks)"
    else
        echo "  ✗ $ds: s0 NOT FOUND"
    fi
done

echo ""
echo "Done. Next step: reconvert these 28 crops with reconvert_blank_images.py"
