#!/bin/bash
#SBATCH --job-name=redownload_blank
#SBATCH --partition=batch
#SBATCH --account=rc_alain_pi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/redownload_blank_%j.log

# =============================================================================
# Re-download 28 blank-image crops with --fetch-all-em-resolutions
#
# These crops have ground-truth labels at coarse resolution (8nm+), so the
# default csc fetch-data only downloaded EM at matching scales (s2-s4+),
# skipping s0 (2nm) and s1 (4nm). Our NIfTI conversion reads s0, so the
# images came out all-zero.
#
# With --fetch-all-em-resolutions, the downloader sets ratio_threshold=0
# so ALL EM scales including s0 are fetched. This ensures the converter
# will find real EM data at s0.
#
# Excluded: crop257 (jrc_cos7-1a) — no EM data at any scale.
# =============================================================================

echo "=== Re-downloading 28 blank crops with --fetch-all-em-resolutions ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

CROPS="238,423,179,184,230,231,177,473,282,289,324,337,346,349,351,386,410,412,413,355,356,357,358,367,387,197,379,380"

echo "Crops to re-download: $CROPS"
echo "Total: 28 crops"
echo ""

csc fetch-data \
    --crops "$CROPS" \
    -d data/ \
    --fetch-all-em-resolutions \
    --batch-size 512 \
    --num-workers 64

DL_EXIT=$?
echo ""
echo "Download exit code: $DL_EXIT"
echo "Download complete at $(date)"

if [ $DL_EXIT -ne 0 ]; then
    echo "WARNING: Download exited with non-zero code $DL_EXIT"
fi

# =============================================================================
# Verify: check that s0 raw EM chunks now exist for each dataset
# =============================================================================
echo ""
echo "=== Verifying s0 EM data availability ==="

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

for DS in "${DATASETS[@]}"; do
    ZARR="data/${DS}/${DS}.zarr/recon-1/em/fibsem-uint8/s0"
    if [ -d "$ZARR" ]; then
        CHUNKS=$(find "$ZARR" -name "*.0" -o -name "0" -type f 2>/dev/null | head -5 | wc -l)
        TOTAL=$(find "$ZARR" -not -name ".z*" -type f 2>/dev/null | wc -l)
        echo "  $DS/s0: $TOTAL chunks on disk"
    else
        echo "  $DS/s0: MISSING (no s0 directory)"
    fi
done

echo ""
echo "=== Done ==="
echo "End time: $(date)"
echo ""
echo "NEXT STEPS:"
echo "  1. Run reconvert_blank_images.py for the 28 crops"
echo "  2. Add them back to datalist.json"
echo "  3. Re-run DataAnalyzer"
echo "  4. Resubmit training"
