#!/bin/bash
# Build all LaTeX/TikZ figures to high-res PNGs
# Usage: bash build_figures.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use system pdflatex (TeX Live 2020), NOT conda's broken one
PDFLATEX=/usr/bin/pdflatex

DPI=300
BUILT=0
FAILED=0

echo "═══════════════════════════════════════════════════"
echo "  Building presentation figures ($DPI DPI)"
echo "  Using: $PDFLATEX"
echo "═══════════════════════════════════════════════════"

for texfile in *.tex; do
    base="${texfile%.tex}"
    pngfile="${base}.png"

    echo -n "  [$base] compiling... "

    # Compile LaTeX to PDF
    if ! $PDFLATEX -interaction=nonstopmode -halt-on-error "$texfile" > "${base}.build.log" 2>&1; then
        echo "FAILED (LaTeX)"
        echo "    See ${base}.build.log for details"
        tail -5 "${base}.build.log" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        continue
    fi

    # Convert PDF to high-res PNG
    if ! pdftoppm -png -r "$DPI" -singlefile "${base}.pdf" "$base" >> "${base}.build.log" 2>&1; then
        echo "FAILED (pdftoppm)"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Trim whitespace with ImageMagick
    if command -v convert &>/dev/null; then
        convert "${pngfile}" -trim +repage "${pngfile}" 2>/dev/null || true
    fi

    # Report size
    dims=$(identify -format "%wx%h" "$pngfile" 2>/dev/null || echo "?x?")
    fsize=$(du -h "$pngfile" | cut -f1)
    echo "OK  ($dims, $fsize)"
    BUILT=$((BUILT + 1))
done

echo ""
echo "Done: $BUILT built, $FAILED failed"
echo "═══════════════════════════════════════════════════"
