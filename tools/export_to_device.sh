#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================================="
echo "    NanoEdge Mobile Model Packager & Device Exporter      "
echo "=========================================================="

MODELS_DIR="$ROOT_DIR/models"
mkdir -p "$MODELS_DIR"

MOBILE_MODEL="$MODELS_DIR/mobile_demo_q4.mllm"

if [ ! -f "$MOBILE_MODEL" ]; then
    echo "[1/2] Compiling 16KB page-aligned mobile model: $MOBILE_MODEL ..."
    python3 "$ROOT_DIR/tools/mllm_compiler.py" \
        --synthetic \
        --dim 2048 \
        --hidden-dim 5632 \
        --layers 4 \
        --quant Q4_0 \
        --out "$MOBILE_MODEL"
else
    echo "[1/2] Model already exists: $MOBILE_MODEL"
fi

echo ""
echo "[2/2] Preparing iOS bundle assets..."
mkdir -p "$ROOT_DIR/mobile/ios/NanoEdgeApp/Resources"
cp "$MOBILE_MODEL" "$ROOT_DIR/mobile/ios/NanoEdgeApp/Resources/"

FILE_SIZE=$(du -h "$MOBILE_MODEL" | cut -f1)
echo "✅ Export Complete! Model size: $FILE_SIZE"
echo ""
echo "=========================================================="
echo "          HOW TO TEST ON A PHYSICAL IPHONE / IPAD         "
echo "=========================================================="
echo "Method 1: AirDrop / Files App (Easiest)"
echo "  1. AirDrop '$MOBILE_MODEL' from your Mac directly to your iPhone."
echo "  2. Save it to 'Files' -> 'On My iPhone'."
echo "  3. In the NanoEdge app, tap 'Import .mllm' and select the file."
echo ""
echo "Method 2: Mac Finder Cable Transfer"
echo "  1. Connect your iPhone to your Mac via USB cable."
echo "  2. Open Finder -> Select your iPhone in the sidebar."
echo "  3. Click the 'Files' tab -> Find 'NanoEdge'."
echo "  4. Drag and drop '$MOBILE_MODEL' into the NanoEdge folder."
echo ""
echo "Method 3: Build to Device via Xcode"
echo "  1. Open Xcode -> Open the 'mobile/ios' folder."
echo "  2. Connect your iPhone and select it as the Run Destination."
echo "  3. Under Signing & Capabilities, select your free Personal Team."
echo "  4. Press Cmd+R (Run)."
echo "=========================================================="
