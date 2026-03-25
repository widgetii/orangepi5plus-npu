#!/bin/bash
# Hardware smoke test for CI — runs on the Orange Pi 5 Plus board.
# Requires: NPU device accessible, model files in /root/npu-research/models/
# Exit codes: 0 = all tests pass, 1 = test failure, 2 = setup error
set -euo pipefail

MODEL_DIR="/root/npu-research/models"
MODEL="${MODEL_DIR}/mobilenet_v1_quant.tflite"
INPUT="${MODEL_DIR}/grace_hopper_224.bin"
# Golden from git (TFLite CPU reference, class 653 "military uniform")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GOLDEN="${SCRIPT_DIR}/../../qemu-boot/golden_mobilenet_output.bin"
# MobileNetV1 class 653 = "military uniform" (Grace Hopper photo)
EXPECTED_CLASS=653

echo "=== Hardware Smoke Test ==="
echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"

# Check NPU device
if [ -e /dev/accel/accel0 ]; then
    echo "NPU: Rocket driver (/dev/accel/accel0)"
elif ls /dev/dri/card* &>/dev/null; then
    echo "NPU: RKNPU vendor driver (/dev/dri/card*)"
else
    echo "ERROR: No NPU device found"
    exit 2
fi

# Check required files
for f in "$MODEL" "$INPUT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 2
    fi
done

if [ ! -f "$GOLDEN" ]; then
    echo "ERROR: Golden file not found: $GOLDEN"
    echo "  (golden must be committed in git at qemu-boot/golden_mobilenet_output.bin)"
    exit 2
fi

# Build
echo ""
echo "=== Building librocketnpu ==="
make clean 2>/dev/null || true
make librocketnpu.so test_mobilenet test_sw_ops test_rknpu_abi

# Run CPU-only tests on aarch64
echo ""
echo "=== Running sw_ops tests (aarch64) ==="
LD_LIBRARY_PATH=. ./test_sw_ops

echo ""
echo "=== Running rknpu_abi tests (aarch64) ==="
LD_LIBRARY_PATH=. ./test_rknpu_abi

# Run NPU inference with real image and classification check
echo ""
echo "=== Running MobileNetV1 NPU inference (Grace Hopper) ==="
LD_LIBRARY_PATH=. ./test_mobilenet "$MODEL" 5 "$GOLDEN" "$INPUT" "$EXPECTED_CLASS"

echo ""
echo "=== All hardware smoke tests passed ==="
