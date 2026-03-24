#!/bin/bash
# Hardware smoke test for CI — runs on the Orange Pi 5 Plus board.
# Requires: NPU device accessible, model files in /root/npu-research/models/
# Exit codes: 0 = all tests pass, 1 = test failure, 2 = setup error
set -euo pipefail

MODEL_DIR="/root/npu-research/models"
MODEL="${MODEL_DIR}/mobilenet_v1_quant.tflite"
GOLDEN="${MODEL_DIR}/mobilenet_golden.bin"

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

# Check model files
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
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

# Run NPU inference test
echo ""
echo "=== Running MobileNetV1 NPU inference ==="
if [ -f "$GOLDEN" ]; then
    LD_LIBRARY_PATH=. ./test_mobilenet "$MODEL" 5 "$GOLDEN"
else
    echo "WARNING: No golden file, running without comparison"
    LD_LIBRARY_PATH=. ./test_mobilenet "$MODEL" 5
fi

echo ""
echo "=== All hardware smoke tests passed ==="
