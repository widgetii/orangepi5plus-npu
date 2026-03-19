#!/bin/bash
# Phase 2: Set up RKNN NPU stack on Armbian
# Run this AFTER flashing Armbian with vendor kernel
set -euo pipefail

echo "=== Orange Pi 5 Plus NPU Setup ==="

# Step 1: Verify kernel and NPU driver
echo ""
echo "--- System Info ---"
uname -a
echo ""

echo "--- NPU Driver ---"
if [ -f /sys/module/rknpu/version ]; then
    echo "RKNPU version: $(cat /sys/module/rknpu/version)"
else
    echo "ERROR: rknpu module not loaded!"
    echo "Try: sudo modprobe rknpu"
    exit 1
fi

if [ -c /dev/rknpu0 ]; then
    echo "/dev/rknpu0: OK"
else
    echo "ERROR: /dev/rknpu0 does not exist!"
    exit 1
fi

# Step 2: Install dependencies
echo ""
echo "--- Installing dependencies ---"
sudo apt update
sudo apt install -y python3-pip python3-venv git wget

# Step 3: Create workspace
WORKSPACE="$HOME/npu-research"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# Step 4: Set up Python venv
echo ""
echo "--- Setting up Python environment ---"
python3 -m venv venv
source venv/bin/activate

# Step 5: Clone and install RKNN Toolkit Lite 2
echo ""
echo "--- Installing RKNN Toolkit Lite 2 ---"
if [ ! -d rknn-toolkit2 ]; then
    git clone --depth 1 https://github.com/airockchip/rknn-toolkit2
fi

# Install the lite version (aarch64 on-device)
pip3 install ./rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp312-cp312-linux_aarch64.whl 2>/dev/null || \
pip3 install ./rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl 2>/dev/null || \
echo "WARNING: Could not find matching wheel. Check Python version: $(python3 --version)"

# Step 6: Install RKNPU2 runtime libraries
echo ""
echo "--- Installing RKNPU2 runtime ---"
if [ ! -d rknpu2 ]; then
    git clone --depth 1 https://github.com/rockchip-linux/rknpu2
fi

# Copy runtime libs
RKNPU2_LIB="rknpu2/runtime/RK3588/Linux/librknn_api/aarch64"
if [ -d "$RKNPU2_LIB" ]; then
    sudo cp "$RKNPU2_LIB"/librknnrt.so /usr/lib/
    sudo ldconfig
    echo "RKNPU2 runtime installed"
else
    echo "WARNING: RKNPU2 runtime libs not found at expected path"
fi

# Step 7: Clone RKNN-LLM (optional)
echo ""
echo "--- Cloning RKNN-LLM ---"
if [ ! -d rknn-llm ]; then
    git clone --depth 1 https://github.com/airockchip/rknn-llm
fi

# Step 8: Verify
echo ""
echo "=== Verification ==="
echo "NPU load: $(cat /proc/rknpu/load 2>/dev/null || echo 'not available')"
echo "RKNPU version: $(cat /sys/module/rknpu/version)"
ls -la /dev/rknpu*
echo ""
echo "Workspace: $WORKSPACE"
echo "To activate: source $WORKSPACE/venv/bin/activate"
echo ""
echo "Next steps:"
echo "  cd $WORKSPACE/rknn-toolkit2/rknn-toolkit-lite2/examples"
echo "  python3 resnet18/test.py  # Run example inference"
