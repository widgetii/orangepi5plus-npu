#!/bin/bash
# Phase 4: NPU research and probing tools
set -euo pipefail

echo "=== RK3588 NPU Deep Probe ==="

# NPU MMIO base addresses (from devicetree)
# Core 0: 0xfdab0000
# Core 1: 0xfdac0000
# Core 2: 0xfdad0000

echo ""
echo "--- Kernel module info ---"
modinfo rknpu 2>/dev/null || echo "rknpu module info not available"

echo ""
echo "--- NPU device tree ---"
if [ -d /proc/device-tree/npu@fdab0000 ]; then
    echo "NPU node found in device tree"
    ls /proc/device-tree/npu@fdab0000/ 2>/dev/null
else
    echo "NPU device tree node not found at expected path"
    # Try to find it
    find /proc/device-tree -name "*npu*" -o -name "*rknpu*" 2>/dev/null || true
fi

echo ""
echo "--- NPU status ---"
cat /sys/module/rknpu/version 2>/dev/null && echo "" || echo "rknpu not loaded"
cat /proc/rknpu/load 2>/dev/null || echo "/proc/rknpu/load not available"

echo ""
echo "--- IOMMU groups ---"
for g in /sys/kernel/iommu_groups/*/devices/*; do
    if [ -e "$g" ]; then
        dev=$(basename "$g")
        case "$dev" in *fdab* | *fdac* | *fdad*)
            echo "NPU IOMMU: $g"
            ;;
        esac
    fi
done

echo ""
echo "--- Power domain ---"
if [ -d /sys/class/devfreq ]; then
    for d in /sys/class/devfreq/*npu* /sys/class/devfreq/*fdab*; do
        if [ -e "$d" ]; then
            echo "NPU devfreq: $d"
            cat "$d/cur_freq" 2>/dev/null && echo " Hz"
            echo "Available: $(cat "$d/available_frequencies" 2>/dev/null)"
        fi
    done
fi

echo ""
echo "--- To trace IOCTL calls ---"
echo "Install strace and run:"
echo "  strace -e trace=ioctl -f <npu_program> 2>&1 | tee ioctl_trace.log"
echo ""
echo "--- To probe MMIO registers ---"
echo "Install devmem2 and run:"
echo "  sudo devmem2 0xfdab0000 w  # NPU core 0 base"
echo "  sudo devmem2 0xfdac0000 w  # NPU core 1 base"
echo "  sudo devmem2 0xfdad0000 w  # NPU core 2 base"
