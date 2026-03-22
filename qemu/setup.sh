#!/bin/bash
#
# Set up QEMU build tree with RK3588 NPU emulator support.
# Run from the repository root:
#
#   bash qemu/setup.sh
#
# This will:
#   1. Clone QEMU 10.2.0 into qemu-src/
#   2. Copy all NPU/IOMMU/machine source files
#   3. Patch Kconfig and meson.build
#   4. Configure and build (aarch64 target only)
#
set -e

QEMU_VERSION="v10.2.0"
QEMU_DIR="qemu-src"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# ── 1. Clone QEMU ──────────────────────────────────────────────────────
if [ -d "$QEMU_DIR" ]; then
    echo "qemu-src/ already exists, skipping clone"
else
    echo "Cloning QEMU $QEMU_VERSION..."
    git clone --depth 1 --branch "$QEMU_VERSION" \
        https://gitlab.com/qemu-project/qemu.git "$QEMU_DIR"
fi

# ── 2. Copy source files ───────────────────────────────────────────────
echo "Copying NPU emulator sources..."

cp qemu/hw/arm/rk3588.c            "$QEMU_DIR/hw/arm/"
cp qemu/include/hw/arm/rk3588.h    "$QEMU_DIR/include/hw/arm/"

for f in rockchip-npu.c rockchip-npu.h rockchip-iommu.c rockchip-iommu.h; do
    cp "qemu/hw/misc/$f" "$QEMU_DIR/hw/misc/"
done

# QEMU resolves headers from both hw/misc/ and include/hw/misc/
cp qemu/hw/misc/rockchip-npu.h     "$QEMU_DIR/include/hw/misc/"
cp qemu/hw/misc/rockchip-iommu.h   "$QEMU_DIR/include/hw/misc/"

# ── 3. Patch build system ──────────────────────────────────────────────
echo "Patching build system..."

# hw/arm/Kconfig
if ! grep -q ORANGEPI5PLUS "$QEMU_DIR/hw/arm/Kconfig"; then
    cat >> "$QEMU_DIR/hw/arm/Kconfig" <<'KCONFIG'

config ORANGEPI5PLUS
    bool
    default y
    depends on TCG && AARCH64
    select ARM_GICV3
    select SERIAL
    select UNIMP
    select ROCKCHIP_NPU
KCONFIG
    echo "  patched hw/arm/Kconfig"
else
    echo "  hw/arm/Kconfig already patched"
fi

# hw/arm/meson.build
if ! grep -q rk3588 "$QEMU_DIR/hw/arm/meson.build"; then
    echo "arm_ss.add(when: 'CONFIG_ORANGEPI5PLUS', if_true: files('rk3588.c'))" \
        >> "$QEMU_DIR/hw/arm/meson.build"
    echo "  patched hw/arm/meson.build"
else
    echo "  hw/arm/meson.build already patched"
fi

# hw/misc/Kconfig
if ! grep -q ROCKCHIP_NPU "$QEMU_DIR/hw/misc/Kconfig"; then
    cat >> "$QEMU_DIR/hw/misc/Kconfig" <<'KCONFIG'

config ROCKCHIP_NPU
    bool
KCONFIG
    echo "  patched hw/misc/Kconfig"
else
    echo "  hw/misc/Kconfig already patched"
fi

# hw/misc/meson.build
if ! grep -q rockchip-npu "$QEMU_DIR/hw/misc/meson.build"; then
    echo "system_ss.add(when: 'CONFIG_ROCKCHIP_NPU', if_true: files('rockchip-npu.c', 'rockchip-iommu.c'))" \
        >> "$QEMU_DIR/hw/misc/meson.build"
    echo "  patched hw/misc/meson.build"
else
    echo "  hw/misc/meson.build already patched"
fi

# ── 4. Build ────────────────────────────────────────────────────────────
echo "Building QEMU (aarch64 target)..."
cd "$QEMU_DIR"
mkdir -p build && cd build
../configure --target-list=aarch64-softmmu --disable-docs
ninja -j$(nproc)

echo ""
echo "Done! Verify:"
echo "  ./qemu-src/build/qemu-system-aarch64 -machine help | grep orangepi"
