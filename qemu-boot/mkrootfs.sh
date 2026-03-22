#!/bin/bash
#
# Prepare an Armbian disk image for QEMU booting.
#
# Downloads the Armbian Noble minimal image, installs mainline 6.18 kernel
# modules (the stock image ships only vendor 6.1, which lacks virtio), and
# produces a ready-to-boot full disk image. The QEMU machine's built-in CRU
# reset controller eliminates the need for any custom kernel modules.
#
# Usage:
#   bash qemu-boot/mkrootfs.sh [output.img]
#
# Requirements: curl, xz, fdisk, qemu-aarch64-static (optional, for depmod)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${1:-$SCRIPT_DIR/armbian.img}"
WORK="/tmp/mkrootfs-$$"
MNT="$WORK/mnt"

ARMBIAN_URL="https://dl.armbian.com/orangepi5-plus/Noble_vendor_minimal"
KERNEL_DEB_URL="https://apt.armbian.com/pool/main/l/linux-6.18.10/linux-image-current-rockchip64_26.2.1_arm64__6.18.10-S41ce-D1d59-P992d-C2cce-H050e-HK01ba-Vc222-B052a-R448a.deb"
KERNEL_VER="6.18.10-current-rockchip64"

cleanup() { sudo umount "$MNT" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

mkdir -p "$WORK" "$MNT"

# ── 1. Download Armbian image ────────────────────────────────────────────
ARMBIAN_XZ="$WORK/armbian.img.xz"
if [ -f "$OUTPUT" ]; then
    echo "Output image already exists, skipping download"
    ARMBIAN_IMG="$OUTPUT"
else
    echo "Downloading Armbian Noble minimal..."
    curl -L -o "$ARMBIAN_XZ" "$ARMBIAN_URL"
    echo "Decompressing..."
    xz -d -c "$ARMBIAN_XZ" > "$OUTPUT"
    ARMBIAN_IMG="$OUTPUT"
fi

# ── 2. Install mainline kernel modules ───────────────────────────────────
# The stock Armbian image ships vendor kernel 6.1 which lacks virtio-blk
# and virtio-net drivers. Install mainline 6.18 modules so the kernel
# (passed via -kernel) can find its modules.
echo "Downloading kernel modules..."
DEB="$WORK/linux-image.deb"
curl -sL -o "$DEB" "$KERNEL_DEB_URL"

echo "Extracting modules..."
mkdir -p "$WORK/deb" && cd "$WORK/deb"
ar x "$DEB"

# Mount the rootfs partition (first partition in the GPT image)
PART_INFO=$(fdisk -l "$OUTPUT" 2>/dev/null | grep "${OUTPUT}1\|\.img1")
START=$(echo "$PART_INFO" | awk '{print $2}')
sudo mount -o loop,offset=$((START * 512)) "$OUTPUT" "$MNT"

sudo tar xf data.tar.xz -C "$MNT" ./lib/modules/"$KERNEL_VER"/

# Run depmod (needs qemu-aarch64-static for cross-arch chroot)
if command -v qemu-aarch64-static >/dev/null 2>&1; then
    sudo cp "$(command -v qemu-aarch64-static)" "$MNT/usr/bin/"
    sudo chroot "$MNT" /usr/sbin/depmod "$KERNEL_VER" 2>/dev/null
    sudo rm -f "$MNT/usr/bin/qemu-aarch64-static"
else
    echo "  WARNING: qemu-aarch64-static not found, skipping depmod"
    echo "  Modules will still work but modprobe may not auto-resolve deps"
fi

sudo umount "$MNT"

echo ""
echo "Done! Boot with:"
echo "  qemu-system-aarch64 -M orangepi5plus -m 2G -nographic -smp 4 \\"
echo "    -kernel $SCRIPT_DIR/Image-6.18 \\"
echo "    -append 'console=ttyS0,1500000 earlycon panic=10 root=LABEL=armbi_root rw' \\"
echo "    -drive file=$OUTPUT,format=raw,if=none,id=hd0 \\"
echo "    -device virtio-blk-device,drive=hd0 \\"
echo "    -netdev user,id=net0 -device virtio-net-device,netdev=net0"
