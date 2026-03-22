#!/bin/bash
#
# Build a QEMU-bootable rootfs image from the official Armbian image.
#
# Usage:
#   bash qemu-boot/mkrootfs.sh [output.img]
#
# The script downloads the Armbian Noble minimal image, extracts the rootfs
# partition, installs mainline 6.18 kernel modules + QEMU helper modules,
# fixes fstab, and produces a ready-to-boot ext4 image.
#
# Requirements: curl, xz, fdisk, e2fsck, resize2fs, qemu-aarch64-static
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${1:-$SCRIPT_DIR/armbian_rootfs.img}"
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
ARMBIAN_IMG="$WORK/armbian.img"
if [ -f "$ARMBIAN_IMG" ]; then
    echo "Armbian image already exists, skipping download"
else
    echo "Downloading Armbian Noble minimal..."
    curl -L -o "$ARMBIAN_XZ" "$ARMBIAN_URL"
    echo "Decompressing..."
    xz -d "$ARMBIAN_XZ"
fi

# ── 2. Extract rootfs partition ──────────────────────────────────────────
echo "Extracting rootfs partition..."
# Parse partition offset from fdisk
PART_INFO=$(fdisk -l "$ARMBIAN_IMG" 2>/dev/null | grep '\.img1')
START=$(echo "$PART_INFO" | awk '{print $2}')
COUNT=$(echo "$PART_INFO" | awk '{print $4}')

dd if="$ARMBIAN_IMG" of="$OUTPUT" bs=512 skip="$START" count="$COUNT" \
    status=progress 2>&1
rm -f "$ARMBIAN_IMG"

# Resize to 2 GB
truncate -s 2G "$OUTPUT"
e2fsck -f -y "$OUTPUT" >/dev/null 2>&1 || true
resize2fs "$OUTPUT" >/dev/null 2>&1
echo "Rootfs image: $(ls -lh "$OUTPUT" | awk '{print $5}')"

# ── 3. Install mainline kernel modules ───────────────────────────────────
echo "Downloading kernel modules..."
DEB="$WORK/linux-image.deb"
curl -sL -o "$DEB" "$KERNEL_DEB_URL"

echo "Extracting modules..."
mkdir -p "$WORK/deb" && cd "$WORK/deb"
ar x "$DEB"

sudo mount -o loop "$OUTPUT" "$MNT"
sudo tar xf data.tar.xz -C "$MNT" ./lib/modules/"$KERNEL_VER"/

# ── 4. Install QEMU helper modules ──────────────────────────────────────
echo "Installing QEMU helper modules..."
MODDIR="$MNT/lib/modules/$KERNEL_VER"
sudo cp "$SCRIPT_DIR/qemu_reset.ko" "$MODDIR/kernel/drivers/"
sudo cp "$SCRIPT_DIR/qemu_iommu.ko" "$MODDIR/kernel/drivers/"

# Run depmod (needs qemu-aarch64-static for cross-arch chroot)
if command -v qemu-aarch64-static >/dev/null 2>&1; then
    sudo cp "$(command -v qemu-aarch64-static)" "$MNT/usr/bin/"
    sudo chroot "$MNT" /usr/sbin/depmod "$KERNEL_VER" 2>/dev/null
    sudo rm -f "$MNT/usr/bin/qemu-aarch64-static"
else
    echo "  WARNING: qemu-aarch64-static not found, skipping depmod"
    echo "  Modules will still work but modprobe may not auto-resolve deps"
fi

sudo mkdir -p "$MNT/etc/modules-load.d"
echo -e "qemu_reset\nqemu_iommu" | sudo tee "$MNT/etc/modules-load.d/qemu.conf" >/dev/null

# ── 5. Fix fstab and serial console ─────────────────────────────────────
echo "Configuring fstab and serial console..."
printf '/dev/vda / ext4 defaults,noatime 0 1\ntmpfs /tmp tmpfs defaults,nosuid 0 0\n' \
    | sudo tee "$MNT/etc/fstab" >/dev/null

# Enable serial console on ttyS0 (QEMU UART, not ttyFIQ0)
sudo mkdir -p "$MNT/etc/systemd/system/getty.target.wants"
sudo ln -sf /lib/systemd/system/serial-getty@.service \
    "$MNT/etc/systemd/system/getty.target.wants/serial-getty@ttyS0.service"

# Ensure systemd-networkd is enabled for virtio-net DHCP
sudo mkdir -p "$MNT/etc/systemd/network"
printf '[Match]\nName=eth* en*\n\n[Network]\nDHCP=yes\n' \
    | sudo tee "$MNT/etc/systemd/network/20-wired.network" >/dev/null

sudo umount "$MNT"

echo ""
echo "Done! Boot with:"
echo "  qemu-system-aarch64 -M orangepi5plus -m 2G -nographic -smp 4 \\"
echo "    -kernel $SCRIPT_DIR/Image-6.18 \\"
echo "    -append 'console=ttyS0,1500000 earlycon panic=10 root=/dev/vda rw' \\"
echo "    -drive file=$OUTPUT,format=raw,if=none,id=hd0 \\"
echo "    -device virtio-blk-device,drive=hd0 \\"
echo "    -netdev user,id=net0 -device virtio-net-device,netdev=net0"
