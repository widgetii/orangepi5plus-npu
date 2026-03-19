#!/bin/bash
# In-place eMMC flash script for Orange Pi 5 Plus
# Run this ON the board via SSH
# WARNING: This will destroy all data on eMMC!
set -euo pipefail

IMAGE_URL="https://dl.armbian.com/orangepi5-plus/Noble_vendor_minimal"
EMMC_DEV=""  # Will be detected

echo "=== Orange Pi 5 Plus eMMC Flash Script ==="
echo "WARNING: This will ERASE ALL DATA on the eMMC!"
echo ""

# Detect eMMC device
if [ -b /dev/mmcblk1 ]; then
    EMMC_DEV="/dev/mmcblk1"
elif [ -b /dev/mmcblk0 ]; then
    EMMC_DEV="/dev/mmcblk0"
else
    echo "ERROR: No eMMC device found!"
    lsblk
    exit 1
fi

echo "Detected eMMC: $EMMC_DEV"
echo "Current block devices:"
lsblk
echo ""

# Check available RAM
TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
echo "Total RAM: ${TOTAL_RAM_MB}MB"
if [ "$TOTAL_RAM_MB" -lt 8000 ]; then
    echo "WARNING: Less than 8GB RAM. Image might not fit in tmpfs."
fi

# Confirm
read -p "Type 'YES' to proceed with flashing $EMMC_DEV: " CONFIRM
if [ "$CONFIRM" != "YES" ]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Mount tmpfs and download image
echo ""
echo "=== Step 1: Downloading image to RAM ==="
sudo mkdir -p /tmp/ramfs
sudo mount -t tmpfs -o size=4G tmpfs /tmp/ramfs
cd /tmp/ramfs
wget "$IMAGE_URL" -O armbian.img.xz
echo "Decompressing..."
xz -d armbian.img.xz

IMAGE_SIZE=$(stat -c %s armbian.img)
echo "Image size: $((IMAGE_SIZE / 1024 / 1024))MB"

# Step 2: Install screen if needed
echo ""
echo "=== Step 2: Preparing environment ==="
which screen >/dev/null 2>&1 || sudo apt install -y screen

# Step 3: Flash
echo ""
echo "=== Step 3: Flashing eMMC ==="
echo "POINT OF NO RETURN!"
echo "Run the following inside a screen session:"
echo ""
echo "  screen -S flash"
echo "  sudo dd if=/tmp/ramfs/armbian.img of=$EMMC_DEV bs=4M status=progress conv=fsync"
echo "  sudo sync"
echo "  sudo reboot"
echo ""
echo "Or to run automatically (will reboot when done):"
echo "  sudo nohup bash -c 'dd if=/tmp/ramfs/armbian.img of=$EMMC_DEV bs=4M conv=fsync && sync && reboot' > /tmp/ramfs/flash.log 2>&1 &"
echo "  tail -f /tmp/ramfs/flash.log"
