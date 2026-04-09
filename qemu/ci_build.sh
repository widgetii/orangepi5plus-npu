#!/bin/bash
# Build everything needed for QEMU NPU tests from source.
# Runs on ubuntu-24.04 CI runner with aarch64 cross-compiler.
#
# Produces:
#   qemu-src/build/qemu-system-aarch64  (QEMU binary)
#   qemu-boot/Image-6.18               (mainline kernel)
#   qemu-boot/Image-vendor              (vendor kernel)
#   qemu-boot/initrd.gz                 (Rocket initrd)
#   qemu-boot/initrd-vendor.gz          (RKNPU initrd)
#
# Prerequisites (installed by CI):
#   build-essential ninja-build libglib2.0-dev libfdt-dev libpixman-1-dev
#   gcc-aarch64-linux-gnu libdrm-dev wget cpio
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CROSS=aarch64-linux-gnu-
BOOT="$REPO_ROOT/qemu-boot"
ARMBIAN_REPO="http://apt.armbian.com"
ARMBIAN_SUITE="noble"

echo "=== Step 1: Build QEMU ==="
if [ -x qemu-src/build/qemu-system-aarch64 ]; then
    echo "QEMU binary found (cached), re-copying NPU sources..."
    # Re-copy our sources in case they changed (cache key should prevent this,
    # but setup.sh handles it gracefully)
    bash qemu/setup.sh 2>&1 | tail -5
else
    bash qemu/setup.sh
fi
echo "QEMU: $(qemu-src/build/qemu-system-aarch64 --version | head -1)"

echo ""
echo "=== Step 2: Download Armbian kernels ==="
mkdir -p "$BOOT"
KERN_DIR=$(mktemp -d)

# Download kernel packages from Armbian pool.
# Armbian pool structure: pool/main/l/linux-<version>/<package>_<version>_arm64__<hash>.deb
# We search the kernel version directory for the matching package name.
download_armbian_kernel() {
    local PKG="$1" KVER="$2" DESTDIR="$3"
    local POOL_URL="$ARMBIAN_REPO/pool/main/l/linux-${KVER}"
    local DEB_NAME
    DEB_NAME=$(wget -q -O- "$POOL_URL/" 2>/dev/null | \
               grep -oP "href=\"\./\K${PKG}_[^\"]+_arm64[^\"]*\.deb" | sort -V | tail -1)
    if [ -z "$DEB_NAME" ]; then
        echo "WARNING: Could not find $PKG in $POOL_URL"
        return 1
    fi
    echo "  Downloading $DEB_NAME..."
    wget -q "$POOL_URL/$DEB_NAME" -O "/tmp/$DEB_NAME"
    dpkg-deb -x "/tmp/$DEB_NAME" "$DESTDIR"
    rm -f "/tmp/$DEB_NAME"
}

if [ ! -f "$BOOT/Image-6.18" ]; then
    echo "Downloading mainline kernel..."
    if download_armbian_kernel "linux-image-current-rockchip64" "6.18.10" "$KERN_DIR/mainline"; then
        cp "$KERN_DIR/mainline/boot/vmlinuz-"* "$BOOT/Image-6.18"
        echo "Mainline kernel: $(ls -lh "$BOOT/Image-6.18" | awk '{print $5}')"
    else
        echo "ERROR: Failed to download mainline kernel"
        exit 1
    fi
fi

if [ ! -f "$BOOT/Image-vendor" ]; then
    echo "Downloading vendor kernel..."
    if download_armbian_kernel "linux-image-vendor-rk35xx" "6.1.115" "$KERN_DIR/vendor"; then
        cp "$KERN_DIR/vendor/boot/vmlinuz-"* "$BOOT/Image-vendor"
        echo "Vendor kernel: $(ls -lh "$BOOT/Image-vendor" | awk '{print $5}')"
    else
        echo "ERROR: Failed to download vendor kernel"
        exit 1
    fi
fi

# Extract kernel modules (Rocket driver from mainline)
if [ -d "$KERN_DIR/mainline" ]; then
    MODDIR=$(find "$KERN_DIR/mainline/lib/modules" -maxdepth 1 -mindepth 1 -type d | head -1)
    if [ -n "$MODDIR" ]; then
        echo "Extracting kernel modules from $MODDIR..."
        mkdir -p "$BOOT/modules"
        for mod in rocket.ko drm_shmem_helper.ko gpu-sched.ko; do
            found=$(find "$MODDIR" -name "$mod" | head -1)
            if [ -n "$found" ]; then
                cp "$found" "$BOOT/modules/"
                echo "  $mod: $(ls -lh "$BOOT/modules/$mod" | awk '{print $5}')"
            else
                echo "  WARNING: $mod not found"
            fi
        done
    fi
fi
rm -rf "$KERN_DIR"

echo ""
echo "=== Step 3: Download busybox ==="
if [ ! -f "$BOOT/busybox" ]; then
    echo "Extracting aarch64 busybox from Docker image..."
    # Use the official busybox Docker image (static, multi-arch)
    docker pull --platform linux/arm64 busybox:stable-musl >/dev/null 2>&1
    CID=$(docker create --platform linux/arm64 busybox:stable-musl)
    docker cp "$CID:/bin/busybox" "$BOOT/busybox"
    docker rm "$CID" >/dev/null
    chmod +x "$BOOT/busybox"
    echo "busybox: $(file "$BOOT/busybox" | grep -o 'ARM.*')"
fi

echo ""
echo "=== Step 4: Cross-compile test binaries ==="

# npu_test
${CROSS}gcc -static -O2 -o "$BOOT/npu_test" qemu/tests/npu_test.c -lm
echo "npu_test: $(file "$BOOT/npu_test" | grep -o 'ARM aarch64.*linked')"

# npu_conv_tests
${CROSS}gcc -static -O2 -o "$BOOT/npu_conv_tests" qemu/tests/npu_conv_tests.c -lm
echo "npu_conv_tests: $(file "$BOOT/npu_conv_tests" | grep -o 'ARM aarch64.*linked')"

# test_mobilenet — needs librocketnpu compiled as .a
echo "Building librocketnpu (aarch64 static)..."
LIBSRC="librocketnpu/src"
OBJDIR=$(mktemp -d)
for f in rnpu_drm.c rnpu_tflite.c rnpu_coefs.c rnpu_task.c rnpu_regcmd.c \
         rnpu_convert.c rnpu_sw_ops.c rnpu_rknn.c rnpu_model.c; do
    ${CROSS}gcc -O2 -Wall -Wno-unused-function \
        -I"$LIBSRC" -Ilibrocketnpu/include -march=armv8-a \
        -fPIC -c -o "$OBJDIR/$(basename "$f" .c).o" "$LIBSRC/$f"
done
${CROSS}ar rcs "$OBJDIR/librocketnpu.a" "$OBJDIR"/*.o
echo "librocketnpu.a: $(ls -lh "$OBJDIR/librocketnpu.a" | awk '{print $5}')"

${CROSS}gcc -static -O2 -Ilibrocketnpu/include -Ilibrocketnpu/src \
    -o "$BOOT/test_mobilenet" librocketnpu/tests/test_mobilenet.c \
    "$OBJDIR/librocketnpu.a" -lm
echo "test_mobilenet: $(file "$BOOT/test_mobilenet" | grep -o 'ARM aarch64.*linked')"

# test_fc
${CROSS}gcc -static -O2 -Ilibrocketnpu/include -Ilibrocketnpu/src \
    -o "$BOOT/test_fc" librocketnpu/tests/test_fc.c \
    "$OBJDIR/librocketnpu.a" -lm
echo "test_fc: $(file "$BOOT/test_fc" | grep -o 'ARM aarch64.*linked')"
rm -rf "$OBJDIR"

echo ""
echo "=== Step 5: Pack initrds ==="

# Download MobileNetV1 INT8 model (public TFLite model)
MODEL="$BOOT/mobilenet_v1.tflite"
if [ ! -f "$MODEL" ]; then
    echo "Downloading MobileNetV1 INT8 model..."
    wget -q "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz" \
         -O /tmp/mobilenet.tgz
    tar xzf /tmp/mobilenet.tgz -C /tmp ./mobilenet_v1_1.0_224_quant.tflite
    mv /tmp/mobilenet_v1_1.0_224_quant.tflite "$MODEL"
    rm /tmp/mobilenet.tgz
    echo "Model: $(ls -lh "$MODEL" | awk '{print $5}')"
fi
GOLDEN="$BOOT/golden_mobilenet_output.bin"

# Pack Rocket initrd
echo "Packing Rocket initrd..."
ROOTFS=$(mktemp -d)
mkdir -p "$ROOTFS"/{bin,lib/modules,dev,proc,sys,tmp}

# Busybox + symlinks
cp "$BOOT/busybox" "$ROOTFS/bin/busybox"
for cmd in sh cat ls mount insmod sleep ln dmesg mknod; do
    ln -sf busybox "$ROOTFS/bin/$cmd"
done

# Test binaries
cp "$BOOT/npu_test" "$ROOTFS/bin/"
cp "$BOOT/npu_conv_tests" "$ROOTFS/bin/"
cp "$BOOT/test_mobilenet" "$ROOTFS/bin/"
cp "$BOOT/test_fc" "$ROOTFS/bin/"

# Kernel modules
if [ -d "$BOOT/modules" ]; then
    cp "$BOOT/modules/"*.ko "$ROOTFS/lib/modules/" 2>/dev/null || true
fi

# Model + golden + Grace Hopper input
if [ -f "$MODEL" ]; then cp "$MODEL" "$ROOTFS/model.tflite"; fi
if [ -f "$GOLDEN" ]; then cp "$GOLDEN" "$ROOTFS/golden.bin"; fi
if [ -f "$BOOT/grace_hopper_224.bin" ]; then cp "$BOOT/grace_hopper_224.bin" "$ROOTFS/input.bin"; fi

# FC model + test data
FC_DIR="$REPO_ROOT/librocketnpu/tests/models"
if [ -f "$FC_DIR/fc_model_int8.tflite" ]; then
    cp "$FC_DIR/fc_model_int8.tflite" "$ROOTFS/fc_model.tflite"
    cp "$FC_DIR/fc_test_input.bin" "$ROOTFS/fc_input.bin"
    cp "$FC_DIR/fc_cpu_golden.bin" "$ROOTFS/fc_golden.bin"
fi

# Init script (from git)
cp "$BOOT/rootfs/init" "$ROOTFS/init"
chmod +x "$ROOTFS/init"

(cd "$ROOTFS" && find . | cpio -o -H newc 2>/dev/null | gzip > "$BOOT/initrd.gz")
echo "initrd.gz: $(ls -lh "$BOOT/initrd.gz" | awk '{print $5}')"

# Pack vendor initrd
echo "Packing vendor initrd..."
ROOTFS_V=$(mktemp -d)
mkdir -p "$ROOTFS_V"/{bin,lib/modules,dev,proc,sys,tmp}

cp "$BOOT/busybox" "$ROOTFS_V/bin/busybox"
for cmd in sh cat ls mount insmod sleep ln dmesg mknod grep head; do
    ln -sf busybox "$ROOTFS_V/bin/$cmd"
done

cp "$BOOT/npu_conv_tests" "$ROOTFS_V/bin/" 2>/dev/null || true
cp "$BOOT/test_mobilenet" "$ROOTFS_V/bin/" 2>/dev/null || true
cp "$BOOT/test_fc" "$ROOTFS_V/bin/" 2>/dev/null || true
if [ -f "$MODEL" ]; then cp "$MODEL" "$ROOTFS_V/model.tflite"; fi
if [ -f "$GOLDEN" ]; then cp "$GOLDEN" "$ROOTFS_V/golden.bin"; fi
if [ -f "$BOOT/grace_hopper_224.bin" ]; then cp "$BOOT/grace_hopper_224.bin" "$ROOTFS_V/input.bin"; fi
if [ -f "$FC_DIR/fc_model_int8.tflite" ]; then
    cp "$FC_DIR/fc_model_int8.tflite" "$ROOTFS_V/fc_model.tflite"
    cp "$FC_DIR/fc_test_input.bin" "$ROOTFS_V/fc_input.bin"
    cp "$FC_DIR/fc_cpu_golden.bin" "$ROOTFS_V/fc_golden.bin"
fi
cp "$BOOT/rootfs-vendor/init" "$ROOTFS_V/init"
chmod +x "$ROOTFS_V/init"

(cd "$ROOTFS_V" && find . | cpio -o -H newc 2>/dev/null | gzip > "$BOOT/initrd-vendor.gz")
echo "initrd-vendor.gz: $(ls -lh "$BOOT/initrd-vendor.gz" | awk '{print $5}')"

rm -rf "$ROOTFS" "$ROOTFS_V"

echo ""
echo "=== Build complete ==="
echo "QEMU:    $(ls -lh qemu-src/build/qemu-system-aarch64 | awk '{print $5}')"
echo "Kernels: $(ls -lh "$BOOT/Image-6.18" "$BOOT/Image-vendor" 2>/dev/null | awk '{print $NF, $5}')"
echo "Initrds: $(ls -lh "$BOOT/initrd.gz" "$BOOT/initrd-vendor.gz" | awk '{print $NF, $5}')"
