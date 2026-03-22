# QEMU RK3588 NPU Emulator — Boot & Test Guide

This directory contains everything needed to run the RK3588 NPU emulator in
QEMU and test INT8 convolution models on both the mainline Rocket driver and
the vendor RKNPU driver.

## Building QEMU

A setup script handles everything — cloning QEMU, copying source files,
patching the build system, and compiling. Run it once from the repo root:

```sh
bash qemu/setup.sh
```

This takes a few minutes. When it finishes, verify:

```sh
./qemu-src/build/qemu-system-aarch64 -machine help | grep orangepi
# Expected: orangepi5plus   Rockchip RK3588 (Orange Pi 5 Plus) with NPU
```

The script is safe to re-run — it skips steps that are already done.

> **What it does under the hood:** clones QEMU v10.2.0 into `qemu-src/`,
> copies the NPU/IOMMU/machine model files from `qemu/`, appends Kconfig
> and meson.build entries, then runs `configure` + `ninja`. See
> `qemu/setup.sh` for details.

## Cross-Compiling Test Binaries

Requires `aarch64-linux-gnu-gcc` and libdrm headers
(`libdrm-dev:arm64` or equivalent). Run from the repo root:

```sh
# npu_conv_tests (standalone, no dependencies)
aarch64-linux-gnu-gcc -static -O2 -o qemu-boot/npu_conv_tests \
  qemu/tests/npu_conv_tests.c

# test_mobilenet (librocketnpu, supports both Rocket and RKNPU)
aarch64-linux-gnu-gcc -static -O2 -Wall -Wno-unused-function \
  -I/usr/aarch64-linux-gnu/include/drm \
  -Ilibrocketnpu/include -Ilibrocketnpu/src \
  -o qemu-boot/test_mobilenet \
  librocketnpu/tests/test_mobilenet.c \
  librocketnpu/src/rnpu_drm.c librocketnpu/src/rnpu_tflite.c \
  librocketnpu/src/rnpu_coefs.c librocketnpu/src/rnpu_task.c \
  librocketnpu/src/rnpu_regcmd.c librocketnpu/src/rnpu_convert.c \
  librocketnpu/src/rnpu_sw_ops.c librocketnpu/src/rnpu_model.c -lm
```

After rebuilding test binaries, copy them into the rootfs trees and
regenerate the initrds (see "Rebuilding Initrds" below).

## Running — Minimal Initrd (NPU Tests Only)

```sh
qemu-system-aarch64 \
  -M orangepi5plus -m 2G -nographic -smp 4 \
  -kernel Image-6.18 \
  -initrd initrd.gz \
  -append "console=ttyS0,1500000 earlycon panic=10"
```

The init script loads the Rocket kernel module, waits for `/dev/accel/accel0`,
then runs three tests in order:

1. **npu_test** — single identity convolution sanity check
2. **npu_conv_tests** — 6 parameterised convolution tests (stride, padding,
   depthwise, DMA bias, BN+ReLU)
3. **test_mobilenet** — full MobileNetV1 INT8 inference with golden comparison

Expected output:
```
PASS: Identity conv with NVDLA truncation correct
=== CONV TESTS: 6/6 passed ===
RESULT: PASS (bit-exact)
```

A few `NPU job timed out` messages during MobileNetV1 are normal — the
deferred-IRQ timer in the QEMU NPU model fires asynchronously with TCG
timeslicing, so the DRM scheduler occasionally retries. Results are correct.

## Running — Vendor Kernel (RKNPU Driver)

The vendor kernel requires more memory and an explicit CMA reservation:

```sh
qemu-system-aarch64 \
  -M orangepi5plus -m 3G -nographic -smp 4 \
  -kernel Image-vendor \
  -initrd initrd-vendor.gz \
  -dtb vendor.dtb \
  -append "console=ttyS0,1500000 earlycon panic=10 cma=64M"
```

**Important constraints:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `-m` | 2G-4G+ | Any size works. RAM >3838 MiB is capped at the MMIO boundary with a warning |
| `cma=64M` | Required | Default 16MB CMA is too small; the RKNPU driver allocates ~8MB contiguous for the activation buffer |
| `-dtb vendor.dtb` | Required | Vendor kernel needs the single-node `rockchip,rk3588-rknpu` binding |

The RKNPU driver probe takes ~15 seconds (deferred probe). The init script
waits, then runs:

1. **rknpu_test** — driver probe check, HW version query, memory allocation
2. **npu_conv_tests** — same 6 tests, auto-detects RKNPU driver
3. **test_mobilenet** — MobileNetV1 via librocketnpu's RKNPU backend

Expected output:
```
=== RKNPU TESTS: 3/4 passed ===
=== CONV TESTS: 6/6 passed ===
RESULT: PASS (bit-exact)
```

The `test_dmesg` sub-test reports PARTIAL (dmesg format parsing), which is
harmless.

## Running — Full Armbian Rootfs

Boots the real Armbian 25.11.1 Noble image with networking, package
management, and all Armbian services (zram, ramlog, hw-monitor, etc.):

```sh
qemu-system-aarch64 \
  -M orangepi5plus -m 2G -nographic -smp 4 \
  -kernel Image-6.18 \
  -append "console=ttyS0,1500000 earlycon panic=10 root=/dev/vda rw" \
  -drive file=armbian_rootfs.img,format=raw,if=none,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -netdev user,id=net0 -device virtio-net-device,netdev=net0
```

### Preparing the rootfs image

1. Download Armbian Noble minimal for Orange Pi 5 Plus:
   ```sh
   curl -L -o armbian.img.xz \
     "https://dl.armbian.com/orangepi5-plus/Noble_vendor_minimal"
   xz -d armbian.img.xz
   ```

2. Extract the rootfs partition to a standalone image:
   ```sh
   # Find partition offset (typically 32768 sectors × 512 bytes)
   fdisk -l armbian.img
   dd if=armbian.img of=armbian_rootfs.img bs=512 skip=32768 count=3235807
   truncate -s 2G armbian_rootfs.img
   e2fsck -f armbian_rootfs.img && resize2fs armbian_rootfs.img
   ```

3. Install mainline kernel modules (the stock image ships vendor 6.1 only):
   ```sh
   # Download Armbian kernel deb
   curl -L -o linux-image.deb \
     "https://apt.armbian.com/pool/main/l/linux-6.18.10/linux-image-current-rockchip64_26.2.1_arm64.deb"
   mkdir deb && cd deb && ar x ../linux-image.deb
   sudo mount -o loop armbian_rootfs.img /mnt
   sudo tar xf data.tar.xz -C /mnt ./lib/modules/6.18.10-current-rockchip64/
   ```

4. Install QEMU helper modules and configure:
   ```sh
   MODDIR="/mnt/lib/modules/6.18.10-current-rockchip64"
   sudo cp qemu_reset.ko qemu_iommu.ko "$MODDIR/kernel/drivers/"
   sudo chroot /mnt depmod 6.18.10-current-rockchip64
   echo -e "qemu_reset\nqemu_iommu" | sudo tee /mnt/etc/modules-load.d/qemu.conf
   ```

5. Fix fstab (stock image uses UUID that won't match):
   ```sh
   echo "/dev/vda / ext4 defaults,noatime 0 1" | sudo tee /mnt/etc/fstab
   echo "tmpfs /tmp tmpfs defaults,nosuid 0 0" | sudo tee -a /mnt/etc/fstab
   sudo umount /mnt
   ```

### What works

- **Full Armbian**: systemd, zram, ramlog, hw-monitor, resize-fs, firstrun
- **Networking**: `eth0` via virtio-net, DHCP at `10.0.2.x`, full internet
  (`wget https://example.com` works)
- **NPU**: All 3 Rocket cores probe, `/dev/accel/accel0` available
- **Root disk**: virtio-blk at `/dev/vda`
- **0 failed systemd units** (only `serial-getty@ttyFIQ0` dep fails — expected,
  no Rockchip FIQ debugger UART in QEMU)
- Armbian auto-login on serial console, first-run password wizard works

## File Inventory

| File | Description |
|------|-------------|
| `Image-6.18` | Mainline Linux 6.18 kernel (aarch64) |
| `Image-vendor` | Vendor kernel 6.1.115-vendor-rk35xx |
| `vendor.dtb` | Device tree for vendor kernel (single rknpu node) |
| `initrd.gz` | Rocket initrd (busybox + modules + tests + model) |
| `initrd-vendor.gz` | RKNPU initrd (busybox + tests + model) |
| `model.tflite` | MobileNetV1 INT8 quantised TFLite model |
| `golden.bin` | Expected softmax output (1001 bytes, all-zero for uniform input) |
| `rootfs/` | Rocket initrd source tree |
| `rootfs-vendor/` | Vendor initrd source tree |

## Rebuilding Initrds

After modifying files in `rootfs/` or `rootfs-vendor/`:

```sh
# Rocket
cd rootfs && find . | cpio -o -H newc 2>/dev/null | gzip > ../initrd.gz

# Vendor
cd rootfs-vendor && find . | cpio -o -H newc 2>/dev/null | gzip > ../initrd-vendor.gz
```

## Capturing Serial Output to a File

For CI or scripted runs, use `-serial file:/tmp/output.log` instead of
`-nographic`:

```sh
qemu-system-aarch64 \
  -M orangepi5plus -m 2G -nographic -smp 4 \
  -kernel Image-6.18 -initrd initrd.gz \
  -append "console=ttyS0,1500000 earlycon panic=10" \
  -serial file:/tmp/rocket_test.log
```

Then grep for results:
```sh
grep -E "CONV TESTS|RESULT|exit code" /tmp/rocket_test.log
```

## Architecture

The QEMU machine (`orangepi5plus`) emulates:

- 4× Cortex-A55 (EL3 enabled for vendor kernel SMC calls)
- GICv3 interrupt controller
- DW UART2 at 0xfeb50000
- CRU/PMU stubs (PLL lock bits)
- GRF/IOC stubs (for vendor kernel built-in drivers)
- 3× NPU cores at 0xfdab0000/0xfdac0000/0xfdad0000
- Rockchip IOMMU v2 (page-table walk)
- 8× virtio-mmio transports (for virtio-blk root disk and virtio-net)

The NPU model executes INT8 convolutions in software (C loops) inside the
MMIO write handler for `PC_OPERATION_ENABLE`, then defers the completion IRQ
via a 1ms QEMU virtual timer. This lets the DRM scheduler's ioctl return
path complete before the fence signals, avoiding a race condition with batch
job submissions.

Both the Rocket (upstream DRM/accel) and RKNPU (vendor DRM) drivers program
the same hardware registers — the emulator handles both transparently.

## Networking

Networking uses **virtio-net** via the virtio-mmio transports. Add
`-netdev user,id=net0 -device virtio-net-device,netdev=net0` for NAT
networking with DHCP (guest gets `10.0.2.x`).

The real RK3588 GMAC (Synopsys DWMAC 4.20a) is not emulated — QEMU's
NPCM GMAC model implements DWMAC3 registers which are incompatible with
the DWMAC4 DMA layout that the Linux `stmmac` driver expects. Virtio-net
is the practical solution and works reliably.

## Known Issues

- **RAM capped at ~3838 MiB** — the mainline `rockchip-iommu` driver
  allocates its page table directory with `GFP_KERNEL`, which can land
  above 4GB when a Normal zone exists. The 32-bit `DTE_ADDR` register
  truncates the address, breaking all IOMMU translations. Any `-m` value
  is accepted but silently capped. See
  [`bugs/rockchip-iommu-gfp-kernel.md`](../bugs/rockchip-iommu-gfp-kernel.md)
  for the full analysis and proposed kernel fix.
