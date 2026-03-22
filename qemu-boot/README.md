# QEMU RK3588 NPU Emulator — Boot & Test Guide

This directory contains everything needed to run the RK3588 NPU emulator in
QEMU and test INT8 convolution models on both the mainline Rocket driver and
the vendor RKNPU driver.

## Building QEMU

The NPU emulator is a set of source files in `qemu/` that plug into a
vanilla QEMU 10.x tree. You need to clone QEMU, copy our files in, patch
two Kconfig/meson.build files, and build.

### 1. Clone QEMU

```sh
git clone --depth 1 --branch v10.2.0 https://gitlab.com/qemu-project/qemu.git qemu-src
cd qemu-src
```

Any QEMU 10.x release should work (tested with 10.2.0).

### 2. Copy NPU source files

From the repo root:

```sh
# Machine definition
cp qemu/hw/arm/rk3588.c          qemu-src/hw/arm/
cp qemu/include/hw/arm/rk3588.h  qemu-src/include/hw/arm/

# NPU + IOMMU device models
cp qemu/hw/misc/rockchip-npu.c   qemu-src/hw/misc/
cp qemu/hw/misc/rockchip-npu.h   qemu-src/hw/misc/
cp qemu/hw/misc/rockchip-iommu.c qemu-src/hw/misc/
cp qemu/hw/misc/rockchip-iommu.h qemu-src/hw/misc/

# Headers go to both locations (QEMU includes from both)
cp qemu/hw/misc/rockchip-npu.h   qemu-src/include/hw/misc/
cp qemu/hw/misc/rockchip-iommu.h qemu-src/include/hw/misc/
```

### 3. Patch build system

Add our Kconfig entries and meson.build lines. The fragment files
(`Kconfig.rk3588`, `meson.build.rk3588`, etc.) show exactly what to add.

**hw/arm/Kconfig** — append:
```
config ORANGEPI5PLUS
    bool
    default y
    depends on TCG && AARCH64
    select ARM_GICV3
    select SERIAL
    select UNIMP
    select ROCKCHIP_NPU
```

**hw/arm/meson.build** — add to the `arm_ss.add(...)` block:
```meson
arm_ss.add(when: 'CONFIG_ORANGEPI5PLUS', if_true: files('rk3588.c'))
```

**hw/misc/Kconfig** — append:
```
config ROCKCHIP_NPU
    bool
```

**hw/misc/meson.build** — add to the `system_ss.add(...)` block:
```meson
system_ss.add(when: 'CONFIG_ROCKCHIP_NPU', if_true: files('rockchip-npu.c', 'rockchip-iommu.c'))
```

### 4. Configure and build

```sh
cd qemu-src
mkdir -p build && cd build
../configure --target-list=aarch64-softmmu --disable-docs
ninja -j$(nproc)
```

The binary is `qemu-src/build/qemu-system-aarch64`. Verify the machine is
registered:

```sh
./qemu-system-aarch64 -machine help | grep orangepi
# Expected: orangepi5plus   Rockchip RK3588 (Orange Pi 5 Plus) with NPU
```

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

## Running — Mainline Kernel (Rocket Driver)

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
| `-m` | 3G (max ~3.9G) | RAM >4GB places a fragment at 0x100000000 which crashes vendor kernel early boot (IPI oops) |
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

The NPU model executes INT8 convolutions in software (C loops) inside the
MMIO write handler for `PC_OPERATION_ENABLE`, then defers the completion IRQ
via a 1ms QEMU virtual timer. This lets the DRM scheduler's ioctl return
path complete before the fence signals, avoiding a race condition with batch
job submissions.

Both the Rocket (upstream DRM/accel) and RKNPU (vendor DRM) drivers program
the same hardware registers — the emulator handles both transparently.
