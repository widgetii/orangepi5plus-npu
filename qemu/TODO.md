# QEMU RK3588 NPU — Remaining Work

## Goal

Run `bench_teflon.py mobilenet_v1_int8.tflite --verify golden.bin` inside QEMU guest with bit-exact output matching real hardware.

## 1. Golden Output Capture (on real board)

Capture reference outputs from the real Orange Pi 5 Plus NPU for verification.

```bash
# On board (10.216.128.51):
cd /root/npu-research
source venv/bin/activate

# MobileNetV1 — dump NPU output tensor
python3 -c "
import ai_edge_litert as tfl
import numpy as np
interp = tfl.Interpreter('/root/npu-research/mobilenet_v1_1.0_224_quant.tflite',
                          experimental_delegates=[tfl.load_delegate('libteflon.so')])
interp.allocate_tensors()
inp = np.random.RandomState(42).randint(0, 256, (1,224,224,3), dtype=np.uint8)
interp.set_tensor(interp.get_input_details()[0]['index'], inp)
interp.invoke()
out = interp.get_tensor(interp.get_output_details()[0]['index'])
np.save('/tmp/mbv1_input.npy', inp)
np.save('/tmp/mbv1_golden.npy', out)
print(f'Output shape: {out.shape}, first 10: {out.ravel()[:10]}')
"
```

Save `mbv1_input.npy` and `mbv1_golden.npy` to `qemu/golden/`.

## 2. CI Rootfs Image

Build a self-contained rootfs with everything needed to run inference.

### Contents needed:
- Static busybox (shell, mount, insmod, ls, cat)
- Kernel modules: `qemu_iommu.ko`, `qemu_reset.ko`, `drm_shmem_helper.ko`, `gpu-sched.ko`, `rocket.ko`
- `libteflon.so` (system or custom build)
- Python 3 + numpy + ai_edge_litert (or a static C test binary)
- `mobilenet_v1_1.0_224_quant.tflite`
- `mbv1_input.npy`, `mbv1_golden.npy`

### Simpler alternative: static C test binary

Skip Python entirely. Write a C program that:
1. Loads tflite model via librocketnpu
2. Feeds the golden input
3. Compares output byte-by-byte against golden
4. Prints PASS/FAIL + max_diff

This is much smaller than a Python rootfs and eliminates numpy/tflite-runtime dependencies.

```bash
# Build on board:
gcc -static -O2 -o verify_npu verify_npu.c -I librocketnpu/include librocketnpu/src/*.c -ldrm -lm
```

### Rootfs build:
```bash
mkdir -p rootfs/{bin,lib/modules/...,...}
# Copy: busybox, verify_npu, modules, model, golden data
# Init script: load modules → run verify_npu → poweroff
find rootfs | cpio -o -H newc | gzip > ci-initrd.gz
```

## 3. Convolution Engine Fixes

Issues found during testing that need fixing before bit-exact output:

### 3a. Weight packing read function

The `read_weight()` function in `rockchip-npu.c` must exactly match Mesa's `rkt_fill_weights()` packing order. Current implementation has the index calculation but the test showed alternating zeros for odd channels. Verify against a known-good weight buffer dumped from real hardware:

```bash
# On board, enable BO dumps:
ROCKET_DEBUG=dump_bos python3 bench_teflon.py mobilenet_v1_int8.tflite
# Produces mesa-weights-000-000.bin, mesa-biases-000-000.bin etc.
```

Compare QEMU's `read_weight()` output against these dumps for the first conv layer.

### 3b. Bias handling modes

Two bias paths exist:
- **Standard**: bias from DMA (`BRDMA_DATA_USE=1`, `BS_BASE_ADDR` points to bias BO)
- **Per-channel**: bias from register (`BS_CFG=0x13F`, bias in `BS_ALU_CFG`)

Both are implemented but the standard path needs verification that the bias correction (zero-point cross terms) computed by Mesa matches what the QEMU engine expects.

### 3c. Output conversion verification

The requantization formula:
```
result = clamp((acc * scale + (1 << (shift-1))) >> shift + offset, -128, 127)
```

Verify rounding matches hardware. The NPU may use banker's rounding (round-half-to-even) rather than round-half-up. Test with values at the rounding boundary.

### 3d. Multi-task spatial tiling

When a conv layer is split into multiple tasks (spatial tiling), each task has:
- `input_offset` / `output_offset` into the activation tensor
- Overlap slices for filter context
- Chain pointer to next task's regcmd

The chain-following logic in `execute_job` works but hasn't been tested with actual multi-task jobs. MobileNetV1's first conv (224→112, stride=2) has 1 task; deeper layers may have more.

## 4. IOMMU Improvements

### 4a. Remove iommu.passthrough=1 requirement

Currently the dummy IOMMU module must be loaded before the rocket driver, and `iommu.passthrough=1` is needed because the module registers late. Options:

- **Build IOMMU into kernel**: Add `qemu_iommu.c` to the kernel tree and compile with `CONFIG_QEMU_IOMMU=y`. Cleanest but requires custom kernel build.
- **Initramfs early-load**: Load `qemu_iommu.ko` from initramfs before `init` runs, using a custom initramfs init that does `insmod` first.

### 4b. IOMMU mailbox scalability

Current mailbox uses linear scan of up to 4096 entries. For large models (YOLO with hundreds of BOs), this could be slow. Options:
- Hash table lookup by IOVA page
- Sorted array with binary search
- Increase `NPU_IOMMU_MAX_PAGES` if needed

Not urgent — MobileNetV1 uses ~170 BOs = ~170 pages.

## 5. CI Automation

### Test script
```bash
#!/bin/bash
QEMU=qemu-src/build/qemu-system-aarch64
KERNEL=/tmp/Image-6.18
INITRD=ci-initrd.gz

timeout 120 $QEMU -M orangepi5plus -smp 4 -m 1G \
  -kernel $KERNEL -initrd $INITRD \
  -append "console=ttyS0 earlycon rdinit=/init iommu.passthrough=1" \
  -nographic 2>&1 | tee /tmp/qemu-npu-test.log

grep -q "PASS" /tmp/qemu-npu-test.log && echo "CI: PASS" || echo "CI: FAIL"
```

### GitHub Actions integration
```yaml
jobs:
  npu-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build QEMU
        run: |
          apt-get install -y build-essential ninja-build libglib2.0-dev libfdt-dev libpixman-1-dev
          # clone qemu, copy files, build
      - name: Run NPU test
        run: ./ci/run-npu-test.sh
```

## 6. Extended Model Support

After MobileNetV1 passes:

### 6a. SSD MobileNetV1
Same conv engine but more ops. Should work if MobileNetV1 is bit-exact.

### 6b. YOLOv5s-relu
Requires SW ops in the test binary (CONCATENATION, MAX_POOL_2D, PAD, RESIZE_NEAREST, LOGISTIC). These are CPU-side ops that librocketnpu already implements. The QEMU NPU model only handles CONV — SW ops run on the emulated CPU.

### 6c. Per-axis quantization
YOLO uses per-axis weights → per-group or per-channel decomposition. The QEMU conv engine supports per-channel mode (RKNN-style BS bypass). Verify with YOLO golden outputs.

## Priority Order

1. **Golden capture** — 10 minutes on real board, blocks everything else
2. **Weight packing verification** — dump mesa weights, compare with `read_weight()` output
3. **Static C test binary** — avoids Python/numpy in rootfs, much simpler CI
4. **CI initrd** — combine modules + test binary + model + golden data
5. **End-to-end verification** — run in QEMU, compare output
6. **CI script** — automate the above
7. **IOMMU cleanup** — remove passthrough requirement (nice-to-have)
8. **Extended models** — SSD, YOLO (stretch goal)
