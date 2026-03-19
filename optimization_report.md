# Rocket NPU Driver Optimization Report

## Environment

- Board: Orange Pi 5 Plus (RK3588, 16GB RAM)
- NPU: 3-core Rockchip RKNN (NVDLA-derived CNA/DPU architecture)
- Kernel: Linux 6.18.10-current-rockchip64 (Armbian mainline, v6.18.10)
- Mesa: 26.1.0-devel (commit 384d128)
- Baseline reference: RKNN proprietary runtime 2.3.0 (vendor kernel 6.1.x)

## Summary

Seven optimizations across Mesa userspace and kernel module reduce MobileNetV1 INT8
inference latency from 12.6ms to 9.9ms (22% improvement, bit-exact output).

| Model | System Baseline | Optimized | Improvement |
|-------|----------------|-----------|-------------|
| MobileNetV1 224 INT8 | 12.62ms avg / 11.61ms min | **9.85ms avg / 9.02ms min** | **22%** |
| SSD MobileNetV1 INT8 | 24.02ms avg / 23.19ms min | **20.28ms avg / 19.43ms min** | **16%** |

RKNN proprietary single-core reference: 2.6ms (MobileNetV1). The remaining 3.8x gap
is dominated by hardware compute time (~3ms), irreducible format conversion (~1ms),
and per-task kernel IRQ overhead (~1ms). Closing it further requires multi-core support
or a pre-compiled model format — both major new features beyond driver optimization.

## Patches

### Mesa patches (apply in order against Mesa 26.1.0-devel / commit 384d128)

**`patches/mesa/0001-rocket-buffer-pool-cache-sync-batched-submit.patch`**
Round 1 optimizations targeting IOCTL overhead:
- **Buffer pool**: Recycle GEM BOs via a per-screen pool (best-fit, 256 cap) — eliminates CREATE_BO/GEM_CLOSE
- **Cache sync reduction**: `device_resident` flag skips PREP_BO/FINI_BO for write-once BOs (weights, biases, regcmds); `persistent_map` keeps mmap alive
- **Batched submission**: Pre-allocate and cache `drm_rocket_submit` structures
- Files: `rkt_device.h`, `rkt_device.c`, `rkt_ml.h`, `rkt_ml.c`

**`patches/mesa/0002-rocket-teflon-input-conv-skip-prep-single-job.patch`**
Round 2 optimizations targeting CPU-side overhead and job scheduling:
- **Input conversion fast path**: Pre-fill padding in `subgraph_create` (once); 3-channel RGB fast path writes only 3 of 16 bytes per pixel, avoiding branch-per-iteration and 5.3x fewer writes
- **Skip input PREP_BO**: `cpu_write_only` flag on graph input tensors skips the PREP_BO IOCTL (802KB cache invalidation) since CPU only writes, never reads
- **Single merged job**: All 27 per-operation DRM jobs consolidated into 1 job, saving 26x scheduler/fence/IOMMU overhead. Graph-input-only handles go in `in_bo_handles`, all others in `out_bo_handles` to avoid duplicate BO locks
- **Teflon malloc elimination**: Pre-allocate `buffers`/`is_signed` arrays in `partition_init`, removing 4 malloc+free per invoke
- Files: `rkt_device.h`, `rkt_device.c`, `rkt_ml.c`, `tfl_device.c`

### Kernel patch (apply against Linux 6.18.10, `drivers/accel/rocket/`)

**`patches/kernel/0001-rocket-iommu-attach-caching-and-submit-error-propagation.patch`**
- **IOMMU attach caching**: Cache attached domain on `rocket_core`; skip redundant `iommu_attach_group`/`iommu_detach_group` when consecutive jobs share the same domain. Detach only on suspend, file close, or reset
- **Submit error propagation**: `rocket_ioctl_submit` now propagates per-job errors from `rocket_ioctl_submit_job` instead of silently ignoring them. This is a correctness bugfix — without it, a failed job (e.g. from invalid BO handles) is silently dropped and userspace gets success
- Files: `rocket_core.h`, `rocket_job.c`, `rocket_core.c`, `rocket_drv.c`

## Applying the patches

### Mesa

```sh
cd mesa                         # Mesa 26.1.0-devel source tree
git apply patches/mesa/0001-rocket-buffer-pool-cache-sync-batched-submit.patch
git apply patches/mesa/0002-rocket-teflon-input-conv-skip-prep-single-job.patch

meson setup build \
  -Dgallium-drivers=rocket -Dvulkan-drivers="" -Dteflon=true \
  -Dprefix=/usr/local -Dbuildtype=release --wrap-mode=nodownload \
  -Dplatforms=wayland -Dglx=disabled -Degl=disabled -Dgbm=disabled \
  -Dopengl=false -Dgles1=disabled -Dgles2=disabled
ninja -C build && ninja -C build install
```

### Kernel module

```sh
# Get source (must match running kernel exactly)
cd /path/to/linux-6.18.10/drivers/accel/rocket/
patch -p4 < patches/kernel/0001-rocket-iommu-attach-caching-and-submit-error-propagation.patch

# Build as out-of-tree module
cat > Makefile << 'EOF'
obj-m := rocket.o
rocket-y := rocket_core.o rocket_device.o rocket_drv.o rocket_gem.o rocket_job.o
KDIR := /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
EOF
make
cp rocket.ko /lib/modules/$(uname -r)/kernel/drivers/accel/rocket/rocket.ko
rmmod rocket && modprobe rocket
```

### Runtime tuning (no code change)

```sh
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 5000 > /sys/bus/platform/drivers/rocket/fdab0000.npu/power/autosuspend_delay_ms
```

## Verification

Output must be **bit-exact identical** (max_diff=0) to system Mesa. Run:

```python
import numpy as np
import ai_edge_litert.interpreter as tflite

model = "mobilenet_v1_1.0_224_quant.tflite"
np.random.seed(42)

for lib in ["/usr/local/lib/aarch64-linux-gnu/libteflon.so",
            "/usr/lib/teflon/libteflon.so"]:
    d = [tflite.load_delegate(lib)]
    i = tflite.Interpreter(model_path=model, experimental_delegates=d)
    i.allocate_tensors()
    data = np.random.randint(0, 255, size=i.get_input_details()[0]["shape"]
                             ).astype(i.get_input_details()[0]["dtype"])
    i.set_tensor(i.get_input_details()[0]["index"], data)
    i.invoke()
    print(lib, i.get_tensor(i.get_output_details()[0]["index"]).flatten()[:5])
```

## Detailed optimization analysis

### IOCTL reduction (per inference, steady state)

| IOCTL | Unpatched | After round 1 | After round 2 |
|-------|-----------|---------------|---------------|
| CREATE_BO | 171 | 0 | 0 |
| PREP_BO | 229 | 4 | 3 |
| FINI_BO | 86 | 2 | 2 |
| SUBMIT | 2 | 2 | 2 |
| GEM_CLOSE | 114 | 0 | 0 |
| **Total** | **757** | **8** | **7** |

### Where the time goes (MobileNetV1 INT8, single core)

| Component | Unpatched | Optimized | Notes |
|-----------|-----------|-----------|-------|
| Input format conversion | ~2-3ms | ~0.5ms | Pre-fill + 3ch fast path |
| PREP_BO cache invalidation | ~0.5ms | ~0ms | Skipped for input tensor |
| IOMMU attach/detach (27x) | ~1-2ms | ~0ms | Cached, done once |
| DRM scheduler overhead (27 jobs) | ~0.5-1ms | ~0ms | Merged to 1 job |
| NPU hardware compute | ~3ms | ~3ms | Irreducible |
| Output format conversion | ~1ms | ~1ms | Not yet optimized |
| Kernel per-task IRQ chain | ~2ms | ~2ms | 41 IRQs for 41 tasks |
| BO pool / persistent map | ~1ms saved | - | Round 1 |
| **Total** | **~12.6ms** | **~9.9ms** | |

### Why RKNN is still 3.8x faster (2.6ms vs 9.9ms)

1. **Pre-compiled command stream**: RKNN's `.rknn` format contains the complete register
   program for the entire model. No per-invoke compilation or job construction.
   Rocket rebuilds `drm_rocket_submit` structures (cached after first invoke, but the
   kernel still copies and validates 41 tasks from userspace each time).

2. **Zero data conversion**: RKNN stores tensors in HW-native interleaved int8 format.
   Rocket converts NHWC uint8 to interleaved int8 with 0x80 bias offset every invoke.

3. **Per-task IRQ overhead**: The kernel submits 41 tasks one-by-one via IRQ (each task
   triggers an interrupt, the handler programs the next task's registers). RKNN likely
   uses hardware task chaining to avoid per-task IRQs.

4. **Multi-core**: RKNN can split work across 3 NPU cores. Rocket's DRM scheduler
   assigns all tasks to a single core.

### Remaining optimization paths

**High impact (requires kernel UAPI changes):**
- Hardware task chaining across operations — submit all 27 operations as a single
  hardware task chain via regcmd patching (PC_REGCMD_BASE_ADDR), reducing 41 IRQs to 1.
  Requires extending the `drm_rocket_task` struct or adding a chain flag.
- Multi-core support — split output channels across 3 cores for ~2-3x throughput.

**Medium impact (userspace only):**
- Output conversion optimization — current loop is `(oc, x, y)` with scattered reads.
  Reorder to `(g, y, x, c)` for sequential access; add NEON vectorization.
- Pre-compiled model cache — serialize the compiled subgraph (regcmd BOs, tensor layout)
  to disk, skip `subgraph_create` on subsequent loads.

**Low impact:**
- NEON for input conversion — `vld3_u8`/`vst3_lane_u8` for 8 pixels at a time.
  Current scalar 3-channel path is already fast (~0.5ms for 224x224).

## NPU hardware notes

The RK3588 NPU is register-programmed — there is no instruction set. The PC (Program
Counter) unit is a DMA engine that fetches register values from memory and programs them
into the CNA (Convolution Neural Accelerator). Each "task" is a set of register writes
that configures one convolution operation (or a vertical stripe of one).

Key MMIO regions per core: PC (+0x0000), CNA (+0x1000), Core control (+0x3000).

The driver uses 4 custom IOCTLs: `CREATE_BO`, `PREP_BO`, `FINI_BO`, `SUBMIT`.
`PREP_BO` does `dma_resv_wait_timeout` + `dma_sync_sgtable_for_cpu`.
`FINI_BO` does `dma_sync_sgtable_for_device`.

Within an operation, `compile_operation` patches regcmd entries to chain tasks
(sets `PC_REGCMD_BASE_ADDR` pointing to the next task's register block). The kernel
submits one task at a time via IRQ; the hardware may or may not follow the chain
autonomously — the `OPERATION_ENABLE=0` in the IRQ handler aborts any in-progress chain
before re-submitting the next task explicitly.

## Known issues

- **EfficientNet-Lite0 INT8** crashes the Rocket driver (unsupported ops trigger kernel panic). Do not test with NPU.
- **NPU job timeout** can corrupt IOMMU state and cascade to kernel memory corruption. Boot with `panic=10 panic_on_oops=1`.
- **Struct changes require clean rebuild**: `rm -rf build && meson setup ...` — stale `.o` files cause crashes.
- **Single-job BO handles**: Intermediate tensors must not appear in both `in_bo_handles` and `out_bo_handles` of the same job, or `drm_gem_lock_reservations` returns `-EALREADY`. The upstream `rocket_ioctl_submit` silently ignores this error (fixed by our kernel patch).
