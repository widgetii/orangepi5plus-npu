# Rocket NPU Driver Optimization Report

## Environment

- Board: Orange Pi 5 Plus (RK3588, 16GB RAM)
- NPU: 3-core Rockchip RKNN (NVDLA-derived CNA/DPU architecture)
- Kernel: Linux 6.18.10-current-rockchip64 (Armbian mainline, v6.18.10)
- Mesa: 26.1.0-devel (commit 384d128)
- Baseline reference: RKNN proprietary runtime 2.3.0 (vendor kernel 6.1.x)

## Summary

Optimizations across Mesa userspace and kernel module reduce MobileNetV1 INT8
inference latency by 12% (bit-exact output):

| Model | System Baseline | Optimized | Improvement |
|-------|----------------|-----------|-------------|
| MobileNetV1 224 INT8 | 11.61ms avg / 10.91ms min | **10.23ms avg / 9.51ms min** | **12%** |
| SSD MobileNetV1 INT8 | 22.90ms avg / 21.65ms min | **19.82ms avg / 18.58ms min** | **13%** |

RKNN proprietary single-core reference: 2.6ms (MobileNetV1). The remaining 3.9x gap
is dominated by hardware compute time (~3ms), per-task kernel IRQ overhead (~2ms),
and irreducible format conversion (~1ms). Closing it further requires multi-core support
or a pre-compiled model format — both major new features beyond driver optimization.

## Patches

### Mesa patch (apply against Mesa 26.1.0-devel / commit 384d128)

**`patches/mesa/0003-rocket-bo-pool-cache-sync-output-reorder-neon-input-cached-submit.patch`**

Standalone patch combining all Mesa optimizations:
- **Buffer pool**: Recycle GEM BOs via a per-screen pool (best-fit, 256 cap) — eliminates CREATE_BO/GEM_CLOSE
- **Cache sync reduction**: `device_resident` flag skips PREP_BO/FINI_BO for write-once BOs (weights, biases, regcmds); `persistent_map` keeps mmap alive; `cpu_write_only` skips PREP_BO for graph input tensors
- **Output conversion reorder**: Loop changed from `(oc, x, y)` to `(g, y, x, c)` for sequential reads from NPU interleaved format; NEON `vld1q/vaddq` fast path for single-group outputs (oc <= 16)
- **Input conversion NEON**: 3-channel RGB fast path using NEON `vst1q_u8` for padding writes; bounded inner loop for general case (avoids branch per iteration for padding channels)
- **Cached per-operation submit**: Pre-build `drm_rocket_job` array in `subgraph_create`, zero malloc/free per invoke
- Files: `rkt_device.h`, `rkt_device.c`, `rkt_ml.h`, `rkt_ml.c`

### Kernel patch (apply against Linux 6.18.10, `drivers/accel/rocket/`)

**`patches/kernel/0001-rocket-iommu-attach-caching-and-submit-error-propagation.patch`**
- **IOMMU attach caching**: Cache attached domain on `rocket_core`; skip redundant `iommu_attach_group`/`iommu_detach_group` when consecutive jobs share the same domain. Detach only on suspend, file close, or reset
- **Submit error propagation**: `rocket_ioctl_submit` now propagates per-job errors from `rocket_ioctl_submit_job` instead of silently ignoring them
- Files: `rocket_core.h`, `rocket_job.c`, `rocket_core.c`, `rocket_drv.c`

## Applying the patches

### Mesa

```sh
cd mesa                         # Mesa 26.1.0-devel source tree
git apply patches/mesa/0003-rocket-bo-pool-cache-sync-output-reorder-neon-input-cached-submit.patch

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

### Where the time goes (MobileNetV1 INT8, perf profile)

`perf record` on 200-iteration benchmark (cortex_a76 event counts):
- 56.5% idle (openblas `blas_thread_server` spin during NPU compute)
- 3.7% `rkt_ml_subgraph_invoke` (input conversion)
- 2.8% `__pi_dcache_clean_poc` (FINI_BO cache flush)
- 2.6% `rkt_ml_subgraph_read_outputs` (output conversion)
- 1.3% `rkt_fill_weights` (warmup only)
- 1.0% `_raw_spin_unlock_irqrestore` (kernel IRQ handling)

### Time budget breakdown (estimated per invoke)

| Component | Current | Notes |
|-----------|---------|-------|
| NPU hardware compute | ~3ms | Irreducible (single core) |
| Per-task IRQ chain (41 tasks) | ~2ms | 41 IRQs, not removable without perf loss |
| Output format conversion | ~0.5ms | Reordered loop + NEON |
| Input format conversion | ~0.3ms | NEON fast path |
| Kernel submit + cache flush | ~0.5ms | FINI_BO + scheduler |
| Misc (fences, scheduling) | ~3.5ms | Partly irreducible |
| **Total** | **~10ms** | |

### Hardware task chaining investigation

**Goal**: Use `PC_TASK_CON_TASK_NUMBER(N)` to process N tasks with 1 IRQ instead of 41.

**Findings**:
- Within-operation task chaining via regcmd `PC_BASE_ADDRESS`/`PC_REGISTER_AMOUNTS` fields already works (tasks within an operation are chained in `compile_operation`)
- Cross-operation chaining was implemented in Mesa: patch last task's regcmd to point to next operation's first task
- `TASK_NUMBER(N)` with N > 1 does work — hardware processes all N tasks before generating an interrupt
- **Problem**: Merging all tasks into 1 DRM job forces all tasks to a single NPU core, losing 3-core spatial parallelism. Result: 14.98ms (50% regression!) vs 10ms with per-task jobs
- Per-operation jobs with `TASK_NUMBER(task_count)`: negligible gain since most operations have only 1 task (MobileNetV1: 27 operations, 41 tasks → only ~14 multi-task operations)
- **Conclusion**: Per-task IRQ overhead (~2ms) is real but cannot be removed without losing multi-core parallelism. The DRM scheduler distributes per-task jobs across 3 cores, which provides more benefit than eliminating IRQ overhead

### Why RKNN is still 3.9x faster (2.6ms vs 10.2ms)

1. **Pre-compiled command stream**: RKNN's `.rknn` format contains the complete register
   program for the entire model. No per-invoke compilation or job construction.
   Rocket rebuilds `drm_rocket_submit` structures (cached after first invoke, but the
   kernel still copies and validates tasks from userspace each time).

2. **Zero data conversion**: RKNN stores tensors in HW-native interleaved int8 format.
   Rocket converts NHWC uint8 to interleaved int8 with 0x80 bias offset every invoke.

3. **Per-task IRQ overhead**: The kernel submits 41 tasks one-by-one via IRQ (each task
   triggers an interrupt, the handler programs the next task's registers). RKNN likely
   uses hardware task chaining on a single core since it doesn't use the DRM scheduler.

4. **Multi-core**: RKNN can split work across 3 NPU cores. Rocket's DRM scheduler
   distributes tasks across cores but with per-task job overhead.

### Remaining optimization paths

**High impact (requires kernel UAPI changes):**
- Single-core hardware task chaining with dedicated multi-core split — instead of relying on
  the DRM scheduler, split operations across cores explicitly in Mesa and use TASK_NUMBER(N)
  within each core's task chain. This could eliminate both IRQ overhead AND get multi-core.
- Pre-compiled model cache — serialize the compiled subgraph to disk.

**Medium impact (userspace only):**
- Further output conversion NEON — the multi-group scatter-store path is still scalar.
  For large channel counts, NEON gather-scatter could help.
- Kernel cached task submission — avoid `copy_from_user` for repeated identical task arrays.
  Measured at ~0.3ms via perf; moderate effort for small gain.

**Low impact:**
- Input conversion is already well-optimized (~0.3ms for 224x224x3).
- BO pool is effective; further tuning unlikely to yield measurable gains.

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
submits one task at a time via IRQ; the hardware follows the chain after
`OPERATION_ENABLE` is written via the regcmd's final entry.

`PC_TASK_CON_TASK_NUMBER(N)` tells the hardware to count N task completions before
generating an interrupt. With N=1 (default), every task generates an IRQ. With N>1,
the hardware auto-chains tasks via the regcmd linkage and only interrupts after N
completions. This works correctly but is only beneficial when all tasks run on the
same core (no multi-core parallelism).

## Software ML ops for YOLO (patch 0004)

### Motivation

The Rocket NPU driver only supports CONV_2D and ADD operations. YOLO models need
13+ additional ops (CONCATENATION, MAX_POOL_2D, PAD, RESIZE_NEAREST_NEIGHBOR,
LOGISTIC, etc.). When unsupported, the Teflon delegate splits the graph at each
boundary, requiring NPU→NHWC→CPU→NHWC→NPU format conversions per split.

### Implementation

Five CPU-side software ops. Two execution modes:
- **Mixed HW/SW subgraphs** (CONV + SW ops): SW ops use NPU interleaved int8 format
- **SW-only subgraphs** (no CONVs): bypass NPU format, use raw NHWC directly

| Op | Strategy | Notes |
|----|----------|-------|
| CONCATENATION | Per-pixel channel append (NHWC) or group memcpy (NPU) | Fast path for 16-aligned |
| MAX_POOL_2D | Sliding-window max with `padding_same` | NHWC int8 comparison for sw_only |
| PAD | Spatial zero-padding | Quantization-aware fill value |
| RESIZE_NEAREST | `floor(ox * in_w / out_w)` per pixel | General scale factors |
| LOGISTIC | 256-entry LUT | `raw_lut` (int8→int8) for sw_only, `lut` (NPU) for mixed |

### Execution architecture

Mixed HW/SW execution via a segment-based plan built at `subgraph_create` time:
1. Consecutive CONV ops are grouped into one HW segment → single `DRM_IOCTL_ROCKET_SUBMIT`
2. Each SW op is its own segment → executed on CPU between HW batches
3. `pipe_buffer_map`/`pipe_buffer_unmap` handles cache coherency (PREP_BO/FINI_BO)
4. Pre-built job/task arrays avoid per-invoke allocation

### SW-only subgraph bypass

When a delegate partition contains no CONV ops (`sw_only=true`), NPU interleaved format
is bypassed entirely:
- **Input**: `memcpy` raw TFLite bytes (no 0x80 bias, no group interleaving)
- **Output**: `memcpy` back (no deinterleaving)
- **Spatial ops**: use flat NHWC addressing `data[(x*h+y)*C+c]`
- **Rationale**: the NPU tensor format has an implicit spatial transpose (input x-major,
  output y-major) and a 0x80 bias that inverts int8 sign ordering — both break spatial
  neighbor access in SW ops when no NPU hardware is involved

### Validation

- All 7 test models produce bit-exact output (max_diff=0): POOL_ONLY, CONCAT, MAXPOOL,
  LOGISTIC, RESIZE, 2CONV_INT8, MobileNetV1
- `is_quantized_feature_tensor()` rejects non-quantized/non-4D tensors (e.g., int32 shape
  constants from TFLite's UpSampling2D decomposition into RESHAPE/TILE/CONCATENATION ops)

## YOLOv5s-relu end-to-end results

With patches 0004+0005+0006 applied (Mesa 26.1.0-devel, debugoptimized build):

| Metric | Value |
|--------|-------|
| Model | YOLOv5s-relu INT8, 640x640x3 input, 3 detection heads |
| Delegate partitions | 3 (57 + 9 + 32 ops) |
| Total delegated ops | 98 / ~100 |
| SW ops executed | CONCAT(13), MAX_POOL_2D(6), PAD(6), RESIZE_NN(2), LOGISTIC(3) |
| Inference time | 1120ms (vs 143ms CPU-only, vs 16.7ms RKNN single-core) |
| Output correctness | **Incorrect** — constant values per head |
| NPU timeouts | 0 (individual CONVs all succeed) |

### Root cause: per-axis quantization not supported

All 61 CONV operations complete without error (verified by submitting each one
individually with `ROCKET_DEBUG=dbg_msgs`). The incorrect output is caused by
**per-axis quantization being silently treated as per-tensor**.

YOLO weight tensors have per-axis quantization: `scale->size = N` (one per output
channel) but `zero_point->size = 1`. The Teflon fix (patch 0006) prevents crashing
by skipping per-axis scale storage when sizes mismatch, which causes the Rocket driver
to accept the CONV using only the first channel's scale for all channels.

The per-axis scale ratios in YOLO weights are significant — up to 27x between channels
in some layers. Using a single scale for all channels causes massive quantization errors
in `rkt_fill_biases()` (which precomputes `bias * input_scale * weight_scale`), producing
overflow/saturation that propagates through 61 layers to produce constant output.

This is a fundamental limitation of the Rocket driver's convolution implementation.
Fixing it requires per-axis support in:
- `rkt_coefs.c`: compute per-channel biases with per-channel scales
- `rkt_regcmd.c`: program per-channel accumulator truncation (if HW supports it)

All SW ops (CONCAT, MAX_POOL_2D, PAD, RESIZE, LOGISTIC) execute correctly. The
correctness problem is entirely in the HW CONV quantization path.

Patches required for YOLO:
- **0006**: Removes per-axis quantization assertion in `tfl_device.c` (YOLO weight tensors
  have `scale->size != zero_point->size`). Without this, the delegate crashes on load.
- **0005**: Fixes INT8 regression (per-task job splitting). Without this, all INT8 CONVs
  produce wrong output.
- **0004**: Adds SW ops. Without this, the graph splits at every non-CONV op, requiring
  costly format conversions at each boundary.

## Known issues

- **EfficientNet-Lite0 INT8** crashes the Rocket driver (unsupported ops trigger kernel panic). Do not test with NPU.
- **NPU job timeout** can corrupt IOMMU state and cascade to kernel memory corruption. Boot with `panic=10 panic_on_oops=1`.
- **Struct changes require clean rebuild**: `rm -rf build && meson setup ...` — stale `.o` files cause crashes.
- **Single-job BO handles**: Intermediate tensors must not appear in both `in_bo_handles` and `out_bo_handles` of the same job, or `drm_gem_lock_reservations` returns `-EALREADY`. The upstream `rocket_ioctl_submit` silently ignores this error (fixed by our kernel patch).
- **Upstream int8 regression (FIXED by patch 0005)**: The git HEAD's per-task job
  splitting (`reuse_weights_cbuf == false`) breaks INT8 models. Fix: batch all tasks
  into one job per operation (matches Mesa 26.0.2 behavior).
