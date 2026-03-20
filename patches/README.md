# Rocket NPU Performance Patches

Optimizations for the open-source Rocket NPU driver (RK3588), reducing MobileNetV1 INT8
inference from 11.6ms to 10.2ms (12% improvement, bit-exact output). Also fixes a Teflon
delegate crash (per-axis quantization assertion) that prevented loading YOLO models.

## Tested against

- **Mesa**: 26.1.0-devel, commit `384d128` (gitlab.freedesktop.org/mesa/mesa)
- **Kernel**: Linux 6.18.10 (git.kernel.org/stable/linux, tag `v6.18.10`)
- **Board**: Orange Pi 5 Plus (RK3588), Armbian Noble

## Mesa patches

Four generations of patches exist. Apply 0003 for performance, and 0004 for new ML op
support. 0004 applies independently on top of stock Mesa (does not require 0003):

```sh
cd mesa
git apply 0003-rocket-bo-pool-cache-sync-output-reorder-neon-input-cached-submit.patch
git apply 0004-rocket-add-sw-ops-concat-maxpool-pad-resize-logistic.patch
```

### 0004: Software ML ops for YOLO (CONCAT, MAX_POOL, PAD, RESIZE, LOGISTIC)

Adds 5 CPU-side software ops that execute directly on the NPU's interleaved int8 format,
avoiding costly NPU→NHWC→CPU→NHWC→NPU format conversions at graph split points:
- **CONCATENATION**: Channel-axis concat; fast `memcpy` path when all inputs are 16-aligned,
  per-element path with channel remapping otherwise
- **MAX_POOL_2D**: Sliding-window max on raw int8 bytes (0x80 bias is monotonic, so
  `max(a,b)` on biased values is correct); `padding_same` support
- **PAD**: Spatial zero-padding with quantization-aware fill (`zero_point - 0x80`)
- **RESIZE_NEAREST_NEIGHBOR**: Nearest-neighbor upscale with general scale factors
  (`floor(ox * in / out)`)
- **LOGISTIC**: Sigmoid via 256-entry LUT built at `subgraph_create` from input/output
  quantization parameters
- **Execution architecture**: Mixed HW/SW segment-based execution — consecutive CONV ops
  batched into one `DRM_IOCTL_ROCKET_SUBMIT`, each SW op runs on CPU between batches
- **Validation**: `is_quantized_feature_tensor()` rejects non-quantized/non-4D tensors
  (prevents crash from TFLite's UpSampling2D decomposition into int32 RESHAPE/TILE ops)
- Files: `rkt_ml.h`, `rkt_ml.c`

### 0003: BO pool, cache sync, output reorder, NEON input, cached submit

Standalone patch combining all Mesa performance optimizations (supersedes 0001 + 0002):
- **Buffer pool**: Recycle GEM BOs via a per-screen pool (best-fit, 256 cap)
- **Cache sync reduction**: `device_resident` flag skips PREP_BO/FINI_BO for write-once BOs;
  `persistent_map` keeps mmap alive; `cpu_write_only` skips PREP_BO for graph input tensors
- **Output conversion reorder**: Loop changed from `(oc, x, y)` to `(g, y, x, c)` for
  sequential reads from NPU interleaved format; NEON fast path for single-group outputs
- **Input conversion NEON**: 3-channel RGB fast path using NEON `vst1q_u8` for padding;
  bounded inner loop for general case (only iterate real channels)
- **Cached per-operation submit**: Pre-build `drm_rocket_job` array in `subgraph_create`,
  zero malloc/free per invoke
- **Teflon per-axis quant fix**: Remove assertion crash in `tfl_device.c` `fill_tensor()`
  when `quant->scale->size != quant->zero_point->size` (happens with YOLO models).
  Gracefully skips per-axis storage when sizes mismatch.
- Files: `rkt_device.h`, `rkt_device.c`, `rkt_ml.h`, `rkt_ml.c`, `tfl_device.c`

### 0001 + 0002 (legacy, superseded by 0003)

Earlier incremental patches. Not needed if using 0003. Kept for reference:
- 0001: Buffer pool, cache sync reduction, batched submission
- 0002: Input conversion fast path, skip PREP_BO, single-job merge, teflon alloc

## Kernel patch

```sh
cd linux/drivers/accel/rocket
patch -p4 < 0001-rocket-iommu-attach-caching-and-submit-error-propagation.patch
```

### 0001: IOMMU attach caching + submit error propagation

- Cache IOMMU domain attachment on `rocket_core`; skip redundant attach/detach between
  jobs sharing the same domain. Detach on suspend, file close, or reset.
- **Bugfix**: `rocket_ioctl_submit` now propagates per-job errors instead of silently
  ignoring them (a failed job was previously dropped with success returned to userspace).

## Porting to other versions

The patches touch a small number of files and structures. Key things to watch when
porting to a different Mesa or kernel version:

**Mesa (0003 patch):**
- `struct rkt_resource` in `rkt_device.h` — new fields: `device_resident`,
  `cpu_write_only`, `persistent_map`
- `struct rkt_screen` in `rkt_device.h` — new fields: `bo_pool`, `pool_mutex`
- `struct rkt_ml_subgraph` in `rkt_ml.h` — new fields for cached submit structures
- `rkt_device.c` — BO pool in create/destroy, persistent map, cache sync skip
- `rkt_ml.c` — output conversion reorder, input NEON, cached per-op submit,
  `chain_operations` (cross-op regcmd linking, currently unused but available)

**Mesa (0004 patch):**
- `enum rkt_op_type` in `rkt_ml.h` — new op type enum
- `struct rkt_operation` in `rkt_ml.h` — new `type` field + `sw` union for op params
- `struct rkt_exec_segment` in `rkt_ml.h` — segment-based execution plan
- `struct rkt_ml_subgraph` in `rkt_ml.h` — execution segments replace cached submit
- `rkt_ml.c` — lowering/execution for 5 new ops, `build_execution_plan`,
  `is_quantized_feature_tensor` validation, segment-based invoke loop

**Kernel:**
- `struct rocket_core` in `rocket_core.h` — new field: `attached_domain`
- `rocket_job_run` / `rocket_job_handle_irq` in `rocket_job.c` — IOMMU logic
- `rocket_core_fini`, `rocket_postclose` — cleanup of cached domain
- `rocket_ioctl_submit` — error propagation loop
