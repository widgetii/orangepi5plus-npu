# Rocket NPU Performance Patches

Optimizations for the open-source Rocket NPU driver (RK3588), reducing MobileNetV1 INT8
inference from 11.6ms to 10.2ms (12% improvement, bit-exact output). Also fixes a Teflon
delegate crash (per-axis quantization assertion) that prevented loading YOLO models.

## Tested against

- **Mesa**: 26.1.0-devel, commit `384d128` (gitlab.freedesktop.org/mesa/mesa)
- **Kernel**: Linux 6.18.10 (git.kernel.org/stable/linux, tag `v6.18.10`)
- **Board**: Orange Pi 5 Plus (RK3588), Armbian Noble

## Mesa patches

Apply 0006+0005 first (fix YOLO crash + INT8 regression), then 0004 for SW ops, and
optionally 0003 for performance:

```sh
cd mesa
git apply 0006-teflon-fix-per-axis-quant-assertion-for-yolo.patch
git apply 0005-rocket-fix-int8-regression-batch-tasks-per-operation.patch
git apply 0004-rocket-add-sw-ops-concat-maxpool-pad-resize-logistic.patch
git apply 0003-rocket-bo-pool-cache-sync-output-reorder-neon-input-cached-submit.patch  # optional perf
```

### 0006: Fix Teflon per-axis quantization assertion for YOLO

Removes `assert(quant->scale->size == quant->zero_point->size)` in `fill_tensor()`
that crashes on YOLO models (per-axis weights have N scales but 1 zero_point). Adds
`scale->size == zero_point->size` to the per-axis storage condition so mismatched
per-axis tensors are treated as per-tensor (using the first scale/zero_point value).
This allows the Rocket driver to accept per-axis convolutions without graph fragmentation.
- Files: `tfl_device.c` (2 lines changed)

### 0005: Fix INT8 regression — batch tasks per operation

Fixes an upstream regression in Mesa git HEAD where per-task job splitting
(`reuse_weights_cbuf == false`) produces incorrect output for INT8 quantized models.
The per-task split distributes tasks across NPU cores, but tasks within an operation
have inter-task data dependencies via the regcmd chain that require sequential execution.
- **Fix**: Always batch all tasks into one job per operation (matching Mesa 26.0.2 behavior)
- Files: `rkt_ml.c` (submit loop only, ~30 lines changed)

### 0004: Software ML ops for YOLO (CONCAT, MAX_POOL, PAD, RESIZE, LOGISTIC)

Adds 5 CPU-side software ops that reduce graph splits for YOLO models. All 7 test
models produce bit-exact output (max_diff=0) vs CPU baseline.

Two execution modes based on subgraph composition:
- **Mixed HW/SW** (CONV + SW ops together): SW ops use NPU interleaved int8 format
  (`NPU_OFFSET` addressing, 0x80 bias) to match NPU hardware input/output
- **SW-only** (no CONVs in subgraph): bypass NPU format entirely — raw NHWC flat copy
  in/out, standard NHWC addressing in SW ops. Avoids NPU x-major/y-major layout
  mismatch and 0x80 bias sign inversion that would break spatial ops like MAX_POOL_2D

Ops:
- **CONCATENATION**: Channel-axis concat; fast `memcpy` for 16-aligned, per-element remap otherwise
- **MAX_POOL_2D**: Sliding-window max with `padding_same`; NHWC path for sw_only, NPU path for mixed
- **PAD**: Spatial zero-padding; NHWC or NPU format depending on subgraph type
- **RESIZE_NEAREST_NEIGHBOR**: Nearest-neighbor upscale (`floor(ox * in / out)`)
- **LOGISTIC**: 256-entry LUT; `raw_lut` (int8→int8) for sw_only, `lut` (NPU byte→NPU byte) for mixed

Other features:
- **Execution architecture**: Segment-based plan — consecutive CONV ops batched into one
  `DRM_IOCTL_ROCKET_SUBMIT`, each SW op runs on CPU between batches
- **Validation**: `is_quantized_feature_tensor()` rejects non-quantized/non-4D tensors
- **Multi-input fix**: CONCAT input conversion writes to `input_idxs[i]`, not first tensor
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
- `struct rkt_operation` in `rkt_ml.h` — new `type` field + `sw` union (incl. `raw_lut`)
- `struct rkt_exec_segment` in `rkt_ml.h` — segment-based execution plan
- `struct rkt_ml_subgraph` in `rkt_ml.h` — execution segments, `sw_only` flag
- `rkt_ml.c` — lowering/execution for 5 new ops, `build_execution_plan`,
  sw_only NHWC bypass in input/output conversion and spatial ops,
  `is_quantized_feature_tensor` validation, segment-based invoke loop

**Kernel:**
- `struct rocket_core` in `rocket_core.h` — new field: `attached_domain`
- `rocket_job_run` / `rocket_job_handle_irq` in `rocket_job.c` — IOMMU logic
- `rocket_core_fini`, `rocket_postclose` — cleanup of cached domain
- `rocket_ioctl_submit` — error propagation loop
