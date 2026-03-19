# Rocket NPU Performance Patches

Optimizations for the open-source Rocket NPU driver (RK3588), reducing MobileNetV1 INT8
inference from 11.6ms to 10.2ms (12% improvement, bit-exact output).

## Tested against

- **Mesa**: 26.1.0-devel, commit `384d128` (gitlab.freedesktop.org/mesa/mesa)
- **Kernel**: Linux 6.18.10 (git.kernel.org/stable/linux, tag `v6.18.10`)
- **Board**: Orange Pi 5 Plus (RK3588), Armbian Noble

## Mesa patches

Three generations of patches exist. Only the latest (0003) is needed — it is standalone
and applies directly to stock Mesa without 0001 or 0002:

```sh
cd mesa
git apply 0003-rocket-bo-pool-cache-sync-output-reorder-neon-input-cached-submit.patch
```

### 0003: BO pool, cache sync, output reorder, NEON input, cached submit (recommended)

Standalone patch combining all Mesa optimizations (supersedes 0001 + 0002):
- **Buffer pool**: Recycle GEM BOs via a per-screen pool (best-fit, 256 cap)
- **Cache sync reduction**: `device_resident` flag skips PREP_BO/FINI_BO for write-once BOs;
  `persistent_map` keeps mmap alive; `cpu_write_only` skips PREP_BO for graph input tensors
- **Output conversion reorder**: Loop changed from `(oc, x, y)` to `(g, y, x, c)` for
  sequential reads from NPU interleaved format; NEON fast path for single-group outputs
- **Input conversion NEON**: 3-channel RGB fast path using NEON `vst1q_u8` for padding;
  bounded inner loop for general case (only iterate real channels)
- **Cached per-operation submit**: Pre-build `drm_rocket_job` array in `subgraph_create`,
  zero malloc/free per invoke
- Files: `rkt_device.h`, `rkt_device.c`, `rkt_ml.h`, `rkt_ml.c`

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

**Kernel:**
- `struct rocket_core` in `rocket_core.h` — new field: `attached_domain`
- `rocket_job_run` / `rocket_job_handle_irq` in `rocket_job.c` — IOMMU logic
- `rocket_core_fini`, `rocket_postclose` — cleanup of cached domain
- `rocket_ioctl_submit` — error propagation loop
