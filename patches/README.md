# Rocket NPU Performance Patches

Optimizations for the open-source Rocket NPU driver (RK3588), reducing MobileNetV1 INT8
inference from 12.6ms to 9.9ms (22% improvement, bit-exact output).

## Tested against

- **Mesa**: 26.1.0-devel, commit `384d128` (gitlab.freedesktop.org/mesa/mesa)
- **Kernel**: Linux 6.18.10 (git.kernel.org/stable/linux, tag `v6.18.10`)
- **Board**: Orange Pi 5 Plus (RK3588), Armbian Noble

## Mesa patches

Apply in order:

```sh
cd mesa
git apply 0001-rocket-buffer-pool-cache-sync-batched-submit.patch
git apply 0002-rocket-teflon-input-conv-skip-prep-single-job.patch
```

### 0001: Buffer pool, cache sync reduction, batched submission

Targets IOCTL overhead (757 → 8 per invoke):
- GEM BO pool with best-fit reuse (eliminates CREATE_BO/GEM_CLOSE)
- `device_resident` flag skips PREP_BO/FINI_BO for write-once BOs
- `persistent_map` avoids repeated mmap/munmap
- Pre-allocated cached submit structures

### 0002: Input conversion, skip PREP_BO, single job, teflon alloc

Targets CPU-side and scheduling overhead (8 → 7 IOCTLs, 27 → 1 DRM job):
- Pre-fill input tensor padding at compile time; 3-channel fast path (5.3x fewer writes)
- `cpu_write_only` flag skips PREP_BO for graph input tensors
- Merge all per-operation DRM jobs into single job (graph-input-only BOs in `in_handles`,
  all output/intermediate BOs in `out_handles` — no overlap)
- Pre-allocate invoke buffers in teflon delegate

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

**Mesa:**
- `struct rkt_resource` in `rkt_device.h` — new fields: `pooled`, `device_resident`,
  `cpu_write_only`, `persistent_map`
- `struct rkt_screen` in `rkt_device.h` — new fields: `bo_pool`, `pool_mutex`
- `struct rkt_ml_subgraph` in `rkt_ml.h` — new fields: `cached_submit`, `cached_jobs`,
  `cached_in_handles`, `cached_out_handles`, `cached_job_count`, `submit_cached`
- `rkt_ml_subgraph_invoke` job construction — the single-job merge requires
  `in_bo_handles` and `out_bo_handles` to be disjoint sets
- `tfl_device.c` `struct teflon_subgraph` — new fields for pre-allocated buffers

**Kernel:**
- `struct rocket_core` in `rocket_core.h` — new field: `attached_domain`
- `rocket_job_run` / `rocket_job_handle_irq` in `rocket_job.c` — IOMMU logic
- `rocket_core_fini`, `rocket_postclose` — cleanup of cached domain
- `rocket_ioctl_submit` — error propagation loop

If the upstream driver adds new fields or changes the job submission flow, the patches
may need rebasing but the optimization strategies remain valid.
