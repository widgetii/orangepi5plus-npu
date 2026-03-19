# Rocket NPU IOCTL Analysis

## Raw IOCTL Numbers (from strace)

| Hex Code | Count | DRM Command | Rocket IOCTL | Direction | Size |
|----------|-------|-------------|--------------|-----------|------|
| 0xc0186440 | 171 | DRM_CMD_BASE+0x00 | DRM_ROCKET_CREATE_BO | RW | 24 bytes |
| 0x40186441 | 10 | DRM_CMD_BASE+0x01 | DRM_ROCKET_SUBMIT | W | 24 bytes |
| 0x40106442 | 245 | DRM_CMD_BASE+0x02 | DRM_ROCKET_PREP_BO | W | 16 bytes |
| 0x40086443 | 94 | DRM_CMD_BASE+0x03 | DRM_ROCKET_FINI_BO | W | 8 bytes |
| 0x40086409 | 114 | DRM_GEM_CLOSE | GEM_CLOSE | W | 8 bytes |

## Decoding the hex:
- Bits 31-30: direction (0x40=W, 0xc0=RW, 0x80=R)
- Bits 29-16: size of struct
- Bits 15-8: type ('d' = 0x64 = DRM)
- Bits 7-0: command number

## Inference Pattern (MobileNetV1, single inference)
Total: 634 IOCTLs on the accel device

1. **CREATE_BO** (171 calls): Allocate GPU buffer objects for weights, activations, register command buffers
2. **PREP_BO** (245 calls): Prepare BOs for CPU access (cache sync before CPU writes)
3. **FINI_BO** (94 calls): Finalize BOs after CPU writes (cache sync before NPU access)
4. **SUBMIT** (10 calls): Submit jobs to NPU - each submit contains one or more jobs with tasks
5. **GEM_CLOSE** (114 calls): Free buffer objects after inference

## Command Submission Flow
1. Allocate BOs for weights + activations + register cmd buffers (CREATE_BO)
2. Map BOs to CPU, write data (PREP_BO + mmap + FINI_BO)
3. Submit register command buffers to NPU (SUBMIT)
4. Read results (PREP_BO)
5. Clean up (GEM_CLOSE)

## Key Structures (from rocket_accel.h)
- drm_rocket_create_bo: {size, flags} -> {handle, dma_addr, offset}
- drm_rocket_submit: {jobs_ptr, job_count, flags}
- drm_rocket_job: {tasks_ptr, task_count, in_bo_handles_ptr, out_bo_handles_ptr, ...}
- drm_rocket_task: {dma_addr, cmd_count} — register writes to NPU

---

# RKNPU Proprietary IOCTL Analysis (vendor kernel 6.1.115, driver v0.9.8)

## Raw IOCTL Numbers (from strace, ResNet18 inference)

| Hex Code | Count | DRM Command | RKNPU IOCTL | Direction | Size |
|----------|-------|-------------|-------------|-----------|------|
| 0xc0406400 | 6 | DRM_IOCTL_VERSION | (standard) | RW | 64B |
| 0xc0106401 | 2 | DRM_IOCTL_GET_UNIQUE | (standard) | RW | 16B |
| 0xc008640a | 5 | DRM_IOCTL_GEM_CLOSE | (standard) | RW | 8B |
| 0xc00c642d | 5 | DRM_IOCTL_PRIME_FD_TO_HANDLE | (standard) | RW | 12B |
| 0xc0086440 | 13 | DRM_CMD_BASE+0x00 | RKNPU_ACTION | RW | 8B |
| 0xc0686441 | 1 | DRM_CMD_BASE+0x01 | RKNPU_SUBMIT | RW | 104B |
| 0xc0306442 | 5 | DRM_CMD_BASE+0x02 | RKNPU_MEM_CREATE | RW | 48B |
| 0xc0106443 | 5 | DRM_CMD_BASE+0x03 | RKNPU_MEM_MAP | RW | 16B |
| 0xc0106444 | 5 | DRM_CMD_BASE+0x04 | RKNPU_MEM_DESTROY | RW | 16B |
| 0xc0206445 | 16 | DRM_CMD_BASE+0x05 | RKNPU_MEM_SYNC | RW | 32B |

Total: 63 real IOCTLs (excluding TCGETS noise)

## RKNPU Inference Sequence (ResNet18, single core)

```
1. DRM_IOCTL_VERSION (x6)            — identify driver
2. DRM_IOCTL_GET_UNIQUE (x2)         — get device ID
3. RKNPU_ACTION (x6)                 — init/query device state
4. RKNPU_MEM_CREATE                  — allocate weight memory
5. GEM_CLOSE + PRIME_FD_TO_HANDLE    — export as DMA-buf, reimport
6. RKNPU_MEM_MAP                     — map to NPU address space
7. RKNPU_MEM_SYNC                    — sync cache for data upload
   (repeat MEM_CREATE/MAP/SYNC for activation buffers)
8. RKNPU_MEM_SYNC (x4)              — sync all input buffers
9. RKNPU_SUBMIT (x1)                — submit ENTIRE model in one call
10. RKNPU_MEM_SYNC (x5)             — sync output/intermediate buffers
11. RKNPU_ACTION                     — query completion status
12. RKNPU_MEM_DESTROY (x5)          — free all buffers
```

## Key Differences: Rocket vs RKNPU

| Aspect | Rocket (open-source) | RKNPU (proprietary) |
|--------|---------------------|---------------------|
| Total IOCTLs per inference | **634** | **63** (10x fewer) |
| Submit calls | 10 (layer-by-layer) | **1** (whole model) |
| Buffer allocations | 171 CREATE_BO | 5 MEM_CREATE |
| Cache syncs | 339 (PREP+FINI) | 16 MEM_SYNC |
| Submit struct size | 24 bytes | **104 bytes** |
| Model compilation | JIT per layer | Pre-compiled .rknn graph |
| DMA-buf export | None | Yes (PRIME_FD_TO_HANDLE) |

## Analysis

The RKNPU driver is dramatically more efficient in IOCTL usage:

1. **Single submit**: The entire model graph is submitted in ONE 104-byte RKNPU_SUBMIT
   call. The pre-compiled .rknn model contains the full execution plan — all layer
   configs, weight addresses, and data flow. The driver/firmware handles layer
   sequencing internally.

2. **Fewer buffers**: RKNPU allocates only 5 large buffers (weights, input, output,
   scratch, command buffer) vs Rocket's 171 small per-layer buffers.

3. **Fewer cache syncs**: RKNPU does 16 syncs (pre/post inference) vs Rocket's 339
   per-layer syncs. This is a major source of Rocket's overhead.

4. **DMA-buf interop**: RKNPU exports buffers via PRIME for zero-copy sharing with
   other subsystems (camera, display). Rocket uses private GEM objects.

5. **ACTION ioctl**: RKNPU has a general-purpose ACTION command for device queries,
   power management, and firmware communication — no equivalent in Rocket.

This IOCTL efficiency explains much of the 4.8x performance gap: Rocket makes ~10x more
kernel transitions per inference, each involving syscall overhead, cache maintenance,
and scheduler interaction.
