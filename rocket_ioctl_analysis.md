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
