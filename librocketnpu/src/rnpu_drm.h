/*
 * DRM/ioctl wrapper for Rocket NPU
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_DRM_H
#define RNPU_DRM_H

#include <stdint.h>
#include <stddef.h>

struct rnpu_bo {
   uint32_t handle;
   uint64_t dma_addr;
   void *map;
   uint32_t size;
};

/* 4 Rocket ioctls */
int rnpu_bo_create(int fd, uint32_t size, struct rnpu_bo *bo);
int rnpu_bo_prep(int fd, struct rnpu_bo *bo);
int rnpu_bo_fini(int fd, struct rnpu_bo *bo);
void rnpu_bo_destroy(int fd, struct rnpu_bo *bo);

/* Submit jobs to NPU. Returns 0 on success. */
int rnpu_submit(int fd, struct drm_rocket_job *jobs, uint32_t job_count);

#endif /* RNPU_DRM_H */
