/*
 * DRM/ioctl wrapper for Rocket NPU and RKNPU vendor driver
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_DRM_H
#define RNPU_DRM_H

#include <stdint.h>
#include <stddef.h>

enum rnpu_driver_type {
   RNPU_DRIVER_ROCKET,
   RNPU_DRIVER_RKNPU,
};

/* Global driver type — set by rnpu_open(), used by all DRM functions */
extern enum rnpu_driver_type rnpu_active_driver;

struct rnpu_bo {
   uint32_t handle;
   uint64_t dma_addr;
   uint64_t obj_addr;   /* RKNPU only — kernel object address */
   void *map;
   uint32_t size;
};

/* BO lifecycle */
int rnpu_bo_create(int fd, uint32_t size, struct rnpu_bo *bo);
int rnpu_bo_prep(int fd, struct rnpu_bo *bo);
int rnpu_bo_fini(int fd, struct rnpu_bo *bo);
void rnpu_bo_destroy(int fd, struct rnpu_bo *bo);

/* Submit jobs to NPU. Returns 0 on success. */
int rnpu_submit(int fd, struct drm_rocket_job *jobs, uint32_t job_count);

#endif /* RNPU_DRM_H */
