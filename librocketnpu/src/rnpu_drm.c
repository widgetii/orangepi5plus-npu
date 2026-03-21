/*
 * DRM/ioctl wrapper for Rocket NPU — 4 ioctls only
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>

#include <drm/drm.h>

/* Rocket UAPI — matches kernel's rocket_accel.h */
#define DRM_ROCKET_CREATE_BO  0x00
#define DRM_ROCKET_SUBMIT     0x01
#define DRM_ROCKET_PREP_BO    0x02
#define DRM_ROCKET_FINI_BO    0x03

#define DRM_IOCTL_ROCKET_CREATE_BO \
   DRM_IOWR(DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT \
   DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO \
   DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO \
   DRM_IOW(DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, struct drm_rocket_fini_bo)

struct drm_rocket_create_bo {
   uint32_t size;
   uint32_t handle;
   uint64_t dma_address;
   uint64_t offset;
};

struct drm_rocket_prep_bo {
   uint32_t handle;
   uint32_t reserved;
   int64_t timeout_ns;
};

struct drm_rocket_fini_bo {
   uint32_t handle;
   uint32_t reserved;
};

struct drm_rocket_task {
   uint32_t regcmd;
   uint32_t regcmd_count;
};

struct drm_rocket_job {
   uint64_t tasks;
   uint64_t in_bo_handles;
   uint64_t out_bo_handles;
   uint32_t task_count;
   uint32_t task_struct_size;
   uint32_t in_bo_handle_count;
   uint32_t out_bo_handle_count;
};

struct drm_rocket_submit {
   uint64_t jobs;
   uint32_t job_count;
   uint32_t job_struct_size;
   uint64_t reserved;
};

#include "rnpu_drm.h"

int rnpu_bo_create(int fd, uint32_t size, struct rnpu_bo *bo)
{
   struct drm_rocket_create_bo req = { .size = size };
   int ret = ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &req);
   if (ret) {
      fprintf(stderr, "rnpu: CREATE_BO failed: %s (size=%u)\n",
              strerror(errno), size);
      return -1;
   }

   bo->handle = req.handle;
   bo->dma_addr = req.dma_address;
   bo->size = size;

   bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                  fd, req.offset);
   if (bo->map == MAP_FAILED) {
      fprintf(stderr, "rnpu: mmap failed: %s\n", strerror(errno));
      struct drm_gem_close close_req = { .handle = req.handle };
      ioctl(fd, DRM_IOCTL_GEM_CLOSE, &close_req);
      return -1;
   }

   return 0;
}

int rnpu_bo_prep(int fd, struct rnpu_bo *bo)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   int64_t abs_ns = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec + 5000000000LL;
   struct drm_rocket_prep_bo req = {
      .handle = bo->handle,
      .timeout_ns = abs_ns, /* absolute: now + 5s */
   };
   int ret = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &req);
   /* Kernel returns remaining jiffies (>0) on success, -1 on error */
   if (ret > 0) ret = 0;
   return ret;
}

int rnpu_bo_fini(int fd, struct rnpu_bo *bo)
{
   struct drm_rocket_fini_bo req = { .handle = bo->handle };
   return ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &req);
}

void rnpu_bo_destroy(int fd, struct rnpu_bo *bo)
{
   if (bo->map && bo->map != MAP_FAILED) {
      munmap(bo->map, bo->size);
      bo->map = NULL;
   }
   if (bo->handle) {
      struct drm_gem_close req = { .handle = bo->handle };
      ioctl(fd, DRM_IOCTL_GEM_CLOSE, &req);
      bo->handle = 0;
   }
}

int rnpu_submit(int fd, struct drm_rocket_job *jobs, uint32_t job_count)
{
   struct drm_rocket_submit req = {
      .jobs = (uint64_t)(uintptr_t)jobs,
      .job_count = job_count,
      .job_struct_size = sizeof(struct drm_rocket_job),
   };
   /* Use drmIoctl for EINTR/EAGAIN retry (matches Mesa) */
   extern int drmIoctl(int fd, unsigned long request, void *arg);
   int ret = drmIoctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &req);
   if (ret) {
      fprintf(stderr, "rnpu: SUBMIT failed: %s (jobs=%u)\n",
              strerror(errno), job_count);
   }
   return ret;
}
