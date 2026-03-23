/*
 * DRM/ioctl wrapper — supports both Rocket (upstream) and RKNPU (vendor)
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

/* ======================================================================
 * Rocket UAPI — matches kernel's rocket_accel.h
 * ====================================================================== */

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

/* ======================================================================
 * RKNPU UAPI — matches vendor kernel's rknpu_ioctl.h
 * ====================================================================== */

struct rknpu_mem_create {
   uint32_t handle;
   uint32_t flags;
   uint64_t size;
   uint64_t obj_addr;
   uint64_t dma_addr;
   uint64_t sram_size;
   int32_t  iommu_domain_id;
   uint32_t core_mask;
};

struct rknpu_mem_map {
   uint32_t handle;
   uint32_t reserved;
   uint64_t offset;
};

struct rknpu_mem_destroy {
   uint32_t handle;
   uint32_t reserved;
   uint64_t obj_addr;
};

struct rknpu_mem_sync {
   uint32_t flags;
   uint32_t reserved;
   uint64_t obj_addr;
   uint64_t offset;
   uint64_t size;
};

struct rknpu_task {
   uint32_t flags;
   uint32_t op_idx;
   uint32_t enable_mask;
   uint32_t int_mask;
   uint32_t int_clear;
   uint32_t int_status;
   uint32_t regcfg_amount;
   uint32_t regcfg_offset;
   uint64_t regcmd_addr;
} __attribute__((packed));

struct rknpu_subcore_task {
   uint32_t task_start;
   uint32_t task_number;
};

struct rknpu_submit {
   uint32_t flags;
   uint32_t timeout;
   uint32_t task_start;
   uint32_t task_number;
   uint32_t task_counter;
   int32_t  priority;
   uint64_t task_obj_addr;
   uint32_t iommu_domain_id;
   uint32_t reserved;
   uint64_t task_base_addr;
   int64_t  hw_elapse_time;
   uint32_t core_mask;
   int32_t  fence_fd;
   struct rknpu_subcore_task subcore_task[5];
};

#define RKNPU_JOB_PC       (1 << 0)
#define RKNPU_JOB_BLOCK    0
#define RKNPU_JOB_PINGPONG (1 << 2)
#define RKNPU_MEM_NON_CONTIGUOUS (1 << 0)
#define RKNPU_MEM_CACHEABLE      (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING (1 << 3)
#define RKNPU_MEM_IOMMU          (1 << 4)
#define RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT (1 << 10)

struct rknpu_action { uint32_t flags; uint32_t value; };
#define RKNPU_POWER_ON  20
#define DRM_IOCTL_RKNPU_ACTION      _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x00, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT      _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

#define RKNPU_PC_DATA_EXTRA_AMOUNT 4

/* ======================================================================
 * Driver state
 * ====================================================================== */

#include "rnpu_drm.h"

enum rnpu_driver_type rnpu_active_driver = RNPU_DRIVER_ROCKET;

/* ======================================================================
 * BO operations — dual-driver
 * ====================================================================== */

int rnpu_bo_create(int fd, uint32_t size, struct rnpu_bo *bo)
{
   memset(bo, 0, sizeof(*bo));
   bo->size = size;

   if (rnpu_active_driver == RNPU_DRIVER_ROCKET) {
      struct drm_rocket_create_bo req = { .size = size };
      int ret = ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &req);
      if (ret) {
         fprintf(stderr, "rnpu: CREATE_BO failed: %s (size=%u)\n",
                 strerror(errno), size);
         return -1;
      }
      bo->handle = req.handle;
      bo->dma_addr = req.dma_address;
      bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, req.offset);
   } else {
      struct rknpu_mem_create mc = {
         .size = size,
         .flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE
                  | RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT,
         .iommu_domain_id = 0,
      };
      if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc)) {
         fprintf(stderr, "rnpu: RKNPU MEM_CREATE failed: %s (size=%u)\n",
                 strerror(errno), size);
         return -1;
      }
      bo->handle = mc.handle;
      bo->dma_addr = mc.dma_addr;
      bo->obj_addr = mc.obj_addr;

      struct rknpu_mem_map mm = { .handle = mc.handle };
      if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) {
         fprintf(stderr, "rnpu: RKNPU MEM_MAP failed: %s\n", strerror(errno));
         struct rknpu_mem_destroy md = { .handle = mc.handle, .obj_addr = mc.obj_addr };
         ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);
         return -1;
      }
      bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, mm.offset);
   }

   if (bo->map == MAP_FAILED) {
      fprintf(stderr, "rnpu: mmap failed: %s\n", strerror(errno));
      rnpu_bo_destroy(fd, bo);
      return -1;
   }
   return 0;
}

int rnpu_bo_prep(int fd, struct rnpu_bo *bo)
{
   if (rnpu_active_driver == RNPU_DRIVER_ROCKET) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      int64_t abs_ns = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec + 5000000000LL;
      struct drm_rocket_prep_bo req = {
         .handle = bo->handle,
         .timeout_ns = abs_ns,
      };
      int ret = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &req);
      if (ret > 0) ret = 0;
      return ret;
   } else {
      /* RKNPU BLOCK mode submit already waits — prep is just cache sync */
      struct rknpu_mem_sync ms = {
         .flags = 2, /* SYNC_FROM_DEVICE */
         .obj_addr = bo->obj_addr,
         .size = bo->size,
      };
      ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
      return 0;
   }
}

int rnpu_bo_fini(int fd, struct rnpu_bo *bo)
{
   if (rnpu_active_driver == RNPU_DRIVER_ROCKET) {
      struct drm_rocket_fini_bo req = { .handle = bo->handle };
      return ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &req);
   } else {
      struct rknpu_mem_sync ms = {
         .flags = 1, /* SYNC_TO_DEVICE */
         .obj_addr = bo->obj_addr,
         .size = bo->size,
      };
      return ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
   }
}

void rnpu_bo_destroy(int fd, struct rnpu_bo *bo)
{
   if (bo->map && bo->map != MAP_FAILED) {
      munmap(bo->map, bo->size);
      bo->map = NULL;
   }
   if (bo->handle) {
      if (rnpu_active_driver == RNPU_DRIVER_ROCKET) {
         struct drm_gem_close req = { .handle = bo->handle };
         ioctl(fd, DRM_IOCTL_GEM_CLOSE, &req);
      } else {
         struct rknpu_mem_destroy md = {
            .handle = bo->handle,
            .obj_addr = bo->obj_addr,
         };
         ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);
      }
      bo->handle = 0;
   }
}

/* ======================================================================
 * Job submission — dual-driver
 * ====================================================================== */

int rnpu_submit(int fd, struct drm_rocket_job *jobs, uint32_t job_count)
{
   if (rnpu_active_driver == RNPU_DRIVER_ROCKET) {
      struct drm_rocket_submit req = {
         .jobs = (uint64_t)(uintptr_t)jobs,
         .job_count = job_count,
         .job_struct_size = sizeof(struct drm_rocket_job),
      };
      int ret;
      do {
         ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &req);
      } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
      if (ret) {
         fprintf(stderr, "rnpu: SUBMIT failed: %s (jobs=%u)\n",
                 strerror(errno), job_count);
      }
      return ret;
   }

   /* RKNPU path: submit each job in blocking mode.
    * Uses a cached task BO to avoid per-job alloc/free overhead. */

   /* Ensure NPU is powered on (runtime PM) */
   static int npu_powered = 0;
   if (!npu_powered) {
      struct rknpu_action act = { .flags = RKNPU_POWER_ON };
      ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
      npu_powered = 1;
   }

   /* Cached task BO — grows as needed, never shrinks */
   static struct {
      uint32_t handle;
      uint64_t obj_addr;
      void *map;
      uint32_t size;
      int fd;
   } task_bo_cache = {0};

   /* Find max tasks across all jobs to size the cached BO */
   uint32_t max_tasks = 0;
   for (uint32_t j = 0; j < job_count; j++)
      if (jobs[j].task_count > max_tasks)
         max_tasks = jobs[j].task_count;

   uint32_t needed_size = ((max_tasks * sizeof(struct rknpu_task)) + 4095) & ~4095u;
   if (needed_size < 4096) needed_size = 4096;

   /* Grow cached BO if needed or if fd changed */
   if (task_bo_cache.size < needed_size || task_bo_cache.fd != fd) {
      if (task_bo_cache.map) {
         munmap(task_bo_cache.map, task_bo_cache.size);
         struct rknpu_mem_destroy tmd = {
            .handle = task_bo_cache.handle,
            .obj_addr = task_bo_cache.obj_addr
         };
         ioctl(task_bo_cache.fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &tmd);
         memset(&task_bo_cache, 0, sizeof(task_bo_cache));
      }
      struct rknpu_mem_create tmc = {
         .size = needed_size,
         .flags = RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE
                  | RKNPU_MEM_KERNEL_MAPPING | RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT,
         .iommu_domain_id = 0,
      };
      if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &tmc)) {
         fprintf(stderr, "rnpu: RKNPU task BO alloc failed (size=%u)\n", needed_size);
         return -1;
      }
      struct rknpu_mem_map tmm = { .handle = tmc.handle };
      if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &tmm)) {
         struct rknpu_mem_destroy tmd = { .handle = tmc.handle, .obj_addr = tmc.obj_addr };
         ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &tmd);
         return -1;
      }
      void *tmap = mmap(NULL, needed_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, tmm.offset);
      if (tmap == MAP_FAILED) {
         struct rknpu_mem_destroy tmd = { .handle = tmc.handle, .obj_addr = tmc.obj_addr };
         ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &tmd);
         return -1;
      }
      task_bo_cache.handle = tmc.handle;
      task_bo_cache.obj_addr = tmc.obj_addr;
      task_bo_cache.map = tmap;
      task_bo_cache.size = needed_size;
      task_bo_cache.fd = fd;
   }

   for (uint32_t j = 0; j < job_count; j++) {
      struct drm_rocket_job *job = &jobs[j];
      uint32_t ntasks = job->task_count;

      /* Fill task descriptors into cached BO */
      struct rknpu_task *task = (struct rknpu_task *)task_bo_cache.map;
      for (uint32_t t = 0; t < ntasks; t++) {
         struct drm_rocket_task *rt =
            &((struct drm_rocket_task *)(uintptr_t)job->tasks)[t];
         struct rknpu_task *tp = &task[t];
         memset(tp, 0, sizeof(*tp));
         tp->op_idx = j;
         tp->enable_mask = 0xf;
         tp->int_mask = 0x300;
         tp->int_clear = 0x1ffff;
         tp->regcfg_amount = rt->regcmd_count - RKNPU_PC_DATA_EXTRA_AMOUNT;
         tp->regcmd_addr = (uint64_t)rt->regcmd;
      }

      struct rknpu_submit submit = {
         .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
         .timeout = 6000,
         .task_number = ntasks,
         .task_obj_addr = task_bo_cache.obj_addr,
         .core_mask = 0x0,
         .fence_fd = -1,
         .subcore_task = { {0, ntasks}, {0, ntasks}, {0, ntasks}, {0, ntasks}, {0, ntasks} },
      };

      int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
      if (ret) {
         fprintf(stderr, "rnpu: RKNPU SUBMIT failed: %s (job %u/%u, tasks=%u)\n",
                 strerror(errno), j, job_count, ntasks);
         return ret;
      }
   }

   return 0;
}
