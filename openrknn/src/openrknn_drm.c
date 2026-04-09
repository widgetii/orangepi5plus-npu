/*
 * openrknn — RKNPU kernel driver interface (from scratch)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>

/* ======================================================================
 * RKNPU ioctl structures — must match vendor kernel exactly
 * ====================================================================== */

struct rknpu_action {
    uint32_t flags;
    uint32_t value;
};

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

/* Task descriptor: 40 bytes packed */
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

/* ioctl codes */
#define DRM_IOCTL_BASE      'd'
#define DRM_COMMAND_BASE     0x40

#define RKNPU_POWER_ON       20
#define RKNPU_JOB_PC         (1 << 0)
#define RKNPU_JOB_BLOCK      0
#define RKNPU_JOB_PINGPONG   (1 << 2)
#define RKNPU_MEM_NON_CONTIGUOUS      (1 << 0)
#define RKNPU_MEM_CACHEABLE           (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING      (1 << 3)
#define RKNPU_MEM_IOMMU_LIMIT_IOVA   (1 << 10)

#define DRM_IOCTL_RKNPU_ACTION \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x00, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC \
    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

/* ======================================================================
 * Device open
 * ====================================================================== */

int orknn_drm_open(void)
{
    /* Scan /dev/dri/ for RKNPU render node */
    DIR *dir = opendir("/dev/dri");
    if (!dir) {
        orknn_log(0, "drm: cannot open /dev/dri");
        return -1;
    }

    int fd = -1;
    struct dirent *ent;
    while ((ent = readdir(dir))) {
        if (strncmp(ent->d_name, "renderD", 7) != 0)
            continue;
        /* Check sysfs to identify RKNPU device */
        char sysfs[128];
        snprintf(sysfs, sizeof(sysfs), "/sys/class/drm/%s/device/uevent", ent->d_name);
        FILE *f = fopen(sysfs, "r");
        if (!f) continue;
        char buf[256];
        int is_npu = 0;
        while (fgets(buf, sizeof(buf), f))
            if (strstr(buf, "RKNPU") || strstr(buf, "rknpu"))
                is_npu = 1;
        fclose(f);
        if (!is_npu) continue;

        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/%s", ent->d_name);
        int try_fd = open(path, O_RDWR);
        if (try_fd < 0) continue;

        /* Power on */
        struct rknpu_action act = { .flags = RKNPU_POWER_ON };
        ioctl(try_fd, DRM_IOCTL_RKNPU_ACTION, &act);

        /* Verify with a small MEM_CREATE */
        struct rknpu_mem_create test_mc = { .size = 4096, .flags = 0x403 };
        if (ioctl(try_fd, DRM_IOCTL_RKNPU_MEM_CREATE, &test_mc) == 0) {
            /* Clean up test alloc */
            struct rknpu_mem_destroy md = {
                .handle = test_mc.handle, .obj_addr = test_mc.obj_addr
            };
            ioctl(try_fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);

            fd = try_fd;
            orknn_log(1, "drm: opened %s (RKNPU)", path);
            break;
        }
        close(try_fd);
    }
    closedir(dir);

    if (fd < 0)
        orknn_log(0, "drm: no RKNPU device found");
    return fd;
}

/* ======================================================================
 * BO operations
 * ====================================================================== */

int orknn_bo_create_flags(int fd, uint32_t size, uint32_t flags, struct orknn_bo *bo)
{
    memset(bo, 0, sizeof(*bo));
    bo->size = size;

    struct rknpu_mem_create mc = {
        .size = size,
        .flags = flags,
        .iommu_domain_id = 0,
    };
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc)) {
        orknn_log(0, "drm: MEM_CREATE failed: %s (size=%u)", strerror(errno), size);
        return -1;
    }
    bo->handle = mc.handle;
    bo->dma_addr = mc.dma_addr;
    bo->obj_addr = mc.obj_addr;

    struct rknpu_mem_map mm = { .handle = mc.handle };
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) {
        orknn_log(0, "drm: MEM_MAP failed: %s", strerror(errno));
        orknn_bo_destroy(fd, bo);
        return -1;
    }
    bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
    if (bo->map == MAP_FAILED) {
        orknn_log(0, "drm: mmap failed: %s", strerror(errno));
        bo->map = NULL;
        orknn_bo_destroy(fd, bo);
        return -1;
    }
    return 0;
}

int orknn_bo_create(int fd, uint32_t size, struct orknn_bo *bo)
{
    return orknn_bo_create_flags(fd, size, 0x403, bo);
}

void orknn_bo_destroy(int fd, struct orknn_bo *bo)
{
    if (bo->map) {
        munmap(bo->map, bo->size);
        bo->map = NULL;
    }
    if (bo->handle) {
        struct rknpu_mem_destroy md = {
            .handle = bo->handle,
            .obj_addr = bo->obj_addr,
        };
        ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);
        bo->handle = 0;
    }
}

int orknn_bo_sync_to_device(int fd, struct orknn_bo *bo)
{
    struct rknpu_mem_sync ms = {
        .flags = 1, /* TO_DEVICE */
        .obj_addr = bo->obj_addr,
        .size = bo->size,
    };
    return ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
}

int orknn_bo_sync_from_device(int fd, struct orknn_bo *bo)
{
    struct rknpu_mem_sync ms = {
        .flags = 2, /* FROM_DEVICE */
        .obj_addr = bo->obj_addr,
        .size = bo->size,
    };
    return ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
}

/* ======================================================================
 * NPU task submission
 * ====================================================================== */

int orknn_npu_submit(int fd, struct orknn_bo *task_bo,
                     struct orknn_segment *seg)
{
    struct rknpu_submit sub = {
        .flags = seg->flags | RKNPU_JOB_BLOCK,
        .timeout = 6000,
        .task_start = seg->sc_start,
        .task_number = seg->task_number,
        .task_obj_addr = task_bo->obj_addr,
        .core_mask = 0x0, /* auto — matches proxy */
        .fence_fd = -1,
        .subcore_task = {
            { seg->sc_start, seg->sc_count },
            { seg->sc_start, seg->sc_count },
            { seg->sc_start, seg->sc_count },
            { seg->sc_start, seg->sc_count },
            { seg->sc_start, seg->sc_count },
        },
    };

    int ret;
    do {
        ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

    if (ret)
        orknn_log(0, "drm: SUBMIT failed: %s (tasks=%u sc={%u,%u})",
                  strerror(errno), seg->task_number, seg->sc_start, seg->sc_count);

    return ret;
}
