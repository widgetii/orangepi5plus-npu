/*
 * verify_intercept.c — LD_PRELOAD interceptor for RKNN golden state capture
 *
 * Captures COMPLETE DMA BO state at every checkpoint during RKNN inference:
 * - post_init: after all MEM_CREATE calls
 * - pre_submit_N: before each SUBMIT ioctl
 * - post_submit_N: after each SUBMIT returns (with MEM_SYNC invalidate!)
 *
 * Output: /tmp/verify_golden/
 *   meta.txt, post_init_bo_NNN.bin, pre_submit_N_bo_NNN.bin, etc.
 *
 * Usage:
 *   mkdir -p /tmp/verify_golden
 *   VERIFY_GOLDEN=1 LD_PRELOAD=./verify_intercept.so python3 -c "..."
 *
 * Build:
 *   gcc -shared -fPIC -o verify_intercept.so verify_intercept.c -ldl
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

/* RKNPU ioctl structs (from rnpu_drm.c) */
#define IOCTL_MEM_CREATE  0xc0306442
#define IOCTL_MEM_MAP     0xc0106443
#define IOCTL_MEM_SYNC    0xc0186445
#define IOCTL_SUBMIT      0xc0686441

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
    uint32_t pad;
    uint64_t offset;
};

struct rknpu_mem_sync {
    uint32_t flags;
    uint32_t reserved;
    uint64_t obj_addr;
    uint64_t offset;
    uint64_t size;
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
    struct { uint32_t task_start; uint32_t task_number; } subcore_task[5];
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
};

/* State */
static int drm_fd = -1;
static int submit_count = 0;
static int (*real_ioctl)(int, unsigned long, ...) = NULL;
static const char *golden_dir = NULL;

/* BO tracking */
#define MAX_BOS 64
static struct {
    uint64_t obj;
    uint32_t handle;
    uint64_t dma;
    uint64_t size;
    uint32_t flags;
    void *map;          /* lazily mmapped */
    uint64_t map_offset; /* mmap offset from MEM_MAP */
} bo_table[MAX_BOS];
static int bo_count = 0;

static void init_real_ioctl(void) {
    if (!real_ioctl)
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
}

/* Map a BO for reading (lazy, cached) */
static void *map_bo(int fd, int idx) {
    if (bo_table[idx].map)
        return bo_table[idx].map;

    struct rknpu_mem_map mm = { .handle = bo_table[idx].handle };
    if (real_ioctl(fd, IOCTL_MEM_MAP, &mm) != 0)
        return NULL;

    bo_table[idx].map_offset = mm.offset;
    bo_table[idx].map = mmap(NULL, bo_table[idx].size,
                              PROT_READ | PROT_WRITE, MAP_SHARED,
                              fd, mm.offset);
    if (bo_table[idx].map == MAP_FAILED) {
        bo_table[idx].map = NULL;
        return NULL;
    }
    return bo_table[idx].map;
}

/* Invalidate cache for a BO (MEM_SYNC flags=2) */
static void sync_bo_from_device(int fd, int idx) {
    struct rknpu_mem_sync ms = {
        .flags = 2, /* SYNC_FROM_DEVICE */
        .obj_addr = bo_table[idx].obj,
        .size = bo_table[idx].size,
    };
    real_ioctl(fd, IOCTL_MEM_SYNC, &ms);
}

/* Dump all BOs to files with a given prefix */
static void dump_all_bos(int fd, const char *prefix) {
    char path[256];
    for (int i = 0; i < bo_count; i++) {
        void *map = map_bo(fd, i);
        if (!map) continue;

        snprintf(path, sizeof(path), "%s/%s_bo_%03d_%lluB.bin",
                 golden_dir, prefix, i, (unsigned long long)bo_table[i].size);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(map, 1, bo_table[i].size, f);
            fclose(f);
        }
    }
}

/* Write metadata file */
static void write_meta(void) {
    char path[256];
    snprintf(path, sizeof(path), "%s/meta.txt", golden_dir);
    FILE *f = fopen(path, "a"); /* append mode for incremental writes */
    if (!f) return;

    fprintf(f, "bo_count=%d\n", bo_count);
    for (int i = 0; i < bo_count; i++) {
        fprintf(f, "bo[%d] handle=%u dma=0x%llx obj=0x%llx size=%llu flags=0x%x\n",
                i, bo_table[i].handle,
                (unsigned long long)bo_table[i].dma,
                (unsigned long long)bo_table[i].obj,
                (unsigned long long)bo_table[i].size,
                bo_table[i].flags);
    }
    fclose(f);
}

/* Dump submit descriptor */
static void dump_submit(const struct rknpu_submit *s, int idx) {
    char path[256];
    snprintf(path, sizeof(path), "%s/submit_%d_descriptor.bin", golden_dir, idx);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(s, 1, sizeof(*s), f); fclose(f); }

    /* Also write human-readable */
    snprintf(path, sizeof(path), "%s/submit_%d.txt", golden_dir, idx);
    f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "submit=%d flags=0x%x timeout=%u task_start=%u task_number=%u\n",
            idx, s->flags, s->timeout, s->task_start, s->task_number);
    fprintf(f, "task_counter=%u priority=%d task_obj_addr=0x%llx\n",
            s->task_counter, s->priority, (unsigned long long)s->task_obj_addr);
    fprintf(f, "task_base_addr=0x%llx core_mask=0x%x fence_fd=%d\n",
            (unsigned long long)s->task_base_addr, s->core_mask, s->fence_fd);
    for (int i = 0; i < 5; i++)
        fprintf(f, "subcore[%d] start=%u count=%u\n",
                i, s->subcore_task[i].task_start, s->subcore_task[i].task_number);

    /* Dump task descriptors from the task BO */
    for (int bi = 0; bi < bo_count; bi++) {
        if (bo_table[bi].obj == s->task_obj_addr) {
            void *map = map_bo(drm_fd, bi);
            if (!map) break;
            struct rknpu_task *tasks = (struct rknpu_task *)map;
            int n_tasks = bo_table[bi].size / sizeof(struct rknpu_task);
            if (n_tasks > (int)s->task_number) n_tasks = s->task_number;
            fprintf(f, "\ntasks (%d):\n", n_tasks);
            for (int t = 0; t < n_tasks; t++) {
                fprintf(f, "  [%2d] op=%u en=0x%x amt=%u addr=0x%llx\n",
                        t, tasks[t].op_idx, tasks[t].enable_mask,
                        tasks[t].regcfg_amount,
                        (unsigned long long)tasks[t].regcmd_addr);
            }
            break;
        }
    }
    fclose(f);
}

int ioctl(int fd, unsigned long request, ...) {
    init_real_ioctl();

    va_list ap;
    __builtin_va_start(ap, request);
    void *arg = __builtin_va_arg(ap, void *);
    __builtin_va_end(ap);

    if (!golden_dir) {
        golden_dir = getenv("VERIFY_GOLDEN");
        if (!golden_dir) golden_dir = "/tmp/verify_golden";
    }

    /* Track MEM_CREATE */
    if (request == IOCTL_MEM_CREATE) {
        int ret = real_ioctl(fd, request, arg);
        if (ret == 0 && bo_count < MAX_BOS) {
            struct rknpu_mem_create *mc = arg;
            bo_table[bo_count].obj = mc->obj_addr;
            bo_table[bo_count].handle = mc->handle;
            bo_table[bo_count].dma = mc->dma_addr;
            bo_table[bo_count].size = mc->size;
            bo_table[bo_count].flags = mc->flags;
            bo_table[bo_count].map = NULL;
            fprintf(stderr, "VERIFY: MEM_CREATE[%d] handle=%u dma=0x%llx size=%llu flags=0x%x\n",
                    bo_count, mc->handle,
                    (unsigned long long)mc->dma_addr,
                    (unsigned long long)mc->size, mc->flags);
            bo_count++;
            if (drm_fd < 0) drm_fd = fd;
        }
        return ret;
    }

    /* Intercept SUBMIT */
    if (request == IOCTL_SUBMIT && getenv("VERIFY_GOLDEN")) {
        struct rknpu_submit *s = (struct rknpu_submit *)arg;
        submit_count++;

        if (drm_fd < 0) drm_fd = fd;

        /* Checkpoint: pre_submit */
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "pre_submit_%d", submit_count);
        dump_all_bos(fd, prefix);
        dump_submit(s, submit_count);

        if (submit_count == 1)
            write_meta();

        fprintf(stderr, "VERIFY: SUBMIT[%d] flags=0x%x tasks=%u sc={%u,%u}\n",
                submit_count, s->flags, s->task_number,
                s->subcore_task[0].task_start, s->subcore_task[0].task_number);

        /* Call real submit (blocking) */
        int ret = real_ioctl(fd, request, arg);

        /* Checkpoint: post_submit — invalidate ALL BOs first! */
        for (int i = 0; i < bo_count; i++)
            sync_bo_from_device(fd, i);

        snprintf(prefix, sizeof(prefix), "post_submit_%d", submit_count);
        dump_all_bos(fd, prefix);

        fprintf(stderr, "VERIFY: SUBMIT[%d] returned %d (elapse=%lld us)\n",
                submit_count, ret, (long long)s->hw_elapse_time);

        return ret;
    }

    return real_ioctl(fd, request, arg);
}
