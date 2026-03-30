/*
 * Test: Replay captured RKNN register commands via RKNPU driver.
 *
 * Loads pre-captured BO data from /tmp/rknn_dump/, allocates new BOs,
 * patches DMA addresses in regcmd, and submits to NPU.
 *
 * Usage: ./test_replay [input.bin]
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <xf86drm.h>

/* RKNPU structs */
struct rknpu_task {
    uint32_t flags, op_idx, enable_mask, int_mask, int_clear, int_status;
    uint32_t regcfg_amount, regcfg_offset;
    uint64_t regcmd_addr;
} __attribute__((packed));

struct rknpu_subcore_task { uint32_t task_start, task_number; };

struct rknpu_submit {
    uint32_t flags, timeout, task_start, task_number, task_counter;
    int32_t priority;
    uint64_t task_obj_addr;
    uint32_t iommu_domain_id, reserved;
    uint64_t task_base_addr;
    int64_t hw_elapse_time;
    uint32_t core_mask;
    int32_t fence_fd;
    struct rknpu_subcore_task subcore_task[5];
};

struct rknpu_mem_create {
    uint32_t handle, flags;
    uint64_t size, obj_addr, dma_addr, sram_size;
    int32_t iommu_domain_id;
    uint32_t core_mask;
};

struct rknpu_mem_map {
    uint32_t handle, reserved;
    uint64_t offset;
};

struct rknpu_mem_sync {
    uint32_t flags, reserved;
    uint64_t obj_addr, offset, size;
};

/* Use DRM_IOCTL_BASE macros for proper ioctl numbers */
#include <drm/drm.h>
#define RKNPU_IOCTL(nr, type)  _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + (nr), type)
#define IOCTL_ACTION      RKNPU_IOCTL(0x00, struct { uint32_t f; uint32_t v; })
#define IOCTL_SUBMIT      RKNPU_IOCTL(0x01, struct rknpu_submit)
#define IOCTL_MEM_CREATE  RKNPU_IOCTL(0x02, struct rknpu_mem_create)
#define IOCTL_MEM_MAP     RKNPU_IOCTL(0x03, struct rknpu_mem_map)
#define IOCTL_MEM_SYNC    RKNPU_IOCTL(0x05, struct rknpu_mem_sync)

struct bo {
    uint32_t handle;
    uint64_t dma_addr;
    uint64_t obj_addr;
    uint64_t size;
    void *map;
};

static int create_bo(int fd, struct bo *bo, uint64_t size)
{
    struct rknpu_mem_create mc = { .size = size, .flags = 0x403 };
    if (ioctl(fd, IOCTL_MEM_CREATE, &mc)) return -1;
    bo->handle = mc.handle;
    bo->dma_addr = mc.dma_addr;
    bo->obj_addr = mc.obj_addr;
    bo->size = size;

    struct rknpu_mem_map mm = { .handle = mc.handle };
    if (ioctl(fd, IOCTL_MEM_MAP, &mm)) return -1;
    bo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
    if (bo->map == MAP_FAILED) return -1;
    memset(bo->map, 0, size);
    return 0;
}

static void sync_bo(int fd, struct bo *bo, int to_device)
{
    struct rknpu_mem_sync ms = {
        .flags = to_device ? 1 : 2,
        .obj_addr = bo->obj_addr,
        .size = bo->size
    };
    ioctl(fd, IOCTL_MEM_SYNC, &ms);
}

static uint8_t *load_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *out_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc(*out_size);
    fread(data, 1, *out_size, f);
    fclose(f);
    return data;
}

/* DMA address registers that need patching */
static const uint16_t dma_regs[] = {
    0x0010, /* PC_BASE_ADDRESS (chain pointer) */
    0x1070, /* CNA_WT_BASE_ADDR */
    0x1110, /* RDMA_SRC_BASE_ADDR */
    0x1184, /* Unknown DMA */
    0x4020, /* DPU_DST_BASE_ADDR */
    0x4080, /* DPU_OUT_CVT_OFFSET (sometimes DMA!) */
    0x5018, /* RDMA related */
    0x5020, /* RDMA_BS_BASE_ADDR */
    0x5038, /* RDMA related */
    0x6070, /* PC related */
    0x701c, /* PC related */
};
#define N_DMA_REGS (sizeof(dma_regs) / sizeof(dma_regs[0]))

static int is_dma_reg(uint16_t reg)
{
    for (unsigned i = 0; i < N_DMA_REGS; i++)
        if (dma_regs[i] == reg) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    const char *dump_dir = "/tmp/rknn_dump";
    const char *input_path = argc > 1 ? argv[1] : "/root/npu-research/grace_hopper_224.bin";
    int dry_run = (argc > 2 && strcmp(argv[2], "--dry-run") == 0);

    /* Open RKNPU device — try card1 first (usually NPU), then card0 */
    int fd = open("/dev/dri/card1", O_RDWR);
    if (fd < 0) {
        fd = open("/dev/dri/card0", O_RDWR);
        if (fd < 0) { perror("open /dev/dri"); return 1; }
        printf("Using /dev/dri/card0\n");
    } else {
        printf("Using /dev/dri/card1\n");
    }

    /* Power on NPU via ACTION IOCTL */
    {
        struct { uint32_t flags; uint32_t value; } action = { .flags = 20, .value = 1 };
        if (ioctl(fd, IOCTL_ACTION, &action)) {
            /* Not fatal — NPU may already be powered on */
            fprintf(stderr, "Note: ACTION power-on returned error (may be OK)\n");
        } else {
            printf("NPU powered on\n");
        }
    }

    /* Parse submit_1.txt to get original BO info */
    uint64_t orig_dma[8] = {0};
    uint64_t orig_size[8] = {0};
    int orig_bo_count = 0;
    {
        char path[256];
        snprintf(path, sizeof(path), "%s/submit_1.txt", dump_dir);
        FILE *f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            int idx;
            unsigned handle;
            unsigned long long dma, obj, sz;
            if (sscanf(line, "bo[%d] handle=%u dma=0x%llx obj=0x%llx size=%llu",
                        &idx, &handle, &dma, &obj, &sz) == 5) {
                if (idx < 8) {
                    orig_dma[idx] = dma;
                    orig_size[idx] = sz;
                    orig_bo_count = idx + 1;
                }
            }
        }
        fclose(f);
    }
    printf("Original BOs: %d\n", orig_bo_count);
    for (int i = 0; i < orig_bo_count; i++)
        printf("  BO[%d]: dma=0x%llx size=%llu\n", i,
               (unsigned long long)orig_dma[i], (unsigned long long)orig_size[i]);

    /* Allocate new BOs with same sizes */
    struct bo bos[8];
    for (int i = 0; i < orig_bo_count; i++) {
        if (create_bo(fd, &bos[i], orig_size[i]) < 0) {
            fprintf(stderr, "Failed to create BO[%d] size=%llu\n",
                    i, (unsigned long long)orig_size[i]);
            return 1;
        }
        printf("  New BO[%d]: dma=0x%llx\n", i, (unsigned long long)bos[i].dma_addr);
    }

    /* Load captured BO[1] (weights + regcmd) into our BO[1] */
    {
        char path[256];
        snprintf(path, sizeof(path), "%s/sub1_bo_001_%lluB.bin",
                 dump_dir, (unsigned long long)orig_size[1]);
        size_t sz;
        uint8_t *data = load_file(path, &sz);
        if (!data || sz != orig_size[1]) {
            fprintf(stderr, "Failed to load BO[1] from %s\n", path);
            return 1;
        }
        memcpy(bos[1].map, data, sz);
        free(data);
        printf("Loaded BO[1]: %zu bytes (weights + regcmd)\n", sz);
    }

    /* Patch DMA addresses in BO[1]'s regcmd region ONLY.
     * Regcmd starts after the weight data. Find the offset from task[0].regcmd_addr. */
    uint32_t rc_region_start = 0;
    {
        /* Parse first task to find regcmd offset within BO[1] */
        char path[256];
        snprintf(path, sizeof(path), "%s/sub1_tasks.txt", dump_dir);
        FILE *f = fopen(path, "r");
        char line[512];
        while (f && fgets(line, sizeof(line), f)) {
            unsigned long long rc_addr;
            if (sscanf(line, "task[0] regcmd_addr=0x%llx", &rc_addr) == 1) {
                rc_region_start = (uint32_t)rc_addr - (uint32_t)orig_dma[1];
                break;
            }
        }
        if (f) fclose(f);
    }
    printf("Regcmd region starts at BO[1]+0x%x\n", rc_region_start);

    {
        uint64_t *entries = (uint64_t *)((uint8_t *)bos[1].map + rc_region_start);
        unsigned n = (orig_size[1] - rc_region_start) / 8;
        unsigned patched = 0;

        for (unsigned i = 0; i < n; i++) {
            uint16_t reg = entries[i] & 0xFFFF;
            uint32_t val = (entries[i] >> 16) & 0xFFFFFFFF;

            if (!is_dma_reg(reg)) continue;
            if (val == 0) continue; /* Skip zero values */

            /* Check which original BO this address belongs to */
            uint32_t new_val = val;
            for (int b = 0; b < orig_bo_count; b++) {
                if (val >= (uint32_t)orig_dma[b] &&
                    val < (uint32_t)(orig_dma[b] + orig_size[b])) {
                    uint32_t offset = val - (uint32_t)orig_dma[b];
                    new_val = (uint32_t)bos[b].dma_addr + offset;
                    break;
                }
            }

            if (new_val != val) {
                entries[i] = (entries[i] & 0xFFFF000000000000ULL) |
                             ((uint64_t)new_val << 16) |
                             (entries[i] & 0xFFFF);
                patched++;
            }
        }
        printf("Patched %u DMA addresses in regcmd\n", patched);
    }

    /* Load input image into BO[3] */
    {
        size_t inp_sz;
        uint8_t *inp = load_file(input_path, &inp_sz);
        if (!inp) { fprintf(stderr, "Cannot load %s\n", input_path); return 1; }
        if (inp_sz > orig_size[3]) inp_sz = orig_size[3];
        memcpy(bos[3].map, inp, inp_sz);
        free(inp);
        printf("Loaded input: %zu bytes\n", inp_sz);
    }

    /* Build task array in BO[0] */
    /* Parse original task info and patch regcmd_addr */
    unsigned task_count = 0;
    {
        char path[256];
        snprintf(path, sizeof(path), "%s/sub1_tasks.txt", dump_dir);
        FILE *f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Cannot open tasks\n"); return 1; }

        struct rknpu_task *tasks = (struct rknpu_task *)bos[0].map;
        char line[512];
        unsigned cur_task = 0;
        int in_task = 0;

        while (fgets(line, sizeof(line), f)) {
            unsigned tidx, amt, flags, op_idx;
            unsigned long long rc_addr;
            if (sscanf(line, "task[%u] regcmd_addr=0x%llx regcfg_amount=%u flags=0x%x op_idx=%u",
                        &tidx, &rc_addr, &amt, &flags, &op_idx) == 5) {
                /* Patch regcmd_addr: it pointed into old BO[1], remap to new BO[1] */
                uint32_t rc_offset = (uint32_t)rc_addr - (uint32_t)orig_dma[1];
                uint64_t new_rc_addr = bos[1].dma_addr + rc_offset;

                tasks[tidx].flags = flags;
                tasks[tidx].op_idx = op_idx;
                tasks[tidx].enable_mask = 0x70001;
                tasks[tidx].int_mask = 0x300;
                tasks[tidx].int_clear = 0x1FFFF;
                tasks[tidx].int_status = 0;
                tasks[tidx].regcfg_amount = amt;
                tasks[tidx].regcfg_offset = 0;
                tasks[tidx].regcmd_addr = new_rc_addr;
                task_count = tidx + 1;
            }
        }
        fclose(f);
        printf("Built %u tasks\n", task_count);
    }

    if (dry_run) {
        printf("\n=== DRY RUN — not submitting to NPU ===\n");
        printf("Tasks: %u, BOs: %d, regcmd at BO[1]+0x%x\n",
               task_count, orig_bo_count, rc_region_start);
        for (int i = 0; i < orig_bo_count; i++)
            munmap(bos[i].map, bos[i].size);
        close(fd);
        return 0;
    }

    /* Sync all BOs to device */
    for (int i = 0; i < orig_bo_count; i++)
        sync_bo(fd, &bos[i], 1);

    /* Submit HW tasks */
    struct rknpu_submit submit = {
        .flags = 0x5,  /* HW tasks */
        .timeout = 6000,
        .task_start = 0,
        .task_number = task_count,
        .task_counter = 0,
        .priority = 0,
        .task_obj_addr = bos[0].obj_addr,
        .iommu_domain_id = 0,
        .task_base_addr = 0,
        .core_mask = 0x1,
        .fence_fd = -1,
    };
    for (int i = 0; i < 5; i++) {
        submit.subcore_task[i].task_start = 0;
        submit.subcore_task[i].task_number = task_count;
    }

    printf("Submitting %u tasks...\n", task_count);
    int ret = ioctl(fd, IOCTL_SUBMIT, &submit);
    printf("Submit returned %d (hw_time=%lld ms)\n", ret,
           (long long)submit.hw_elapse_time / 1000);

    if (ret == 0) {
        /* Read output from activation BO (BO[2]) */
        sync_bo(fd, &bos[2], 0);

        /* For MBv1, the output should be in the activation BO.
         * We need to find where — let's dump the first few values
         * and compare with RKNN golden output. */
        printf("\nActivation BO[2] first 64 bytes:\n");
        uint8_t *act = (uint8_t *)bos[2].map;
        for (int i = 0; i < 64; i++) printf("%02x ", act[i]);
        printf("\n");

        /* Also check BO[4] (small output BO) */
        sync_bo(fd, &bos[4], 0);
        printf("\nBO[4] first 64 bytes:\n");
        uint8_t *bo4 = (uint8_t *)bos[4].map;
        for (int i = 0; i < 64; i++) printf("%02x ", bo4[i]);
        printf("\n");

        /* Search activation BO for non-zero regions */
        unsigned first_nz = 0, last_nz = 0;
        for (unsigned i = 0; i < orig_size[2]; i++) {
            if (act[i] != 0) {
                if (!first_nz) first_nz = i;
                last_nz = i;
            }
        }
        printf("\nActivation BO: first non-zero at 0x%x, last at 0x%x\n",
               first_nz, last_nz);
    }

    /* Cleanup */
    for (int i = 0; i < orig_bo_count; i++)
        munmap(bos[i].map, bos[i].size);
    close(fd);
    return ret;
}
