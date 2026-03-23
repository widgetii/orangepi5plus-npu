/*
 * LD_PRELOAD intercept: hijack RKNN's first SUBMIT to inject our regcmd.
 *
 * Strategy:
 * 1. Let RKNN init normally (all ACTION/MEM_CREATE/etc calls pass through)
 * 2. On the FIRST SUBMIT, intercept the task BO:
 *    - Read RKNN's task_obj_addr to find the task BO's kv_addr
 *    - Create our own regcmd BO with our MobileNetV1 regcmd data
 *    - Overwrite task[0].regcmd_addr to point to our regcmd
 *    - Change task_number to 1 (single task)
 *    - Let the submit proceed
 * 3. If it succeeds, RKNN's init is needed. If it fails, something else is wrong.
 *
 * Build: gcc -shared -fPIC -o intercept_swap.so intercept_swap.c -ldl
 * Usage: LD_PRELOAD=./intercept_swap.so LD_LIBRARY_PATH=... ./bench_rknn model.rknn 1
 *
 * Set SWAP_REGCMD=/path/to/our_regcmd0.bin to inject our regcmd.
 * If not set, just logs RKNN's submit params without swapping.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/* RKNPU structs (must match kernel exactly) */
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

#define IOCTL_MEM_CREATE  0xc0306442
#define IOCTL_MEM_MAP     0xc0106443
#define IOCTL_MEM_SYNC    0xc0206445
#define IOCTL_SUBMIT      0xc0686441

/* State */
static int drm_fd = -1;
static int submit_count = 0;

/* Our injected regcmd BO */
static uint32_t our_rc_handle = 0;
static uint64_t our_rc_dma = 0;
static uint64_t our_rc_obj = 0;
static void *our_rc_map = NULL;
static uint32_t our_rc_entries = 0;

/* RKNN's task BO — we need to find it and mmap it */
static uint64_t rknn_task_obj = 0;
static uint32_t rknn_task_handle = 0;
static void *rknn_task_map = NULL;

/* Track MEM_CREATE results to map obj_addr -> handle */
#define MAX_BOS 512
static struct { uint64_t obj; uint32_t handle; uint64_t dma; uint64_t size; } bo_table[MAX_BOS];
static int bo_count = 0;

static int (*real_ioctl)(int, unsigned long, ...) = NULL;

static void init_real_ioctl(void) {
    if (!real_ioctl)
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
}

static uint64_t create_bo_dma(int fd, uint32_t size) {
    struct rknpu_mem_create mc = { .size = size, .flags = 0x403 };
    if (real_ioctl(fd, IOCTL_MEM_CREATE, &mc)) return 0;
    /* Sync (zeros) to device */
    struct rknpu_mem_sync ms = { .flags = 1, .obj_addr = mc.obj_addr, .size = size };
    real_ioctl(fd, IOCTL_MEM_SYNC, &ms);
    return mc.dma_addr;
}

static int create_our_regcmd(int fd) {
    const char *path = getenv("SWAP_REGCMD");
    if (!path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "SWAP: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *regcmd_data = malloc(sz);
    fread(regcmd_data, 1, sz, f);
    fclose(f);
    our_rc_entries = sz / 8;

    /* Create fresh BOs for weight/activation/bias so DMA addresses are valid.
     * Sizes from MobileNetV1 model. Data is zeros (wrong output but HW should complete). */
    uint64_t act_dma  = create_bo_dma(fd, 11699200); /* activation */
    uint64_t wt_dma   = create_bo_dma(fd, 8438976);  /* weights */
    uint64_t bias_dma = create_bo_dma(fd, 47808);    /* biases */
    fprintf(stderr, "SWAP: Fresh BOs: act=0x%llx wt=0x%llx bias=0x%llx\n",
            (unsigned long long)act_dma, (unsigned long long)wt_dma,
            (unsigned long long)bias_dma);

    /* Patch DMA addresses in regcmd.
     * Our regcmd references: SRC(0x1070)=act, WT(0x1110)=wt, DST(0x4020)=act, BS(0x5020)=bias
     * We need to find and replace the OLD addresses with the new ones.
     * Read the old addresses from the regcmd: */
    uint64_t old_src = 0, old_wt = 0, old_dst = 0, old_bs = 0;
    for (int i = 0; i < (int)(sz/8); i++) {
        uint64_t e; memcpy(&e, regcmd_data + i*8, 8);
        uint16_t a = e & 0xFFFF;
        uint32_t v = (e >> 16) & 0xFFFFFFFF;
        if (a == 0x1070 && !old_src) old_src = v;
        if (a == 0x1110 && !old_wt)  old_wt = v;
        if (a == 0x4020 && !old_dst) old_dst = v;
        if (a == 0x5020 && !old_bs)  old_bs = v;
    }
    fprintf(stderr, "SWAP: Old addrs: src=0x%llx wt=0x%llx dst=0x%llx bs=0x%llx\n",
            (unsigned long long)old_src, (unsigned long long)old_wt,
            (unsigned long long)old_dst, (unsigned long long)old_bs);

    /* Compute offset from base for src/dst (they're offsets into activation BO) */
    /* src_offset = old_src - old_act_base, dst_offset = old_dst - old_act_base
     * We don't know old_act_base exactly, but we can compute offsets from the
     * smallest address among src/dst. Actually, let's just replace ALL values
     * that match old addresses with new ones. */
    for (int i = 0; i < (int)(sz/8); i++) {
        uint64_t e; memcpy(&e, regcmd_data + i*8, 8);
        uint32_t v = (e >> 16) & 0xFFFFFFFF;
        uint32_t nv = v;
        /* Replace weight base */
        if (old_wt && v == (uint32_t)old_wt)
            nv = (uint32_t)wt_dma;
        /* Replace bias base */
        if (old_bs && v == (uint32_t)old_bs)
            nv = (uint32_t)bias_dma;
        /* Replace activation addresses (src/dst are offsets into activation BO) */
        if (old_src && old_dst) {
            uint32_t act_base_old = old_dst < old_src ? old_dst : old_src;
            /* Treat any address within 16MB of act_base_old as activation ref */
            if (v >= act_base_old && v < act_base_old + 16*1024*1024) {
                uint32_t offset = v - act_base_old;
                nv = (uint32_t)act_dma + offset;
            }
        }
        if (nv != v) {
            e = (e & 0xFFFF) | ((uint64_t)nv << 16) | (e & 0xFFFF000000000000ULL);
            memcpy(regcmd_data + i*8, &e, 8);
        }
    }

    /* Create regcmd BO */
    struct rknpu_mem_create mc = { .size = (sz + 4095) & ~4095, .flags = 0x403 };
    if (real_ioctl(fd, IOCTL_MEM_CREATE, &mc)) {
        fprintf(stderr, "SWAP: MEM_CREATE failed\n"); free(regcmd_data); return -1;
    }
    our_rc_handle = mc.handle;
    our_rc_dma = mc.dma_addr;
    our_rc_obj = mc.obj_addr;

    /* Map and fill */
    struct rknpu_mem_map mm = { .handle = mc.handle };
    if (real_ioctl(fd, IOCTL_MEM_MAP, &mm)) {
        free(regcmd_data); return -1;
    }
    our_rc_map = mmap(NULL, mc.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mm.offset);
    if (our_rc_map == MAP_FAILED) {
        free(regcmd_data); return -1;
    }
    memset(our_rc_map, 0, mc.size);
    memcpy(our_rc_map, regcmd_data, sz);
    free(regcmd_data);

    /* Sync */
    struct rknpu_mem_sync ms = { .flags = 1, .obj_addr = mc.obj_addr, .size = mc.size };
    real_ioctl(fd, IOCTL_MEM_SYNC, &ms);

    fprintf(stderr, "SWAP: regcmd BO dma=0x%llx (%ld bytes, %u entries)\n",
            (unsigned long long)our_rc_dma, sz, our_rc_entries);
    return 0;
}

int ioctl(int fd, unsigned long request, ...) {
    init_real_ioctl();

    __builtin_va_list ap;
    __builtin_va_start(ap, request);
    void *arg = __builtin_va_arg(ap, void *);
    __builtin_va_end(ap);

    /* Track MEM_CREATE results */
    if (request == IOCTL_MEM_CREATE) {
        int ret = real_ioctl(fd, request, arg);
        if (ret == 0 && bo_count < MAX_BOS) {
            struct rknpu_mem_create *mc = arg;
            bo_table[bo_count].obj = mc->obj_addr;
            bo_table[bo_count].handle = mc->handle;
            bo_table[bo_count].dma = mc->dma_addr;
            bo_table[bo_count].size = mc->size;
            bo_count++;
            if (drm_fd < 0) drm_fd = fd;
        }
        return ret;
    }

    /* Intercept SUBMIT */
    if (request == IOCTL_SUBMIT) {
        struct rknpu_submit *s = arg;
        submit_count++;

        fprintf(stderr, "SWAP: SUBMIT[%d] flags=0x%x tasks=%u obj=0x%llx mask=0x%x\n",
                submit_count, s->flags, s->task_number,
                (unsigned long long)s->task_obj_addr, s->core_mask);

        /* Mode 0: Dump all task regcmd register values on first SUBMIT */
        if (submit_count == 1 && getenv("DUMP_REGCMD")) {
            /* Find and mmap the task BO */
            uint64_t task_obj = s->task_obj_addr;
            uint32_t task_handle = 0;
            uint64_t task_bo_size = 0;
            for (int i = 0; i < bo_count; i++) {
                if (bo_table[i].obj == task_obj) {
                    task_handle = bo_table[i].handle;
                    task_bo_size = bo_table[i].size;
                    break;
                }
            }
            if (!task_handle) {
                fprintf(stderr, "DUMP: cannot find task BO for obj=0x%llx\n",
                        (unsigned long long)task_obj);
                return real_ioctl(fd, request, arg);
            }

            struct rknpu_mem_map tm = { .handle = task_handle };
            void *task_map = NULL;
            if (real_ioctl(fd, IOCTL_MEM_MAP, &tm) == 0) {
                task_map = mmap(NULL, task_bo_size, PROT_READ, MAP_SHARED, fd, tm.offset);
            }
            if (!task_map || task_map == MAP_FAILED) {
                fprintf(stderr, "DUMP: cannot mmap task BO\n");
                return real_ioctl(fd, request, arg);
            }

            struct rknpu_task *tasks = (struct rknpu_task *)task_map;
            fprintf(stderr, "DUMP: %u tasks\n", s->task_number);

            for (uint32_t t = 0; t < s->task_number; t++) {
                uint64_t rc_addr = tasks[t].regcmd_addr;
                uint32_t rc_amount = tasks[t].regcfg_amount;

                /* Find regcmd BO containing this DMA address */
                void *rc_map = NULL;
                uint64_t rc_bo_dma = 0;
                uint64_t rc_bo_size = 0;
                uint32_t rc_handle = 0;
                for (int i = 0; i < bo_count; i++) {
                    if (rc_addr >= bo_table[i].dma &&
                        rc_addr < bo_table[i].dma + bo_table[i].size) {
                        rc_bo_dma = bo_table[i].dma;
                        rc_bo_size = bo_table[i].size;
                        rc_handle = bo_table[i].handle;
                        break;
                    }
                }
                if (!rc_handle) {
                    fprintf(stderr, "TASK[%u]: regcmd BO not found for dma=0x%llx\n",
                            t, (unsigned long long)rc_addr);
                    continue;
                }

                struct rknpu_mem_map rm = { .handle = rc_handle };
                if (real_ioctl(fd, IOCTL_MEM_MAP, &rm) == 0) {
                    rc_map = mmap(NULL, rc_bo_size, PROT_READ, MAP_SHARED, fd, rm.offset);
                }
                if (!rc_map || rc_map == MAP_FAILED) {
                    fprintf(stderr, "TASK[%u]: cannot mmap regcmd BO\n", t);
                    continue;
                }

                /* Parse regcmd entries starting at offset within BO */
                uint64_t offset_in_bo = rc_addr - rc_bo_dma;
                uint64_t *entries = (uint64_t *)((uint8_t *)rc_map + offset_in_bo);
                /* total entries = regcfg_amount + 4 (the tail includes PC_OP_EN) */
                unsigned total_entries = rc_amount + 4;

                uint32_t bs_cfg = 0, bs_alu_cfg = 0, bs_mul_cfg = 0;
                uint32_t out_cvt_offset = 0, out_cvt_scale = 0, out_cvt_shift = 0;
                uint32_t data_format = 0, bn_cfg = 0, ew_cfg = 0;
                uint32_t bs_relux = 0, brdma_cfg = 0, bs_base_addr = 0;
                uint32_t bs_ow_op = 0, cube_ch = 0, rdma_weight = 0;
                uint32_t rdma_cube_ch = 0;

                for (unsigned e = 0; e < total_entries; e++) {
                    uint64_t entry = entries[e];
                    uint16_t reg = entry & 0xFFFF;
                    uint32_t val = (entry >> 16) & 0xFFFFFFFF;
                    switch (reg) {
                    case 0x4040: bs_cfg = val; break;         /* DPU_BS_CFG */
                    case 0x4044: bs_alu_cfg = val; break;     /* DPU_BS_ALU_CFG */
                    case 0x4048: bs_mul_cfg = val; break;     /* DPU_BS_MUL_CFG */
                    case 0x4058: bs_ow_op = val; break;       /* DPU_BS_OW_OP */
                    case 0x4080: out_cvt_offset = val; break; /* DPU_OUT_CVT_OFFSET */
                    case 0x4084: out_cvt_scale = val; break;  /* DPU_OUT_CVT_SCALE */
                    case 0x4088: out_cvt_shift = val; break;  /* DPU_OUT_CVT_SHIFT */
                    case 0x4010: data_format = val; break;    /* DPU_DATA_FORMAT */
                    case 0x4020: cube_ch = val; break;        /* DPU_DATA_CUBE_CHANNEL */
                    case 0x4060: bn_cfg = val; break;         /* DPU_BN_CFG */
                    case 0x4070: ew_cfg = val; break;         /* DPU_EW_CFG */
                    case 0x404c: bs_relux = val; break;       /* DPU_BS_RELUX_CMP */
                    case 0x501c: brdma_cfg = val; break;      /* RDMA_BRDMA_CFG */
                    case 0x5020: bs_base_addr = val; break;   /* RDMA_BS_BASE_ADDR */
                    case 0x5014: rdma_cube_ch = val; break;   /* RDMA_DATA_CUBE_CH */
                    case 0x5068: rdma_weight = val; break;    /* RDMA_WEIGHT */
                    }
                }

                fprintf(stderr, "TASK[%u]: BS_CFG=0x%x ALU=%d MUL=0x%x OW_OP=0x%x "
                        "CVT_OFS=%d SCALE=%u SHIFT=%u CUBE_CH=0x%x "
                        "DFMT=0x%x BN=0x%x EW=0x%x "
                        "BRDMA=0x%x BS_ADDR=0x%x RDMA_CH=0x%x RDMA_WT=0x%x amt=%u\n",
                        t, bs_cfg, (int32_t)bs_alu_cfg, bs_mul_cfg, bs_ow_op,
                        (int32_t)out_cvt_offset, out_cvt_scale, out_cvt_shift, cube_ch,
                        data_format, bn_cfg, ew_cfg,
                        brdma_cfg, bs_base_addr, rdma_cube_ch, rdma_weight, rc_amount);

                /* Full regcmd dump for task 19 (first "normal" per-channel task) */
                if (t == 19 && getenv("DUMP_FULL")) {
                    fprintf(stderr, "FULL_REGCMD[%u]:\n", t);
                    for (unsigned e = 0; e < total_entries; e++) {
                        uint64_t entry = entries[e];
                        uint16_t reg = entry & 0xFFFF;
                        uint32_t val = (entry >> 16) & 0xFFFFFFFF;
                        uint16_t tgt = (entry >> 48) & 0xFFFF;
                        fprintf(stderr, "  [%3u] tgt=0x%04x reg=0x%04x val=0x%08x\n",
                                e, tgt, reg, val);
                    }
                }

                munmap(rc_map, rc_bo_size);
            }

            munmap(task_map, task_bo_size);
            /* Let the submit proceed normally */
            return real_ioctl(fd, request, arg);
        }

        /* Mode 1: just reduce RKNN's submit to 1 task (RKNN's own regcmd) */
        if (submit_count == 1 && getenv("SWAP_1TASK")) {
            uint32_t saved_task_num = s->task_number;
            struct rknpu_subcore_task saved_sc[5];
            memcpy(saved_sc, s->subcore_task, sizeof(saved_sc));

            s->task_number = 1;
            for (int i = 0; i < 5; i++) {
                s->subcore_task[i].task_start = 0;
                s->subcore_task[i].task_number = 1;
            }

            fprintf(stderr, "SWAP: Reduced to 1 task (was %u)\n", saved_task_num);
            int ret = real_ioctl(fd, request, arg);
            fprintf(stderr, "SWAP: 1-task submit returned %d (hw_time=%lld)\n",
                    ret, (long long)s->hw_elapse_time);

            /* Restore */
            s->task_number = saved_task_num;
            memcpy(s->subcore_task, saved_sc, sizeof(saved_sc));
            return ret;
        }

        /* Mode 2: inject our regcmd into task[0] */
        if (submit_count == 1 && getenv("SWAP_REGCMD")) {
            /* Create our regcmd BO if not done yet */
            if (!our_rc_handle) {
                if (create_our_regcmd(fd) < 0) {
                    fprintf(stderr, "SWAP: failed to create regcmd, passing through\n");
                    return real_ioctl(fd, request, arg);
                }
            }

            /* Find and mmap RKNN's task BO */
            uint64_t task_obj = s->task_obj_addr;
            uint32_t task_handle = 0;
            for (int i = 0; i < bo_count; i++) {
                if (bo_table[i].obj == task_obj) {
                    task_handle = bo_table[i].handle;
                    break;
                }
            }
            if (!task_handle) {
                fprintf(stderr, "SWAP: cannot find task BO handle for obj=0x%llx\n",
                        (unsigned long long)task_obj);
                return real_ioctl(fd, request, arg);
            }

            /* Map task BO */
            struct rknpu_mem_map tm = { .handle = task_handle };
            if (real_ioctl(fd, IOCTL_MEM_MAP, &tm) == 0) {
                rknn_task_map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, tm.offset);
            }
            if (!rknn_task_map || rknn_task_map == MAP_FAILED) {
                fprintf(stderr, "SWAP: cannot mmap task BO\n");
                return real_ioctl(fd, request, arg);
            }

            /* Read RKNN's first task to log it */
            struct rknpu_task *tasks = (struct rknpu_task *)rknn_task_map;
            fprintf(stderr, "SWAP: RKNN task[0]: regcmd_addr=0x%llx amount=%u mask=0x%x\n",
                    (unsigned long long)tasks[0].regcmd_addr,
                    tasks[0].regcfg_amount, tasks[0].int_mask);

            /* Overwrite task[0] to use our regcmd */
            uint64_t saved_addr = tasks[0].regcmd_addr;
            uint32_t saved_amount = tasks[0].regcfg_amount;
            tasks[0].regcmd_addr = our_rc_dma;
            tasks[0].regcfg_amount = our_rc_entries > 8 ? our_rc_entries - 8 : our_rc_entries;

            /* Modify submit to use only 1 task */
            uint32_t saved_task_num = s->task_number;
            struct rknpu_subcore_task saved_sc[5];
            memcpy(saved_sc, s->subcore_task, sizeof(saved_sc));

            s->task_number = 1;
            for (int i = 0; i < 5; i++) {
                s->subcore_task[i].task_start = 0;
                s->subcore_task[i].task_number = 1;
            }

            /* Sync task BO */
            struct rknpu_mem_sync ms = { .flags = 1, .obj_addr = task_obj, .size = 4096 };
            real_ioctl(fd, IOCTL_MEM_SYNC, &ms);

            fprintf(stderr, "SWAP: Injected our regcmd (dma=0x%llx, amount=%u) into task[0]\n",
                    (unsigned long long)our_rc_dma, tasks[0].regcfg_amount);

            /* Submit with our swapped regcmd */
            int ret = real_ioctl(fd, request, arg);

            fprintf(stderr, "SWAP: Submit returned %d (hw_time=%lld)\n",
                    ret, (long long)s->hw_elapse_time);

            /* Restore original task and submit params for subsequent RKNN work */
            tasks[0].regcmd_addr = saved_addr;
            tasks[0].regcfg_amount = saved_amount;
            s->task_number = saved_task_num;
            memcpy(s->subcore_task, saved_sc, sizeof(saved_sc));

            munmap(rknn_task_map, 4096);
            rknn_task_map = NULL;
            return ret;
        }
    }

    return real_ioctl(fd, request, arg);
}
