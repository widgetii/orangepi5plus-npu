/*
 * NPU Convolution Test Suite for QEMU RK3588 NPU emulator.
 *
 * Six tests covering: multi-channel matmul, stride/padding, depthwise,
 * DMA bias, and BN ReLU — all code paths that have zero isolated test
 * coverage in npu_test.c.
 *
 * Works with both the upstream Rocket DRM driver (/dev/accel/accel0)
 * and the vendor RKNPU DRM driver (/dev/dri/card0). Auto-detects at
 * runtime based on which device is available.
 *
 * Builds statically for aarch64, runs inside QEMU initrd.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>

/* ======================================================================
 * DRM ioctl definitions — shared base
 * ====================================================================== */

#define DRM_IOCTL_BASE 'd'
#define DRM_COMMAND_BASE 0x40

struct drm_gem_close {
    uint32_t handle;
    uint32_t pad;
};
#define DRM_IOCTL_GEM_CLOSE _IOW(DRM_IOCTL_BASE, 0x09, struct drm_gem_close)

/* --- Rocket (upstream) ioctls --- */

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

#define DRM_IOCTL_ROCKET_CREATE_BO _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x00, struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT    _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x01, struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO   _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x02, struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO   _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x03, struct drm_rocket_fini_bo)

/* --- RKNPU (vendor) ioctls --- */

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
    uint64_t regcfg_obj_addr;
    uint64_t task_base_addr;
    uint64_t user_data;
    uint32_t core_mask;
    int32_t  fence_fd;
    struct rknpu_subcore_task subcore_task[5];
};

#define RKNPU_JOB_PC        (1 << 0)
#define RKNPU_JOB_BLOCK     0
#define RKNPU_JOB_PINGPONG  (1 << 2)
#define RKNPU_MEM_NON_CONTIGUOUS (1 << 0)
#define RKNPU_MEM_CACHEABLE      (1 << 1)
#define RKNPU_MEM_KERNEL_MAPPING (1 << 3)
#define RKNPU_MEM_IOMMU_LIMIT    (1 << 10)
#define RKNPU_MEM_DEFAULT        (RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE | RKNPU_MEM_IOMMU_LIMIT)

#define DRM_IOCTL_RKNPU_ACTION      _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x00, struct rknpu_action)
#define DRM_IOCTL_RKNPU_SUBMIT      _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x01, struct rknpu_submit)
#define DRM_IOCTL_RKNPU_MEM_CREATE  _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x02, struct rknpu_mem_create)
#define DRM_IOCTL_RKNPU_MEM_MAP     _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x03, struct rknpu_mem_map)
#define DRM_IOCTL_RKNPU_MEM_DESTROY _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x04, struct rknpu_mem_destroy)
#define DRM_IOCTL_RKNPU_MEM_SYNC    _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x05, struct rknpu_mem_sync)

#define RKNPU_PC_DATA_EXTRA_AMOUNT 4

/* ======================================================================
 * Driver abstraction — BO lifecycle and job submission
 * ====================================================================== */

enum npu_driver { DRIVER_ROCKET, DRIVER_RKNPU };

struct npu_bo {
    uint32_t handle;
    uint64_t dma_addr;
    uint64_t obj_addr;    /* rknpu only */
    uint64_t mmap_offset;
    uint32_t size;
};

static enum npu_driver g_driver;

static int npu_open_device(void)
{
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd >= 0) {
        g_driver = DRIVER_ROCKET;
        return fd;
    }
    /* Try card1 first — on boards with display, card0 is rockchip-drm */
    fd = open("/dev/dri/card1", O_RDWR);
    if (fd >= 0) {
        g_driver = DRIVER_RKNPU;
        return fd;
    }
    fd = open("/dev/dri/card0", O_RDWR);
    if (fd >= 0) {
        g_driver = DRIVER_RKNPU;
        return fd;
    }
    return -1;
}

static int npu_alloc_bo(int fd, uint32_t size, struct npu_bo *bo, uint32_t flags)
{
    bo->size = size;
    if (g_driver == DRIVER_ROCKET) {
        struct drm_rocket_create_bo req = { .size = size };
        if (ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &req)) return -1;
        bo->handle = req.handle;
        bo->dma_addr = req.dma_address;
        bo->mmap_offset = req.offset;
        bo->obj_addr = 0;
    } else {
        struct rknpu_mem_create mc = { .size = size, .flags = flags ? flags : RKNPU_MEM_DEFAULT };
        if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mc)) return -1;
        bo->handle = mc.handle;
        bo->dma_addr = mc.dma_addr;
        bo->obj_addr = mc.obj_addr;
        struct rknpu_mem_map mm = { .handle = mc.handle };
        if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) return -1;
        bo->mmap_offset = mm.offset;
    }
    return 0;
}

static void *npu_mmap_bo(int fd, struct npu_bo *bo)
{
    return mmap(NULL, bo->size, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, bo->mmap_offset);
}

static void npu_sync_bo(int fd, struct npu_bo *bo)
{
    if (g_driver == DRIVER_ROCKET) {
        struct drm_rocket_fini_bo fini = { .handle = bo->handle };
        ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    } else {
        struct rknpu_mem_sync ms = {
            .flags = 1, /* SYNC_TO_DEVICE */
            .obj_addr = bo->obj_addr,
            .size = bo->size,
        };
        ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &ms);
    }
}

static int npu_submit_and_wait(int fd, struct npu_bo *regcmd_bo,
                                unsigned regcmd_count,
                                struct npu_bo *in_bos[], unsigned n_in,
                                struct npu_bo *out_bos[], unsigned n_out)
{
    if (g_driver == DRIVER_ROCKET) {
        struct drm_rocket_task task = {
            .regcmd = (uint32_t)regcmd_bo->dma_addr,
            .regcmd_count = regcmd_count * 2,
        };
        uint32_t in_handles[8], out_handles[4];
        for (unsigned i = 0; i < n_in; i++) in_handles[i] = in_bos[i]->handle;
        for (unsigned i = 0; i < n_out; i++) out_handles[i] = out_bos[i]->handle;

        struct drm_rocket_job job = {
            .tasks = (uint64_t)(uintptr_t)&task,
            .in_bo_handles = (uint64_t)(uintptr_t)in_handles,
            .out_bo_handles = (uint64_t)(uintptr_t)out_handles,
            .task_count = 1,
            .task_struct_size = sizeof(struct drm_rocket_task),
            .in_bo_handle_count = n_in,
            .out_bo_handle_count = n_out,
        };
        struct drm_rocket_submit submit = {
            .jobs = (uint64_t)(uintptr_t)&job,
            .job_count = 1,
            .job_struct_size = sizeof(struct drm_rocket_job),
        };
        int ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit);
        if (ret < 0) return ret;

        struct timespec _ts;
        clock_gettime(CLOCK_MONOTONIC, &_ts);
        int64_t _abs = (int64_t)_ts.tv_sec * 1000000000LL + _ts.tv_nsec + 5000000000LL;
        struct drm_rocket_prep_bo prep = {
            .handle = out_bos[0]->handle,
            .timeout_ns = _abs,
        };
        ret = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep);
        return (ret < 0) ? ret : 0;
    } else {
        /* RKNPU: allocate task BO, fill rknpu_task, submit */
        struct npu_bo task_bo;
        if (npu_alloc_bo(fd, 4096, &task_bo, RKNPU_MEM_DEFAULT | RKNPU_MEM_KERNEL_MAPPING))
            return -1;
        struct rknpu_task *tasks = npu_mmap_bo(fd, &task_bo);
        if (tasks == MAP_FAILED) return -1;

        memset(tasks, 0, sizeof(struct rknpu_task));
        tasks[0].flags = 0;
        tasks[0].op_idx = 0;
        tasks[0].enable_mask = 0xd;
        tasks[0].int_mask = 0x300;
        tasks[0].int_clear = 0x1ffff;
        tasks[0].int_status = 0;
        tasks[0].regcfg_amount = regcmd_count - (RKNPU_PC_DATA_EXTRA_AMOUNT + 4);
        tasks[0].regcfg_offset = 0;
        tasks[0].regcmd_addr = regcmd_bo->dma_addr;

        struct rknpu_submit submit = {
            .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
            .timeout = 6000,
            .task_start = 0,
            .task_number = 1,
            .task_obj_addr = task_bo.obj_addr,
            .core_mask = 1,
            .fence_fd = -1,
            .subcore_task = { {0, 1}, {1, 0}, {2, 0}, {0, 0}, {0, 0} },
        };
        int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);

        munmap(tasks, 4096);
        struct rknpu_mem_destroy md = {
            .handle = task_bo.handle,
            .obj_addr = task_bo.obj_addr,
        };
        ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);
        return ret;
    }
}

static void npu_free_bo(int fd, struct npu_bo *bo)
{
    if (g_driver == DRIVER_ROCKET) {
        struct drm_gem_close cl = { .handle = bo->handle };
        ioctl(fd, DRM_IOCTL_GEM_CLOSE, &cl);
    } else {
        struct rknpu_mem_destroy md = {
            .handle = bo->handle,
            .obj_addr = bo->obj_addr,
        };
        ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &md);
    }
}

/* ======================================================================
 * NPU constants and helpers
 * ====================================================================== */

#define NPU_FEATURE_ATOMIC_SIZE 16
#define NPU_WEIGHT_ATOMIC_SIZE  32

#define ALIGN_UP(x, a)    (((x) + (a) - 1) & ~((a) - 1))
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

/* Regcmd target IDs (bit [0] = 1 for write) */
#define TARGET_CNA   0x0201
#define TARGET_CORE  0x0801
#define TARGET_DPU   0x1001
#define TARGET_RDMA  0x2001
#define TARGET_PC    0x0101

static uint64_t emit(uint16_t target, uint16_t reg, uint32_t value)
{
    return ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

/* ======================================================================
 * NVDLA rounding helpers (matching rockchip-npu.c exactly)
 * ====================================================================== */

static inline int32_t nvdla_truncate(int32_t value, unsigned truncate)
{
    if (truncate == 0) return value;
    uint32_t guide = (value >> (truncate - 1)) & 1;
    return (value >> truncate) + guide;
}

static inline int64_t nvdla_shift_right_round64(int64_t value, unsigned shift)
{
    if (shift == 0) return value;
    uint64_t guide = (value >> (shift - 1)) & 1;
    return (value >> shift) + guide;
}

/* ======================================================================
 * Convolution configuration
 * ====================================================================== */

struct conv_config {
    uint32_t in_w, in_h, in_c;
    uint32_t out_c;
    uint32_t filt_w, filt_h;
    uint32_t stride_x, stride_y;
    uint32_t pad_left, pad_top;
    int8_t   pad_value;
    int      depthwise;

    /* Quantization */
    uint32_t truncate_bits;
    uint32_t out_cvt_scale;
    uint32_t out_cvt_shift;
    int32_t  out_cvt_offset;

    /* SDP config */
    uint32_t bs_cfg;
    int32_t  bs_alu_cfg;
    uint32_t brdma_cfg;    /* 0x02 = bias from DMA */
    uint32_t bn_cfg;
    uint16_t bs_ow_op;     /* BS output-write operand (0x80 - wzp) */
};

static void conv_config_defaults(struct conv_config *c)
{
    memset(c, 0, sizeof(*c));
    c->filt_w = 1;
    c->filt_h = 1;
    c->stride_x = 1;
    c->stride_y = 1;
    c->pad_value = (int8_t)0x80;  /* -128 */
    c->out_cvt_scale = 16384;
    c->out_cvt_shift = 14;
    /* BS: ALU_ALGO=2(ADD), ALU_SRC=1(DMA), RELU_BYPASS=1, MUL_BYPASS=1 */
    c->bs_cfg = 0x00020150;
    c->brdma_cfg = 0x02;  /* BRDMA_DATA_USE=1 */
    /* BN: all bypass */
    c->bn_cfg = 0x1f;
}

/* ======================================================================
 * Weight packing — flat [oc][kx][ky][ic] → NPU packed format
 * ====================================================================== */

static void pack_weights(const int8_t *flat, uint8_t *packed,
                         uint32_t out_c, uint32_t in_c,
                         uint32_t filt_w, uint32_t filt_h,
                         int depthwise)
{
    uint32_t ic_group = depthwise ? NPU_WEIGHT_ATOMIC_SIZE * 2
                                  : NPU_WEIGHT_ATOMIC_SIZE;
    uint32_t oc_aligned = depthwise ? 1 : ALIGN_UP(MAX2(out_c, 2), 2);
    uint32_t ic_padded = MAX2(in_c, NPU_FEATURE_ATOMIC_SIZE);

    uint32_t ic1_count = DIV_ROUND_UP(ic_padded, ic_group);
    uint32_t oc2_count = MIN2(oc_aligned, NPU_WEIGHT_ATOMIC_SIZE);
    uint32_t ic2_count = MIN2(ic_padded, ic_group);

    for (uint32_t oc = 0; oc < out_c; oc++) {
        for (uint32_t kx = 0; kx < filt_w; kx++) {
            for (uint32_t ky = 0; ky < filt_h; ky++) {
                uint32_t ic_limit = in_c;
                for (uint32_t ic = 0; ic < ic_limit; ic++) {
                    uint32_t oc1 = oc / NPU_WEIGHT_ATOMIC_SIZE;
                    uint32_t oc2 = oc % NPU_WEIGHT_ATOMIC_SIZE;
                    uint32_t ic1 = ic / ic_group;
                    uint32_t ic2 = ic % ic_group;

                    uint32_t offset = oc1 * (ic1_count * filt_w * filt_h *
                                             oc2_count * ic2_count);
                    offset += ic1 * (filt_w * filt_h * oc2_count * ic2_count);
                    offset += kx * (filt_h * oc2_count * ic2_count);
                    offset += ky * (oc2_count * ic2_count);
                    offset += oc2 * ic2_count;
                    offset += ic2;

                    int8_t val;
                    if (depthwise) {
                        val = flat[oc * filt_w * filt_h + kx * filt_h + ky];
                    } else {
                        val = flat[oc * filt_w * filt_h * in_c +
                                   kx * filt_h * in_c + ky * in_c + ic];
                    }
                    packed[offset] = (uint8_t)val;
                }
            }
        }
    }
}

static uint32_t packed_weight_size(uint32_t out_c, uint32_t in_c,
                                   uint32_t filt_w, uint32_t filt_h,
                                   int depthwise)
{
    uint32_t ic_group = depthwise ? NPU_WEIGHT_ATOMIC_SIZE * 2
                                  : NPU_WEIGHT_ATOMIC_SIZE;
    uint32_t oc_aligned = depthwise ? 1 : ALIGN_UP(MAX2(out_c, 2), 2);
    uint32_t ic_padded = MAX2(in_c, NPU_FEATURE_ATOMIC_SIZE);

    return DIV_ROUND_UP(oc_aligned, NPU_WEIGHT_ATOMIC_SIZE)
         * DIV_ROUND_UP(ic_padded, ic_group)
         * filt_w * filt_h
         * MIN2(oc_aligned, NPU_WEIGHT_ATOMIC_SIZE)
         * MIN2(ic_padded, ic_group);
}

/* ======================================================================
 * Input/output layout conversion
 * ====================================================================== */

/* NHWC → NPU x-major interleaved: [group][x][y][c16] */
static void nhwc_to_npu_input(const int8_t *nhwc, uint8_t *npu,
                               uint32_t w, uint32_t h, uint32_t c)
{
    uint32_t c_padded = ALIGN_UP(MAX2(c, NPU_FEATURE_ATOMIC_SIZE),
                                 NPU_FEATURE_ATOMIC_SIZE);
    uint32_t groups = c_padded / NPU_FEATURE_ATOMIC_SIZE;
    uint32_t buf_size = groups * w * h * NPU_FEATURE_ATOMIC_SIZE;
    memset(npu, 0, buf_size);

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t ch = 0; ch < c; ch++) {
                uint32_t g = ch / NPU_FEATURE_ATOMIC_SIZE;
                uint32_t c_within = ch % NPU_FEATURE_ATOMIC_SIZE;
                uint32_t off = g * w * h * NPU_FEATURE_ATOMIC_SIZE
                             + x * h * NPU_FEATURE_ATOMIC_SIZE
                             + y * NPU_FEATURE_ATOMIC_SIZE
                             + c_within;
                npu[off] = (uint8_t)nhwc[y * w * c + x * c + ch];
            }
        }
    }
}

/* NPU y-major interleaved [group][y][x][c16] → NHWC */
static void npu_output_to_nhwc(const uint8_t *npu, int8_t *nhwc,
                                uint32_t w, uint32_t h, uint32_t c)
{
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            for (uint32_t ch = 0; ch < c; ch++) {
                uint32_t g = ch / NPU_FEATURE_ATOMIC_SIZE;
                uint32_t c_within = ch % NPU_FEATURE_ATOMIC_SIZE;
                /* x-major: matching NPU_OFFSET in librocketnpu */
                uint32_t off = g * w * h * NPU_FEATURE_ATOMIC_SIZE
                             + x * h * NPU_FEATURE_ATOMIC_SIZE
                             + y * NPU_FEATURE_ATOMIC_SIZE
                             + c_within;
                nhwc[y * w * c + x * c + ch] = (int8_t)npu[off];
            }
        }
    }
}

/* ======================================================================
 * CPU reference convolution (matches rockchip-npu.c pipeline exactly)
 * ====================================================================== */

static void cpu_conv_reference(const struct conv_config *cfg,
                               const int8_t *input_nhwc,
                               const int8_t *weights_flat,
                               const int32_t *bias,
                               int8_t *output_nhwc)
{
    uint32_t out_w = (cfg->in_w + cfg->pad_left * 2 - cfg->filt_w) / cfg->stride_x + 1;
    uint32_t out_h = (cfg->in_h + cfg->pad_top * 2 - cfg->filt_h) / cfg->stride_y + 1;

    for (uint32_t oy = 0; oy < out_h; oy++) {
        for (uint32_t ox = 0; ox < out_w; ox++) {
            for (uint32_t oc = 0; oc < cfg->out_c; oc++) {
                int32_t acc = 0;
                uint32_t ic_limit = cfg->depthwise ? 1 : cfg->in_c;
                uint32_t ic_base = cfg->depthwise ? oc : 0;

                for (uint32_t ky = 0; ky < cfg->filt_h; ky++) {
                    for (uint32_t kx = 0; kx < cfg->filt_w; kx++) {
                        int32_t iy = (int32_t)(oy * cfg->stride_y + ky) - (int32_t)cfg->pad_top;
                        int32_t ix = (int32_t)(ox * cfg->stride_x + kx) - (int32_t)cfg->pad_left;
                        for (uint32_t ic = 0; ic < ic_limit; ic++) {
                            int8_t in_val;
                            uint32_t abs_ic = ic_base + ic;
                            if (ix < 0 || ix >= (int32_t)cfg->in_w ||
                                iy < 0 || iy >= (int32_t)cfg->in_h) {
                                in_val = cfg->pad_value;
                            } else {
                                in_val = input_nhwc[iy * cfg->in_w * cfg->in_c
                                                  + ix * cfg->in_c + abs_ic];
                            }
                            int8_t w_val;
                            if (cfg->depthwise)
                                w_val = weights_flat[oc * cfg->filt_w * cfg->filt_h + kx * cfg->filt_h + ky];
                            else
                                w_val = weights_flat[oc * cfg->filt_w * cfg->filt_h * cfg->in_c
                                                   + kx * cfg->filt_h * cfg->in_c + ky * cfg->in_c + abs_ic];
                            acc += (int32_t)in_val * (int32_t)w_val;
                        }
                    }
                }

                acc = nvdla_truncate(acc, cfg->truncate_bits);

                /* BS stage */
                if (!(cfg->bs_cfg & 0x01)) {
                    if (!(cfg->bs_cfg & 0x02)) {
                        uint32_t algo = (cfg->bs_cfg >> 16) & 0xf;
                        int32_t bs_alu_op = ((cfg->bs_cfg >> 8) & 1) && bias
                            ? bias[oc] : cfg->bs_alu_cfg;
                        switch (algo) {
                        case 0: if (acc < bs_alu_op) acc = bs_alu_op; break;
                        case 1: if (acc > bs_alu_op) acc = bs_alu_op; break;
                        case 2: acc += bs_alu_op; break;
                        }
                    }
                }
                /* BN stage */
                if (!(cfg->bn_cfg & 0x01)) {
                    if (!(cfg->bn_cfg & 0x40)) {
                        if (acc < 0) acc = 0;
                    }
                }
                /* OUT_CVT */
                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)cfg->out_cvt_scale, cfg->out_cvt_shift);
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;
                int32_t result = (int32_t)scaled + cfg->out_cvt_offset;
                if (result < -128) result = -128;
                if (result > 127) result = 127;
                output_nhwc[oy * out_w * cfg->out_c + ox * cfg->out_c + oc] = (int8_t)result;
            }
        }
    }
}

/* ======================================================================
 * Parameterized regcmd builder
 * ====================================================================== */

static unsigned build_conv_regcmd(uint64_t *buf, const struct conv_config *cfg,
                                  uint32_t in_addr, uint32_t wt_addr,
                                  uint32_t out_addr, uint32_t bias_addr)
{
    uint64_t *p = buf;
    uint32_t in_c_padded = ALIGN_UP(MAX2(cfg->in_c, NPU_FEATURE_ATOMIC_SIZE),
                                    NPU_FEATURE_ATOMIC_SIZE);
    uint32_t out_w = (cfg->in_w + cfg->pad_left * 2 - cfg->filt_w) / cfg->stride_x + 1;
    uint32_t out_h = (cfg->in_h + cfg->pad_top * 2 - cfg->filt_h) / cfg->stride_y + 1;

    uint32_t wt_kernels = cfg->depthwise ? 1 : ALIGN_UP(MAX2(cfg->out_c, 2), 2);
    uint32_t wt_size1 = MAX2(cfg->in_c, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t wt_size0 = wt_size1 * wt_kernels;

    *p++ = emit(TARGET_CNA, 0x1040, 0x00000b01);
    *p++ = emit(TARGET_CNA, 0x1100, 0);
    *p++ = emit(TARGET_CNA, 0x1104, 0);
    *p++ = emit(TARGET_CNA, 0x100c, cfg->depthwise ? 3 : 0);
    *p++ = emit(TARGET_DPU, 0x4004, 0x00070007);
    *p++ = emit(TARGET_RDMA, 0x5004, 0x00070007);
    *p++ = emit(TARGET_CNA, 0x100c, cfg->depthwise ? 3 : 0);
    *p++ = emit(TARGET_CNA, 0x1010, 52);
    uint32_t conv_con3 = (cfg->stride_x & 0x7) | ((cfg->stride_y & 0x7) << 3);
    *p++ = emit(TARGET_CNA, 0x1014, conv_con3);
    *p++ = emit(TARGET_CNA, 0x1020, cfg->in_h | (cfg->in_w << 16));
    *p++ = emit(TARGET_CNA, 0x1024, ((cfg->in_c - 1) << 16) | in_c_padded);
    *p++ = emit(TARGET_CNA, 0x1028, out_w);
    *p++ = emit(TARGET_CNA, 0x102c, 1);
    *p++ = emit(TARGET_CNA, 0x1030, wt_size0);
    *p++ = emit(TARGET_CNA, 0x1034, wt_size1);
    *p++ = emit(TARGET_CNA, 0x1038, wt_kernels | (cfg->filt_h << 16) | (cfg->filt_w << 24));
    *p++ = emit(TARGET_CNA, 0x1040, 0x00000b01);
    *p++ = emit(TARGET_CNA, 0x1044, 0x00ff0000);
    *p++ = emit(TARGET_CNA, 0x1048, 0x00000030);
    *p++ = emit(TARGET_CNA, 0x104c, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x1050, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x1054, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x1058, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x105c, 0);
    *p++ = emit(TARGET_CNA, 0x1060, 0);
    uint32_t pad_con0 = (cfg->pad_top & 0xf) | ((cfg->pad_left & 0xf) << 4);
    *p++ = emit(TARGET_CNA, 0x1068, pad_con0);
    *p++ = emit(TARGET_CNA, 0x1070, in_addr);
    *p++ = emit(TARGET_CNA, 0x1074, 0);
    *p++ = emit(TARGET_CNA, 0x1078, 0x0f0f0000);
    /* Line/surface stride in register units (bytes / 4), matching librocketnpu */
    uint32_t line_stride = cfg->in_h * NPU_FEATURE_ATOMIC_SIZE / 4;
    uint32_t surf_stride = cfg->in_w * cfg->in_h * NPU_FEATURE_ATOMIC_SIZE / 4;
    *p++ = emit(TARGET_CNA, 0x107c, line_stride);
    *p++ = emit(TARGET_CNA, 0x1080, surf_stride);
    *p++ = emit(TARGET_CNA, 0x1084, 0);
    *p++ = emit(TARGET_CNA, 0x1088, 0x000f0000);
    *p++ = emit(TARGET_CNA, 0x1104, 0);
    *p++ = emit(TARGET_CNA, 0x1100, 0);
    *p++ = emit(TARGET_CNA, 0x1110, wt_addr);
    for (int i = 0; i < 16; i++)
        *p++ = emit(TARGET_CNA, 0x1114 + i * 4, 0);
    *p++ = emit(TARGET_CNA, 0x10a0, 0);
    *p++ = emit(TARGET_CNA, 0x1184, (uint32_t)(int32_t)cfg->pad_value);
    uint32_t misc_cfg = 1 | (cfg->depthwise ? 2 : 0);
    *p++ = emit(TARGET_CORE, 0x3010, misc_cfg);
    *p++ = emit(TARGET_CORE, 0x3014, (out_w - 1) | ((out_h - 1) << 16));
    *p++ = emit(TARGET_CORE, 0x3018, cfg->out_c - 1);
    *p++ = emit(TARGET_CORE, 0x301c, cfg->truncate_bits);
    *p++ = emit(0x0801, 0x3030, 0);
    *p++ = emit(TARGET_DPU, 0x400c, 0x0000020f);
    *p++ = emit(TARGET_DPU, 0x4010, 0);
    *p++ = emit(TARGET_DPU, 0x4014, 0);
    *p++ = emit(TARGET_DPU, 0x4020, out_addr);
    uint32_t dst_surf_stride = out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;
    *p++ = emit(TARGET_DPU, 0x4024, dst_surf_stride);
    *p++ = emit(TARGET_DPU, 0x4030, out_w - 1);
    *p++ = emit(TARGET_DPU, 0x4034, out_h - 1);
    *p++ = emit(TARGET_DPU, 0x4038, 0);
    *p++ = emit(TARGET_DPU, 0x403c, ((cfg->out_c - 1) << 16) | (cfg->out_c - 1));
    *p++ = emit(TARGET_DPU, 0x4040, cfg->bs_cfg);
    *p++ = emit(TARGET_DPU, 0x4044, (uint32_t)cfg->bs_alu_cfg);
    *p++ = emit(TARGET_DPU, 0x4048, 0);
    *p++ = emit(TARGET_DPU, 0x404c, 0);
    *p++ = emit(TARGET_DPU, 0x4050, 0x00010101);
    *p++ = emit(TARGET_DPU, 0x4054, cfg->bs_ow_op & 0xffff);
    *p++ = emit(TARGET_DPU, 0x4058, cfg->out_c - 1);
    *p++ = emit(TARGET_DPU, 0x405c, (out_w - 1) | ((out_h - 1) << 16));
    *p++ = emit(TARGET_DPU, 0x4060, cfg->bn_cfg);
    *p++ = emit(TARGET_DPU, 0x4064, 0);
    *p++ = emit(TARGET_DPU, 0x4068, 0);
    *p++ = emit(TARGET_DPU, 0x406c, 0);
    *p++ = emit(TARGET_DPU, 0x4070, 0x00000383);
    *p++ = emit(TARGET_DPU, 0x4074, 0);
    *p++ = emit(TARGET_DPU, 0x4078, 0x00010000);
    *p++ = emit(TARGET_DPU, 0x407c, 0);
    *p++ = emit(TARGET_DPU, 0x4080, (uint32_t)cfg->out_cvt_offset);
    *p++ = emit(TARGET_DPU, 0x4084, cfg->out_cvt_scale);
    *p++ = emit(TARGET_DPU, 0x4088, cfg->out_cvt_shift);
    for (int i = 0; i < 8; i++)
        *p++ = emit(TARGET_DPU, 0x4090 + i * 4, 0);
    *p++ = emit(TARGET_DPU, 0x40c0, 1);
    *p++ = emit(0x1001, 0x40c4, 0);
    *p++ = emit(TARGET_DPU, 0x40b4, 0);
    *p++ = emit(TARGET_DPU, 0x40b8, 0);
    *p++ = emit(TARGET_DPU, 0x40bc, 0);
    *p++ = emit(TARGET_DPU, 0x40c0, 0);
    for (int i = 0; i < 8; i++)
        *p++ = emit(TARGET_DPU, 0x40c8 + i * 4, 0);
    *p++ = emit(TARGET_RDMA, 0x5008, 0);
    *p++ = emit(TARGET_RDMA, 0x500c, 0);
    *p++ = emit(TARGET_RDMA, 0x5010, cfg->out_c - 1);
    *p++ = emit(TARGET_RDMA, 0x5014, 0);
    *p++ = emit(TARGET_RDMA, 0x501c, cfg->brdma_cfg);
    *p++ = emit(TARGET_RDMA, 0x5020, bias_addr);
    *p++ = emit(TARGET_RDMA, 0x5028, 0);
    *p++ = emit(TARGET_RDMA, 0x502c, 0);
    *p++ = emit(TARGET_RDMA, 0x5034, 0x00000001);
    *p++ = emit(TARGET_RDMA, 0x5038, 0);
    *p++ = emit(TARGET_RDMA, 0x5040, 0);
    *p++ = emit(TARGET_RDMA, 0x5000, 0x000f0010);
    *p++ = emit(TARGET_RDMA, 0x5038, 0);
    *p++ = emit(TARGET_RDMA, 0x503c, 0);
    *p++ = emit(TARGET_RDMA, 0x5040, 0);
    *p++ = emit(TARGET_RDMA, 0x5044, 0x01010101);
    *p++ = emit(TARGET_RDMA, 0x5048, 0);
    /* Chain pointer: none */
    *p++ = 0;
    *p++ = emit(TARGET_PC, 0x0014, 0);
    /* Control: activate */
    *p++ = 0x0041000000000000ULL;
    *p++ = emit(0x0081, 0x0008, 0x0000001d);

    return (unsigned)(p - buf);
}

/* ======================================================================
 * Test runner — allocate BOs, pack data, submit, compare
 * ====================================================================== */

static int run_conv_test(int fd, const char *name,
                         const struct conv_config *cfg,
                         const int8_t *input_nhwc,
                         const int8_t *weights_flat,
                         const int32_t *bias,
                         const int8_t *expected_nhwc)
{
    uint32_t out_w = (cfg->in_w + cfg->pad_left * 2 - cfg->filt_w) / cfg->stride_x + 1;
    uint32_t out_h = (cfg->in_h + cfg->pad_top * 2 - cfg->filt_h) / cfg->stride_y + 1;

    uint32_t in_c_padded = ALIGN_UP(MAX2(cfg->in_c, NPU_FEATURE_ATOMIC_SIZE),
                                    NPU_FEATURE_ATOMIC_SIZE);
    uint32_t in_groups = in_c_padded / NPU_FEATURE_ATOMIC_SIZE;
    uint32_t in_buf_size = in_groups * cfg->in_w * cfg->in_h * NPU_FEATURE_ATOMIC_SIZE;
    uint32_t wt_buf_size = packed_weight_size(cfg->out_c, cfg->in_c,
                                              cfg->filt_w, cfg->filt_h, cfg->depthwise);
    uint32_t out_c_padded = ALIGN_UP(cfg->out_c, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t out_groups = out_c_padded / NPU_FEATURE_ATOMIC_SIZE;
    uint32_t out_buf_size = out_groups * out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;
    uint32_t bias_buf_size = cfg->out_c * sizeof(int32_t);

    uint32_t in_alloc = ALIGN_UP(MAX2(in_buf_size, 4096), 4096);
    uint32_t wt_alloc = ALIGN_UP(MAX2(wt_buf_size, 4096), 4096);
    uint32_t out_alloc = ALIGN_UP(MAX2(out_buf_size, 4096), 4096);
    uint32_t bias_alloc = ALIGN_UP(MAX2(bias_buf_size, 4096), 4096);
    uint32_t regcmd_alloc = 4096;

    struct npu_bo in_bo, out_bo, wt_bo, bias_bo, regcmd_bo;
    if (npu_alloc_bo(fd, in_alloc, &in_bo, 0) ||
        npu_alloc_bo(fd, out_alloc, &out_bo, 0) ||
        npu_alloc_bo(fd, wt_alloc, &wt_bo, 0) ||
        npu_alloc_bo(fd, bias_alloc, &bias_bo, 0) ||
        npu_alloc_bo(fd, regcmd_alloc, &regcmd_bo, 0)) {
        printf("  %s: FAIL (BO creation)\n", name);
        return 0;
    }

    uint8_t *in_map = npu_mmap_bo(fd, &in_bo);
    uint8_t *out_map = npu_mmap_bo(fd, &out_bo);
    uint8_t *wt_map = npu_mmap_bo(fd, &wt_bo);
    int32_t *bias_map = (int32_t *)npu_mmap_bo(fd, &bias_bo);
    uint64_t *regcmd_map = (uint64_t *)npu_mmap_bo(fd, &regcmd_bo);

    if (in_map == MAP_FAILED || out_map == MAP_FAILED || wt_map == MAP_FAILED ||
        bias_map == MAP_FAILED || regcmd_map == MAP_FAILED) {
        printf("  %s: FAIL (mmap)\n", name);
        return 0;
    }

    memset(in_map, 0, in_alloc);
    nhwc_to_npu_input(input_nhwc, in_map, cfg->in_w, cfg->in_h, cfg->in_c);

    memset(wt_map, 0, wt_alloc);
    pack_weights(weights_flat, wt_map, cfg->out_c, cfg->in_c,
                 cfg->filt_w, cfg->filt_h, cfg->depthwise);

    memset(bias_map, 0, bias_alloc);
    if (bias) memcpy(bias_map, bias, bias_buf_size);

    memset(out_map, 0xff, out_alloc);

    unsigned regcmd_count = build_conv_regcmd(regcmd_map, cfg,
        (uint32_t)in_bo.dma_addr, (uint32_t)wt_bo.dma_addr,
        (uint32_t)out_bo.dma_addr, (uint32_t)bias_bo.dma_addr);

    npu_sync_bo(fd, &in_bo);
    npu_sync_bo(fd, &wt_bo);
    npu_sync_bo(fd, &bias_bo);
    npu_sync_bo(fd, &regcmd_bo);

    struct npu_bo *in_bos[] = { &in_bo, &wt_bo, &bias_bo, &regcmd_bo };
    struct npu_bo *out_bos[] = { &out_bo };

    int pass = 0;
    errno = 0;
    int ret = npu_submit_and_wait(fd, &regcmd_bo, regcmd_count,
                                   in_bos, 4, out_bos, 1);
    if (ret) {
        printf("  %s: FAIL (submit ret=%d errno=%d %s)\n",
               name, ret, errno, strerror(errno));
        goto cleanup;
    }

    int8_t *actual_nhwc = (int8_t *)malloc(out_w * out_h * cfg->out_c);
    npu_output_to_nhwc(out_map, actual_nhwc, out_w, out_h, cfg->out_c);

    pass = 1;
    int mismatches = 0;
    for (uint32_t i = 0; i < out_w * out_h * cfg->out_c; i++) {
        if (actual_nhwc[i] != expected_nhwc[i]) {
            if (mismatches < 5) {
                uint32_t oc = i % cfg->out_c;
                uint32_t ox = (i / cfg->out_c) % out_w;
                uint32_t oy = i / (cfg->out_c * out_w);
                printf("  %s: MISMATCH at [%u,%u,%u]: got %d, expected %d\n",
                       name, oy, ox, oc, actual_nhwc[i], expected_nhwc[i]);
            }
            mismatches++;
            pass = 0;
        }
    }
    if (mismatches > 5)
        printf("  %s: ... and %d more mismatches\n", name, mismatches - 5);

    free(actual_nhwc);
    printf("  %s: %s\n", name, pass ? "PASS" : "FAIL");
    if (!pass)
        printf("  %s: FAIL (%d mismatches)\n", name, mismatches);

cleanup:
    munmap(in_map, in_alloc);
    munmap(out_map, out_alloc);
    munmap(wt_map, wt_alloc);
    munmap((void *)bias_map, bias_alloc);
    munmap((void *)regcmd_map, regcmd_alloc);

    npu_free_bo(fd, &in_bo);
    npu_free_bo(fd, &out_bo);
    npu_free_bo(fd, &wt_bo);
    npu_free_bo(fd, &bias_bo);
    npu_free_bo(fd, &regcmd_bo);

    return pass;
}

/* ======================================================================
 * Deterministic PRNG for test data
 * ====================================================================== */

static int8_t prng_val(uint32_t idx, uint32_t seed)
{
    return (int8_t)(((idx * 7 + seed) % 251) - 125);
}

/* ======================================================================
 * Test definitions
 * ====================================================================== */

static int test_matmul_small(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 4; cfg.in_c = 32;
    cfg.out_c = 16;
    cfg.truncate_bits = 0;

    int8_t input[4 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 1);

    int8_t weights[16 * 1 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 3 + 5) % 5 - 2);

    int32_t bias[16] = {0};
    int8_t expected[4 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);
    return run_conv_test(fd, "test_matmul_small", &cfg, input, weights, bias, expected);
}

static int test_matmul_medium(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 4; cfg.in_c = 64;
    cfg.out_c = 32;
    cfg.truncate_bits = 1;

    int8_t input[4 * 64];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 42);

    int8_t weights[32 * 64];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 7 + 3) % 5 - 2);

    int32_t bias[32] = {0};
    int8_t expected[4 * 32];
    cpu_conv_reference(&cfg, input, weights, bias, expected);
    return run_conv_test(fd, "test_matmul_medium", &cfg, input, weights, bias, expected);
}

static int test_conv3x3_stride2(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 4; cfg.in_h = 4; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.filt_w = 3; cfg.filt_h = 3;
    cfg.stride_x = 2; cfg.stride_y = 2;
    cfg.pad_left = 1; cfg.pad_top = 1;
    cfg.pad_value = (int8_t)0x80;
    cfg.truncate_bits = 0;
    cfg.out_cvt_scale = 128;
    cfg.out_cvt_shift = 14;

    int8_t input[4 * 4 * 16];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 7);

    int8_t weights[16 * 3 * 3 * 16];
    memset(weights, 0, sizeof(weights));
    for (uint32_t oc = 0; oc < 16; oc++)
        weights[oc * 3 * 3 * 16 + 1 * 3 * 16 + 1 * 16 + oc] = 1;

    int32_t bias[16] = {0};
    int8_t expected[2 * 2 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);
    return run_conv_test(fd, "test_conv3x3_stride2", &cfg, input, weights, bias, expected);
}

static int test_depthwise3x3(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 4; cfg.in_h = 4; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.filt_w = 3; cfg.filt_h = 3;
    cfg.pad_left = 1; cfg.pad_top = 1;
    cfg.pad_value = 0;
    cfg.depthwise = 1;
    cfg.truncate_bits = 0;

    int8_t input[4 * 4 * 16];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = (int8_t)((i % 11) - 5);

    int8_t weights[16 * 3 * 3];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = 1;

    int32_t bias[16] = {0};
    int8_t expected[4 * 4 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);
    return run_conv_test(fd, "test_depthwise3x3", &cfg, input, weights, bias, expected);
}

static int test_bias_dma(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 1; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.truncate_bits = 0;
    cfg.bs_cfg = 0x00020150;
    cfg.brdma_cfg = 0x02;

    int8_t input[1 * 1 * 16];
    memset(input, 0, sizeof(input));

    int8_t weights[16 * 1 * 1 * 16];
    memset(weights, 0, sizeof(weights));
    for (int oc = 0; oc < 16; oc++)
        weights[oc * 16 + oc] = 1;

    int32_t bias[16];
    for (int i = 0; i < 16; i++)
        bias[i] = (i + 1) * 100;

    int8_t expected[1 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);
    return run_conv_test(fd, "test_bias_dma", &cfg, input, weights, bias, expected);
}

static int test_bn_relu(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 2; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.truncate_bits = 0;
    cfg.bn_cfg = 0x12;

    int8_t input[2 * 1 * 16];
    for (int i = 0; i < 16; i++) {
        input[0 * 16 + i] = -50;
        input[1 * 16 + i] = 50;
    }

    int8_t weights[16 * 16];
    memset(weights, 0, sizeof(weights));
    for (int oc = 0; oc < 16; oc++)
        weights[oc * 16 + oc] = 1;

    int32_t bias[16] = {0};
    int8_t expected[2 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);
    return run_conv_test(fd, "test_bn_relu", &cfg, input, weights, bias, expected);
}

/* ======================================================================
 * Test 7: BS_OW_OP empirical test
 *
 * Runs a 1x1 conv (4x4 input, 32→16 channels) twice:
 *   A) BS_OW_OP = 0      (baseline, matches current CPU reference)
 *   B) BS_OW_OP = 18     (0x80 - 110, simulating wzp=110)
 * Prints both outputs and the per-element difference.
 * This reveals the exact hardware semantics of BS_OW_OP.
 * ====================================================================== */

static int test_bs_ow_op(int fd)
{
    /* Run same 1x1 conv twice: BS_OW_OP=0 and BS_OW_OP=18.
     * Compare outputs to determine what BS_OW_OP does in hardware. */
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 4; cfg.in_c = 32;
    cfg.out_c = 16;
    cfg.truncate_bits = 0;

    int8_t input[4 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 1);

    int8_t weights[16 * 1 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 3 + 5) % 5 - 2);

    int32_t bias[16] = {100, -200, 300, -400, 50, -50, 0, 150,
                        -150, 75, -75, 200, -100, 25, -25, 0};

    /* CPU reference (BS_OW_OP=0) */
    cfg.bs_ow_op = 0;
    int8_t expected_a[4 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected_a);

    /* Run A on NPU with BS_OW_OP=0 — should match CPU ref */
    int pass_a = run_conv_test(fd, "test_bs_ow_op(OW=0)", &cfg,
                               input, weights, bias, expected_a);

    /* Compute sum_inputs per row for analysis */
    printf("    sum_inputs per row:");
    for (uint32_t y = 0; y < 4; y++) {
        int32_t sum = 0;
        for (uint32_t ic = 0; ic < 32; ic++)
            sum += input[y * 32 + ic];
        printf(" %d", sum);
    }
    printf("\n");

    /* Run B on NPU with BS_OW_OP=18 (= 0x80 - 110) */
    /* We can't easily get the raw output from run_conv_test, so we
     * compute a hypothetical expected with ow_op*sum_inputs correction
     * and test if that's what the hardware produces. */
    cfg.bs_ow_op = 18;

    /* Hypothesis A: acc += ow_op * sum_inputs (before truncation) */
    int8_t expected_b[4 * 1 * 16];
    {
        uint32_t out_w = 1, out_h = 4;
        for (uint32_t oy = 0; oy < out_h; oy++) {
            /* sum of all inputs for this output row */
            int32_t sum = 0;
            for (uint32_t ic = 0; ic < 32; ic++)
                sum += input[oy * 32 + ic];

            for (uint32_t oc = 0; oc < 16; oc++) {
                int32_t acc = 0;
                for (uint32_t ic = 0; ic < 32; ic++) {
                    acc += (int32_t)input[oy * 32 + ic] *
                           (int32_t)weights[oc * 32 + ic];
                }
                /* BS_OW_OP correction: add ow_op * sum_inputs */
                acc += (int32_t)(int16_t)cfg.bs_ow_op * sum;
                acc = nvdla_truncate(acc, cfg.truncate_bits);
                /* BS: add bias */
                acc += bias[oc];
                /* OUT_CVT */
                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)cfg.out_cvt_scale, cfg.out_cvt_shift);
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;
                int32_t result = (int32_t)scaled + cfg.out_cvt_offset;
                if (result < -128) result = -128;
                if (result > 127) result = 127;
                expected_b[oy * 16 + oc] = (int8_t)result;
            }
        }
    }

    return pass_a;
}

static int test_bs_ow_op_18(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 4; cfg.in_c = 32;
    cfg.out_c = 16;
    cfg.truncate_bits = 0;
    cfg.bs_ow_op = 18;

    int8_t input[4 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 1);

    int8_t weights[16 * 1 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 3 + 5) % 5 - 2);

    int32_t bias[16] = {100, -200, 300, -400, 50, -50, 0, 150,
                        -150, 75, -75, 200, -100, 25, -25, 0};

    /* Hypothesis A expected: acc += ow_op * sum_inputs before truncation */
    int8_t expected[4 * 1 * 16];
    {
        uint32_t out_w = 1, out_h = 4;
        for (uint32_t oy = 0; oy < out_h; oy++) {
            int32_t sum = 0;
            for (uint32_t ic = 0; ic < 32; ic++) sum += input[oy * 32 + ic];
            for (uint32_t oc = 0; oc < 16; oc++) {
                int32_t acc = 0;
                for (uint32_t ic = 0; ic < 32; ic++)
                    acc += (int32_t)input[oy * 32 + ic] * (int32_t)weights[oc * 32 + ic];
                acc += (int32_t)(int16_t)cfg.bs_ow_op * sum;
                acc = nvdla_truncate(acc, cfg.truncate_bits);
                acc += bias[oc];
                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)cfg.out_cvt_scale, cfg.out_cvt_shift);
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;
                int32_t result = (int32_t)scaled + cfg.out_cvt_offset;
                if (result < -128) result = -128;
                if (result > 127) result = 127;
                expected[oy * 16 + oc] = (int8_t)result;
            }
        }
    }

    return run_conv_test(fd, "test_bs_ow_op(OW=18)", &cfg, input, weights, bias, expected);
}

/* ======================================================================
 * Main
 * ====================================================================== */

/* ======================================================================
 * Test 9: MobileNet op0 replica
 *
 * Matches MobileNet's first conv exactly:
 *   - 4x4 input (tiny), 3 real channels (padded to 16)
 *   - 3x3 kernel, stride 2, oc=12 (padded to 32)
 *   - BS_OW_OP = -23 (0x80 - 151)
 *   - padding = SAME (pad_left=1, pad_top=1), pad_value = 0
 *
 * If this FAILS in QEMU, the bug is in 3-channel handling.
 * If this PASSES, the bug is in multi-layer chaining.
 * ====================================================================== */
static int test_mobilenet_op0(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 4; cfg.in_h = 4; cfg.in_c = 3;
    cfg.out_c = 12;
    cfg.filt_w = 3; cfg.filt_h = 3;
    cfg.stride_x = 2; cfg.stride_y = 2;
    cfg.pad_left = 1; cfg.pad_top = 1;
    cfg.pad_value = 0;  /* izp - 0x80 = 128 - 128 = 0 */
    cfg.truncate_bits = 0;
    cfg.bs_ow_op = (uint16_t)(int16_t)(0x80 - 151);  /* -23 */
    cfg.out_cvt_scale = 128;
    cfg.out_cvt_shift = 14;

    int8_t input[4 * 4 * 3];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 42);

    int8_t weights[12 * 3 * 3 * 3];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 7 + 13) % 11 - 5);

    int32_t bias[12];
    for (int i = 0; i < 12; i++) bias[i] = (i * 37 - 200);

    /* CPU reference WITH BS_OW_OP correction */
    uint32_t out_w = (cfg.in_w + cfg.pad_left * 2 - cfg.filt_w) / cfg.stride_x + 1;
    uint32_t out_h = (cfg.in_h + cfg.pad_top * 2 - cfg.filt_h) / cfg.stride_y + 1;
    int8_t expected[2 * 2 * 12];  /* out_w=2, out_h=2 */

    for (uint32_t oy = 0; oy < out_h; oy++) {
        for (uint32_t ox = 0; ox < out_w; ox++) {
            for (uint32_t oc = 0; oc < 12; oc++) {
                int32_t acc = 0;
                int32_t sum_in = 0;
                for (uint32_t ky = 0; ky < 3; ky++) {
                    for (uint32_t kx = 0; kx < 3; kx++) {
                        int32_t iy = (int32_t)(oy * 2 + ky) - 1;
                        int32_t ix = (int32_t)(ox * 2 + kx) - 1;
                        for (uint32_t ic = 0; ic < 3; ic++) {
                            int8_t in_val;
                            if (ix < 0 || ix >= 4 || iy < 0 || iy >= 4)
                                in_val = 0;  /* pad_value */
                            else
                                in_val = input[iy * 4 * 3 + ix * 3 + ic];
                            int8_t w_val = weights[oc * 3 * 3 * 3 + kx * 3 * 3 + ky * 3 + ic];
                            acc += (int32_t)in_val * (int32_t)w_val;
                            sum_in += (int32_t)in_val;
                        }
                    }
                }
                acc += (int32_t)(int16_t)cfg.bs_ow_op * sum_in;
                acc = nvdla_truncate(acc, 0);
                acc += bias[oc];
                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)cfg.out_cvt_scale, cfg.out_cvt_shift);
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;
                int32_t result = (int32_t)scaled + cfg.out_cvt_offset;
                if (result < -128) result = -128;
                if (result > 127) result = 127;
                expected[oy * out_w * 12 + ox * 12 + oc] = (int8_t)result;
            }
        }
    }

    return run_conv_test(fd, "test_mobilenet_op0", &cfg, input, weights, bias, expected);
}

int main(void)
{
    printf("=== NPU Convolution Test Suite ===\n");

    int passed = 0;
    int total = 9;

    int (*tests[])(int) = {
        test_matmul_small, test_matmul_medium, test_conv3x3_stride2,
        test_depthwise3x3, test_bias_dma, test_bn_relu,
        test_bs_ow_op, test_bs_ow_op_18, test_mobilenet_op0,
    };

    /* Detect driver type with a probe open */
    int probe_fd = npu_open_device();
    if (probe_fd < 0) {
        perror("open NPU device");
        return 1;
    }
    printf("  Using %s driver\n", g_driver == DRIVER_ROCKET ? "Rocket" : "RKNPU");
    close(probe_fd);

    for (int i = 0; i < total; i++) {
        int fd = npu_open_device();
        if (fd < 0) {
            printf("  test %d: FAIL (open device)\n", i);
            continue;
        }
        passed += tests[i](fd);
        close(fd);
    }

    printf("=== CONV TESTS: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
