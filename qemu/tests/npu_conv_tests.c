/*
 * NPU Convolution Test Suite for QEMU Rocket NPU emulator.
 *
 * Six tests covering: multi-channel matmul, stride/padding, depthwise,
 * DMA bias, and BN ReLU — all code paths that have zero isolated test
 * coverage in npu_test.c.
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

/* ======================================================================
 * DRM / Rocket ioctl definitions (copied from npu_test.c)
 * ====================================================================== */

#define DRM_IOCTL_BASE 'd'
#define DRM_COMMAND_BASE 0x40

#define DRM_ROCKET_CREATE_BO 0x00
#define DRM_ROCKET_SUBMIT    0x01
#define DRM_ROCKET_PREP_BO   0x02
#define DRM_ROCKET_FINI_BO   0x03

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

#define DRM_IOCTL_ROCKET_CREATE_BO _IOWR(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_CREATE_BO, struct drm_rocket_create_bo)
#define DRM_IOCTL_ROCKET_SUBMIT    _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_SUBMIT, struct drm_rocket_submit)
#define DRM_IOCTL_ROCKET_PREP_BO   _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_PREP_BO, struct drm_rocket_prep_bo)
#define DRM_IOCTL_ROCKET_FINI_BO   _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_ROCKET_FINI_BO, struct drm_rocket_fini_bo)

struct drm_gem_close {
    uint32_t handle;
    uint32_t pad;
};
#define DRM_IOCTL_GEM_CLOSE _IOW(DRM_IOCTL_BASE, 0x09, struct drm_gem_close)

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

static int create_bo(int fd, uint32_t size, uint32_t *handle,
                     uint64_t *dma_addr, uint64_t *offset)
{
    struct drm_rocket_create_bo req = { .size = size };
    int ret = ioctl(fd, DRM_IOCTL_ROCKET_CREATE_BO, &req);
    if (ret) { perror("CREATE_BO"); return -1; }
    *handle = req.handle;
    *dma_addr = req.dma_address;
    *offset = req.offset;
    return 0;
}

/* ======================================================================
 * NVDLA rounding helpers (matching rockchip-npu.c exactly)
 * ====================================================================== */

static inline int32_t nvdla_truncate(int32_t value, unsigned truncate)
{
    if (truncate == 0) return value;
    int32_t sign = (value < 0) ? 1 : 0;
    uint32_t guide = (value >> (truncate - 1)) & 1;
    uint32_t sticky = (truncate > 1)
        ? ((value & ((1u << (truncate - 1)) - 1)) != 0) : 0;
    int32_t round_up = guide & ((!sign) | sticky);
    return (value >> truncate) + round_up;
}

static inline int64_t nvdla_shift_right_round64(int64_t value, unsigned shift)
{
    if (shift == 0) return value;
    int64_t sign = (value < 0) ? 1 : 0;
    uint64_t guide = (value >> (shift - 1)) & 1;
    uint64_t sticky = (shift > 1)
        ? ((value & ((1ULL << (shift - 1)) - 1)) != 0) : 0;
    int64_t round_up = guide & ((!sign) | sticky);
    return (value >> shift) + round_up;
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
 * Must match read_weight() in rockchip-npu.c
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
                uint32_t ic_limit = depthwise ? in_c : in_c;
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
                        /* flat layout for depthwise: [ic][kx][ky] */
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
                /* x-major: [g][x][y][c16] */
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
                /* y-major: [g][y][x][c16] */
                uint32_t off = g * w * h * NPU_FEATURE_ATOMIC_SIZE
                             + y * w * NPU_FEATURE_ATOMIC_SIZE
                             + x * NPU_FEATURE_ATOMIC_SIZE
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
                        int32_t iy = (int32_t)(oy * cfg->stride_y + ky)
                                   - (int32_t)cfg->pad_top;
                        int32_t ix = (int32_t)(ox * cfg->stride_x + kx)
                                   - (int32_t)cfg->pad_left;

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
                            if (cfg->depthwise) {
                                w_val = weights_flat[oc * cfg->filt_w * cfg->filt_h
                                                   + kx * cfg->filt_h + ky];
                            } else {
                                w_val = weights_flat[oc * cfg->filt_w * cfg->filt_h * cfg->in_c
                                                   + kx * cfg->filt_h * cfg->in_c
                                                   + ky * cfg->in_c + abs_ic];
                            }

                            acc += (int32_t)in_val * (int32_t)w_val;
                        }
                    }
                }

                /* CACC truncation */
                acc = nvdla_truncate(acc, cfg->truncate_bits);

                /* BS stage */
                if (!(cfg->bs_cfg & 0x01)) {
                    /* ALU sub-stage */
                    if (!(cfg->bs_cfg & 0x02)) {
                        uint32_t algo = (cfg->bs_cfg >> 16) & 0xf;
                        int32_t bs_alu_op;
                        int bs_alu_src = (cfg->bs_cfg >> 8) & 1;
                        if (bs_alu_src && bias) {
                            bs_alu_op = bias[oc];
                        } else {
                            bs_alu_op = cfg->bs_alu_cfg;
                        }
                        switch (algo) {
                        case 0: if (acc < bs_alu_op) acc = bs_alu_op; break;
                        case 1: if (acc > bs_alu_op) acc = bs_alu_op; break;
                        case 2: acc += bs_alu_op; break;
                        }
                    }
                    /* MUL and ReLU bypass in default config */
                }

                /* BN stage */
                if (!(cfg->bn_cfg & 0x01)) {
                    /* ReLU sub-stage */
                    if (!(cfg->bn_cfg & 0x40)) {
                        if (acc < 0) acc = 0;
                    }
                }

                /* OUT_CVT */
                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)cfg->out_cvt_scale,
                    cfg->out_cvt_shift);
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;
                int32_t result = (int32_t)scaled + cfg->out_cvt_offset;
                if (result < -128) result = -128;
                if (result > 127) result = 127;

                output_nhwc[oy * out_w * cfg->out_c + ox * cfg->out_c + oc]
                    = (int8_t)result;
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

    /* CNA registers */
    *p++ = emit(TARGET_CNA, 0x1040, 0x00000b01);  /* CBUF_CON0 */
    *p++ = emit(TARGET_CNA, 0x1100, 0);           /* DCOMP_REGNUM */
    *p++ = emit(TARGET_CNA, 0x1104, 0);           /* DCOMP_CTRL */
    *p++ = emit(TARGET_CNA, 0x100c, cfg->depthwise ? 3 : 0);  /* CONV_CON1 */

    /* DPU/RDMA S_POINTER */
    *p++ = emit(TARGET_DPU, 0x4004, 0x00070007);
    *p++ = emit(TARGET_RDMA, 0x5004, 0x00070007);

    *p++ = emit(TARGET_CNA, 0x100c, cfg->depthwise ? 3 : 0);
    *p++ = emit(TARGET_CNA, 0x1010, 52);          /* CONV_CON2: feature_grains */

    /* CONV_CON3: stride */
    uint32_t conv_con3 = (cfg->stride_x & 0x7) | ((cfg->stride_y & 0x7) << 3);
    *p++ = emit(TARGET_CNA, 0x1014, conv_con3);

    /* Input dimensions (CNA stores raw values) */
    *p++ = emit(TARGET_CNA, 0x1020, cfg->in_h | (cfg->in_w << 16));
    *p++ = emit(TARGET_CNA, 0x1024, ((cfg->in_c - 1) << 16) | in_c_padded);
    *p++ = emit(TARGET_CNA, 0x1028, out_w);       /* DATA_SIZE2 */
    *p++ = emit(TARGET_CNA, 0x102c, 1);           /* DATA_SIZE3: atomics */

    /* Weights */
    *p++ = emit(TARGET_CNA, 0x1030, wt_size0);
    *p++ = emit(TARGET_CNA, 0x1034, wt_size1);
    *p++ = emit(TARGET_CNA, 0x1038, wt_kernels
                | (cfg->filt_h << 16) | (cfg->filt_w << 24));

    *p++ = emit(TARGET_CNA, 0x1040, 0x00000b01);
    *p++ = emit(TARGET_CNA, 0x1044, 0x00ff0000);  /* CBUF_CON1 */

    /* CVT: signed bypass */
    *p++ = emit(TARGET_CNA, 0x1048, 0x00000030);
    *p++ = emit(TARGET_CNA, 0x104c, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x1050, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x1054, 0x00010000);
    *p++ = emit(TARGET_CNA, 0x1058, 0x00010000);

    *p++ = emit(TARGET_CNA, 0x105c, 0);           /* FC_CON0 */
    *p++ = emit(TARGET_CNA, 0x1060, 0);           /* FC_CON1 */

    /* Padding */
    uint32_t pad_con0 = (cfg->pad_top & 0xf) | ((cfg->pad_left & 0xf) << 4);
    *p++ = emit(TARGET_CNA, 0x1068, pad_con0);

    /* Input DMA address */
    *p++ = emit(TARGET_CNA, 0x1070, in_addr);
    *p++ = emit(TARGET_CNA, 0x1074, 0);           /* FC_CON2 */
    *p++ = emit(TARGET_CNA, 0x1078, 0x0f0f0000);  /* DMA_CON0 */

    uint32_t line_stride = cfg->in_h * NPU_FEATURE_ATOMIC_SIZE;
    uint32_t surf_stride = cfg->in_w * cfg->in_h * NPU_FEATURE_ATOMIC_SIZE;
    *p++ = emit(TARGET_CNA, 0x107c, line_stride);
    *p++ = emit(TARGET_CNA, 0x1080, surf_stride);

    *p++ = emit(TARGET_CNA, 0x1084, 0);
    *p++ = emit(TARGET_CNA, 0x1088, 0x000f0000);
    *p++ = emit(TARGET_CNA, 0x1104, 0);
    *p++ = emit(TARGET_CNA, 0x1100, 0);

    /* Weight DMA address */
    *p++ = emit(TARGET_CNA, 0x1110, wt_addr);

    /* DCOMP_AMOUNT0..15 = 0 */
    for (int i = 0; i < 16; i++)
        *p++ = emit(TARGET_CNA, 0x1114 + i * 4, 0);

    *p++ = emit(TARGET_CNA, 0x10a0, 0);           /* CVT_CON5 */
    *p++ = emit(TARGET_CNA, 0x1184, (uint32_t)(int32_t)cfg->pad_value);

    /* CORE registers */
    uint32_t misc_cfg = 1 | (cfg->depthwise ? 2 : 0);
    *p++ = emit(TARGET_CORE, 0x3010, misc_cfg);
    *p++ = emit(TARGET_CORE, 0x3014, (out_w - 1) | ((out_h - 1) << 16));
    *p++ = emit(TARGET_CORE, 0x3018, cfg->out_c - 1);
    *p++ = emit(TARGET_CORE, 0x301c, cfg->truncate_bits);
    *p++ = emit(0x0801, 0x3030, 0);

    /* DPU registers */
    *p++ = emit(TARGET_DPU, 0x400c, 0x0000020f);  /* FEAT_MODE_CFG */
    *p++ = emit(TARGET_DPU, 0x4010, 0);           /* DATA_FORMAT */
    *p++ = emit(TARGET_DPU, 0x4014, 0);           /* OFFSET_PEND */

    /* Output DMA */
    *p++ = emit(TARGET_DPU, 0x4020, out_addr);
    uint32_t dst_surf_stride = out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;
    *p++ = emit(TARGET_DPU, 0x4024, dst_surf_stride << 4);
    *p++ = emit(TARGET_DPU, 0x4030, out_w - 1);
    *p++ = emit(TARGET_DPU, 0x4034, out_h - 1);
    *p++ = emit(TARGET_DPU, 0x4038, 0);           /* DATA_CUBE_NOTCH */
    *p++ = emit(TARGET_DPU, 0x403c,
                ((cfg->out_c - 1) << 16) | (cfg->out_c - 1));

    /* BS stage */
    *p++ = emit(TARGET_DPU, 0x4040, cfg->bs_cfg);
    *p++ = emit(TARGET_DPU, 0x4044, (uint32_t)cfg->bs_alu_cfg);
    *p++ = emit(TARGET_DPU, 0x4048, 0);           /* BS_MUL_CFG */
    *p++ = emit(TARGET_DPU, 0x404c, 0);           /* BS_RELUX_CMP */
    *p++ = emit(TARGET_DPU, 0x4050, 0x00010101);  /* BS_OW_CFG */
    *p++ = emit(TARGET_DPU, 0x4054, 0);           /* BS_OW_OP */

    *p++ = emit(TARGET_DPU, 0x4058, cfg->out_c - 1);  /* WDMA_SIZE_0 */
    *p++ = emit(TARGET_DPU, 0x405c, (out_w - 1) | ((out_h - 1) << 16));

    /* BN stage */
    *p++ = emit(TARGET_DPU, 0x4060, cfg->bn_cfg);
    *p++ = emit(TARGET_DPU, 0x4064, 0);
    *p++ = emit(TARGET_DPU, 0x4068, 0);
    *p++ = emit(TARGET_DPU, 0x406c, 0);

    /* EW: all bypass */
    *p++ = emit(TARGET_DPU, 0x4070, 0x00000383);
    *p++ = emit(TARGET_DPU, 0x4074, 0);
    *p++ = emit(TARGET_DPU, 0x4078, 0x00010000);
    *p++ = emit(TARGET_DPU, 0x407c, 0);

    /* OUT_CVT */
    *p++ = emit(TARGET_DPU, 0x4080, (uint32_t)cfg->out_cvt_offset);
    *p++ = emit(TARGET_DPU, 0x4084, cfg->out_cvt_scale);
    *p++ = emit(TARGET_DPU, 0x4088, cfg->out_cvt_shift);

    /* EW_OP values */
    for (int i = 0; i < 8; i++)
        *p++ = emit(TARGET_DPU, 0x4090 + i * 4, 0);

    *p++ = emit(TARGET_DPU, 0x40c0, 1);           /* SURFACE_ADD */
    *p++ = emit(0x1001, 0x40c4, 0);

    /* LUT registers (all zero) */
    *p++ = emit(TARGET_DPU, 0x40b4, 0);
    *p++ = emit(TARGET_DPU, 0x40b8, 0);
    *p++ = emit(TARGET_DPU, 0x40bc, 0);
    *p++ = emit(TARGET_DPU, 0x40c0, 0);
    for (int i = 0; i < 8; i++)
        *p++ = emit(TARGET_DPU, 0x40c8 + i * 4, 0);

    /* RDMA registers */
    *p++ = emit(TARGET_RDMA, 0x5008, 0);
    *p++ = emit(TARGET_RDMA, 0x500c, 0);
    *p++ = emit(TARGET_RDMA, 0x5010, cfg->out_c - 1);

    *p++ = emit(TARGET_RDMA, 0x5014, 0);          /* RDMA_SRC_BASE_ADDR */

    /* BRDMA: bias config */
    *p++ = emit(TARGET_RDMA, 0x501c, cfg->brdma_cfg);
    *p++ = emit(TARGET_RDMA, 0x5020, bias_addr);
    *p++ = emit(TARGET_RDMA, 0x5028, 0);          /* NRDMA_CFG */
    *p++ = emit(TARGET_RDMA, 0x502c, 0);          /* BN_BASE_ADDR */
    *p++ = emit(TARGET_RDMA, 0x5034, 0x00000001); /* ERDMA_CFG: disable */
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
                                              cfg->filt_w, cfg->filt_h,
                                              cfg->depthwise);

    uint32_t out_c_padded = ALIGN_UP(cfg->out_c, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t out_groups = out_c_padded / NPU_FEATURE_ATOMIC_SIZE;
    uint32_t out_buf_size = out_groups * out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;

    uint32_t bias_buf_size = cfg->out_c * sizeof(int32_t);

    /* Align sizes to page */
    uint32_t in_alloc = ALIGN_UP(MAX2(in_buf_size, 4096), 4096);
    uint32_t wt_alloc = ALIGN_UP(MAX2(wt_buf_size, 4096), 4096);
    uint32_t out_alloc = ALIGN_UP(MAX2(out_buf_size, 4096), 4096);
    uint32_t bias_alloc = ALIGN_UP(MAX2(bias_buf_size, 4096), 4096);
    uint32_t regcmd_alloc = 4096;

    /* Create BOs */
    uint32_t in_h_bo, out_h_bo, wt_h_bo, bias_h_bo, regcmd_h_bo;
    uint64_t in_dma, out_dma, wt_dma, bias_dma, regcmd_dma;
    uint64_t in_off, out_off, wt_off, bias_off, regcmd_off;

    if (create_bo(fd, in_alloc, &in_h_bo, &in_dma, &in_off) ||
        create_bo(fd, out_alloc, &out_h_bo, &out_dma, &out_off) ||
        create_bo(fd, wt_alloc, &wt_h_bo, &wt_dma, &wt_off) ||
        create_bo(fd, bias_alloc, &bias_h_bo, &bias_dma, &bias_off) ||
        create_bo(fd, regcmd_alloc, &regcmd_h_bo, &regcmd_dma, &regcmd_off)) {
        printf("  %s: FAIL (BO creation)\n", name);
        return 0;
    }

    /* Map BOs */
    uint8_t *in_map = mmap(NULL, in_alloc, PROT_READ|PROT_WRITE, MAP_SHARED, fd, in_off);
    uint8_t *out_map = mmap(NULL, out_alloc, PROT_READ|PROT_WRITE, MAP_SHARED, fd, out_off);
    uint8_t *wt_map = mmap(NULL, wt_alloc, PROT_READ|PROT_WRITE, MAP_SHARED, fd, wt_off);
    int32_t *bias_map = (int32_t *)mmap(NULL, bias_alloc, PROT_READ|PROT_WRITE, MAP_SHARED, fd, bias_off);
    uint64_t *regcmd_map = (uint64_t *)mmap(NULL, regcmd_alloc, PROT_READ|PROT_WRITE, MAP_SHARED, fd, regcmd_off);

    if (in_map == MAP_FAILED || out_map == MAP_FAILED || wt_map == MAP_FAILED ||
        bias_map == MAP_FAILED || regcmd_map == MAP_FAILED) {
        printf("  %s: FAIL (mmap)\n", name);
        return 0;
    }

    /* Fill input */
    memset(in_map, 0, in_alloc);
    nhwc_to_npu_input(input_nhwc, in_map, cfg->in_w, cfg->in_h, cfg->in_c);

    /* Fill weights */
    memset(wt_map, 0, wt_alloc);
    pack_weights(weights_flat, wt_map, cfg->out_c, cfg->in_c,
                 cfg->filt_w, cfg->filt_h, cfg->depthwise);

    /* Fill bias */
    memset(bias_map, 0, bias_alloc);
    if (bias) {
        memcpy(bias_map, bias, bias_buf_size);
    }

    /* Clear output */
    memset(out_map, 0xff, out_alloc);

    /* Build regcmd */
    unsigned regcmd_count = build_conv_regcmd(regcmd_map, cfg,
        (uint32_t)in_dma, (uint32_t)wt_dma,
        (uint32_t)out_dma, (uint32_t)bias_dma);

    /* Sync BOs */
    struct drm_rocket_fini_bo fini = { .reserved = 0 };
    fini.handle = in_h_bo; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    fini.handle = wt_h_bo; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    fini.handle = bias_h_bo; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);
    fini.handle = regcmd_h_bo; ioctl(fd, DRM_IOCTL_ROCKET_FINI_BO, &fini);

    /* Submit */
    struct drm_rocket_task task = {
        .regcmd = (uint32_t)regcmd_dma,
        .regcmd_count = regcmd_count * 2,
    };
    uint32_t in_handles[] = { in_h_bo, wt_h_bo, bias_h_bo, regcmd_h_bo };
    uint32_t out_handles[] = { out_h_bo };
    struct drm_rocket_job job = {
        .tasks = (uint64_t)(uintptr_t)&task,
        .in_bo_handles = (uint64_t)(uintptr_t)in_handles,
        .out_bo_handles = (uint64_t)(uintptr_t)out_handles,
        .task_count = 1,
        .task_struct_size = sizeof(struct drm_rocket_task),
        .in_bo_handle_count = 4,
        .out_bo_handle_count = 1,
    };
    struct drm_rocket_submit submit = {
        .jobs = (uint64_t)(uintptr_t)&job,
        .job_count = 1,
        .job_struct_size = sizeof(struct drm_rocket_job),
    };

    int pass = 0;
    int ret = ioctl(fd, DRM_IOCTL_ROCKET_SUBMIT, &submit);
    if (ret) {
        printf("  %s: FAIL (SUBMIT: %s)\n", name, strerror(errno));
        goto cleanup;
    }

    /* Wait for completion */
    struct drm_rocket_prep_bo prep = {
        .handle = out_h_bo,
        .timeout_ns = 5000000000LL,
    };
    ret = ioctl(fd, DRM_IOCTL_ROCKET_PREP_BO, &prep);
    if (ret < 0) {
        printf("  %s: FAIL (PREP_BO ret=%d errno=%d %s)\n", name, ret, errno, strerror(errno));
        goto cleanup;
    }

    /* Read output and compare */
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
    if (mismatches > 5) {
        printf("  %s: ... and %d more mismatches\n", name, mismatches - 5);
    }

    free(actual_nhwc);

    if (pass) {
        printf("  %s: PASS\n", name);
    } else {
        printf("  %s: FAIL (%d mismatches)\n", name, mismatches);
    }

cleanup:
    munmap(in_map, in_alloc);
    munmap(out_map, out_alloc);
    munmap(wt_map, wt_alloc);
    munmap((void *)bias_map, bias_alloc);
    munmap((void *)regcmd_map, regcmd_alloc);

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
 * Test 1: 1x1 conv, M=4 K=32 N=16 (matmul small)
 * ====================================================================== */

static int test_matmul_small(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 4; cfg.in_c = 32;
    cfg.out_c = 16;
    cfg.filt_w = 1; cfg.filt_h = 1;
    cfg.truncate_bits = 0;

    /* Generate input */
    int8_t input[4 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 1);

    /* Generate weights: small values to avoid overflow */
    int8_t weights[16 * 1 * 1 * 32];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 3 + 5) % 5 - 2);  /* range: -2..2 */

    /* Zero bias */
    int32_t bias[16] = {0};

    /* CPU reference */
    int8_t expected[4 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);

    return run_conv_test(fd, "test_matmul_small", &cfg,
                         input, weights, bias, expected);
}

/* ======================================================================
 * Test 2: 1x1 conv, M=4 K=64 N=32 (matmul medium, truncation)
 * ====================================================================== */

static int test_matmul_medium(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 4; cfg.in_c = 64;
    cfg.out_c = 32;
    cfg.filt_w = 1; cfg.filt_h = 1;
    cfg.truncate_bits = 1;

    /* Generate input */
    int8_t input[4 * 64];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 42);

    /* Weights: small to keep accumulators manageable */
    int8_t weights[32 * 64];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = (int8_t)((i * 7 + 3) % 5 - 2);

    int32_t bias[32] = {0};

    int8_t expected[4 * 32];
    cpu_conv_reference(&cfg, input, weights, bias, expected);

    return run_conv_test(fd, "test_matmul_medium", &cfg,
                         input, weights, bias, expected);
}

/* ======================================================================
 * Test 3: 3x3 conv, stride=2, padding=1
 * ====================================================================== */

static int test_conv3x3_stride2(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 4; cfg.in_h = 4; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.filt_w = 3; cfg.filt_h = 3;
    cfg.stride_x = 2; cfg.stride_y = 2;
    cfg.pad_left = 1; cfg.pad_top = 1;
    cfg.pad_value = (int8_t)0x80;  /* -128 */
    cfg.truncate_bits = 0;

    /* Scale down to avoid overflow with 3x3x16 accumulations */
    cfg.out_cvt_scale = 128;
    cfg.out_cvt_shift = 14;

    /* Input */
    int8_t input[4 * 4 * 16];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = prng_val(i, 7);

    /* Weights: very small to prevent saturation */
    int8_t weights[16 * 3 * 3 * 16];
    memset(weights, 0, sizeof(weights));
    for (uint32_t oc = 0; oc < 16; oc++) {
        /* Just one non-zero weight per output channel: center pixel, same ic */
        weights[oc * 3 * 3 * 16 + 1 * 3 * 16 + 1 * 16 + oc] = 1;
    }

    int32_t bias[16] = {0};

    int8_t expected[2 * 2 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);

    return run_conv_test(fd, "test_conv3x3_stride2", &cfg,
                         input, weights, bias, expected);
}

/* ======================================================================
 * Test 4: Depthwise 3x3 conv, stride=1, padding=1
 * ====================================================================== */

static int test_depthwise3x3(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 4; cfg.in_h = 4; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.filt_w = 3; cfg.filt_h = 3;
    cfg.stride_x = 1; cfg.stride_y = 1;
    cfg.pad_left = 1; cfg.pad_top = 1;
    cfg.pad_value = 0;
    cfg.depthwise = 1;
    cfg.truncate_bits = 0;

    /* Input: small values */
    int8_t input[4 * 4 * 16];
    for (uint32_t i = 0; i < sizeof(input); i++)
        input[i] = (int8_t)((i % 11) - 5);  /* range: -5..5 */

    /* Depthwise weights: [ic][kx][ky], all 1 (box filter) */
    int8_t weights[16 * 3 * 3];
    for (uint32_t i = 0; i < sizeof(weights); i++)
        weights[i] = 1;

    int32_t bias[16] = {0};

    int8_t expected[4 * 4 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);

    return run_conv_test(fd, "test_depthwise3x3", &cfg,
                         input, weights, bias, expected);
}

/* ======================================================================
 * Test 5: 1x1 conv with per-channel DMA bias
 * ====================================================================== */

static int test_bias_dma(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 1; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.filt_w = 1; cfg.filt_h = 1;
    cfg.truncate_bits = 0;
    /* BS: bypass=0, ALU_BYPASS=0, ALU_SRC=1(DMA), ALU_ALGO=2(ADD),
     * MUL_BYPASS=1, RELU_BYPASS=1 */
    cfg.bs_cfg = 0x00020150;
    cfg.brdma_cfg = 0x02;

    /* Input: all zeros */
    int8_t input[1 * 1 * 16];
    memset(input, 0, sizeof(input));

    /* Weights: identity */
    int8_t weights[16 * 1 * 1 * 16];
    memset(weights, 0, sizeof(weights));
    for (int oc = 0; oc < 16; oc++)
        weights[oc * 16 + oc] = 1;

    /* Bias: 100, 200, ..., 1600 */
    int32_t bias[16];
    for (int i = 0; i < 16; i++)
        bias[i] = (i + 1) * 100;

    int8_t expected[1 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);

    return run_conv_test(fd, "test_bias_dma", &cfg,
                         input, weights, bias, expected);
}

/* ======================================================================
 * Test 6: BN stage with ReLU (clamp negatives to 0)
 * ====================================================================== */

static int test_bn_relu(int fd)
{
    struct conv_config cfg;
    conv_config_defaults(&cfg);
    cfg.in_w = 1; cfg.in_h = 2; cfg.in_c = 16;
    cfg.out_c = 16;
    cfg.filt_w = 1; cfg.filt_h = 1;
    cfg.truncate_bits = 0;

    /* BN_CFG=0x12: bypass=0, alu_bypass=1, mul_bypass=1(bit4),
     * relu_bypass=0(bit6=0) → ReLU active
     * 0x12 = 0b00010010 → bypass=0, alu_bypass=1, bit2=0, bit3=0,
     *                      mul_bypass=1, prelu=0, relu_bypass=0
     */
    cfg.bn_cfg = 0x12;

    /* Input: row0 = -50 (all channels), row1 = +50 */
    int8_t input[2 * 1 * 16];
    for (int i = 0; i < 16; i++) {
        input[0 * 16 + i] = -50;
        input[1 * 16 + i] = 50;
    }

    /* Weights: identity */
    int8_t weights[16 * 16];
    memset(weights, 0, sizeof(weights));
    for (int oc = 0; oc < 16; oc++)
        weights[oc * 16 + oc] = 1;

    int32_t bias[16] = {0};

    /* Expected: row0 = 0 (ReLU clamps -50), row1 = 50 */
    int8_t expected[2 * 1 * 16];
    cpu_conv_reference(&cfg, input, weights, bias, expected);

    return run_conv_test(fd, "test_bn_relu", &cfg,
                         input, weights, bias, expected);
}

/* ======================================================================
 * Main
 * ====================================================================== */

int main(void)
{
    printf("=== NPU Convolution Test Suite ===\n");

    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/accel/accel0");
        return 1;
    }

    int passed = 0;
    int total = 6;

    passed += test_matmul_small(fd);
    passed += test_matmul_medium(fd);
    passed += test_conv3x3_stride2(fd);
    passed += test_depthwise3x3(fd);
    passed += test_bias_dma(fd);
    passed += test_bn_relu(fd);

    close(fd);

    printf("=== CONV TESTS: %d/%d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
