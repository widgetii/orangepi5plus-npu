/*
 * Rockchip RK3588 NPU device model
 *
 * Emulates the RK3588 NPU's register interface, regcmd parser, and INT8
 * convolution engine. Supports both the upstream Rocket DRM driver and the
 * vendor rknpu DRM driver — both program the same hardware registers via
 * MMIO, differing only in kernel-side ioctl handling and DT bindings.
 *
 * IOVA→GPA translation uses the Rockchip IOMMU v2 model (page table walk).
 * The Rocket driver additionally requires iommu.passthrough=1 on the kernel
 * command line (bypassing the IOMMU for DMA addresses).
 *
 * Execution model:
 *   1. Kernel writes PC_BASE_ADDRESS and PC_REGISTER_AMOUNTS via MMIO
 *   2. Kernel writes PC_OPERATION_ENABLE = 1
 *   3. QEMU: fetch regcmd buffer from guest memory (address_space_read)
 *   4. Parse regcmd entries into RocketConvTask
 *   5. Execute INT8 convolution in software
 *   6. Raise IRQ (lowered when guest writes PC_INTERRUPT_CLEAR)
 *
 * Multi-core: 3 independent cores, each with own MMIO + IRQ.
 * Execution is synchronous within the MMIO write handler (deterministic).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/error-report.h"
#include "qemu/timer.h"
#include "hw/misc/rockchip-npu.h"
#include "hw/misc/rockchip-iommu.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "system/address-spaces.h"
#include "migration/vmstate.h"
#include "trace.h"


/* Version that the Rocket kernel driver checks during probe */
#define ROCKET_PC_VERSION      0x00010001
#define ROCKET_PC_VERSION_NUM  0x00000001

/* Helper macros */
#define ALIGN_UP(x, a)    (((x) + (a) - 1) & ~((a) - 1))
#define DIV_ROUND_UP(n,d) (((n) + (d) - 1) / (d))
#define MIN2(a, b)        ((a) < (b) ? (a) : (b))
#define MAX2(a, b)        ((a) > (b) ? (a) : (b))

/* ======================================================================
 * Regcmd Parser — extract convolution parameters from packed 64-bit entries
 * ====================================================================== */

static void parse_regcmd_entry(RocketConvTask *task, uint64_t entry)
{
    uint32_t reg = (uint32_t)(entry & REGCMD_REG_MASK);
    uint32_t val = (uint32_t)((entry & REGCMD_VALUE_MASK) >> REGCMD_VALUE_SHIFT);
    uint16_t target = (uint16_t)(entry >> REGCMD_TARGET_SHIFT);

    if (target == 0x0041 || target == 0x0081) {
        return;
    }
    if (entry == 0) {
        return;
    }

    switch (reg) {
    case 0x1070: task->src_addr = val; break;
    case 0x1020:
        task->input_height = val & 0x7ff;
        task->input_width = (val >> 16) & 0x7ff;
        break;
    case 0x1024:
        task->input_channels = val & 0xffff;
        task->input_channels_real = ((val >> 16) & 0x3fff) + 1;
        break;
    case 0x1028: break;
    case 0x1110: task->weight_addr = val; break;
    case 0x1038:
        task->weight_kernels = val & 0x3fff;
        task->weight_height = (val >> 16) & 0x1f;
        task->weight_width = (val >> 24) & 0x1f;
        break;
    case 0x1030: task->weight_size0 = val; break;
    case 0x1034: task->weight_size1 = val; break;
    case 0x100c:
        task->conv_con1 = val;
        task->depthwise = (val & 0xf) == 3;
        break;
    case 0x1014:
        task->stride_x = val & 0x7;
        task->stride_y = (val >> 3) & 0x7;
        if (task->stride_x == 0) task->stride_x = 1;
        if (task->stride_y == 0) task->stride_y = 1;
        break;
    case 0x1068:
        task->pad_top = val & 0xf;
        task->pad_left = (val >> 4) & 0xf;
        break;
    case 0x1184: task->pad_value = (int32_t)val; break;
    case 0x107c: task->input_line_stride = val & 0x0fffffff; break;
    case 0x1080: task->input_surface_stride = val & 0x0fffffff; break;
    case 0x3010:
        task->depthwise = task->depthwise || ((val >> 1) & 1);
        break;
    case 0x3014:
        task->output_width = (val & 0xffff) + 1;
        task->output_height = ((val >> 16) & 0xffff) + 1;
        break;
    case 0x3018:
        task->output_channels = (val & 0x1fff) + 1;
        task->output_channels_real = task->output_channels;
        break;
    case 0x301c: task->truncate_bits = val & 0x1f; break;
    case 0x4020: task->dst_addr = val; break;
    case 0x4024: task->output_surface_stride = (val >> 4) & 0x0fffffff; break;
    case 0x4030: break;
    case 0x4034: break;
    case 0x403c:
        task->output_channels = (val & 0x1fff) + 1;
        task->output_channels_real = ((val >> 16) & 0x1fff) + 1;
        break;
    case 0x4040: task->bs_cfg = val; break;
    case 0x4044: task->bs_alu_cfg = (int32_t)val; break;
    case 0x4048: task->bs_mul_cfg = val; break;
    case 0x404c: task->bs_relux_cmp = val; break;
    case 0x4054: task->bs_ow_op = val & 0xffff; break;
    case 0x4060: task->bn_cfg = val; break;
    case 0x4064: task->bn_alu_cfg = (int32_t)val; break;
    case 0x4068: task->bn_mul_cfg = val; break;
    case 0x406c: task->bn_relux_cmp = val; break;
    case 0x4070: task->ew_cfg = val; break;
    case 0x4074: task->ew_cvt_offset = (int32_t)val; break;
    case 0x4078: task->ew_cvt_scale = val; break;
    case 0x4080: task->out_cvt_offset = val; break;
    case 0x4084: task->out_cvt_scale = val & 0xffff; break;
    case 0x4088: task->out_cvt_shift = val & 0xfff; break;
    case 0x400c: task->feature_mode_cfg = val; break;
    case 0x4010: task->data_format = val; break;
    case 0x4050: task->bs_ow_cfg = val; break;
    case 0x4058: task->wdma_size_0 = val; break;
    case 0x405c: task->wdma_size_1 = val; break;
    case 0x407c: task->ew_relux_cmp = val; break;
    case 0x40c0: task->surface_add = val & 0xfffff; break;
    case 0x5018: task->rdma_src_base_addr = val; break;
    case 0x501c: task->brdma_cfg = val; break;
    case 0x5020: task->bias_addr = val; break;
    case 0x5028: task->nrdma_cfg = val; break;
    case 0x502c: task->bn_base_addr = val; break;
    case 0x5034: task->erdma_cfg = val; break;
    case 0x5038: task->ew_base_addr = val; break;
    case 0x5040: task->ew_surf_stride = (val >> 4) & 0x0fffffff; break;
    case 0x5044: task->rdma_feat_mode_cfg = val; break;
    case 0x5068: task->rdma_weight = val; break;
    case 0x0010: task->next_base_addr = val; break;
    case 0x0014: task->next_reg_amounts = val & 0xffff; break;
    default: break;
    }
}

static bool parse_regcmd_buffer(RocketConvTask *task, const uint64_t *entries,
                                unsigned count)
{
    memset(task, 0, sizeof(*task));
    task->stride_x = 1;
    task->stride_y = 1;

    for (unsigned i = 0; i < count; i++) {
        parse_regcmd_entry(task, entries[i]);
    }

    return task->output_width > 0 && task->output_height > 0;
}

/* ======================================================================
 * Weight unpacking — read from packed NPU weight format
 * ====================================================================== */

static int8_t read_weight(const uint8_t *weights, unsigned oc, unsigned ic,
                           unsigned kx, unsigned ky,
                           unsigned filter_w, unsigned filter_h,
                           unsigned in_channels, unsigned out_channels,
                           bool depthwise)
{
    unsigned ic_group = NPU_WEIGHT_ATOMIC_SIZE;
    if (depthwise) {
        ic_group *= 2;
    }
    unsigned oc_aligned = ALIGN_UP(MAX2(out_channels, 2), 2);
    if (depthwise) {
        oc_aligned = 1;
    }
    unsigned ic_padded = MAX2(in_channels, NPU_FEATURE_ATOMIC_SIZE);

    unsigned oc1 = oc / NPU_WEIGHT_ATOMIC_SIZE;
    unsigned oc2 = oc % NPU_WEIGHT_ATOMIC_SIZE;
    unsigned ic1 = ic / ic_group;
    unsigned ic2 = ic % ic_group;

    unsigned ic1_count = DIV_ROUND_UP(ic_padded, ic_group);
    unsigned ic2_count = MIN2(ic_padded, ic_group);
    unsigned oc2_count = MIN2(oc_aligned, NPU_WEIGHT_ATOMIC_SIZE);

    unsigned offset = oc1 * (ic1_count * filter_w * filter_h *
                             oc2_count * ic2_count);
    offset += ic1 * (filter_w * filter_h * oc2_count * ic2_count);
    offset += kx * (filter_h * oc2_count * ic2_count);
    offset += ky * (oc2_count * ic2_count);
    offset += oc2 * ic2_count;
    offset += ic2;

    return (int8_t)weights[offset];
}

/* ======================================================================
 * IOVA → GPA translation via Rockchip IOMMU page table walk
 * ====================================================================== */

static hwaddr npu_iova_to_gpa(RockchipNPUState *s, uint32_t iova)
{
    if (s->rk_iommu) {
        return rk_iommu_translate(s->rk_iommu, iova);
    }
    /* Fallback: identity mapping (iommu.passthrough=1 mode) */
    return (hwaddr)iova;
}

static void npu_dma_read(RockchipNPUState *s, uint32_t iova,
                         void *buf, uint32_t size)
{
    uint8_t *dst = buf;
    while (size > 0) {
        hwaddr gpa = npu_iova_to_gpa(s, iova);
        uint32_t page_remain = 0x1000 - (iova & 0xFFF);
        uint32_t chunk = MIN2(size, page_remain);
        address_space_read(s->dma_as, gpa, MEMTXATTRS_UNSPECIFIED,
                           dst, chunk);
        dst += chunk;
        iova += chunk;
        size -= chunk;
    }
}

static void npu_dma_write(RockchipNPUState *s, uint32_t iova,
                          const void *buf, uint32_t size)
{
    const uint8_t *src = buf;
    while (size > 0) {
        hwaddr gpa = npu_iova_to_gpa(s, iova);
        uint32_t page_remain = 0x1000 - (iova & 0xFFF);
        uint32_t chunk = MIN2(size, page_remain);
        address_space_write(s->dma_as, gpa, MEMTXATTRS_UNSPECIFIED,
                            src, chunk);
        src += chunk;
        iova += chunk;
        size -= chunk;
    }
}

/* ======================================================================
 * NVDLA-referenced rounding helpers
 * ====================================================================== */

static inline int32_t nvdla_truncate(int32_t value, unsigned truncate)
{
    if (truncate == 0) {
        return value;
    }
    int32_t sign = (value < 0) ? 1 : 0;
    uint32_t guide = (value >> (truncate - 1)) & 1;
    uint32_t sticky = (truncate > 1)
        ? ((value & ((1u << (truncate - 1)) - 1)) != 0) : 0;
    int32_t round_up = guide & ((!sign) | sticky);
    return (value >> truncate) + round_up;
}

static inline int64_t nvdla_shift_right_round64(int64_t value, unsigned shift)
{
    if (shift == 0) {
        return value;
    }
    int64_t sign = (value < 0) ? 1 : 0;
    uint64_t guide = (value >> (shift - 1)) & 1;
    uint64_t sticky = (shift > 1)
        ? ((value & ((1ULL << (shift - 1)) - 1)) != 0) : 0;
    int64_t round_up = guide & ((!sign) | sticky);
    return (value >> shift) + round_up;
}

/* ======================================================================
 * SDP X-stage pipeline (shared by BS and BN)
 * ====================================================================== */

static inline int32_t apply_sdp_x_stage(int32_t value, uint32_t cfg,
                                         int32_t alu_operand,
                                         uint32_t mul_cfg,
                                         int32_t relux_cmp)
{
    if (cfg & 0x01) {
        return value;
    }

    if (!(cfg & 0x02)) {
        uint32_t algo = (cfg >> 16) & 0xf;
        switch (algo) {
        case 0: if (value < alu_operand) value = alu_operand; break;
        case 1: if (value > alu_operand) value = alu_operand; break;
        case 2: value += alu_operand; break;
        default: break;
        }
    }

    if (!(cfg & 0x10)) {
        int16_t mul_operand = (int16_t)((mul_cfg >> 16) & 0xffff);
        unsigned mul_shift = (mul_cfg >> 8) & 0x3f;
        bool mul_prelu = (cfg >> 5) & 1;

        if (!mul_prelu || value < 0) {
            int64_t product = (int64_t)value * (int64_t)mul_operand;
            value = (int32_t)nvdla_shift_right_round64(product, mul_shift);
        }
    }

    if (!(cfg & 0x40)) {
        bool relux_en = (cfg >> 7) & 1;
        if (relux_en) {
            if (value < 0) value = 0;
            if (value > relux_cmp) value = relux_cmp;
        } else {
            if (value < 0) value = 0;
        }
    }

    return value;
}

/* ======================================================================
 * INT8 Convolution Engine
 * ====================================================================== */

static void execute_convolution(RockchipNPUState *s, RocketNPUCore *core,
                                RocketConvTask *task)
{
    uint32_t out_w = task->output_width;
    uint32_t out_h = task->output_height;
    uint32_t out_c = task->output_channels;
    uint32_t in_c_real = task->input_channels_real;
    uint32_t in_c = task->input_channels;
    uint32_t filt_w = task->weight_width;
    uint32_t filt_h = task->weight_height;
    uint32_t stride_x = task->stride_x;
    uint32_t stride_y = task->stride_y;
    uint32_t pad_left = task->pad_left;
    uint32_t pad_top = task->pad_top;
    bool depthwise = task->depthwise;

    uint32_t in_w = task->input_width;
    uint32_t in_h = task->input_height;

    uint32_t in_groups = DIV_ROUND_UP(in_c, NPU_FEATURE_ATOMIC_SIZE);

    /* DMA input stride: register values are in 4-byte units.
     * For tiled ops, the stride covers the full tensor (not just the tile). */
    uint32_t in_line_bytes = task->input_line_stride
        ? task->input_line_stride * 4
        : in_h * NPU_FEATURE_ATOMIC_SIZE;
    uint32_t in_surf_bytes = task->input_surface_stride
        ? task->input_surface_stride * 4
        : in_w * in_line_bytes;
    uint32_t in_buf_size = (in_groups > 1)
        ? (in_groups - 1) * in_surf_bytes + (in_w - 1) * in_line_bytes
          + in_h * NPU_FEATURE_ATOMIC_SIZE
        : (in_w - 1) * in_line_bytes + in_h * NPU_FEATURE_ATOMIC_SIZE;

    uint32_t wt_oc = depthwise ? 1 : ALIGN_UP(MAX2(out_c, 2), 2);
    uint32_t wt_ic = MAX2(in_c_real, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t wt_ic_group = depthwise ? NPU_WEIGHT_ATOMIC_SIZE * 2
                                      : NPU_WEIGHT_ATOMIC_SIZE;
    uint32_t wt_buf_size = DIV_ROUND_UP(wt_oc, NPU_WEIGHT_ATOMIC_SIZE)
                         * DIV_ROUND_UP(wt_ic, wt_ic_group)
                         * filt_w * filt_h
                         * MIN2(wt_oc, NPU_WEIGHT_ATOMIC_SIZE)
                         * MIN2(wt_ic, wt_ic_group);

    uint32_t brdma_data_use = (task->brdma_cfg >> 1) & 0xf;
    bool bias_from_dma = (brdma_data_use >= 1);
    bool mul_from_dma = (brdma_data_use >= 3);
    uint32_t oc_pad = ALIGN_UP(MAX2(out_c, 32), 16);
    uint32_t bias_buf_size = oc_pad * sizeof(int32_t);

    /* Per-channel MUL scale buffer (int16, from BRDMA) */
    int16_t *mul_scale_buf = NULL;

    bool ew_bypass = (task->ew_cfg & 0x01);
    bool ew_op_bypass = (task->ew_cfg >> 1) & 1;
    bool ew_active = !ew_bypass && !ew_op_bypass;
    bool erdma_enabled = !(task->erdma_cfg & 0x01);
    uint8_t *ew_buf = NULL;

    uint32_t out_groups = DIV_ROUND_UP(out_c, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t out_buf_size = out_groups * out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;

    uint8_t *in_buf = g_malloc0(in_buf_size);
    uint8_t *wt_buf = g_malloc0(wt_buf_size);
    int32_t *bias_buf = g_malloc0(bias_buf_size);
    uint8_t *out_buf = g_malloc0(out_buf_size);

    npu_dma_read(s, task->src_addr, in_buf, in_buf_size);
    npu_dma_read(s, task->weight_addr, wt_buf, wt_buf_size);

    if (bias_from_dma && task->bias_addr != 0) {
        if (mul_from_dma) {
            /* RKNN BRDMA layout: groups of 8 channels per 64-byte chunk.
             * [0..31]: 8 × int32 bias, [32..47]: padding, [48..63]: 8 × int16 mul */
            uint32_t num_groups = oc_pad / 8;
            uint32_t total_brdma = num_groups * 64;
            uint8_t *brdma_raw = g_malloc0(total_brdma);
            npu_dma_read(s, task->bias_addr, brdma_raw, total_brdma);
            mul_scale_buf = g_malloc0(oc_pad * sizeof(int16_t));
            for (uint32_t g = 0; g < num_groups; g++) {
                uint8_t *chunk = brdma_raw + g * 64;
                memcpy(&bias_buf[g * 8], chunk, 8 * sizeof(int32_t));
                memcpy(&mul_scale_buf[g * 8], chunk + 48, 8 * sizeof(int16_t));
            }
            g_free(brdma_raw);
        } else {
            npu_dma_read(s, task->bias_addr, bias_buf, out_c * sizeof(int32_t));
        }
    }

    if (ew_active && erdma_enabled && task->ew_base_addr != 0) {
        uint32_t ew_buf_size = out_groups * out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;
        ew_buf = g_malloc0(ew_buf_size);
        npu_dma_read(s, task->ew_base_addr, ew_buf, ew_buf_size);
    }

    uint32_t out_cvt_scale = task->out_cvt_scale;
    uint32_t out_cvt_shift = task->out_cvt_shift;
    int32_t out_cvt_offset = (int32_t)task->out_cvt_offset;
    int8_t pad_val = (int8_t)(task->pad_value & 0xff);

    for (uint32_t oy = 0; oy < out_h; oy++) {
        for (uint32_t ox = 0; ox < out_w; ox++) {
            for (uint32_t oc = 0; oc < out_c; oc++) {
                int32_t acc = 0;
                int32_t sum_inputs = 0;
                uint32_t ic_limit = depthwise ? 1 : in_c_real;
                uint32_t ic_base = depthwise ? oc : 0;

                for (uint32_t ky = 0; ky < filt_h; ky++) {
                    for (uint32_t kx = 0; kx < filt_w; kx++) {
                        int32_t iy = (int32_t)(oy * stride_y + ky) -
                                     (int32_t)pad_top;
                        int32_t ix = (int32_t)(ox * stride_x + kx) -
                                     (int32_t)pad_left;

                        for (uint32_t ic = 0; ic < ic_limit; ic++) {
                            int8_t in_val;
                            uint32_t abs_ic = ic_base + ic;

                            if (ix < 0 || ix >= (int32_t)in_w ||
                                iy < 0 || iy >= (int32_t)in_h) {
                                in_val = pad_val;
                            } else {
                                uint32_t g = abs_ic / NPU_FEATURE_ATOMIC_SIZE;
                                uint32_t c = abs_ic % NPU_FEATURE_ATOMIC_SIZE;
                                uint32_t off = g * in_surf_bytes
                                             + ix * in_line_bytes
                                             + iy * NPU_FEATURE_ATOMIC_SIZE;
                                in_val = (off + c < in_buf_size)
                                    ? (int8_t)in_buf[off + c] : pad_val;
                            }

                            int8_t w_val;
                            if (depthwise) {
                                w_val = read_weight(wt_buf, 0, abs_ic,
                                                    kx, ky, filt_w, filt_h,
                                                    in_c_real, 1, true);
                            } else {
                                w_val = read_weight(wt_buf, oc, abs_ic,
                                                    kx, ky, filt_w, filt_h,
                                                    in_c_real, out_c, false);
                            }

                            acc += (int32_t)in_val * (int32_t)w_val;
                            sum_inputs += (int32_t)in_val;
                        }
                    }
                }

                /* BS_OW_OP: weight zero-point compensation.
                 * Weights are stored as (w - 0x80), but correct is (w - wzp).
                 * Per output pixel, this introduces error = (wzp - 0x80) * sum(inputs).
                 * BS_OW_OP = (0x80 - wzp) cancels it: acc += ow_op * sum_inputs. */
                if (task->bs_ow_op) {
                    int16_t ow_op = (int16_t)task->bs_ow_op;
                    acc += (int32_t)ow_op * sum_inputs;
                }

                acc = nvdla_truncate(acc, task->truncate_bits);

                /* SDP Pipeline: BS → BN → EW → OUT_CVT */
                {
                    bool bs_alu_src = (task->bs_cfg >> 8) & 1;
                    int32_t bs_alu_op = bs_alu_src ? bias_buf[oc]
                                                   : task->bs_alu_cfg;

                    /* Per-channel MUL: when MUL_SRC=DMA (bit 0 of bs_mul_cfg),
                     * override the scalar mul operand with per-channel value */
                    uint32_t eff_bs_mul_cfg = task->bs_mul_cfg;
                    if (mul_scale_buf && (task->bs_mul_cfg & 1)) {
                        int16_t per_ch_mul = mul_scale_buf[oc];
                        eff_bs_mul_cfg = (eff_bs_mul_cfg & 0x0000ffff) |
                                         ((uint32_t)(uint16_t)per_ch_mul << 16);
                    }
                    acc = apply_sdp_x_stage(acc, task->bs_cfg, bs_alu_op,
                                             eff_bs_mul_cfg,
                                             (int32_t)task->bs_relux_cmp);
                }

                acc = apply_sdp_x_stage(acc, task->bn_cfg,
                                         task->bn_alu_cfg,
                                         task->bn_mul_cfg,
                                         (int32_t)task->bn_relux_cmp);

                if (!ew_bypass) {
                    if (!ew_op_bypass) {
                        int32_t ew_operand = 0;
                        uint32_t ew_algo = (task->ew_cfg >> 16) & 0xf;

                        if (ew_buf != NULL) {
                            uint32_t og = oc / NPU_FEATURE_ATOMIC_SIZE;
                            uint32_t oc_within = oc % NPU_FEATURE_ATOMIC_SIZE;
                            uint32_t ew_off = npu_output_offset(og, ox, oy,
                                                                 out_w, out_h);
                            ew_operand = (int32_t)(int8_t)ew_buf[ew_off +
                                                                  oc_within];
                        }

                        switch (ew_algo) {
                        case 0: if (acc < ew_operand) acc = ew_operand; break;
                        case 1: if (acc > ew_operand) acc = ew_operand; break;
                        case 2: acc += ew_operand; break;
                        default: break;
                        }
                    }

                    bool ew_cvt_bypass = (task->ew_cfg >> 8) & 1;
                    if (!ew_cvt_bypass) {
                        acc += task->ew_cvt_offset;
                        uint32_t ew_trunc = (task->ew_cvt_scale >> 22) & 0x3ff;
                        acc = (int32_t)nvdla_truncate(acc, ew_trunc);
                    }

                    if (!((task->ew_cfg >> 9) & 1)) {
                        bool ew_relux_en = (task->ew_cfg >> 10) & 1;
                        if (ew_relux_en) {
                            if (acc < 0) acc = 0;
                            if (acc > (int32_t)task->ew_relux_cmp)
                                acc = (int32_t)task->ew_relux_cmp;
                        } else {
                            if (acc < 0) acc = 0;
                        }
                    }
                }

                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)out_cvt_scale, out_cvt_shift);
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;

                int32_t result = (int32_t)scaled + out_cvt_offset;
                if (result < -128) result = -128;
                if (result > 127) result = 127;

                uint32_t og = oc / NPU_FEATURE_ATOMIC_SIZE;
                uint32_t oc_within = oc % NPU_FEATURE_ATOMIC_SIZE;
                uint32_t out_off = npu_output_offset(og, ox, oy,
                                                      out_w, out_h);
                out_buf[out_off + oc_within] = (uint8_t)(int8_t)result;
            }
        }
    }

    /* Output is col-major (matching npu_input_offset) — no transpose needed. */

    if (task->output_surface_stride > 0 && out_groups > 1) {
        uint32_t group_size = out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;
        for (uint32_t g = 0; g < out_groups; g++) {
            uint32_t dst = task->dst_addr + g * task->output_surface_stride;
            npu_dma_write(s, dst, out_buf + g * group_size, group_size);
        }
    } else {
        npu_dma_write(s, task->dst_addr, out_buf, out_buf_size);
    }

    g_free(in_buf);
    g_free(wt_buf);
    g_free(bias_buf);
    g_free(mul_scale_buf);
    g_free(ew_buf);
    g_free(out_buf);
}

/* ======================================================================
 * Job execution — process a complete regcmd buffer
 * ====================================================================== */

static void execute_job(RockchipNPUState *s, RocketNPUCore *core,
                        uint32_t base_addr, uint32_t encoded_amounts)
{
    uint32_t current_addr = base_addr;
    uint32_t current_count = (encoded_amounts + 1) * 2;

    while (current_count > 0 && current_addr != 0) {
        if (current_count > NPU_MAX_REGCMD_ENTRIES) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "rockchip-npu: core%d: regcmd count %u exceeds max\n",
                          core->core_id, current_count);
            break;
        }

        uint64_t *entries = g_malloc(current_count * sizeof(uint64_t));
        npu_dma_read(s, current_addr, entries,
                     current_count * sizeof(uint64_t));

        RocketConvTask task;
        qemu_log_mask(LOG_UNIMP,
                      "rockchip-npu: parsing %u entries at 0x%08x\n",
                      current_count, current_addr);
        for (unsigned di = 0; di < MIN2(5, current_count); di++) {
            qemu_log_mask(LOG_UNIMP,
                          "  regcmd[%u] = 0x%016llx (target=%04x reg=%04x val=%08x)\n",
                          di, (unsigned long long)entries[di],
                          (unsigned)(entries[di] >> 48),
                          (unsigned)(entries[di] & 0xFFFF),
                          (unsigned)((entries[di] >> 16) & 0xFFFFFFFF));
        }
        if (!parse_regcmd_buffer(&task, entries, current_count)) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "rockchip-npu: core%d: failed to parse regcmd "
                          "at 0x%08x (%u entries)\n",
                          core->core_id, current_addr, current_count);
            g_free(entries);
            break;
        }

        qemu_log_mask(LOG_UNIMP,
                      "rockchip-npu: conv %ux%ux%u -> %ux%ux%u "
                      "src=0x%x dst=0x%x wt=0x%x bias=0x%x\n",
                      task.input_width, task.input_height,
                      task.input_channels_real,
                      task.output_width, task.output_height,
                      task.output_channels,
                      task.src_addr, task.dst_addr,
                      task.weight_addr, task.bias_addr);
        execute_convolution(s, core, &task);

        uint32_t next_addr = task.next_base_addr;
        uint32_t next_encoded = task.next_reg_amounts;
        g_free(entries);

        if (next_addr != 0 && next_encoded != 0) {
            current_addr = next_addr;
            current_count = (next_encoded + 1) * 2;
        } else {
            break;
        }
    }

    core->pc_task_status = 0; /* execution complete */
    core->pc_irq_raw_status |= NPU_IDLE_RAW_BITS | 0x0300; /* idle + DPU_0 | DPU_1 */
    core->pc_irq_status = core->pc_irq_raw_status & core->pc_irq_mask;
    if (core->pc_irq_status) {
        int64_t now = qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL);
        timer_mod(core->irq_timer, now + 1);
    }
}

/* ======================================================================
 * Deferred IRQ timer callback — fires after MMIO write returns to guest
 * ====================================================================== */

static void npu_irq_timer_cb(void *opaque)
{
    RocketNPUCore *core = opaque;
    if (core->pc_irq_status) {
        qemu_irq_raise(core->irq);
    }
}

/* ======================================================================
 * MMIO read/write handlers
 * ====================================================================== */

/*
 * Rockchip IOMMU registers are embedded within the NPU's 0x10000 per-core
 * region at offsets 0x9000 and 0xa000. Forward these to the IOMMU model.
 */
static bool is_iommu_offset(hwaddr addr)
{
    return (addr >= 0x9000 && addr < 0x9100) ||
           (addr >= 0xa000 && addr < 0xa100);
}

static unsigned iommu_instance_for_offset(hwaddr addr, unsigned core_id)
{
    if (core_id == 0 && addr >= 0x9000 && addr < 0x9100) return 0;
    if (core_id == 0 && addr >= 0xa000 && addr < 0xa100) return 1;
    if (core_id == 1 && addr >= 0xa000 && addr < 0xa100) return 2;
    if (core_id == 2 && addr >= 0xa000 && addr < 0xa100) return 3;
    return 0;
}

static uint64_t rockchip_npu_read(void *opaque, hwaddr addr, unsigned size)
{
    RocketNPUCore *core = opaque;
    RockchipNPUState *s = container_of(core, RockchipNPUState,
                                       cores[core->core_id]);

    if (is_iommu_offset(addr) && s->rk_iommu) {
        unsigned inst = iommu_instance_for_offset(addr, core->core_id);
        return rk_iommu_instance_read(&s->rk_iommu->instances[inst],
                                       addr & 0xFF, size);
    }

    switch (addr) {
    case REG_PC_VERSION:
        return ROCKET_PC_VERSION;
    case REG_PC_VERSION_NUM:
        return ROCKET_PC_VERSION_NUM;
    case REG_PC_OP_ENABLE:
        return 0;
    case REG_PC_BASE_ADDRESS:
        return core->pc_base_addr;
    case REG_PC_REGISTER_AMOUNTS:
        return core->pc_reg_amounts;
    case REG_PC_IRQ_MASK:
        return core->pc_irq_mask;
    case REG_PC_IRQ_STATUS:
        return core->pc_irq_status;
    case REG_PC_IRQ_RAW_STATUS:
        return core->pc_irq_raw_status;
    case REG_PC_TASK_CON:
        return core->pc_task_con;
    case REG_PC_TASK_STATUS:
        return core->pc_task_status;
    default:
        if (addr < NPU_REGION_SIZE) {
            return core->regs[addr / 4];
        }
        return 0;
    }
}

static void rockchip_npu_write(void *opaque, hwaddr addr, uint64_t val,
                                unsigned size)
{
    RocketNPUCore *core = opaque;
    RockchipNPUState *s = container_of(core, RockchipNPUState,
                                       cores[core->core_id]);

    if (is_iommu_offset(addr) && s->rk_iommu) {
        unsigned inst = iommu_instance_for_offset(addr, core->core_id);
        hwaddr iommu_reg = addr & 0xFF;
        rk_iommu_instance_write(&s->rk_iommu->instances[inst],
                                 iommu_reg, val, size);
        /* Track the most recent ENABLE_PAGING globally */
        if (iommu_reg == RK_IOMMU_COMMAND && val == RK_IOMMU_CMD_ENABLE_PAGING) {
            s->rk_iommu->last_active_dte =
                s->rk_iommu->instances[inst].active_dte_addr;
        }
        return;
    }

    if (addr < 0x40) {
        qemu_log_mask(LOG_UNIMP,
                      "rockchip-npu: core%d write addr=0x%04x val=0x%08x\n",
                      core->core_id, (unsigned)addr, (unsigned)val);
    }

    switch (addr) {
    case REG_PC_BASE_ADDRESS:
        core->pc_base_addr = (uint32_t)val;
        break;

    case REG_PC_REGISTER_AMOUNTS:
        core->pc_reg_amounts = (uint32_t)(val & 0xffff);
        break;

    case REG_PC_OP_ENABLE:
        if (val & 1) {
            qemu_log_mask(LOG_UNIMP,
                          "rockchip-npu: core%d OP_ENABLE! base=0x%08x amounts=%u\n",
                          core->core_id, core->pc_base_addr,
                          core->pc_reg_amounts);
            execute_job(s, core, core->pc_base_addr, core->pc_reg_amounts);
        }
        break;

    case REG_PC_IRQ_MASK:
        core->pc_irq_mask = (uint32_t)val;
        core->pc_irq_status = core->pc_irq_raw_status & core->pc_irq_mask;
        if (!core->pc_irq_status) {
            qemu_irq_lower(core->irq);
        }
        break;

    case REG_PC_IRQ_CLEAR:
        core->pc_irq_raw_status &= ~(uint32_t)val;
        core->pc_irq_raw_status |= NPU_IDLE_RAW_BITS; /* idle bits always set */
        core->pc_irq_status = core->pc_irq_raw_status & core->pc_irq_mask;
        if (!core->pc_irq_status) {
            qemu_irq_lower(core->irq);
        }
        break;

    case REG_PC_TASK_CON:
        core->pc_task_con = (uint32_t)val;
        break;

    default:
        if (addr < NPU_REGION_SIZE) {
            core->regs[addr / 4] = (uint32_t)val;
        }
        break;
    }
}

static const MemoryRegionOps rockchip_npu_ops = {
    .read = rockchip_npu_read,
    .write = rockchip_npu_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* ======================================================================
 * Device lifecycle
 * ====================================================================== */

static void rockchip_npu_realize(DeviceState *dev, Error **errp)
{
    RockchipNPUState *s = ROCKCHIP_NPU(dev);
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);
    char name[32];

    s->dma_as = &address_space_memory;

    for (int i = 0; i < s->num_cores; i++) {
        RocketNPUCore *core = &s->cores[i];
        core->core_id = i;
        core->pc_irq_mask = 0;
        core->irq_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                        npu_irq_timer_cb, core);

        snprintf(name, sizeof(name), "rockchip-npu-core%d", i);
        memory_region_init_io(&core->iomem, OBJECT(dev), &rockchip_npu_ops,
                              core, name, NPU_REGION_SIZE);
        sysbus_init_mmio(sbd, &core->iomem);
        sysbus_init_irq(sbd, &core->irq);
    }
}

static void rockchip_npu_reset(DeviceState *dev)
{
    RockchipNPUState *s = ROCKCHIP_NPU(dev);

    for (int i = 0; i < s->num_cores; i++) {
        RocketNPUCore *core = &s->cores[i];
        core->pc_base_addr = 0;
        core->pc_reg_amounts = 0;
        core->pc_irq_mask = 0;
        core->pc_irq_status = 0;
        core->pc_irq_raw_status = NPU_IDLE_RAW_BITS;
        core->pc_task_con = 0;
        core->pc_task_status = 0;
        if (core->irq_timer) {
            timer_del(core->irq_timer);
        }
        memset(core->regs, 0, sizeof(core->regs));
    }
}

static const VMStateDescription vmstate_rockchip_npu_core = {
    .name = "rockchip-npu-core",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT32(pc_base_addr, RocketNPUCore),
        VMSTATE_UINT32(pc_reg_amounts, RocketNPUCore),
        VMSTATE_UINT32(pc_irq_mask, RocketNPUCore),
        VMSTATE_UINT32(pc_irq_status, RocketNPUCore),
        VMSTATE_UINT32(pc_irq_raw_status, RocketNPUCore),
        VMSTATE_UINT32(pc_task_con, RocketNPUCore),
        VMSTATE_UINT32(pc_task_status, RocketNPUCore),
        VMSTATE_UINT32_ARRAY(regs, RocketNPUCore, NPU_REGION_SIZE / 4),
        VMSTATE_END_OF_LIST()
    }
};

static const VMStateDescription vmstate_rockchip_npu = {
    .name = "rockchip-npu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_STRUCT_ARRAY(cores, RockchipNPUState, 3, 1,
                             vmstate_rockchip_npu_core, RocketNPUCore),
        VMSTATE_END_OF_LIST()
    }
};

static const Property rockchip_npu_properties[] = {
    DEFINE_PROP_UINT32("num-cores", RockchipNPUState, num_cores, 3),
};

static void rockchip_npu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = rockchip_npu_realize;
    device_class_set_legacy_reset(dc, rockchip_npu_reset);
    dc->vmsd = &vmstate_rockchip_npu;
    dc->desc = "Rockchip RK3588 NPU (3-core, INT8 convolution engine)";
    device_class_set_props(dc, rockchip_npu_properties);
}

static const TypeInfo rockchip_npu_info = {
    .name = TYPE_ROCKCHIP_NPU,
    .parent = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(RockchipNPUState),
    .class_init = rockchip_npu_class_init,
};

static void rockchip_npu_register(void)
{
    type_register_static(&rockchip_npu_info);
}

type_init(rockchip_npu_register)
