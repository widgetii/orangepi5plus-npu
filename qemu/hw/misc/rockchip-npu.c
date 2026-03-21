/*
 * Rockchip RK3588 Rocket NPU device model
 *
 * Emulates the Rockchip Rocket NPU's register interface, regcmd parser,
 * and INT8 convolution engine. Allows running the kernel Rocket DRM driver
 * and Mesa/libteflon/librocketnpu userspace without real hardware.
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

/*
 * Map a register offset (from the regcmd entry) to the "flat" index used
 * in the core register file. The regcmd entries use the full SoC-relative
 * offsets (0x1xxx for CNA, 0x3xxx for CORE, etc.), so we just index by
 * (offset / 4).
 */
static void parse_regcmd_entry(RocketConvTask *task, uint64_t entry)
{
    uint32_t reg = (uint32_t)(entry & REGCMD_REG_MASK);
    uint32_t val = (uint32_t)((entry & REGCMD_VALUE_MASK) >> REGCMD_VALUE_SHIFT);
    uint16_t target = (uint16_t)(entry >> REGCMD_TARGET_SHIFT);

    /* Special control entries */
    if (target == 0x0041 || target == 0x0081) {
        return;
    }
    if (entry == 0) {
        return;
    }

    switch (reg) {
    /* CNA: Input tensor
     * CNA registers store RAW values (Mesa emits without -1 adjustment).
     */
    case 0x1070: /* CNA_FEATURE_DATA_ADDR */
        task->src_addr = val;
        break;
    case 0x1020: /* CNA_DATA_SIZE0: HEIGHT[10:0] | WIDTH[26:16] — raw */
        task->input_height = val & 0x7ff;
        task->input_width = (val >> 16) & 0x7ff;
        break;
    case 0x1024: /* CNA_DATA_SIZE1: CHANNEL[15:0](raw) | CHANNEL_REAL[29:16](count-1) */
        task->input_channels = val & 0xffff;
        task->input_channels_real = ((val >> 16) & 0x3fff) + 1;
        break;
    case 0x1028: /* CNA_DATA_SIZE2: DATAOUT_WIDTH — raw */
        /* CNA's view of output width; we use CORE for authoritative dims */
        break;

    /* CNA: Weights — raw values */
    case 0x1110: /* CNA_DCOMP_ADDR0 — weight DMA address */
        task->weight_addr = val;
        break;
    case 0x1038: /* CNA_WEIGHT_SIZE2: KERNELS[13:0] | HEIGHT[20:16] | WIDTH[28:24] */
        task->weight_kernels = val & 0x3fff;
        task->weight_height = (val >> 16) & 0x1f;
        task->weight_width = (val >> 24) & 0x1f;
        break;
    case 0x1030: /* CNA_WEIGHT_SIZE0 */
        task->weight_size0 = val;
        break;
    case 0x1034: /* CNA_WEIGHT_SIZE1 */
        task->weight_size1 = val;
        break;

    /* CNA: Convolution config — raw values */
    case 0x100c: /* CNA_CONV_CON1: CONV_MODE[3:0] */
        task->conv_con1 = val;
        task->depthwise = (val & 0xf) == 3; /* mode 3 = depthwise */
        break;
    case 0x1014: /* CNA_CONV_CON3: STRIDE_X(bits[2:0]) | STRIDE_Y(bits[5:3]) */
        task->stride_x = val & 0x7;
        task->stride_y = (val >> 3) & 0x7;
        if (task->stride_x == 0) task->stride_x = 1;
        if (task->stride_y == 0) task->stride_y = 1;
        break;
    case 0x1068: /* CNA_PAD_CON0: PAD_TOP(bits[3:0]) | PAD_LEFT(bits[7:4]) */
        task->pad_top = val & 0xf;
        task->pad_left = (val >> 4) & 0xf;
        break;
    case 0x1184: /* CNA_PAD_CON1 */
        task->pad_value = (int32_t)val;
        break;
    case 0x107c: /* CNA_DMA_CON1: LINE_STRIDE[27:0] */
        task->input_line_stride = val & 0x0fffffff;
        break;
    case 0x1080: /* CNA_DMA_CON2: SURF_STRIDE[27:0] */
        task->input_surface_stride = val & 0x0fffffff;
        break;

    /* CORE: Output dimensions */
    case 0x3010: /* CORE_MISC_CFG */
        task->depthwise = task->depthwise || ((val >> 1) & 1);
        break;
    case 0x3014: /* CORE_DATAOUT_SIZE_0: WIDTH[15:0] | HEIGHT[31:16] — count-1 */
        task->output_width = (val & 0xffff) + 1;
        task->output_height = ((val >> 16) & 0xffff) + 1;
        break;
    case 0x3018: /* CORE_DATAOUT_SIZE_1: CHANNEL[12:0] — count-1 */
        task->output_channels = (val & 0x1fff) + 1;
        task->output_channels_real = task->output_channels;
        break;
    case 0x301c: /* CORE_CLIP_TRUNCATE */
        task->truncate_bits = val & 0x1f;
        break;

    /* DPU: Output destination */
    case 0x4020: /* DPU_DST_BASE_ADDR */
        task->dst_addr = val;
        break;
    case 0x4024: /* DPU_DST_SURF_STRIDE[31:4] — value is shifted left by 4 */
        task->output_surface_stride = (val >> 4) & 0x0fffffff;
        break;
    case 0x4030: /* DPU_DATA_CUBE_WIDTH */
        /* DPU output width, should match CORE */
        break;
    case 0x4034: /* DPU_DATA_CUBE_HEIGHT */
        /* DPU output height, should match CORE */
        break;
    case 0x403c: /* DPU_DATA_CUBE_CHANNEL: CHANNEL[12:0] | ORIG_CHANNEL[28:16] — count-1 */
        task->output_channels = (val & 0x1fff) + 1;
        task->output_channels_real = ((val >> 16) & 0x1fff) + 1;
        break;

    /* DPU: BS (Bias/Scale) stage */
    case 0x4040: /* DPU_BS_CFG */
        task->bs_cfg = val;
        break;
    case 0x4044: /* DPU_BS_ALU_CFG */
        task->bs_alu_cfg = (int32_t)val;
        break;
    case 0x4048: /* DPU_BS_MUL_CFG */
        task->bs_mul_cfg = val;
        break;
    case 0x404c: /* DPU_BS_RELUX_CMP */
        task->bs_relux_cmp = val;
        break;
    case 0x4054: /* DPU_BS_OW_OP */
        task->bs_ow_op = val & 0xffff;
        break;

    /* DPU: BN stage */
    case 0x4060: /* DPU_BN_CFG */
        task->bn_cfg = val;
        break;
    case 0x4064: /* DPU_BN_ALU_CFG */
        task->bn_alu_cfg = (int32_t)val;
        break;
    case 0x4068: /* DPU_BN_MUL_CFG */
        task->bn_mul_cfg = val;
        break;
    case 0x406c: /* DPU_BN_RELUX_CMP */
        task->bn_relux_cmp = val;
        break;

    /* DPU: EW (Element-Wise) stage */
    case 0x4070: /* DPU_EW_CFG */
        task->ew_cfg = val;
        break;
    case 0x4074: /* DPU_EW_CVT_OFFSET */
        task->ew_cvt_offset = (int32_t)val;
        break;
    case 0x4078: /* DPU_EW_CVT_SCALE */
        task->ew_cvt_scale = val;
        break;

    /* DPU: Output conversion (requantization) */
    case 0x4080: /* DPU_OUT_CVT_OFFSET */
        task->out_cvt_offset = val;
        break;
    case 0x4084: /* DPU_OUT_CVT_SCALE */
        task->out_cvt_scale = val & 0xffff;
        break;
    case 0x4088: /* DPU_OUT_CVT_SHIFT[11:0] */
        task->out_cvt_shift = val & 0xfff;
        break;

    /* DPU: Data format and additional parsed registers */
    case 0x400c: /* DPU_FEATURE_MODE_CFG */
        task->feature_mode_cfg = val;
        break;
    case 0x4010: /* DPU_DATA_FORMAT */
        task->data_format = val;
        break;
    case 0x4050: /* DPU_BS_OW_CFG */
        task->bs_ow_cfg = val;
        break;
    case 0x4058: /* DPU_WDMA_SIZE_0 */
        task->wdma_size_0 = val;
        break;
    case 0x405c: /* DPU_WDMA_SIZE_1 */
        task->wdma_size_1 = val;
        break;
    case 0x407c: /* DPU_EW_RELUX_CMP */
        task->ew_relux_cmp = val;
        break;
    case 0x40c0: /* DPU_SURFACE_ADD */
        task->surface_add = val & 0xfffff;
        break;

    /* RDMA: DMA config */
    case 0x5018: /* RDMA_SRC_BASE_ADDR */
        task->rdma_src_base_addr = val;
        break;
    case 0x501c: /* RDMA_BRDMA_CFG */
        task->brdma_cfg = val;
        break;
    case 0x5020: /* RDMA_BS_BASE_ADDR */
        task->bias_addr = val;
        break;
    case 0x5028: /* RDMA_NRDMA_CFG */
        task->nrdma_cfg = val;
        break;
    case 0x502c: /* RDMA_BN_BASE_ADDR */
        task->bn_base_addr = val;
        break;
    case 0x5034: /* RDMA_ERDMA_CFG */
        task->erdma_cfg = val;
        break;
    case 0x5038: /* RDMA_EW_BASE_ADDR */
        task->ew_base_addr = val;
        break;
    case 0x5040: /* RDMA_EW_SURF_STRIDE */
        task->ew_surf_stride = (val >> 4) & 0x0fffffff;
        break;
    case 0x5044: /* RDMA_FEATURE_MODE_CFG */
        task->rdma_feat_mode_cfg = val;
        break;
    case 0x5068: /* RDMA_WEIGHT */
        task->rdma_weight = val;
        break;

    /* Task chaining (last entries of regcmd) */
    case 0x0010: /* PC_BASE_ADDRESS */
        task->next_base_addr = val;
        break;
    case 0x0014: /* PC_REGISTER_AMOUNTS */
        task->next_reg_amounts = val & 0xffff;
        break;

    default:
        /* Other registers: silently ignore */
        break;
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

/*
 * Read a single weight value from the packed weight buffer.
 * Weight layout: [oc1][ic1][x][y][oc2][ic2] where
 *   oc1 = oc / WEIGHT_ATOMIC_SIZE, oc2 = oc % WEIGHT_ATOMIC_SIZE
 *   ic1 = ic / ic_group, ic2 = ic % ic_group
 *   ic_group = WEIGHT_ATOMIC_SIZE (32 for normal, 64 for depthwise)
 * Values are stored as signed int8 (already bias-shifted by 0x80).
 */
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
 * IOVA → GPA translation (populated by kernel IOMMU module via mailbox)
 * ====================================================================== */

#define NPU_IOMMU_BASE 0xfdaf0000ULL  /* Mailbox MMIO address */
#define NPU_IOMMU_SIZE 0x1000
#define NPU_IOMMU_REG_IOVA  0x00  /* Write IOVA (page-aligned) */
#define NPU_IOMMU_REG_PHYS  0x04  /* Write GPA → adds mapping */
#define NPU_IOMMU_REG_UNMAP 0x08  /* Write IOVA → removes mapping */
#define NPU_IOMMU_REG_COUNT 0x0c  /* Read: number of mappings */

static hwaddr npu_iova_to_gpa(RockchipNPUState *s, uint32_t iova)
{
    /* In rknpu mode, use Rockchip IOMMU page table walk */
    if (s->driver_mode == NPU_DRIVER_MODE_RKNPU && s->rk_iommu) {
        return rk_iommu_translate(s->rk_iommu, iova);
    }

    /* Rocket mode: use mailbox-populated table */
    uint32_t page = iova & ~0xFFF;
    uint32_t offset = iova & 0xFFF;

    for (uint32_t i = 0; i < s->iommu_entry_count; i++) {
        if (s->iommu_table[i].iova == page) {
            return (hwaddr)s->iommu_table[i].phys + offset;
        }
    }
    /* Fallback: identity mapping (for passthrough mode) */
    return (hwaddr)iova;
}

/*
 * Read data from guest memory using IOVA translation.
 * Handles reads that span multiple IOMMU pages.
 */
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

static uint64_t npu_iommu_read(void *opaque, hwaddr addr, unsigned size)
{
    RockchipNPUState *s = opaque;
    if (addr == NPU_IOMMU_REG_COUNT)
        return s->iommu_entry_count;
    return 0;
}

static void npu_iommu_write(void *opaque, hwaddr addr, uint64_t val,
                            unsigned size)
{
    RockchipNPUState *s = opaque;

    switch (addr) {
    case NPU_IOMMU_REG_IOVA:
        s->iommu_iova = (uint32_t)val;
        break;
    case NPU_IOMMU_REG_PHYS:
        s->iommu_phys = (uint32_t)val;
        /* Add mapping */
        if (s->iommu_entry_count < NPU_IOMMU_MAX_PAGES) {
            s->iommu_table[s->iommu_entry_count].iova = s->iommu_iova;
            s->iommu_table[s->iommu_entry_count].phys = s->iommu_phys;
            s->iommu_entry_count++;
        }
        break;
    case NPU_IOMMU_REG_UNMAP: {
        uint32_t page = (uint32_t)val;
        for (uint32_t i = 0; i < s->iommu_entry_count; i++) {
            if (s->iommu_table[i].iova == page) {
                s->iommu_table[i] = s->iommu_table[--s->iommu_entry_count];
                break;
            }
        }
        break;
    }
    default:
        break;
    }
}

static const MemoryRegionOps npu_iommu_ops = {
    .read = npu_iommu_read,
    .write = npu_iommu_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = { .min_access_size = 4, .max_access_size = 4 },
};

/* ======================================================================
 * NVDLA-referenced rounding helpers
 *
 * The Rockchip NPU is based on NVDLA. The truncation and shift-right
 * operations use round-half-to-even (banker's rounding) with guide and
 * sticky bits, NOT simple round-half-up.
 *
 * From NV_NVDLA_CACC_CALC_int8.v:
 *   {shifted, guide, sticky} = ($signed({value, 16'b0}) >>> truncate)
 *   round_up = guide & (~sign | (|sticky))
 *   result = shifted + round_up
 *
 * For negative values at an exact tie (guide=1, sticky=0), this does NOT
 * round up — it rounds toward zero (banker's rounding for negative).
 * ====================================================================== */

/*
 * NVDLA truncation: round-half-to-even for 32-bit accumulator.
 * Used after CACC accumulation (CLIP_TRUNCATE register).
 */
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

/*
 * NVDLA shift-right with rounding for 64-bit values.
 * Used in OUT_CVT: (acc * scale) >> shift, and EW CVT.
 * From NV_NVDLA_SDP_HLS_C_int.v / NV_NVDLA_HLS_shiftrightsatsu.
 */
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
 *
 * Each X-stage (BS, BN) has: ALU → MUL → ReLU sub-stages.
 * The cfg register has identical bit layout for both:
 *   [0]     BYPASS      — skip entire stage
 *   [1]     ALU_BYPASS  — skip ALU sub-stage
 *   [4]     MUL_BYPASS  — skip MUL sub-stage
 *   [5]     MUL_PRELU   — MUL acts as PReLU (multiply only if negative)
 *   [6]     RELU_BYPASS — skip ReLU sub-stage
 *   [7]     RELUX_EN    — use ReLUx (clamp to relux_cmp) instead of ReLU
 *   [8]     ALU_SRC     — 0=register, 1=DMA
 *   [19:16] ALU_ALGO    — 0=max, 1=min, 2=add
 * ====================================================================== */

static inline int32_t apply_sdp_x_stage(int32_t value, uint32_t cfg,
                                         int32_t alu_operand,
                                         uint32_t mul_cfg,
                                         int32_t relux_cmp)
{
    /* Stage bypass */
    if (cfg & 0x01) {
        return value;
    }

    /* ALU sub-stage */
    if (!(cfg & 0x02)) { /* ALU_BYPASS */
        uint32_t algo = (cfg >> 16) & 0xf;
        switch (algo) {
        case 0: /* MAX */
            if (value < alu_operand) value = alu_operand;
            break;
        case 1: /* MIN */
            if (value > alu_operand) value = alu_operand;
            break;
        case 2: /* ADD */
            value += alu_operand;
            break;
        default:
            break;
        }
    }

    /* MUL sub-stage */
    if (!(cfg & 0x10)) { /* MUL_BYPASS */
        int16_t mul_operand = (int16_t)((mul_cfg >> 16) & 0xffff);
        unsigned mul_shift = (mul_cfg >> 8) & 0x3f;
        bool mul_prelu = (cfg >> 5) & 1;

        if (!mul_prelu || value < 0) {
            int64_t product = (int64_t)value * (int64_t)mul_operand;
            value = (int32_t)nvdla_shift_right_round64(product, mul_shift);
        }
    }

    /* ReLU sub-stage */
    if (!(cfg & 0x40)) { /* RELU_BYPASS */
        bool relux_en = (cfg >> 7) & 1;
        if (relux_en) {
            /* ReLUx: clamp to [0, relux_cmp] */
            if (value < 0) value = 0;
            if (value > relux_cmp) value = relux_cmp;
        } else {
            /* Standard ReLU */
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

    /*
     * Determine input dimensions for address calculation.
     * The regcmd provides input_width/input_height from CNA_DATA_SIZE0.
     */
    uint32_t in_w = task->input_width;
    uint32_t in_h = task->input_height;

    /* Compute buffer sizes */
    uint32_t in_groups = DIV_ROUND_UP(in_c, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t in_buf_size = in_groups * in_w * in_h * NPU_FEATURE_ATOMIC_SIZE;

    uint32_t wt_oc = depthwise ? 1 : ALIGN_UP(MAX2(out_c, 2), 2);
    uint32_t wt_ic = MAX2(in_c_real, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t wt_ic_group = depthwise ? NPU_WEIGHT_ATOMIC_SIZE * 2
                                      : NPU_WEIGHT_ATOMIC_SIZE;
    uint32_t wt_buf_size = DIV_ROUND_UP(wt_oc, NPU_WEIGHT_ATOMIC_SIZE)
                         * DIV_ROUND_UP(wt_ic, wt_ic_group)
                         * filt_w * filt_h
                         * MIN2(wt_oc, NPU_WEIGHT_ATOMIC_SIZE)
                         * MIN2(wt_ic, wt_ic_group);

    /*
     * Determine bias source.
     * The BS pipeline is handled generically below via apply_sdp_x_stage(),
     * but we still pre-load bias from DMA into bias_buf[] for the
     * accumulator initialization (bias is added before CACC truncation,
     * not in the BS ALU — it's the CACC bias port).
     *
     * When BS is NOT bypassed: bias comes from BRDMA (DMA) or BS_ALU_CFG
     * register. When BS IS bypassed: per-channel mode uses BS_ALU_CFG.
     */
    bool bias_from_dma = (task->brdma_cfg & 0x1e) == 0x02; /* BRDMA_DATA_USE=1 */
    uint32_t bias_buf_size = out_c * sizeof(int32_t);

    /* EW stage: check if element-wise source tensor needs to be read */
    bool ew_bypass = (task->ew_cfg & 0x01);
    bool ew_op_bypass = (task->ew_cfg >> 1) & 1;
    bool ew_active = !ew_bypass && !ew_op_bypass;
    bool erdma_enabled = !(task->erdma_cfg & 0x01); /* ERDMA_DISABLE bit */
    uint8_t *ew_buf = NULL;
    uint32_t ew_buf_size = 0;

    /* Output buffer: y-major interleaved */
    uint32_t out_groups = DIV_ROUND_UP(out_c, NPU_FEATURE_ATOMIC_SIZE);
    uint32_t out_buf_size = out_groups * out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;

    /* Allocate temporary buffers */
    uint8_t *in_buf = g_malloc0(in_buf_size);
    uint8_t *wt_buf = g_malloc0(wt_buf_size);
    int32_t *bias_buf = g_malloc0(bias_buf_size);
    uint8_t *out_buf = g_malloc0(out_buf_size);

    /* Read input tensor from guest memory (via IOVA translation) */
    npu_dma_read(s, task->src_addr, in_buf, in_buf_size);

    /* Read weight buffer from guest memory */
    npu_dma_read(s, task->weight_addr, wt_buf, wt_buf_size);

    /*
     * Read bias from DMA.
     * In the real hardware pipeline: CACC → truncate → BS(ALU=ADD bias) → BN → EW → OUT_CVT
     * The bias is applied by the BS ALU stage, not inside the CACC accumulator.
     * However, Mesa/librocketnpu configures BS_ALU_SRC=1 (DMA) with BS_ALU_ALGO=ADD,
     * so apply_sdp_x_stage reads the ALU operand from the register. For DMA-sourced
     * bias, we load it into a per-channel buffer and apply it as per-channel ALU operand
     * inside the BS stage.
     */
    if (bias_from_dma && task->bias_addr != 0) {
        npu_dma_read(s, task->bias_addr, bias_buf, bias_buf_size);
    }

    /* Read EW source tensor if element-wise addition is active */
    if (ew_active && erdma_enabled && task->ew_base_addr != 0) {
        ew_buf_size = out_groups * out_w * out_h * NPU_FEATURE_ATOMIC_SIZE;
        ew_buf = g_malloc0(ew_buf_size);
        npu_dma_read(s, task->ew_base_addr, ew_buf, ew_buf_size);
    }

    /* Requantization parameters */
    uint32_t out_cvt_scale = task->out_cvt_scale;
    uint32_t out_cvt_shift = task->out_cvt_shift;
    int32_t out_cvt_offset = (int32_t)task->out_cvt_offset;

    /* Padding value (already in signed domain, typically input_zp - 0x80) */
    int8_t pad_val = (int8_t)(task->pad_value & 0xff);

    /* Convolution loop */
    for (uint32_t oy = 0; oy < out_h; oy++) {
        for (uint32_t ox = 0; ox < out_w; ox++) {
            for (uint32_t oc = 0; oc < out_c; oc++) {
                int32_t acc = 0;

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
                                /* Padding */
                                in_val = pad_val;
                            } else {
                                /* Read from interleaved x-major input */
                                uint32_t g = abs_ic / NPU_FEATURE_ATOMIC_SIZE;
                                uint32_t c = abs_ic % NPU_FEATURE_ATOMIC_SIZE;
                                uint32_t off = npu_input_offset(g, ix, iy,
                                                                in_w, in_h);
                                in_val = (int8_t)in_buf[off + c];
                            }

                            /* Weight value (already signed in buffer) */
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

                            /*
                             * Accumulate: in_val * w_val
                             * Both values are in signed int8 domain:
                             * - input was converted: in_val = orig - 0x80
                             * - weight was packed:   w_val  = orig - 0x80
                             * Bias already accounts for zero-point cross terms.
                             */
                            acc += (int32_t)in_val * (int32_t)w_val;
                        }
                    }
                }

                /*
                 * CACC truncation (CLIP_TRUNCATE).
                 * NVDLA uses round-half-to-even via guide+sticky bits.
                 */
                acc = nvdla_truncate(acc, task->truncate_bits);

                /*
                 * SDP Pipeline: BS → BN → EW → OUT_CVT
                 *
                 * BS stage: ALU operand is per-channel from DMA (ALU_SRC=1)
                 * or scalar from register (ALU_SRC=0).
                 */
                {
                    bool bs_alu_src = (task->bs_cfg >> 8) & 1;
                    int32_t bs_alu_op = bs_alu_src ? bias_buf[oc]
                                                   : task->bs_alu_cfg;
                    acc = apply_sdp_x_stage(acc, task->bs_cfg,
                                             bs_alu_op,
                                             task->bs_mul_cfg,
                                             (int32_t)task->bs_relux_cmp);
                }

                /* BN stage: same structure as BS */
                acc = apply_sdp_x_stage(acc, task->bn_cfg,
                                         task->bn_alu_cfg,
                                         task->bn_mul_cfg,
                                         (int32_t)task->bn_relux_cmp);

                /*
                 * EW stage: element-wise operation with second tensor.
                 * When active, reads from ERDMA source and applies ALU.
                 */
                if (!ew_bypass) {
                    if (!ew_op_bypass) {
                        int32_t ew_operand = 0;
                        uint32_t ew_algo = (task->ew_cfg >> 16) & 0xf;

                        /* Read EW source element from DMA buffer */
                        if (ew_buf != NULL) {
                            uint32_t og = oc / NPU_FEATURE_ATOMIC_SIZE;
                            uint32_t oc_within = oc % NPU_FEATURE_ATOMIC_SIZE;
                            uint32_t ew_off = npu_output_offset(og, oy, ox,
                                                                 out_w, out_h);
                            ew_operand = (int32_t)(int8_t)ew_buf[ew_off +
                                                                  oc_within];
                        }

                        /* Apply EW ALU operation */
                        switch (ew_algo) {
                        case 0: /* MAX */
                            if (acc < ew_operand) acc = ew_operand;
                            break;
                        case 1: /* MIN */
                            if (acc > ew_operand) acc = ew_operand;
                            break;
                        case 2: /* ADD */
                            acc += ew_operand;
                            break;
                        default:
                            break;
                        }
                    }

                    /* EW CVT: offset + truncate */
                    bool ew_cvt_bypass = (task->ew_cfg >> 8) & 1;
                    if (!ew_cvt_bypass) {
                        acc += task->ew_cvt_offset;
                        uint32_t ew_trunc = (task->ew_cvt_scale >> 22) & 0x3ff;
                        acc = (int32_t)nvdla_truncate(acc, ew_trunc);
                    }

                    /* EW ReLU / ReLUx */
                    if (!((task->ew_cfg >> 9) & 1)) { /* EW_RELU_BYPASS */
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

                /*
                 * OUT_CVT: requantization
                 * scaled = (acc * scale) >> shift   (NVDLA rounding)
                 * result = clamp(scaled + offset, -128, 127)
                 */
                int64_t scaled = nvdla_shift_right_round64(
                    (int64_t)acc * (int64_t)out_cvt_scale, out_cvt_shift);

                /* Saturate to 17-bit signed before int8 clamp (NVDLA chain) */
                if (scaled > 65535) scaled = 65535;
                if (scaled < -65536) scaled = -65536;

                int32_t result = (int32_t)scaled + out_cvt_offset;

                /* Clamp to int8 range (signed, -128..127) */
                if (result < -128) result = -128;
                if (result > 127) result = 127;

                /*
                 * Write to output in y-major interleaved format.
                 * The output buffer uses [group][y][x][c16] layout.
                 */
                uint32_t og = oc / NPU_FEATURE_ATOMIC_SIZE;
                uint32_t oc_within = oc % NPU_FEATURE_ATOMIC_SIZE;
                uint32_t out_off = npu_output_offset(og, oy, ox,
                                                      out_w, out_h);
                out_buf[out_off + oc_within] = (uint8_t)(int8_t)result;
            }
        }
    }

    /* Write output tensor to guest memory (via IOVA translation).
     * When output_surface_stride is set, write each group as a separate
     * DMA chunk with stride between groups (sparse output layout from
     * per-channel group decomposition).
     */
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
    g_free(ew_buf);
    g_free(out_buf);
}

/* ======================================================================
 * Job execution — process a complete regcmd buffer
 * ====================================================================== */

static void execute_job(RockchipNPUState *s, RocketNPUCore *core,
                        uint32_t base_addr, uint32_t encoded_amounts)
{
    /*
     * Kernel encodes regcmd count as:
     *   PC_DATA_AMOUNT = (regcmd_count + 1) / 2 - 1
     * where regcmd_count = number of uint64_t entries.
     * To recover: regcmd_count = (PC_DATA_AMOUNT + 1) * 2
     */
    uint32_t current_addr = base_addr;
    uint32_t current_count = (encoded_amounts + 1) * 2;

    while (current_count > 0 && current_addr != 0) {
        /* Sanity check */
        if (current_count > NPU_MAX_REGCMD_ENTRIES) {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "rockchip-npu: core%d: regcmd count %u exceeds max\n",
                          core->core_id, current_count);
            break;
        }

        /* Read regcmd buffer from guest memory (via IOVA translation) */
        uint64_t *entries = g_malloc(current_count * sizeof(uint64_t));
        npu_dma_read(s, current_addr, entries,
                     current_count * sizeof(uint64_t));

        /* Parse into task */
        RocketConvTask task;
        qemu_log_mask(LOG_UNIMP,
                      "rockchip-npu: parsing %u entries at 0x%08x\n",
                      current_count, current_addr);
        /* Dump first 5 regcmd entries for debugging */
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
        /* Execute the convolution */
        execute_convolution(s, core, &task);

        /* Follow task chain if present */
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

    /* Signal completion: raise IRQ
     * The kernel ISR checks for DPU_0 (0x100) or DPU_1 (0x200) in raw status.
     * Set both to indicate successful completion.
     */
    core->pc_irq_raw_status |= 0x0300; /* DPU_0 | DPU_1 */
    core->pc_irq_status = core->pc_irq_raw_status & core->pc_irq_mask;
    if (core->pc_irq_status) {
        qemu_irq_raise(core->irq);
    }
}

/* ======================================================================
 * MMIO read/write handlers
 * ====================================================================== */

/*
 * Check if an address falls within the embedded IOMMU region.
 * In rknpu mode, IOMMU registers appear at offsets 0x9000 and 0xa000
 * within the 0x10000 per-core region.
 */
static bool is_iommu_offset(hwaddr addr)
{
    return (addr >= 0x9000 && addr < 0x9100) ||
           (addr >= 0xa000 && addr < 0xa100);
}

static unsigned iommu_instance_for_offset(hwaddr addr, unsigned core_id)
{
    /* Core 0: 0x9000 → instance 0, 0xa000 → instance 1
     * Core 1: 0xa000 → instance 2
     * Core 2: 0xa000 → instance 3
     */
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

    /* Forward IOMMU register reads in rknpu mode */
    if (s->driver_mode == NPU_DRIVER_MODE_RKNPU && is_iommu_offset(addr)
        && s->rk_iommu) {
        unsigned inst = iommu_instance_for_offset(addr, core->core_id);
        hwaddr iommu_reg = addr & 0xFF;
        return rk_iommu_instance_read(&s->rk_iommu->instances[inst],
                                       iommu_reg, size);
    }

    switch (addr) {
    case REG_PC_VERSION:
        return ROCKET_PC_VERSION;
    case REG_PC_VERSION_NUM:
        return ROCKET_PC_VERSION_NUM;
    case REG_PC_OP_ENABLE:
        return 0; /* Always idle after synchronous execution */
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
        return 0; /* Tasks always complete immediately */
    default:
        /* Return stored register value or 0 */
        if (addr < NPU_REGION_SIZE_RKNPU) {
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

    /* Forward IOMMU register writes in rknpu mode */
    if (s->driver_mode == NPU_DRIVER_MODE_RKNPU && is_iommu_offset(addr)
        && s->rk_iommu) {
        unsigned inst = iommu_instance_for_offset(addr, core->core_id);
        hwaddr iommu_reg = addr & 0xFF;
        rk_iommu_instance_write(&s->rk_iommu->instances[inst],
                                 iommu_reg, val, size);
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
            /* Trigger job execution */
            execute_job(s, core, core->pc_base_addr, core->pc_reg_amounts);
        }
        break;

    case REG_PC_IRQ_MASK:
        core->pc_irq_mask = (uint32_t)val;
        /* Re-evaluate masked status */
        core->pc_irq_status = core->pc_irq_raw_status & core->pc_irq_mask;
        if (!core->pc_irq_status) {
            qemu_irq_lower(core->irq);
        }
        break;

    case REG_PC_IRQ_CLEAR:
        core->pc_irq_raw_status &= ~(uint32_t)val;
        core->pc_irq_status = core->pc_irq_raw_status & core->pc_irq_mask;
        if (!core->pc_irq_status) {
            qemu_irq_lower(core->irq);
        }
        break;

    case REG_PC_TASK_CON:
        core->pc_task_con = (uint32_t)val;
        break;

    default:
        /* Store all register writes (may be read back by guest) */
        if (addr < NPU_REGION_SIZE_RKNPU) {
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

    uint64_t region_size = (s->driver_mode == NPU_DRIVER_MODE_RKNPU)
                           ? NPU_REGION_SIZE_RKNPU : NPU_REGION_SIZE;

    for (int i = 0; i < s->num_cores; i++) {
        RocketNPUCore *core = &s->cores[i];
        core->core_id = i;
        core->pc_irq_mask = 0;

        snprintf(name, sizeof(name), "rockchip-npu-core%d", i);
        memory_region_init_io(&core->iomem, OBJECT(dev), &rockchip_npu_ops,
                              core, name, region_size);
        sysbus_init_mmio(sbd, &core->iomem);
        sysbus_init_irq(sbd, &core->irq);
    }

    /* IOMMU mailbox MMIO — only used in Rocket mode */
    if (s->driver_mode == NPU_DRIVER_MODE_ROCKET) {
        memory_region_init_io(&s->iommu_mmio, OBJECT(dev), &npu_iommu_ops,
                              s, "rockchip-npu-iommu", NPU_IOMMU_SIZE);
        sysbus_init_mmio(sbd, &s->iommu_mmio);
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
        core->pc_irq_raw_status = 0;
        core->pc_task_con = 0;
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
        VMSTATE_UINT32_ARRAY(regs, RocketNPUCore, NPU_REGION_SIZE_RKNPU / 4),
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
    DEFINE_PROP_UINT32("num-cores", RockchipNPUState, num_cores, 1),
    DEFINE_PROP_UINT32("driver-mode", RockchipNPUState, driver_mode, 0),
};

static void rockchip_npu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = rockchip_npu_realize;
    device_class_set_legacy_reset(dc, rockchip_npu_reset);
    dc->vmsd = &vmstate_rockchip_npu;
    dc->desc = "Rockchip RK3588 Rocket NPU (configurable cores, INT8 convolution)";
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
