/*
 * Rockchip RK3588 NPU device model
 *
 * Emulates 3 NPU cores with MMIO register interface, regcmd parser,
 * and software INT8 convolution engine for CI testing.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_MISC_ROCKCHIP_NPU_H
#define HW_MISC_ROCKCHIP_NPU_H

#include "hw/sysbus.h"
#include "qom/object.h"

typedef struct RockchipIOMMUState RockchipIOMMUState;

#define TYPE_ROCKCHIP_NPU "rockchip-npu"
OBJECT_DECLARE_SIMPLE_TYPE(RockchipNPUState, ROCKCHIP_NPU)

/* Hardware constants */
#define NPU_IDLE_RAW_BITS        0xc0000000  /* bits 30-31 always set when idle */
#define NPU_FEATURE_ATOMIC_SIZE  16
#define NPU_WEIGHT_ATOMIC_SIZE   32
#define NPU_ATOMIC_K_SIZE        16
#define NPU_MAX_REGCMD_ENTRIES   512

/* MMIO region sizes — 0x10000 per core (matches real hardware) */
#define NPU_PC_OFFSET    0x0000  /* Program Controller */
#define NPU_CNA_OFFSET   0x1000  /* Convolution engine */
#define NPU_CORE_OFFSET  0x3000  /* Core control */
#define NPU_DPU_OFFSET   0x4000  /* Data Processing Unit */
#define NPU_RDMA_OFFSET  0x5000  /* DPU Read DMA */
#define NPU_REGION_SIZE  0x10000

/* PC registers */
#define REG_PC_VERSION           0x0000
#define REG_PC_VERSION_NUM       0x0004
#define REG_PC_OP_ENABLE         0x0008
#define REG_PC_BASE_ADDRESS      0x0010
#define REG_PC_REGISTER_AMOUNTS  0x0014
#define REG_PC_IRQ_MASK          0x0020
#define REG_PC_IRQ_CLEAR         0x0024
#define REG_PC_IRQ_STATUS        0x0028
#define REG_PC_IRQ_RAW_STATUS    0x002c
#define REG_PC_TASK_CON          0x0030
#define REG_PC_TASK_STATUS       0x003c

/* CNA registers (offsets from CNA base = core_base + 0x1000) */
#define CNA_CONV_CON1     0x100c
#define CNA_CONV_CON3     0x1014
#define CNA_DATA_SIZE0    0x1020
#define CNA_DATA_SIZE1    0x1024
#define CNA_DATA_SIZE2    0x1028
#define CNA_DATA_SIZE3    0x102c
#define CNA_WEIGHT_SIZE0  0x1030
#define CNA_WEIGHT_SIZE1  0x1034
#define CNA_WEIGHT_SIZE2  0x1038
#define CNA_PAD_CON0      0x1068
#define CNA_PAD_CON1      0x1184
#define CNA_FEAT_ADDR     0x1070
#define CNA_DCOMP_ADDR0   0x1110

/* Core registers */
#define CORE_MISC_CFG      0x3010
#define CORE_DATAOUT_SIZE0 0x3014
#define CORE_DATAOUT_SIZE1 0x3018
#define CORE_CLIP_TRUNCATE 0x301c

/* DPU registers */
#define DPU_FEAT_MODE_CFG    0x400c
#define DPU_DATA_FORMAT      0x4010
#define DPU_DST_BASE_ADDR    0x4020
#define DPU_DST_SURF_STRIDE  0x4024
#define DPU_DATA_CUBE_WIDTH  0x4030
#define DPU_DATA_CUBE_HEIGHT 0x4034
#define DPU_DATA_CUBE_CHANNEL 0x403c
#define DPU_BS_CFG           0x4040
#define DPU_BS_ALU_CFG       0x4044
#define DPU_BS_MUL_CFG       0x4048
#define DPU_BS_RELUX_CMP     0x404c
#define DPU_BS_OW_OP         0x4054
#define DPU_BN_CFG           0x4060
#define DPU_BN_ALU_CFG       0x4064
#define DPU_BN_MUL_CFG       0x4068
#define DPU_BN_RELUX_CMP     0x406c
#define DPU_EW_CFG           0x4070
#define DPU_EW_CVT_OFFSET    0x4074
#define DPU_EW_CVT_SCALE     0x4078
#define DPU_OUT_CVT_OFFSET   0x4080
#define DPU_OUT_CVT_SCALE    0x4084
#define DPU_OUT_CVT_SHIFT    0x4088
#define DPU_BS_OW_CFG        0x4050
#define DPU_WDMA_SIZE_0      0x4058
#define DPU_WDMA_SIZE_1      0x405c
#define DPU_EW_RELUX_CMP     0x407c
#define DPU_SURFACE_ADD      0x40c0

/* RDMA registers */
#define RDMA_SRC_BASE_ADDR  0x5018
#define RDMA_BRDMA_CFG      0x501c
#define RDMA_BS_BASE_ADDR   0x5020
#define RDMA_NRDMA_CFG      0x5028
#define RDMA_BN_BASE_ADDR   0x502c
#define RDMA_ERDMA_CFG      0x5034
#define RDMA_EW_BASE_ADDR   0x5038
#define RDMA_EW_SURF_STRIDE 0x5040
#define RDMA_FEAT_MODE_CFG  0x5044
#define RDMA_WEIGHT         0x5068

/* Regcmd format: 64-bit packed entry
 * bits [15:0]  = register offset
 * bits [47:16] = register value
 * bits [63:48] = target (selects which block)
 */
#define REGCMD_REG_MASK    0x000000000000FFFFULL
#define REGCMD_VALUE_SHIFT 16
#define REGCMD_VALUE_MASK  0x0000FFFFFFFF0000ULL
#define REGCMD_TARGET_SHIFT 48

/* Target IDs from the hardware (matched in register XML) */
#define TARGET_PC    0x0100
#define TARGET_CNA   0x0200
#define TARGET_CORE  0x0800
#define TARGET_DPU   0x1000
#define TARGET_RDMA  0x2000

/* Parsed convolution task state from regcmd */
typedef struct RocketConvTask {
    /* Input */
    uint32_t src_addr;
    uint32_t input_width;
    uint32_t input_height;
    uint32_t input_channels;
    uint32_t input_channels_real;
    uint32_t input_line_stride;
    uint32_t input_surface_stride;

    /* Weights */
    uint32_t weight_addr;
    uint32_t weight_width;
    uint32_t weight_height;
    uint32_t weight_kernels;
    uint32_t weight_size0;
    uint32_t weight_size1;

    /* Output */
    uint32_t dst_addr;
    uint32_t output_width;
    uint32_t output_height;
    uint32_t output_channels;
    uint32_t output_channels_real;
    uint32_t output_surface_stride;

    /* Convolution config */
    uint32_t stride_x;
    uint32_t stride_y;
    uint32_t pad_left;
    uint32_t pad_top;
    int32_t  pad_value;
    bool     depthwise;

    /* Quantization */
    uint32_t out_cvt_offset;
    uint32_t out_cvt_scale;
    uint32_t out_cvt_shift;
    uint32_t truncate_bits;

    /* BS (Bias/Scale) stage */
    uint32_t bias_addr;
    uint32_t bs_cfg;
    int32_t  bs_alu_cfg;
    uint32_t bs_mul_cfg;
    uint32_t bs_relux_cmp;
    uint32_t bs_ow_op;
    uint32_t brdma_cfg;

    /* BN (Batch Norm) stage */
    uint32_t bn_cfg;
    int32_t  bn_alu_cfg;
    uint32_t bn_mul_cfg;
    uint32_t bn_relux_cmp;

    /* EW (Element-Wise) stage */
    uint32_t ew_cfg;
    int32_t  ew_cvt_offset;
    uint32_t ew_cvt_scale;
    uint32_t erdma_cfg;
    uint32_t ew_base_addr;
    uint32_t ew_surf_stride;

    /* Data format */
    uint32_t data_format;
    uint32_t surface_add;
    uint32_t feature_mode_cfg;
    uint32_t bs_ow_cfg;
    uint32_t wdma_size_0;
    uint32_t wdma_size_1;
    uint32_t ew_relux_cmp;

    /* RDMA additional */
    uint32_t rdma_src_base_addr;
    uint32_t nrdma_cfg;
    uint32_t bn_base_addr;
    uint32_t rdma_feat_mode_cfg;
    uint32_t rdma_weight;

    /* CNA input conversion */
    uint32_t conv_con1;

    /* Task chaining */
    uint32_t next_base_addr;
    uint32_t next_reg_amounts;
} RocketConvTask;

/* Per-core NPU state */
typedef struct RocketNPUCore {
    MemoryRegion iomem;
    qemu_irq irq;
    unsigned core_id;

    /* Deferred IRQ: convolution runs immediately in MMIO handler, but IRQ
     * is raised after a short timer delay so the DRM scheduler's SUBMIT
     * ioctl return path completes before the fence signals. */
    struct QEMUTimer *irq_timer;

    /* PC registers (directly written by MMIO) */
    uint32_t pc_version;
    uint32_t pc_base_addr;
    uint32_t pc_reg_amounts;
    uint32_t pc_irq_mask;
    uint32_t pc_irq_status;
    uint32_t pc_irq_raw_status;
    uint32_t pc_task_con;
    uint32_t pc_task_status;

    /* Shadow register file — written by regcmd or MMIO */
    uint32_t regs[NPU_REGION_SIZE / 4];
} RocketNPUCore;

struct RockchipNPUState {
    SysBusDevice parent_obj;

    uint32_t num_cores;

    /* Rockchip IOMMU for IOVA→GPA translation (set by machine init) */
    RockchipIOMMUState *rk_iommu;

    RocketNPUCore cores[3];

    /* Guest physical address space for DMA reads/writes */
    AddressSpace *dma_as;
};

/* NPU tensor offset macros.
 * Input: x-major [group][x][y][c16] — matches CNA DMA read convention
 *   and librocketnpu's rnpu_convert_input (loops x outer, y inner).
 * Output: y-major [group][y][x][c16] — matches DPU DMA write convention
 *   and librocketnpu's rnpu_convert_output: offset = (y * W + x) * 16. */
static inline uint32_t npu_input_offset(uint32_t g, uint32_t x, uint32_t y,
                                         uint32_t w, uint32_t h)
{
    return g * w * h * NPU_FEATURE_ATOMIC_SIZE +
           x * h * NPU_FEATURE_ATOMIC_SIZE +
           y * NPU_FEATURE_ATOMIC_SIZE;
}

static inline uint32_t npu_output_offset(uint32_t g, uint32_t x, uint32_t y,
                                          uint32_t w, uint32_t h)
{
    return g * w * h * NPU_FEATURE_ATOMIC_SIZE +
           y * w * NPU_FEATURE_ATOMIC_SIZE +
           x * NPU_FEATURE_ATOMIC_SIZE;
}

#endif /* HW_MISC_ROCKCHIP_NPU_H */
