/*
 * Rockchip RK3588 Rocket NPU device model
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

#define TYPE_ROCKCHIP_NPU "rockchip-npu"
OBJECT_DECLARE_SIMPLE_TYPE(RockchipNPUState, ROCKCHIP_NPU)

/* Hardware constants from the Rocket driver */
#define NPU_FEATURE_ATOMIC_SIZE  16
#define NPU_WEIGHT_ATOMIC_SIZE   32
#define NPU_ATOMIC_K_SIZE        16
#define NPU_MAX_REGCMD_ENTRIES   512

/* MMIO region sizes within each core */
#define NPU_PC_OFFSET    0x0000  /* Program Controller */
#define NPU_CNA_OFFSET   0x1000  /* Convolution engine */
#define NPU_CORE_OFFSET  0x3000  /* Core control */
#define NPU_DPU_OFFSET   0x4000  /* Data Processing Unit */
#define NPU_RDMA_OFFSET  0x5000  /* DPU Read DMA */
#define NPU_REGION_SIZE  0x6000

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
#define CNA_PAD_CON1      0x106c
#define CNA_FEAT_ADDR     0x1070
#define CNA_DCOMP_ADDR0   0x1110

/* Core registers */
#define CORE_MISC_CFG      0x3010
#define CORE_DATAOUT_SIZE0 0x3014
#define CORE_DATAOUT_SIZE1 0x3018
#define CORE_CLIP_TRUNCATE 0x301c

/* DPU registers */
#define DPU_FEAT_MODE_CFG    0x4010
#define DPU_DATA_FORMAT      0x4014
#define DPU_DST_BASE_ADDR    0x4020
#define DPU_DST_SURF_STRIDE  0x4024
#define DPU_DATA_CUBE_WIDTH  0x4030
#define DPU_DATA_CUBE_HEIGHT 0x4034
#define DPU_DATA_CUBE_CHANNEL 0x403c
#define DPU_BS_CFG           0x4040
#define DPU_BS_ALU_CFG       0x4044
#define DPU_BS_MUL_CFG       0x4048
#define DPU_BS_OW_OP         0x4054
#define DPU_BN_CFG           0x4060
#define DPU_OUT_CVT_OFFSET   0x4080
#define DPU_OUT_CVT_SCALE    0x4084
#define DPU_OUT_CVT_SHIFT    0x4088
#define DPU_SURFACE_ADD      0x40b0

/* RDMA registers */
#define RDMA_BRDMA_CFG      0x501c
#define RDMA_BS_BASE_ADDR   0x5020

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
    uint32_t src_addr;         /* DMA address of input tensor */
    uint32_t input_width;      /* DATAIN_WIDTH field */
    uint32_t input_height;     /* DATAIN_HEIGHT field */
    uint32_t input_channels;   /* DATAIN_CHANNEL field (padded to 16) */
    uint32_t input_channels_real; /* DATAIN_CHANNEL_REAL field */
    uint32_t input_line_stride;
    uint32_t input_surface_stride;

    /* Weights */
    uint32_t weight_addr;      /* DMA address of weight buffer */
    uint32_t weight_width;     /* Filter kernel width */
    uint32_t weight_height;    /* Filter kernel height */
    uint32_t weight_kernels;   /* Number of output kernels (output channels) */
    uint32_t weight_size0;     /* Total weight elements */
    uint32_t weight_size1;     /* Weight elements per kernel */

    /* Output */
    uint32_t dst_addr;         /* DMA address of output tensor */
    uint32_t output_width;     /* DATAOUT_WIDTH */
    uint32_t output_height;    /* DATAOUT_HEIGHT */
    uint32_t output_channels;  /* DATAOUT_CHANNEL */
    uint32_t output_channels_real;
    uint32_t output_surface_stride;

    /* Convolution config */
    uint32_t stride_x;
    uint32_t stride_y;
    uint32_t pad_left;
    uint32_t pad_top;
    int32_t  pad_value;        /* Padding fill value (from PAD_CON1) */
    bool     depthwise;        /* CONV_MODE == 3 */

    /* Quantization */
    uint32_t out_cvt_offset;
    uint32_t out_cvt_scale;
    uint32_t out_cvt_shift;
    uint32_t truncate_bits;

    /* Bias */
    uint32_t bias_addr;        /* DMA address of bias buffer */
    uint32_t bs_cfg;           /* BS stage config */
    int32_t  bs_alu_cfg;       /* BS ALU config (per-channel bias) */
    uint32_t bs_ow_op;         /* Weight zero point offset */
    uint32_t bn_cfg;           /* BN stage config */
    uint32_t brdma_cfg;        /* Bias RDMA config */

    /* Data format */
    uint32_t data_format;
    uint32_t surface_add;

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

    /* PC registers (directly written by MMIO) */
    uint32_t pc_version;
    uint32_t pc_base_addr;
    uint32_t pc_reg_amounts;
    uint32_t pc_irq_mask;
    uint32_t pc_irq_status;
    uint32_t pc_irq_raw_status;
    uint32_t pc_task_con;

    /* Shadow register file — written by regcmd or MMIO */
    uint32_t regs[NPU_REGION_SIZE / 4];
} RocketNPUCore;

/*
 * IOMMU page table entry for IOVA→GPA translation.
 * The kernel IOMMU module writes mappings via a mailbox MMIO region,
 * and the NPU model uses them to resolve DMA addresses in regcmds.
 */
#define NPU_IOMMU_MAX_PAGES 4096

typedef struct RocketIOMMUEntry {
    uint32_t iova;
    uint32_t phys;
} RocketIOMMUEntry;

struct RockchipNPUState {
    SysBusDevice parent_obj;

    RocketNPUCore cores[3];

    /* Guest physical address space for DMA reads/writes */
    AddressSpace *dma_as;

    /* IOVA→GPA translation table (written by kernel IOMMU module) */
    RocketIOMMUEntry iommu_table[NPU_IOMMU_MAX_PAGES];
    uint32_t iommu_entry_count;

    /* Mailbox registers for IOMMU map/unmap from kernel module */
    MemoryRegion iommu_mmio;
    uint32_t iommu_iova;    /* written first */
    uint32_t iommu_phys;    /* written second, triggers add */
};

/* NPU tensor offset macros matching Mesa/librocketnpu */
static inline uint32_t npu_input_offset(uint32_t g, uint32_t x, uint32_t y,
                                         uint32_t w, uint32_t h)
{
    /* x-major interleaved: [group][x][y][c16] */
    return g * w * h * NPU_FEATURE_ATOMIC_SIZE +
           x * h * NPU_FEATURE_ATOMIC_SIZE +
           y * NPU_FEATURE_ATOMIC_SIZE;
}

static inline uint32_t npu_output_offset(uint32_t g, uint32_t y, uint32_t x,
                                          uint32_t w, uint32_t h)
{
    /* y-major interleaved: [group][y][x][c16] — NPU DPU output order */
    return g * w * h * NPU_FEATURE_ATOMIC_SIZE +
           y * w * NPU_FEATURE_ATOMIC_SIZE +
           x * NPU_FEATURE_ATOMIC_SIZE;
}

#endif /* HW_MISC_ROCKCHIP_NPU_H */
