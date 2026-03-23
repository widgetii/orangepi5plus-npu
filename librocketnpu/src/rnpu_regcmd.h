/*
 * Register command generation for NPU
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_REGCMD_H
#define RNPU_REGCMD_H

#include "rnpu_internal.h"

/* Generate register commands for one task of an operation.
 * Writes packed 64-bit reg commands into dst. Returns number of uint64_t written.
 * Addresses (weights, biases, activations) are DMA addresses from shared BOs. */
unsigned rnpu_fill_regcmd(const struct rnpu_model *model,
                          const struct rnpu_operation *op,
                          uint64_t *dst, unsigned max_regs,
                          unsigned task_num);

/* Hybrid regcmd mask — controls which registers use per-channel values.
 * UINT32_MAX = standard path, 0x3FFFF = full per-channel, other = hybrid.
 * Bit assignments documented in rnpu_regcmd.c fill_hybrid_regcmd(). */
extern uint32_t rnpu_hybrid_mask;

/* Predefined group masks for binary search */
#define HYBRID_GROUP_A  0x0842   /* DPU output: CUBE_CH, WDMA_SIZE, SURFACE_ADD */
#define HYBRID_GROUP_B  0x062C   /* Bias path: BS_CFG, BS_ALU, BS_OW_OP, BRDMA, BS_ADDR */
#define HYBRID_GROUP_C  0x0191   /* Post-proc: DATA_FMT, RELUX, BN_CFG, EW_CFG */
#define HYBRID_GROUP_D  0x7000   /* Input/misc: CLIP_TRUNC, CVT_CON5, PAD_CON1 */
#define HYBRID_GROUP_E  0x38000  /* Extra: RDMA_CH, CONV_CON2, RDMA_WEIGHT */
#define HYBRID_ALL      0x3FFFF  /* All per-channel */

#endif /* RNPU_REGCMD_H */
