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

#endif /* RNPU_REGCMD_H */
