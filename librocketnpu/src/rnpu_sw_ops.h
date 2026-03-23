/*
 * CPU software ops: concat, max_pool, avg_pool, pad, resize, logistic, reshape, softmax
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_SW_OPS_H
#define RNPU_SW_OPS_H

#include "rnpu_internal.h"

/* Execute a software operation on CPU. */
void rnpu_execute_sw_op(struct rnpu_model *m, unsigned op_index);

/* Apply per-axis scale correction after CONV. */
void rnpu_apply_per_axis_correction(struct rnpu_model *m,
                                    struct rnpu_operation *op);

/* Compact and un-sort per-group output. */
void rnpu_compact_unsort_output(struct rnpu_model *m,
                                unsigned first_group_op,
                                unsigned num_group_ops);

/* Scatter requant group outputs: pick correctly-scaled channels from each
 * group's full output copy into the final output tensor. */
void rnpu_scatter_requant_output(struct rnpu_model *m,
                                  struct rnpu_operation *op);

/* Apply per-channel MUL quantization correction for BRDMA ops.
 * Corrects the residual error from int16 MUL rounding: for each channel,
 * correction = exact_ratio / (round(exact_ratio * 2^shift) / 2^shift). */
void rnpu_apply_brdma_correction(struct rnpu_model *m,
                                  struct rnpu_operation *op);

#endif /* RNPU_SW_OPS_H */
