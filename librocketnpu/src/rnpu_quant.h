/*
 * Per-channel task decomposition for per-axis quantized CONVs
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_QUANT_H
#define RNPU_QUANT_H

#include "rnpu_internal.h"

/* Lower a per-axis CONV into per-group operations with scale-sorted grouping.
 * Appends group_count operations to ops_out, updating *op_count.
 * Returns number of groups created. */
unsigned rnpu_lower_per_group(struct rnpu_model *m,
                              const struct rnpu_tfl_op *top,
                              struct rnpu_operation **ops_out,
                              unsigned *op_count);

#endif /* RNPU_QUANT_H */
