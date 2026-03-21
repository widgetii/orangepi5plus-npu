/*
 * Weight/bias preparation for NPU format
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_COEFS_H
#define RNPU_COEFS_H

#include "rnpu_internal.h"

/* Fill weight data for a standard CONV into dst buffer. Returns bytes written. */
unsigned rnpu_fill_weights(const struct rnpu_tfl_model *tfl,
                           const struct rnpu_tfl_op *op,
                           uint8_t *dst, unsigned dst_size);

/* Fill weight data for a group of output channels. Returns bytes written. */
unsigned rnpu_fill_weights_group(const struct rnpu_tfl_model *tfl,
                                 const struct rnpu_tfl_op *op,
                                 const unsigned *channel_indices,
                                 unsigned group_count,
                                 uint8_t *dst, unsigned dst_size);

/* Compute weight buffer size for given op parameters. */
unsigned rnpu_calc_weight_size(unsigned ww, unsigned wh,
                               unsigned ic, unsigned oc,
                               bool depthwise);

/* Fill bias data. Returns bytes written. */
unsigned rnpu_fill_biases(const struct rnpu_tfl_model *tfl,
                          const struct rnpu_tfl_op *op,
                          unsigned *truncate_bits,
                          uint8_t *dst, unsigned dst_size);

/* Fill biases for a group of channels. Returns bytes written. */
unsigned rnpu_fill_biases_group(const struct rnpu_tfl_model *tfl,
                                const struct rnpu_tfl_op *op,
                                const unsigned *channel_indices,
                                unsigned group_count,
                                unsigned *truncate_bits,
                                uint8_t *dst, unsigned dst_size);

/* Compute single bias scalar for per-channel ops. */
int32_t rnpu_compute_bias_scalar(const struct rnpu_tfl_model *tfl,
                                 const struct rnpu_tfl_op *op,
                                 unsigned channel);

#endif /* RNPU_COEFS_H */
