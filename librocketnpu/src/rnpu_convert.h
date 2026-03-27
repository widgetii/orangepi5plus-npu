/*
 * NHWC <-> NPU format conversion (NEON optimized)
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_CONVERT_H
#define RNPU_CONVERT_H

#include "rnpu_internal.h"

/* Convert NHWC uint8 input to NPU interleaved format with 0x80 bias. */
void rnpu_convert_input(uint8_t *npu_buf, const uint8_t *nhwc,
                        unsigned width, unsigned height, unsigned channels,
                        uint8_t zero_point);

/* Convert NPU output to NHWC uint8, removing 0x80 bias. */
void rnpu_convert_output(uint8_t *nhwc, const uint8_t *npu_buf,
                         unsigned width, unsigned height, unsigned channels);
/* Extended: add_offset=false skips +0x80 (for INT8 output tensors). */
void rnpu_convert_output_ex(uint8_t *nhwc, const uint8_t *npu_buf,
                             unsigned width, unsigned height, unsigned channels,
                             bool add_offset);

/* Compute NPU tensor size (with 2x group padding). */
unsigned rnpu_calc_npu_tensor_size(unsigned w, unsigned h, unsigned c);

/* Compute raw output size (matching Mesa calc_raw_output_size). */
unsigned rnpu_calc_raw_output_size(unsigned w, unsigned h, unsigned oc,
                                   unsigned tensor_oc);

#endif /* RNPU_CONVERT_H */
