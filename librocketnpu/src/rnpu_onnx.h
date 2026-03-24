/*
 * ONNX model parser for librocketnpu
 * Parses ONNX protobuf (especially check0_base_optimize.onnx from RKNN toolkit)
 * into the same rnpu_tfl_model struct used by the TFLite path.
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_ONNX_H
#define RNPU_ONNX_H

#include "rnpu_internal.h"

/* Parse ONNX model file into a rnpu_tfl_model struct.
 * The resulting struct is compatible with the existing rnpu_model pipeline.
 * Supports quantized ONNX graphs with QuantizeLinear/DequantizeLinear nodes.
 * Returns 0 on success, -1 on error. */
int rnpu_onnx_parse(const char *path, struct rnpu_tfl_model *model);

#endif /* RNPU_ONNX_H */
