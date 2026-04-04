/*
 * openrknn — Input processing (Phase 4 stub)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"

int orknn_own_inputs_set(struct orknn_context *ctx, uint32_t n_inputs,
                         rknn_input inputs[])
{
    (void)ctx; (void)n_inputs; (void)inputs;
    orknn_log(0, "orknn_own_inputs_set: NOT IMPLEMENTED");
    return RKNN_ERR_FAIL;
}
