/*
 * openrknn — Output processing (Phase 6 stub)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"

int orknn_own_outputs_get(struct orknn_context *ctx, uint32_t n_outputs,
                          rknn_output outputs[], rknn_output_extend *extend)
{
    (void)ctx; (void)n_outputs; (void)outputs; (void)extend;
    orknn_log(0, "orknn_own_outputs_get: NOT IMPLEMENTED");
    return RKNN_ERR_FAIL;
}
