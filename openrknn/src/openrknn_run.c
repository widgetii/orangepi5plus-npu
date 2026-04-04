/*
 * openrknn — NPU execution (Phase 5 stub)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"

int orknn_own_run(struct orknn_context *ctx, rknn_run_extend *extend)
{
    (void)ctx; (void)extend;
    orknn_log(0, "orknn_own_run: NOT IMPLEMENTED");
    return RKNN_ERR_FAIL;
}
