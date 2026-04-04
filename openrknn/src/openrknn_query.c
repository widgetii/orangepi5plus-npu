/*
 * openrknn — rknn_query implementation (Phase 2 stub)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"

int orknn_own_query(struct orknn_context *ctx, rknn_query_cmd cmd,
                    void *info, uint32_t size)
{
    (void)ctx; (void)cmd; (void)info; (void)size;
    orknn_log(0, "orknn_own_query: NOT IMPLEMENTED (cmd=%d)", cmd);
    return RKNN_ERR_FAIL;
}
