/*
 * openrknn — .rknn model parser (Phase 2 stub)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"

int orknn_own_init(struct orknn_context *ctx, void *model, uint32_t size,
                   uint32_t flag, rknn_init_extend *extend)
{
    (void)ctx; (void)model; (void)size; (void)flag; (void)extend;
    orknn_log(0, "orknn_own_init: NOT IMPLEMENTED");
    return RKNN_ERR_FAIL;
}
