/*
 * openrknn — Memory management (Phase 3 stub)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"

rknn_tensor_mem *orknn_own_create_mem(struct orknn_context *ctx, uint32_t size)
{
    (void)ctx; (void)size;
    orknn_log(0, "orknn_own_create_mem: NOT IMPLEMENTED");
    return NULL;
}

int orknn_own_destroy_mem(struct orknn_context *ctx, rknn_tensor_mem *mem)
{
    (void)ctx; (void)mem;
    orknn_log(0, "orknn_own_destroy_mem: NOT IMPLEMENTED");
    return RKNN_ERR_FAIL;
}
