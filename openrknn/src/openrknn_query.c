/*
 * openrknn — rknn_query implementation
 *
 * Returns tensor metadata from our parsed .rknn model.
 * Cross-validates against proxy when available.
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>

/* Fill rknn_tensor_attr from our tensor info (user-facing format) */
static void fill_tensor_attr(const struct orknn_tensor_info *ti,
                             rknn_tensor_attr *attr, int is_native_nc1hwc2)
{
    uint32_t idx = attr->index; /* preserve caller's index */
    memset(attr, 0, sizeof(*attr));
    attr->index = idx;

    if (is_native_nc1hwc2) {
        attr->n_dims = ti->native_n_dims;
        for (uint32_t i = 0; i < ti->native_n_dims && i < RKNN_MAX_DIMS; i++)
            attr->dims[i] = ti->native_dims[i];
        attr->n_elems = 1;
        for (uint32_t i = 0; i < attr->n_dims; i++)
            attr->n_elems *= attr->dims[i];
        attr->size = ti->native_size;
        /* NC1HWC2 only for 4D+ tensors; non-4D keep original format */
        attr->fmt = (ti->n_dims >= 4) ? RKNN_TENSOR_NC1HWC2 : ti->fmt;
    } else {
        attr->n_dims = ti->n_dims;
        for (uint32_t i = 0; i < ti->n_dims && i < RKNN_MAX_DIMS; i++)
            attr->dims[i] = ti->dims[i];
        attr->n_elems = ti->n_elems;
        attr->size = ti->size;
        attr->fmt = ti->fmt;
    }

    strncpy(attr->name, ti->name, RKNN_MAX_NAME_LEN - 1);
    attr->type = ti->type;
    attr->qnt_type = ti->qnt_type;
    attr->scale = ti->scale;
    attr->zp = ti->zp;
    attr->w_stride = ti->w_stride;
    attr->size_with_stride = is_native_nc1hwc2 ? ti->native_size : ti->size_with_stride;
}

/* Cross-validate: compare our query result against proxy */
static void cross_validate(struct orknn_context *ctx, rknn_query_cmd cmd,
                           const void *own_result, uint32_t size)
{
    if (!ctx->real_ctx) return;
    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy) return;

    void *proxy_result = calloc(1, size);
    /* For ATTR queries, copy the index field from own_result */
    if (size >= 4)
        memcpy(proxy_result, own_result, 4);

    int ret = proxy->rknn_query(ctx->real_ctx, cmd, proxy_result, size);
    if (ret != RKNN_SUCC) {
        orknn_log(2, "query cross-validate: proxy returned %d for cmd=%d", ret, cmd);
        free(proxy_result);
        return;
    }

    if (memcmp(own_result, proxy_result, size) != 0) {
        orknn_log(1, "MISMATCH: query cmd=%d differs from proxy!", cmd);
        /* Log first differing byte */
        const uint8_t *a = own_result, *b = proxy_result;
        for (uint32_t i = 0; i < size; i++) {
            if (a[i] != b[i]) {
                orknn_log(1, "  first diff at offset %u: own=0x%02x proxy=0x%02x",
                          i, a[i], b[i]);
                break;
            }
        }
        /* For tensor attrs, log key fields from proxy */
        if (cmd == RKNN_QUERY_INPUT_ATTR || cmd == RKNN_QUERY_OUTPUT_ATTR ||
            cmd == RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR ||
            cmd == RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR) {
            rknn_tensor_attr *pa = (rknn_tensor_attr *)proxy_result;
            rknn_tensor_attr *oa = (rknn_tensor_attr *)own_result;
            orknn_log(1, "  proxy: dims=%ux%ux%ux%u type=%d fmt=%d qnt=%d "
                      "scale=%.6f zp=%d size=%u name=%s",
                      pa->dims[0], pa->dims[1], pa->dims[2], pa->dims[3],
                      pa->type, pa->fmt, pa->qnt_type,
                      pa->scale, pa->zp, pa->size, pa->name);
            orknn_log(1, "  own:   dims=%ux%ux%ux%u type=%d fmt=%d qnt=%d "
                      "scale=%.6f zp=%d size=%u name=%s",
                      oa->dims[0], oa->dims[1], oa->dims[2], oa->dims[3],
                      oa->type, oa->fmt, oa->qnt_type,
                      oa->scale, oa->zp, oa->size, oa->name);
        }
    } else {
        orknn_log(2, "query cmd=%d: MATCH with proxy", cmd);
    }

    free(proxy_result);
}

int orknn_own_query(struct orknn_context *ctx, rknn_query_cmd cmd,
                    void *info, uint32_t size)
{
    struct orknn_model *m = &ctx->model;

    switch (cmd) {
    case RKNN_QUERY_IN_OUT_NUM: {
        if (size < sizeof(rknn_input_output_num))
            return RKNN_ERR_PARAM_INVALID;
        rknn_input_output_num *io = info;
        io->n_input = m->n_inputs;
        io->n_output = m->n_outputs;
        cross_validate(ctx, cmd, info, sizeof(*io));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_INPUT_ATTR: {
        if (size < sizeof(rknn_tensor_attr))
            return RKNN_ERR_PARAM_INVALID;
        rknn_tensor_attr *attr = info;
        if (attr->index >= m->n_inputs)
            return RKNN_ERR_INPUT_INVALID;
        fill_tensor_attr(&m->inputs[attr->index], attr, 0);
        cross_validate(ctx, cmd, info, sizeof(*attr));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_OUTPUT_ATTR: {
        if (size < sizeof(rknn_tensor_attr))
            return RKNN_ERR_PARAM_INVALID;
        rknn_tensor_attr *attr = info;
        if (attr->index >= m->n_outputs)
            return RKNN_ERR_OUTPUT_INVALID;
        fill_tensor_attr(&m->outputs[attr->index], attr, 0);
        cross_validate(ctx, cmd, info, sizeof(*attr));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_SDK_VERSION: {
        if (size < sizeof(rknn_sdk_version))
            return RKNN_ERR_PARAM_INVALID;
        rknn_sdk_version *ver = info;
        snprintf(ver->api_version, sizeof(ver->api_version),
                 "openrknn 0.1.0 (proxy+own)");
        snprintf(ver->drv_version, sizeof(ver->drv_version), "0.0.0");
        /* Don't cross-validate — our version string intentionally differs */
        return RKNN_SUCC;
    }

    case RKNN_QUERY_MEM_SIZE: {
        if (size < sizeof(rknn_mem_size))
            return RKNN_ERR_PARAM_INVALID;
        rknn_mem_size *ms = info;
        memset(ms, 0, sizeof(*ms));
        ms->total_weight_size = m->total_weight_size;
        ms->total_internal_size = m->total_internal_size;
        cross_validate(ctx, cmd, info, sizeof(*ms));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR:  /* == RKNN_QUERY_NATIVE_INPUT_ATTR (8) */
    {
        if (size < sizeof(rknn_tensor_attr))
            return RKNN_ERR_PARAM_INVALID;
        rknn_tensor_attr *attr = info;
        if (attr->index >= m->n_inputs)
            return RKNN_ERR_INPUT_INVALID;
        fill_tensor_attr(&m->inputs[attr->index], attr, 1);
        cross_validate(ctx, cmd, info, sizeof(*attr));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR:  /* == RKNN_QUERY_NATIVE_OUTPUT_ATTR (9) */
    {
        if (size < sizeof(rknn_tensor_attr))
            return RKNN_ERR_PARAM_INVALID;
        rknn_tensor_attr *attr = info;
        if (attr->index >= m->n_outputs)
            return RKNN_ERR_OUTPUT_INVALID;
        fill_tensor_attr(&m->outputs[attr->index], attr, 1);
        cross_validate(ctx, cmd, info, sizeof(*attr));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR: {
        if (size < sizeof(rknn_tensor_attr))
            return RKNN_ERR_PARAM_INVALID;
        rknn_tensor_attr *attr = info;
        if (attr->index >= m->n_inputs)
            return RKNN_ERR_INPUT_INVALID;
        fill_tensor_attr(&m->inputs[attr->index], attr, 0);
        cross_validate(ctx, cmd, info, sizeof(*attr));
        return RKNN_SUCC;
    }

    case RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR: {
        if (size < sizeof(rknn_tensor_attr))
            return RKNN_ERR_PARAM_INVALID;
        rknn_tensor_attr *attr = info;
        if (attr->index >= m->n_outputs)
            return RKNN_ERR_OUTPUT_INVALID;
        fill_tensor_attr(&m->outputs[attr->index], attr, 0);
        cross_validate(ctx, cmd, info, sizeof(*attr));
        return RKNN_SUCC;
    }

    default:
        /* Proxy remaining commands (PERF_DETAIL, PERF_RUN, CUSTOM_STRING, etc.) */
        if (ctx->real_ctx) {
            struct orknn_proxy *proxy = orknn_proxy_get();
            if (proxy) {
                int ret = proxy->rknn_query(ctx->real_ctx, cmd, info, size);
                orknn_log(2, "query cmd=%d: proxied -> %d", cmd, ret);
                return ret;
            }
        }
        orknn_log(0, "orknn_own_query: unsupported cmd=%d", cmd);
        return RKNN_ERR_FAIL;
    }
}
