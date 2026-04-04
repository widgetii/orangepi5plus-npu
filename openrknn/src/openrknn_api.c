/*
 * openrknn — Exported API symbols with proxy/own dispatch
 *
 * Every function exported by librknn_api.so is implemented here.
 * Each dispatches to either the proxy (real librknnrt.so) or our own
 * implementation based on the ORKNN_OWN environment variable.
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ======================================================================
 * Parse ORKNN_OWN env var to determine which functions we implement
 * ====================================================================== */

static uint32_t parse_own_flags(void)
{
    const char *env = getenv("ORKNN_OWN");
    if (!env)
        return 0; /* all proxied by default */

    uint32_t flags = 0;
    if (strstr(env, "init"))    flags |= ORKNN_OWN_INIT;
    if (strstr(env, "query"))   flags |= ORKNN_OWN_QUERY;
    if (strstr(env, "input"))   flags |= ORKNN_OWN_INPUTS_SET;
    if (strstr(env, "run"))     flags |= ORKNN_OWN_RUN;
    if (strstr(env, "output"))  flags |= ORKNN_OWN_OUTPUTS;
    if (strstr(env, "memory"))  flags |= ORKNN_OWN_MEMORY;
    if (strstr(env, "all"))     flags = 0xFFFFFFFF;
    return flags;
}

/* ======================================================================
 * Context management
 * ====================================================================== */

static struct orknn_context *ctx_from_handle(rknn_context handle)
{
    struct orknn_context *ctx = (struct orknn_context *)(uintptr_t)handle;
    if (!ctx || ctx->magic != ORKNN_MAGIC) {
        orknn_log(0, "ERROR: invalid context handle 0x%lx", (unsigned long)handle);
        return NULL;
    }
    return ctx;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ======================================================================
 * rknn_init
 * ====================================================================== */

int rknn_init(rknn_context *context, void *model, uint32_t size,
              uint32_t flag, rknn_init_extend *extend)
{
    orknn_proxy_init();
    double t0 = now_ms();

    struct orknn_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return RKNN_ERR_MALLOC_FAIL;

    ctx->magic = ORKNN_MAGIC;
    ctx->own_flags = parse_own_flags();
    ctx->npu_fd = -1;
    ctx->core_mask = RKNN_NPU_CORE_AUTO;

    const char *log_env = getenv("ORKNN_LOG_LEVEL");
    ctx->log_level = log_env ? atoi(log_env) : 1;

    int ret;

    if (ctx->own_flags & ORKNN_OWN_INIT) {
        ret = orknn_own_init(ctx, model, size, flag, extend);
        orknn_log(1, "rknn_init(own): model=%p size=%u flags=0x%x -> %d (%.1fms)",
                  model, size, flag, ret, now_ms() - t0);
    } else {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (!proxy) {
            free(ctx);
            return RKNN_ERR_FAIL;
        }
        ret = proxy->rknn_init(&ctx->real_ctx, model, size, flag, extend);
        orknn_log(1, "rknn_init(proxy): model=%p size=%u flags=0x%x -> %d "
                  "(real_ctx=0x%lx, %.1fms)",
                  model, size, flag, ret,
                  (unsigned long)ctx->real_ctx, now_ms() - t0);
    }

    if (ret != RKNN_SUCC) {
        free(ctx);
        return ret;
    }

    *context = (rknn_context)(uintptr_t)ctx;
    return RKNN_SUCC;
}

/* ======================================================================
 * rknn_destroy
 * ====================================================================== */

int rknn_destroy(rknn_context context)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    int ret = RKNN_SUCC;

    if (ctx->real_ctx) {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (proxy)
            ret = proxy->rknn_destroy(ctx->real_ctx);
    }

    /* Free model data */
    free(ctx->model.file_data);
    free(ctx->model.model_name);
    free(ctx->model.target_platform);
    free(ctx->model.inputs);
    free(ctx->model.outputs);
    free(ctx->model.wt_data);
    free(ctx->model.regcmd_data);
    free(ctx->model.task_data);
    free(ctx->model.segments);
    free(ctx->input_bos);
    free(ctx->output_bos);

    orknn_log(1, "rknn_destroy: ctx=%p -> %d", ctx, ret);
    ctx->magic = 0;
    free(ctx);
    return ret;
}

/* ======================================================================
 * rknn_query
 * ====================================================================== */

int rknn_query(rknn_context context, rknn_query_cmd cmd, void *info, uint32_t size)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    int ret;

    if (ctx->own_flags & ORKNN_OWN_QUERY) {
        ret = orknn_own_query(ctx, cmd, info, size);
        orknn_log(2, "rknn_query(own): cmd=%d size=%u -> %d", cmd, size, ret);
    } else {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (!proxy || !ctx->real_ctx) return RKNN_ERR_CTX_INVALID;
        ret = proxy->rknn_query(ctx->real_ctx, cmd, info, size);
        orknn_log(2, "rknn_query(proxy): cmd=%d size=%u -> %d", cmd, size, ret);
    }

    return ret;
}

/* ======================================================================
 * rknn_inputs_set
 * ====================================================================== */

int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[])
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    int ret;

    if (ctx->own_flags & ORKNN_OWN_INPUTS_SET) {
        ret = orknn_own_inputs_set(ctx, n_inputs, inputs);
        orknn_log(2, "rknn_inputs_set(own): n=%u -> %d", n_inputs, ret);
    } else {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (!proxy || !ctx->real_ctx) return RKNN_ERR_CTX_INVALID;
        ret = proxy->rknn_inputs_set(ctx->real_ctx, n_inputs, inputs);
        orknn_log(2, "rknn_inputs_set(proxy): n=%u -> %d", n_inputs, ret);
    }

    return ret;
}

/* ======================================================================
 * rknn_run
 * ====================================================================== */

int rknn_run(rknn_context context, rknn_run_extend *extend)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    double t0 = now_ms();
    int ret;

    if (ctx->own_flags & ORKNN_OWN_RUN) {
        ret = orknn_own_run(ctx, extend);
        orknn_log(2, "rknn_run(own): -> %d (%.2fms)", ret, now_ms() - t0);
    } else {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (!proxy || !ctx->real_ctx) return RKNN_ERR_CTX_INVALID;
        ret = proxy->rknn_run(ctx->real_ctx, extend);
        orknn_log(2, "rknn_run(proxy): -> %d (%.2fms)", ret, now_ms() - t0);
    }

    ctx->frame_id++;
    return ret;
}

/* ======================================================================
 * rknn_wait
 * ====================================================================== */

int rknn_wait(rknn_context context, rknn_run_extend *extend)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_wait || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_wait(ctx->real_ctx, extend);
    orknn_log(2, "rknn_wait(proxy): -> %d", ret);
    return ret;
}

/* ======================================================================
 * rknn_outputs_get
 * ====================================================================== */

int rknn_outputs_get(rknn_context context, uint32_t n_outputs,
                     rknn_output outputs[], rknn_output_extend *extend)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    int ret;

    if (ctx->own_flags & ORKNN_OWN_OUTPUTS) {
        ret = orknn_own_outputs_get(ctx, n_outputs, outputs, extend);
        orknn_log(2, "rknn_outputs_get(own): n=%u -> %d", n_outputs, ret);
    } else {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (!proxy || !ctx->real_ctx) return RKNN_ERR_CTX_INVALID;
        ret = proxy->rknn_outputs_get(ctx->real_ctx, n_outputs, outputs, extend);
        orknn_log(2, "rknn_outputs_get(proxy): n=%u -> %d", n_outputs, ret);
    }

    return ret;
}

/* ======================================================================
 * rknn_outputs_release
 * ====================================================================== */

int rknn_outputs_release(rknn_context context, uint32_t n_ouputs,
                         rknn_output outputs[])
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !ctx->real_ctx) return RKNN_ERR_CTX_INVALID;

    int ret = proxy->rknn_outputs_release(ctx->real_ctx, n_ouputs, outputs);
    orknn_log(2, "rknn_outputs_release(proxy): n=%u -> %d", n_ouputs, ret);
    return ret;
}

/* ======================================================================
 * rknn_set_core_mask
 * ====================================================================== */

int rknn_set_core_mask(rknn_context context, rknn_core_mask core_mask)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    ctx->core_mask = core_mask;

    if (ctx->real_ctx) {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (proxy && proxy->rknn_set_core_mask) {
            int ret = proxy->rknn_set_core_mask(ctx->real_ctx, core_mask);
            orknn_log(2, "rknn_set_core_mask(proxy): mask=0x%x -> %d", core_mask, ret);
            return ret;
        }
    }

    orknn_log(2, "rknn_set_core_mask(own): mask=0x%x -> 0", core_mask);
    return RKNN_SUCC;
}

/* ======================================================================
 * rknn_set_batch_core_num
 * ====================================================================== */

int rknn_set_batch_core_num(rknn_context context, int core_num)
{
    struct orknn_context *ctx = ctx_from_handle(context);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    if (ctx->real_ctx) {
        struct orknn_proxy *proxy = orknn_proxy_get();
        if (proxy && proxy->rknn_set_batch_core_num) {
            int ret = proxy->rknn_set_batch_core_num(ctx->real_ctx, core_num);
            orknn_log(2, "rknn_set_batch_core_num(proxy): n=%d -> %d", core_num, ret);
            return ret;
        }
    }

    orknn_log(2, "rknn_set_batch_core_num: n=%d (ignored)", core_num);
    return RKNN_SUCC;
}

/* ======================================================================
 * Memory management — proxy passthrough
 * ====================================================================== */

rknn_tensor_mem *rknn_create_mem(rknn_context ctx_handle, uint32_t size)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return NULL;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_create_mem || !ctx->real_ctx) return NULL;

    rknn_tensor_mem *mem = proxy->rknn_create_mem(ctx->real_ctx, size);
    orknn_log(2, "rknn_create_mem(proxy): size=%u -> %p", size, (void *)mem);
    return mem;
}

rknn_tensor_mem *rknn_create_mem2(rknn_context ctx_handle, uint64_t size,
                                  uint64_t alloc_flags)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return NULL;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_create_mem2 || !ctx->real_ctx) return NULL;

    rknn_tensor_mem *mem = proxy->rknn_create_mem2(ctx->real_ctx, size, alloc_flags);
    orknn_log(2, "rknn_create_mem2(proxy): size=%lu flags=0x%lx -> %p",
              (unsigned long)size, (unsigned long)alloc_flags, (void *)mem);
    return mem;
}

rknn_tensor_mem *rknn_create_mem_from_fd(rknn_context ctx_handle, int32_t fd,
                                         void *virt_addr, uint32_t size,
                                         int32_t offset)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return NULL;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_create_mem_from_fd || !ctx->real_ctx) return NULL;

    rknn_tensor_mem *mem = proxy->rknn_create_mem_from_fd(
        ctx->real_ctx, fd, virt_addr, size, offset);
    orknn_log(2, "rknn_create_mem_from_fd(proxy): fd=%d size=%u -> %p",
              fd, size, (void *)mem);
    return mem;
}

rknn_tensor_mem *rknn_create_mem_from_phys(rknn_context ctx_handle,
                                           uint64_t phys_addr, void *virt_addr,
                                           uint32_t size)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return NULL;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_create_mem_from_phys || !ctx->real_ctx) return NULL;

    rknn_tensor_mem *mem = proxy->rknn_create_mem_from_phys(
        ctx->real_ctx, phys_addr, virt_addr, size);
    orknn_log(2, "rknn_create_mem_from_phys(proxy): phys=0x%lx size=%u -> %p",
              (unsigned long)phys_addr, size, (void *)mem);
    return mem;
}

rknn_tensor_mem *rknn_create_mem_from_mb_blk(rknn_context ctx_handle,
                                             void *mb_blk, int32_t offset)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return NULL;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_create_mem_from_mb_blk || !ctx->real_ctx) return NULL;

    rknn_tensor_mem *mem = proxy->rknn_create_mem_from_mb_blk(
        ctx->real_ctx, mb_blk, offset);
    orknn_log(2, "rknn_create_mem_from_mb_blk(proxy): -> %p", (void *)mem);
    return mem;
}

int rknn_destroy_mem(rknn_context ctx_handle, rknn_tensor_mem *mem)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_destroy_mem || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_destroy_mem(ctx->real_ctx, mem);
    orknn_log(2, "rknn_destroy_mem(proxy): mem=%p -> %d", (void *)mem, ret);
    return ret;
}

int rknn_set_io_mem(rknn_context ctx_handle, rknn_tensor_mem *mem,
                    rknn_tensor_attr *attr)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_set_io_mem || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_set_io_mem(ctx->real_ctx, mem, attr);
    orknn_log(2, "rknn_set_io_mem(proxy): mem=%p idx=%u -> %d",
              (void *)mem, attr ? attr->index : 0, ret);
    return ret;
}

int rknn_set_weight_mem(rknn_context ctx_handle, rknn_tensor_mem *mem)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_set_weight_mem || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_set_weight_mem(ctx->real_ctx, mem);
    orknn_log(2, "rknn_set_weight_mem(proxy): mem=%p -> %d", (void *)mem, ret);
    return ret;
}

int rknn_set_internal_mem(rknn_context ctx_handle, rknn_tensor_mem *mem)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_set_internal_mem || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_set_internal_mem(ctx->real_ctx, mem);
    orknn_log(2, "rknn_set_internal_mem(proxy): mem=%p -> %d", (void *)mem, ret);
    return ret;
}

int rknn_mem_sync(rknn_context ctx_handle, rknn_tensor_mem *mem,
                  rknn_mem_sync_mode mode)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_mem_sync || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_mem_sync(ctx->real_ctx, mem, mode);
    orknn_log(3, "rknn_mem_sync(proxy): mode=%d -> %d", mode, ret);
    return ret;
}

/* ======================================================================
 * Context duplication
 * ====================================================================== */

int rknn_dup_context(rknn_context *context_in, rknn_context *context_out)
{
    if (!context_in || !context_out) return RKNN_ERR_PARAM_INVALID;

    struct orknn_context *ctx_in = ctx_from_handle(*context_in);
    if (!ctx_in) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_dup_context || !ctx_in->real_ctx)
        return RKNN_ERR_FAIL;

    /* Create new wrapper context */
    struct orknn_context *ctx_out = calloc(1, sizeof(*ctx_out));
    if (!ctx_out) return RKNN_ERR_MALLOC_FAIL;

    ctx_out->magic = ORKNN_MAGIC;
    ctx_out->own_flags = ctx_in->own_flags;
    ctx_out->core_mask = ctx_in->core_mask;
    ctx_out->log_level = ctx_in->log_level;

    rknn_context real_in = ctx_in->real_ctx;
    int ret = proxy->rknn_dup_context(&real_in, &ctx_out->real_ctx);
    if (ret != RKNN_SUCC) {
        free(ctx_out);
        return ret;
    }

    *context_out = (rknn_context)(uintptr_t)ctx_out;
    orknn_log(1, "rknn_dup_context(proxy): -> %d", ret);
    return ret;
}

/* ======================================================================
 * Input shape (dynamic shape models)
 * ====================================================================== */

int rknn_set_input_shape(rknn_context ctx_handle, rknn_tensor_attr *attr)
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_set_input_shape || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_set_input_shape(ctx->real_ctx, attr);
    orknn_log(2, "rknn_set_input_shape(proxy): -> %d", ret);
    return ret;
}

int rknn_set_input_shapes(rknn_context ctx_handle, uint32_t n_inputs,
                          rknn_tensor_attr attr[])
{
    struct orknn_context *ctx = ctx_from_handle(ctx_handle);
    if (!ctx) return RKNN_ERR_CTX_INVALID;

    struct orknn_proxy *proxy = orknn_proxy_get();
    if (!proxy || !proxy->rknn_set_input_shapes || !ctx->real_ctx)
        return RKNN_ERR_FAIL;

    int ret = proxy->rknn_set_input_shapes(ctx->real_ctx, n_inputs, attr);
    orknn_log(2, "rknn_set_input_shapes(proxy): n=%u -> %d", n_inputs, ret);
    return ret;
}
