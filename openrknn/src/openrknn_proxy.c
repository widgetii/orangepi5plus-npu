/*
 * openrknn — Proxy layer: dlopen real librknnrt.so, forward all calls
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct orknn_proxy g_proxy;
static int g_proxy_initialized = 0;
int g_orknn_log_level = 1;

/* ======================================================================
 * Logging
 * ====================================================================== */

void orknn_log(int level, const char *fmt, ...)
{
    if (level > g_orknn_log_level)
        return;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double t = ts.tv_sec + ts.tv_nsec / 1e9;

    fprintf(stderr, "[openrknn %10.3f] ", t);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fprintf(stderr, "\n");
}

/* ======================================================================
 * Proxy initialization — dlopen + dlsym all symbols
 * ====================================================================== */

#define LOAD_SYM(name) do { \
    g_proxy.name = dlsym(g_proxy.handle, #name); \
    if (!g_proxy.name) \
        orknn_log(1, "WARN: dlsym(%s) failed: %s", #name, dlerror()); \
} while (0)

#define LOAD_SYM_REQUIRED(name) do { \
    g_proxy.name = dlsym(g_proxy.handle, #name); \
    if (!g_proxy.name) { \
        orknn_log(0, "ERROR: required symbol %s not found: %s", #name, dlerror()); \
        dlclose(g_proxy.handle); \
        g_proxy.handle = NULL; \
        return -1; \
    } \
} while (0)

int orknn_proxy_init(void)
{
    if (g_proxy_initialized)
        return g_proxy.handle ? 0 : -1;
    g_proxy_initialized = 1;

    memset(&g_proxy, 0, sizeof(g_proxy));

    /* Read configuration from environment */
    const char *log_env = getenv("ORKNN_LOG_LEVEL");
    if (log_env)
        g_orknn_log_level = atoi(log_env);

    const char *lib_path = getenv("ORKNN_REAL_LIB");
    if (!lib_path)
        lib_path = "librknnrt.so";

    orknn_log(1, "proxy: loading real library from %s", lib_path);

    g_proxy.handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_proxy.handle) {
        orknn_log(0, "proxy: dlopen(%s) failed: %s", lib_path, dlerror());
        orknn_log(0, "proxy: running without proxy (own implementations only)");
        return -1;
    }

    /* Required symbols — proxy cannot function without these */
    LOAD_SYM_REQUIRED(rknn_init);
    LOAD_SYM_REQUIRED(rknn_destroy);
    LOAD_SYM_REQUIRED(rknn_query);
    LOAD_SYM_REQUIRED(rknn_inputs_set);
    LOAD_SYM_REQUIRED(rknn_run);
    LOAD_SYM_REQUIRED(rknn_outputs_get);
    LOAD_SYM_REQUIRED(rknn_outputs_release);

    /* Optional symbols — warn but continue if missing */
    LOAD_SYM(rknn_set_core_mask);
    LOAD_SYM(rknn_set_batch_core_num);
    LOAD_SYM(rknn_create_mem);
    LOAD_SYM(rknn_create_mem2);
    LOAD_SYM(rknn_destroy_mem);
    LOAD_SYM(rknn_set_io_mem);
    LOAD_SYM(rknn_set_weight_mem);
    LOAD_SYM(rknn_set_internal_mem);
    LOAD_SYM(rknn_mem_sync);
    LOAD_SYM(rknn_wait);
    LOAD_SYM(rknn_create_mem_from_fd);
    LOAD_SYM(rknn_create_mem_from_phys);
    LOAD_SYM(rknn_create_mem_from_mb_blk);
    LOAD_SYM(rknn_dup_context);
    LOAD_SYM(rknn_set_input_shape);
    LOAD_SYM(rknn_set_input_shapes);
    LOAD_SYM(rknn_register_custom_ops);
    LOAD_SYM(rknn_custom_op_get_op_attr);

    orknn_log(1, "proxy: loaded %s successfully", lib_path);
    return 0;
}

struct orknn_proxy *orknn_proxy_get(void)
{
    if (!g_proxy_initialized)
        orknn_proxy_init();
    return g_proxy.handle ? &g_proxy : NULL;
}
