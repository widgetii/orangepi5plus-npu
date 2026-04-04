/*
 * openrknn — RKNN API reimplementation from scratch
 * Internal header: context, proxy table, types
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef OPENRKNN_H
#define OPENRKNN_H

#include "rknn_api.h"
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>

#define ORKNN_MAGIC 0x4F524B4E4E415049ULL /* "ORKNNAPI" */

/* ======================================================================
 * Proxy function table — loaded from real librknnrt.so via dlopen
 * ====================================================================== */

struct orknn_proxy {
    void *handle;
    int (*rknn_init)(rknn_context *, void *, uint32_t, uint32_t, rknn_init_extend *);
    int (*rknn_destroy)(rknn_context);
    int (*rknn_query)(rknn_context, rknn_query_cmd, void *, uint32_t);
    int (*rknn_inputs_set)(rknn_context, uint32_t, rknn_input[]);
    int (*rknn_run)(rknn_context, rknn_run_extend *);
    int (*rknn_outputs_get)(rknn_context, uint32_t, rknn_output[], rknn_output_extend *);
    int (*rknn_outputs_release)(rknn_context, uint32_t, rknn_output[]);
    int (*rknn_set_core_mask)(rknn_context, rknn_core_mask);
    int (*rknn_set_batch_core_num)(rknn_context, int);
    rknn_tensor_mem *(*rknn_create_mem)(rknn_context, uint32_t);
    rknn_tensor_mem *(*rknn_create_mem2)(rknn_context, uint64_t, uint64_t);
    int (*rknn_destroy_mem)(rknn_context, rknn_tensor_mem *);
    int (*rknn_set_io_mem)(rknn_context, rknn_tensor_mem *, rknn_tensor_attr *);
    int (*rknn_set_weight_mem)(rknn_context, rknn_tensor_mem *);
    int (*rknn_set_internal_mem)(rknn_context, rknn_tensor_mem *);
    int (*rknn_mem_sync)(rknn_context, rknn_tensor_mem *, rknn_mem_sync_mode);
    int (*rknn_wait)(rknn_context, rknn_run_extend *);
    rknn_tensor_mem *(*rknn_create_mem_from_fd)(rknn_context, int32_t, void *, uint32_t, int32_t);
    rknn_tensor_mem *(*rknn_create_mem_from_phys)(rknn_context, uint64_t, void *, uint32_t);
    rknn_tensor_mem *(*rknn_create_mem_from_mb_blk)(rknn_context, void *, int32_t);
    int (*rknn_dup_context)(rknn_context *, rknn_context *);
    int (*rknn_set_input_shape)(rknn_context, rknn_tensor_attr *);
    int (*rknn_set_input_shapes)(rknn_context, uint32_t, rknn_tensor_attr[]);
    int (*rknn_register_custom_ops)(rknn_context, void *, uint32_t);
    int (*rknn_custom_op_get_op_attr)(rknn_context, const char *, void *);
};

/* ======================================================================
 * Bitmask: which functions use our own impl vs proxy
 * ====================================================================== */

enum orknn_own_flags {
    ORKNN_OWN_INIT       = 1 << 0,
    ORKNN_OWN_QUERY      = 1 << 1,
    ORKNN_OWN_INPUTS_SET = 1 << 2,
    ORKNN_OWN_RUN        = 1 << 3,
    ORKNN_OWN_OUTPUTS    = 1 << 4,
    ORKNN_OWN_MEMORY     = 1 << 5,
};

/* ======================================================================
 * Tensor metadata (from .rknn FlatBuffer)
 * ====================================================================== */

struct orknn_tensor_info {
    char name[RKNN_MAX_NAME_LEN];
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
    rknn_tensor_qnt_type qnt_type;
    float scale;
    int32_t zp;
    uint32_t w_stride;
    uint32_t size_with_stride;
    /* NC1HWC2 native layout */
    uint32_t native_n_dims;
    uint32_t native_dims[RKNN_MAX_DIMS];
    uint32_t native_size;
};

/* ======================================================================
 * RKNPU DRM buffer object
 * ====================================================================== */

struct orknn_bo {
    uint32_t handle;
    uint64_t obj_addr;
    uint64_t dma_addr;
    void    *map;
    uint32_t size;
};

/* ======================================================================
 * Submit segment (from .rknn model)
 * ====================================================================== */

struct orknn_segment {
    uint32_t flags;
    uint32_t sc_start;
    uint32_t sc_count;
    uint32_t task_number;
};

/* ======================================================================
 * Parsed .rknn model
 * ====================================================================== */

struct orknn_model {
    uint8_t  *file_data;
    uint32_t  file_size;
    uint64_t  version;
    char     *model_name;
    char     *target_platform;
    uint32_t  n_inputs;
    uint32_t  n_outputs;
    struct orknn_tensor_info *inputs;
    struct orknn_tensor_info *outputs;
    /* Pre-compiled NPU data */
    uint8_t  *wt_data;
    uint32_t  wt_size;
    uint8_t  *regcmd_data;
    uint32_t  regcmd_size;
    uint8_t  *task_data;
    uint32_t  task_count;
    uint32_t  task_data_size;
    struct orknn_segment *segments;
    uint32_t  segment_count;
    uint32_t  total_weight_size;
    uint32_t  total_internal_size;
};

/* ======================================================================
 * Internal context — rknn_context handle points here
 * ====================================================================== */

struct orknn_context {
    uint64_t magic;
    /* Proxy */
    rknn_context real_ctx;
    uint32_t own_flags;
    /* Device */
    int npu_fd;
    /* Model */
    struct orknn_model model;
    /* DMA buffers */
    struct orknn_bo task_bo;
    struct orknn_bo weight_bo;
    struct orknn_bo activation_bo;
    struct orknn_bo *input_bos;
    struct orknn_bo *output_bos;
    /* Execution state */
    rknn_core_mask core_mask;
    int64_t hw_elapse_time;
    uint64_t frame_id;
    /* Logging */
    int log_level;
};

/* ======================================================================
 * Proxy management
 * ====================================================================== */

int orknn_proxy_init(void);
struct orknn_proxy *orknn_proxy_get(void);

/* ======================================================================
 * Logging
 * ====================================================================== */

void orknn_log(int level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* Global log level (set from ORKNN_LOG_LEVEL env) */
extern int g_orknn_log_level;

/* ======================================================================
 * FlatBuffer reader (openrknn_flatbuf.c)
 * ====================================================================== */

uint16_t orknn_fb_u16(const uint8_t *b, uint32_t p);
uint32_t orknn_fb_u32(const uint8_t *b, uint32_t p);
uint64_t orknn_fb_u64(const uint8_t *b, uint32_t p);
int32_t  orknn_fb_i32(const uint8_t *b, uint32_t p);
uint32_t orknn_fb_follow(const uint8_t *b, uint32_t p);
uint32_t orknn_fb_field(const uint8_t *b, uint32_t table, int field);
uint32_t orknn_fb_string(const uint8_t *b, uint32_t fpos,
                         char *out_str, uint32_t max_len);
uint32_t orknn_fb_vec_len(const uint8_t *b, uint32_t fpos);
uint32_t orknn_fb_vec_at(const uint8_t *b, uint32_t fpos, uint32_t index);
const uint8_t *orknn_fb_bytes(const uint8_t *b, uint32_t fpos, uint32_t *len);
uint8_t  orknn_fb_byte(const uint8_t *b, uint32_t fpos);

/* ======================================================================
 * DRM interface (openrknn_drm.c)
 * ====================================================================== */

int  orknn_drm_open(void);
int  orknn_bo_create(int fd, uint32_t size, struct orknn_bo *bo);
void orknn_bo_destroy(int fd, struct orknn_bo *bo);
int  orknn_bo_sync_to_device(int fd, struct orknn_bo *bo);
int  orknn_bo_sync_from_device(int fd, struct orknn_bo *bo);
int  orknn_npu_submit(int fd, struct orknn_bo *task_bo,
                      struct orknn_segment *seg);

/* ======================================================================
 * Own implementations (phases 2-6)
 * ====================================================================== */

/* Phase 2: model parsing */
int orknn_own_init(struct orknn_context *ctx, void *model, uint32_t size,
                   uint32_t flag, rknn_init_extend *extend);
int orknn_own_query(struct orknn_context *ctx, rknn_query_cmd cmd,
                    void *info, uint32_t size);

/* Phase 3: memory */
rknn_tensor_mem *orknn_own_create_mem(struct orknn_context *ctx, uint32_t size);
int orknn_own_destroy_mem(struct orknn_context *ctx, rknn_tensor_mem *mem);

/* Phase 3: allocate all model BOs during init */
int orknn_alloc_model_bos(struct orknn_context *ctx);

/* Phase 4: input */
int orknn_own_inputs_set(struct orknn_context *ctx, uint32_t n_inputs,
                         rknn_input inputs[]);

/* Phase 5: execution */
int orknn_own_run(struct orknn_context *ctx, rknn_run_extend *extend);

/* Phase 6: output */
int orknn_own_outputs_get(struct orknn_context *ctx, uint32_t n_outputs,
                          rknn_output outputs[], rknn_output_extend *extend);

#endif /* OPENRKNN_H */
