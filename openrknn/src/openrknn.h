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

#define ORKNN_MAX_CYCLES 4

struct orknn_segment {
    uint32_t flags;
    uint32_t sc_start;
    uint32_t sc_count;
    uint32_t task_number;
    /* Per-cycle task BO snapshots. The proxy patches task BO data
     * differently between iterations (MBv1 iter 1 writes output BO
     * while warmup doesn't). cycle[0] = warmup/first iter, cycle[1] =
     * second iter, etc. We pick cycle[min(run_count, n-1)] on each
     * orknn_own_run call. n_cycles=0 means no snapshots available. */
    uint8_t *task_bo_data[ORKNN_MAX_CYCLES];
    uint32_t task_bo_size[ORKNN_MAX_CYCLES];
    uint32_t n_cycles;
};

/* ======================================================================
 * Per-operator metadata extracted from the .rknn FlatBuffer operator graph
 * (subgraph field 1). Populated by parse_fb_operators() in openrknn_model.c.
 *
 * For Conv/ConvRelu/ConvClip/ConvExSwish/ConvSwish etc.:
 *   input_tensors[0] = feature data tensor index
 *   input_tensors[1] = weight tensor index
 *   input_tensors[2] = bias tensor index
 *   output_tensors[0] = output tensor index
 *
 * Non-conv ops (BatchNormalization, InputOperator, etc.) have different
 * tensor layouts; `type` is the authoritative string to dispatch on.
 * ====================================================================== */

#define ORKNN_MAX_OP_INPUTS  8
#define ORKNN_MAX_OP_OUTPUTS 4
#define ORKNN_OP_TYPE_LEN    32

struct orknn_op_info {
    char     type[ORKNN_OP_TYPE_LEN];
    uint32_t input_count;
    uint32_t input_tensors[ORKNN_MAX_OP_INPUTS];
    uint32_t output_count;
    uint32_t output_tensors[ORKNN_MAX_OP_OUTPUTS];
    /* Implicit weight/bias tensor indices. Some ops (e.g. DeepLabv3's
     * Resize which lowers to a 1x1 requantization conv) reference
     * compiler-generated weight and bias blobs that are NOT listed in
     * the op's explicit input_tensors[]. The compiler leaves them as
     * top-level tensors whose names start with `{input[0].name}_` and
     * end in `_weight_*` or `_bias_*`. We resolve those by scanning
     * the tensor vector for a name-prefix match during
     * parse_fb_operators and store the tensor indices here.
     *
     * Both fields default to UINT32_MAX (no match found). */
    uint32_t implicit_wt_tidx;
    uint32_t implicit_bs_tidx;

    /* Per-stage weight tensor indices for exSoftmax13 ops. The softmax
     * lowering emits three em=0x0d tasks (ReduceMax -> rescale -> ReduceSum)
     * and each reads CNA_WT from a distinct compile-time blob. Two of
     * the blobs (_ReduceMax_output_weight and _reducesum_output_weight)
     * are top-level tensors that aren't in input_tensors[] and we
     * resolve them via name-suffix matching against the output tensor's
     * name. The rescale blob IS in input_tensors[2] so it doesn't need
     * its own slot.
     *
     * Both fields default to UINT32_MAX (no match found). */
    uint32_t softmax_rmax_tidx;
    uint32_t softmax_rsum_tidx;
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
    /* Per-operator metadata (phase 3). NULL until parse_fb_operators()
     * runs. op_count is the length of ops[]. */
    struct orknn_op_info *ops;
    uint32_t  op_count;
    /* Phase 4b: tensor memory plan — per-tensor byte offset in the
     * activation BO, parsed from FB tensor.f[13]. Indexed by tensor
     * index as stored in ops[i].input_tensors / output_tensors. Weight
     * and constant tensors have offset 0 here and are not referenced
     * in activation BO reads. NULL until parse_fb_tensor_offsets runs. */
    uint32_t *tensor_offsets;
    /* Per-tensor weight-data blob index (FB tensor.f[18]). Non-zero
     * only for tensors whose data is stored in the weight BO.
     * 0xFFFFFFFF = no weight blob index (activation/intermediate tensor).
     * Resolved to a weight-BO byte offset via wt_blob_offsets[]. */
    uint32_t *tensor_weight_blob;
    uint32_t  tensor_count;
    /* Per-weight_data-index offset into BO[1]. Built alongside the
     * tensor memory plan by iterating the FB weight_data vector in the
     * same order extract_npu_data does, with 64-byte alignment between
     * non-empty entries. wt_blob_count is the full weight_data vector
     * length (not compacted), so a lookup tensor_weight_blob[i] indexes
     * directly into wt_blob_offsets[]. */
    uint32_t *wt_blob_offsets;
    uint32_t  wt_blob_count;
    /* Subgraph output tensor indices — the tensors whose data lands in
     * output BOs rather than the activation BO. Used to decide whether
     * a REFORMAT task's DPU_DST_BASE should point at `out_base` or at
     * `act_base + tensor_off`. Parallel to tensor_offsets[], indexed
     * by tensor index: 1 if this tensor is a subgraph output. */
    uint8_t  *tensor_is_sg_output;
    /* Per-subgraph-output-index: which tensor each output maps to and
     * its corresponding output BO index. m->n_outputs entries. */
    uint32_t *sg_output_tensor_idx;
    /* Phase 3b: pre-processing config parsed from the header `attrs`
     * Python-dict string. Used to compute CVT register values for the
     * first conv op that reads raw user input. */
    char     input_attr_dtype[16];          /* "uint8", "int8", "float32" */
    float    input_attr_mean[4];            /* up to 4 channels */
    float    input_attr_std[4];
    int      input_attr_valid;              /* 1 = attrs parsed OK */
    uint32_t input_consuming_op_idx;        /* op_idx whose input[0] == sg input */
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
    uint32_t run_count;
    /* Output discovery: offsets within activation BO where the final
     * output tensors live. Discovered from proxy's rknn_outputs_get
     * during init. */
    uint32_t act_output_offsets[16];
    uint8_t  act_output_valid[16];
    /* Native layout of data at the output offset:
     * 0 = NC1HWC2 ([N, C1, H, W, C2])
     * 1 = HWC1C2  ([N, H, W, C1*C2]) — padded NHWC
     * 3 = HBWCH16 ([H/16, W, padC, 16]) — YOLO output BO layout:
     *     byte = (h/16)*W*padC*16 + w*padC*16 + c*16 + (h%16)
     */
    uint8_t  act_output_layout[16];
    /* User-visible byte order returned by the proxy:
     * 0 = NHWC (c-minor: dst[h*W*C + w*C + c])
     * 1 = NCHW (c-major: dst[c*H*W + h*W + w])
     * Detected during sig search by trying both interpretations. */
    uint8_t  act_output_src_order[16];
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
int  orknn_bo_create_flags(int fd, uint32_t size, uint32_t flags, struct orknn_bo *bo);
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
