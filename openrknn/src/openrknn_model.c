/*
 * openrknn — .rknn model parser
 *
 * Parses RKNN header, JSON config, and FlatBuffer to extract:
 * - Tensor metadata (name, shape, dtype, quantization)
 * - Pre-compiled NPU data (weights, regcmd, task descriptors)
 * - Submit segments
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ======================================================================
 * JSON config mini-parser
 *
 * The .rknn JSON config contains tensor metadata in this format:
 *   "input_num": 1, "output_num": 1,
 *   "input_tensors": [{"url":"input", "dtype":"uint8", ...}],
 *   "output_tensors": [{"url":"output", "dtype":"int8", ...}],
 *   "target_platform": "rk3588"
 *
 * We parse this with simple strstr/sscanf — no full JSON parser needed.
 * ====================================================================== */

/* Find "key" used as a JSON key (followed by :) and return pointer after : */
static const char *json_find_key(const char *json, const char *key)
{
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        const char *after = p + strlen(pat);
        while (*after == ' ' || *after == '\t' || *after == '\n') after++;
        if (*after == ':') return after + 1;
        p = after;
    }
    return NULL;
}

/* Find "key": and extract the integer value after it */
static int json_int(const char *json, const char *key, int *out)
{
    const char *p = json_find_key(json, key);
    if (!p) return -1;
    while (*p == ' ' || *p == '\t') p++;
    *out = atoi(p);
    return 0;
}

/* Find "key": "value" and extract the string */
static int json_str(const char *json, const char *key,
                    char *out, int max_len)
{
    const char *p = json_find_key(json, key);
    if (!p) return -1;
    while (*p && *p != '"') p++;
    if (*p != '"') return -1;
    p++; /* skip opening quote */
    int i = 0;
    while (*p && *p != '"' && i < max_len - 1)
        out[i++] = *p++;
    out[i] = '\0';
    return i;
}

/* Find "key": [val1, val2, ...] and extract float array */
static int json_float_array(const char *json, const char *key,
                            float *out, int max_count)
{
    const char *p = json_find_key(json, key);
    if (!p) return 0;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '[') return 0;
    if (!p) return 0;
    p++;
    int count = 0;
    while (*p && *p != ']' && count < max_count) {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        out[count++] = (float)strtod(p, (char **)&p);
    }
    return count;
}

/* Find "key": [val1, val2, ...] and extract int array */
static int json_int_array(const char *json, const char *key,
                          int *out, int max_count)
{
    const char *p = json_find_key(json, key);
    if (!p) return 0;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '[') return 0;
    if (!p) return 0;
    p++;
    int count = 0;
    while (*p && *p != ']' && count < max_count) {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        out[count++] = (int)strtol(p, (char **)&p, 10);
    }
    return count;
}

/* Find the Nth object {...} in an array after "key": [
 * Searches for "key" followed by : and [ to avoid matching values. */
static const char *json_array_obj(const char *json, const char *key, int idx)
{
    /* Search for "key" as a JSON key (followed by optional whitespace and :) */
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        const char *after = p + strlen(pat);
        /* Skip whitespace */
        while (*after == ' ' || *after == '\t' || *after == '\n') after++;
        if (*after == ':') { p = after; break; } /* Found as key */
        p = after; /* Not a key, keep searching */
    }
    if (!p) return NULL;
    p = strchr(p, '[');
    if (!p) return NULL;
    p++;
    for (int i = 0; i <= idx; i++) {
        p = strchr(p, '{');
        if (!p) return NULL;
        if (i < idx) {
            int depth = 1;
            p++;
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                p++;
            }
        }
    }
    return p;
}

/* Get a NUL-terminated copy of the Nth JSON object in an array.
 * Caller must free the returned string. */
static char *json_array_obj_copy(const char *json, const char *key, int idx)
{
    const char *start = json_array_obj(json, key, idx);
    if (!start || *start != '{') return NULL;
    /* Find matching } */
    int depth = 1;
    const char *p = start + 1;
    while (*p && depth > 0) {
        if (*p == '{') depth++;
        else if (*p == '}') depth--;
        p++;
    }
    size_t len = p - start;
    char *copy = malloc(len + 1);
    memcpy(copy, start, len);
    copy[len] = '\0';
    return copy;
}

/* Map dtype string to rknn_tensor_type */
static rknn_tensor_type dtype_from_string(const char *s)
{
    if (!strcmp(s, "float32") || !strcmp(s, "FLOAT32")) return RKNN_TENSOR_FLOAT32;
    if (!strcmp(s, "float16") || !strcmp(s, "FLOAT16")) return RKNN_TENSOR_FLOAT16;
    if (!strcmp(s, "int8")    || !strcmp(s, "INT8"))    return RKNN_TENSOR_INT8;
    if (!strcmp(s, "uint8")   || !strcmp(s, "UINT8"))   return RKNN_TENSOR_UINT8;
    if (!strcmp(s, "int16")   || !strcmp(s, "INT16"))    return RKNN_TENSOR_INT16;
    if (!strcmp(s, "int32")   || !strcmp(s, "INT32"))    return RKNN_TENSOR_INT32;
    if (!strcmp(s, "int64")   || !strcmp(s, "INT64"))    return RKNN_TENSOR_INT64;
    if (!strcmp(s, "bool")    || !strcmp(s, "BOOL"))     return RKNN_TENSOR_BOOL;
    if (!strcmp(s, "int4")    || !strcmp(s, "INT4"))     return RKNN_TENSOR_INT4;
    if (!strcmp(s, "bfloat16"))                          return RKNN_TENSOR_BFLOAT16;
    return RKNN_TENSOR_INT8; /* default */
}

/* Bytes per element for a tensor type */
static uint32_t dtype_size(rknn_tensor_type t)
{
    switch (t) {
    case RKNN_TENSOR_FLOAT32: case RKNN_TENSOR_INT32: case RKNN_TENSOR_UINT32:
        return 4;
    case RKNN_TENSOR_FLOAT16: case RKNN_TENSOR_INT16: case RKNN_TENSOR_UINT16:
    case RKNN_TENSOR_BFLOAT16:
        return 2;
    case RKNN_TENSOR_INT64:
        return 8;
    default:
        return 1;
    }
}

/* Map qnt_type string to enum */
static rknn_tensor_qnt_type qnt_from_string(const char *s)
{
    if (strstr(s, "affine") || strstr(s, "AFFINE"))
        return RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
    if (strstr(s, "dfp") || strstr(s, "DFP"))
        return RKNN_TENSOR_QNT_DFP;
    return RKNN_TENSOR_QNT_NONE;
}

/* Parse a single tensor's metadata from its JSON object.
 * RKNN JSON norm_tensor format:
 *   {"url":"input", "dim_num":4, "size":[1,3,224,224], "tensor_id":0,
 *    "dtype":{"qnt_method":"layer","qnt_type":"int8","vx_type":"int8"}}
 * Note: scale and zp are NOT in the JSON — they come from FlatBuffer quant tables.
 */
static void parse_tensor_json(const char *obj, struct orknn_tensor_info *ti)
{
    memset(ti, 0, sizeof(*ti));

    json_str(obj, "url", ti->name, sizeof(ti->name));

    /* dtype is nested: find the dtype sub-object */
    char dtype_str[32] = {0};
    const char *dtype_obj = strstr(obj, "\"dtype\"");
    if (dtype_obj) {
        const char *brace = strchr(dtype_obj, '{');
        if (brace) {
            /* vx_type has the actual data type name */
            json_str(brace, "vx_type", dtype_str, sizeof(dtype_str));
            if (!dtype_str[0])
                json_str(brace, "qnt_type", dtype_str, sizeof(dtype_str));
        }
    }
    ti->type = dtype_from_string(dtype_str);

    /* qnt_type from nested dtype object */
    char qnt_str[64] = {0};
    if (dtype_obj) {
        const char *brace = strchr(dtype_obj, '{');
        if (brace)
            json_str(brace, "qnt_method", qnt_str, sizeof(qnt_str));
    }
    /* "layer" → affine asymmetric, "channel" → per-channel affine */
    if (strstr(qnt_str, "layer") || strstr(qnt_str, "channel"))
        ti->qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
    else if (strstr(qnt_str, "dfp"))
        ti->qnt_type = RKNN_TENSOR_QNT_DFP;
    else
        ti->qnt_type = RKNN_TENSOR_QNT_NONE;

    /* Shape: JSON "size" is in NCHW order [N,C,H,W].
     * We need to determine the user-facing format and reorder dims. */
    int shape[RKNN_MAX_DIMS];
    int dim_num = 0;
    json_int(obj, "dim_num", &dim_num);
    int n_read = json_int_array(obj, "size", shape, RKNN_MAX_DIMS);
    ti->n_dims = (dim_num > 0) ? (uint32_t)dim_num : (uint32_t)n_read;

    /* Scale/zp from JSON (may be present in some models) */
    float scales[1] = {0};
    json_float_array(obj, "scale", scales, 1);
    ti->scale = scales[0];

    int zps[1] = {0};
    json_int_array(obj, "zp", zps, 1);
    ti->zp = zps[0];

    /* JSON stores NCHW: [N,C,H,W]. The real library reports in the model's
     * native format. For 4D tensors with small C (<=channels like 3,255),
     * the model is NHWC and we need to reorder to [N,H,W,C]. */
    if (ti->n_dims == 4 && n_read >= 4) {
        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        /* Heuristic: if C <= 3 or C < H, it's NHWC (the common case for inputs).
         * For outputs, NCHW is common when C is large. The real library checks
         * the layout byte from the model. For now, check if this looks like NCHW
         * (C > W or C > H and C > 4) */
        /* The RKNN library always reports dims in NHWC order [N,H,W,C]
         * regardless of fmt. fmt indicates the data memory layout. */
        ti->dims[0] = N; ti->dims[1] = H; ti->dims[2] = W; ti->dims[3] = C;
        if (C > 4 && C > W)
            ti->fmt = RKNN_TENSOR_NCHW;
        else
            ti->fmt = RKNN_TENSOR_NHWC;
    } else {
        /* Non-4D: copy as-is */
        for (int i = 0; i < n_read && i < RKNN_MAX_DIMS; i++)
            ti->dims[i] = (uint32_t)shape[i];
        ti->fmt = RKNN_TENSOR_UNDEFINED;
    }

    ti->n_elems = 1;
    for (uint32_t i = 0; i < ti->n_dims; i++)
        ti->n_elems *= ti->dims[i];
    ti->size = ti->n_elems * dtype_size(ti->type);

    /* w_stride = width dimension for 4D tensors */
    if (ti->n_dims == 4) {
        if (ti->fmt == RKNN_TENSOR_NHWC)
            ti->w_stride = ti->dims[2]; /* W */
        else
            ti->w_stride = ti->dims[3]; /* W for NCHW */
    }

    /* size_with_stride: for now same as size */
    ti->size_with_stride = ti->size;

    /* Compute NC1HWC2 native layout */
    uint32_t c2 = 16; /* int8/uint8 on RK3588 */
    if (ti->type == RKNN_TENSOR_FLOAT16 || ti->type == RKNN_TENSOR_BFLOAT16)
        c2 = 8;

    if (ti->n_dims == 4) {
        /* dims are always in NHWC order [N,H,W,C] in our representation */
        uint32_t N = ti->dims[0], H = ti->dims[1], W = ti->dims[2], C = ti->dims[3];
        uint32_t C1 = (C + c2 - 1) / c2;
        ti->native_n_dims = 5;
        ti->native_dims[0] = N;
        ti->native_dims[1] = C1;
        ti->native_dims[2] = H;
        ti->native_dims[3] = W;
        ti->native_dims[4] = c2;
        ti->native_size = N * C1 * H * W * c2 * dtype_size(ti->type);
    } else {
        /* Non-4D: native dims = user dims (not padded).
         * But native_size IS padded to c2 alignment on last dim. */
        ti->native_n_dims = ti->n_dims;
        for (uint32_t i = 0; i < ti->n_dims; i++)
            ti->native_dims[i] = ti->dims[i];
        /* Size uses padded last dim */
        uint32_t product = 1;
        for (uint32_t i = 0; i < ti->native_n_dims; i++) {
            uint32_t d = ti->native_dims[i];
            if (i == ti->native_n_dims - 1)
                d = ((d + c2 - 1) / c2) * c2;
            product *= d;
        }
        ti->native_size = product * dtype_size(ti->type);
    }
}

/* ======================================================================
 * Weight/task extraction from FlatBuffer
 * ====================================================================== */

#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

/* RKNPU task descriptor layout (40 bytes, packed) */
struct orknn_rknpu_task {
    uint32_t flags;
    uint32_t op_idx;
    uint32_t enable_mask;
    uint32_t int_mask;
    uint32_t int_clear;
    uint32_t int_status;
    uint32_t regcfg_amount;
    uint32_t regcfg_offset;
    uint64_t regcmd_addr;
} __attribute__((packed));

static int extract_npu_data(const uint8_t *fb, uint32_t fb_size,
                            uint64_t version, struct orknn_model *model)
{
    uint32_t root = orknn_fb_u32(fb, 0);

    int wt_field = (version > 5) ? 20 : 4;
    uint32_t wt_fpos = orknn_fb_field(fb, root, wt_field);
    if (!wt_fpos) {
        orknn_log(0, "model: no weight_data field (field %d)", wt_field);
        return -1;
    }

    uint32_t n_entries = orknn_fb_vec_len(fb, wt_fpos);
    orknn_log(2, "model: %u weight_data entries", n_entries);

    /* Collect blobs from weight_data entries */
    struct {
        const uint8_t *data;
        uint32_t len;
        uint8_t type;
    } blobs[256];
    unsigned n_blobs = 0;

    for (unsigned i = 0; i < n_entries && i < 256; i++) {
        uint32_t entry = orknn_fb_vec_at(fb, wt_fpos, i);
        if (!entry || entry + 70 > fb_size) continue;

        uint8_t type_byte = fb[entry + 66];
        uint32_t fo0 = orknn_fb_field(fb, entry, 0);
        if (!fo0) continue;

        uint32_t data_len;
        const uint8_t *data_ptr = orknn_fb_bytes(fb, fo0, &data_len);
        if (!data_ptr || data_len == 0) continue;

        blobs[n_blobs].data = data_ptr;
        blobs[n_blobs].len = data_len;
        blobs[n_blobs].type = type_byte;
        n_blobs++;

        orknn_log(3, "  blob[%u]: type=%u len=%u", i, type_byte, data_len);
    }

    /* Assemble all blobs into one buffer (BO[1]) with 64-byte alignment */
    uint32_t bo1_size = 0;
    uint32_t *blob_offsets = calloc(n_blobs, sizeof(uint32_t));
    for (unsigned i = 0; i < n_blobs; i++) {
        bo1_size = ALIGN_UP(bo1_size, 64);
        blob_offsets[i] = bo1_size;
        bo1_size += blobs[i].len;
    }
    bo1_size = ALIGN_UP(bo1_size, 4096);

    uint8_t *bo1_data = calloc(1, bo1_size);
    for (unsigned i = 0; i < n_blobs; i++)
        memcpy(bo1_data + blob_offsets[i], blobs[i].data, blobs[i].len);

    /* Find task BO and regcmd within type=6 blobs */
    uint32_t rc_offset = 0, rc_size = 0;
    uint32_t tb_offset = 0, tb_size = 0;

    for (unsigned i = 0; i < n_blobs; i++) {
        if (blobs[i].type != 6) continue;

        /* Task BO: divisible by 40, valid enable_mask in first entry */
        if (blobs[i].len >= 40 && blobs[i].len % 40 == 0) {
            uint32_t em = orknn_fb_u32(bo1_data, blob_offsets[i] + 8);
            if (em == 0x1d || em == 0x18 || em == 0x60 || em == 0x0f) {
                tb_offset = blob_offsets[i];
                tb_size = blobs[i].len;
            }
        }
        /* Regcmd: largest type=6 blob (excluding task BO) */
        if (blobs[i].len > rc_size && blob_offsets[i] != tb_offset) {
            rc_offset = blob_offsets[i];
            rc_size = blobs[i].len;
        }
    }

    if (!rc_size || !tb_size) {
        orknn_log(0, "model: regcmd(%u) or taskbo(%u) not found", rc_size, tb_size);
        free(bo1_data);
        free(blob_offsets);
        return -1;
    }

    orknn_log(1, "model: regcmd at +%u (%u bytes), taskbo at +%u (%u bytes)",
              rc_offset, rc_size, tb_offset, tb_size);

    /* Parse task BO — find single-core task count */
    uint32_t total_tasks = tb_size / 40;
    uint32_t sc_tasks = total_tasks;

    /* Detect multi-core duplication: tasks repeat with different regcmd addrs */
    if (total_tasks > 3) {
        for (int div = 3; div >= 2; div--) {
            if (total_tasks % div != 0) continue;
            uint32_t chunk = total_tasks / div;
            int match = 1;
            for (uint32_t t = 0; t < chunk && match; t++) {
                uint32_t *a = (uint32_t *)(bo1_data + tb_offset + t * 40);
                uint32_t *b = (uint32_t *)(bo1_data + tb_offset + (chunk + t) * 40);
                if (a[1] != b[1] || a[2] != b[2]) match = 0;
            }
            if (match) {
                uint64_t rc0 = orknn_fb_u64(bo1_data, tb_offset + 32);
                uint64_t rc1 = orknn_fb_u64(bo1_data, tb_offset + chunk * 40 + 32);
                if (rc0 != rc1) { sc_tasks = chunk; break; }
            }
        }
    }

    /* Build sc_count: tasks in first CONV block (up to REFORMAT transition) */
    uint32_t sc_count = 0;
    {
        int seen_reformat = 0;
        for (uint32_t t = 0; t < sc_tasks; t++) {
            uint32_t em = orknn_fb_u32(bo1_data, tb_offset + t * 40 + 8);
            if (em == 0x18) seen_reformat = 1;
            else if (seen_reformat && em == 0x1d) { sc_count = t + 1; break; }
        }
        if (sc_count == 0) sc_count = sc_tasks;
    }

    orknn_log(1, "model: %u total tasks, %u single-core, sc_count=%u",
              total_tasks, sc_tasks, sc_count);

    /* Copy task data and fix regcmd offsets */
    model->task_count = sc_tasks;
    model->task_data_size = sc_tasks * 40;
    model->task_data = malloc(model->task_data_size);
    memcpy(model->task_data, bo1_data + tb_offset, model->task_data_size);

    struct orknn_rknpu_task *tasks = (struct orknn_rknpu_task *)model->task_data;
    for (uint32_t t = 0; t < sc_tasks; t++)
        tasks[t].regcmd_addr += rc_offset;

    /* Store weight+regcmd combined BO data */
    model->wt_data = bo1_data;
    model->wt_size = bo1_size;
    model->regcmd_data = bo1_data + rc_offset;
    model->regcmd_size = rc_size;
    model->total_weight_size = bo1_size;

    /* Build single submit segment */
    model->segment_count = 1;
    model->segments = calloc(1, sizeof(*model->segments));
    model->segments[0].flags = 0x5; /* PC + PINGPONG */
    model->segments[0].sc_start = 0;
    model->segments[0].sc_count = sc_count;
    model->segments[0].task_number = sc_tasks;

    /* Extract activation BO size from metadata blobs */
    model->total_internal_size = 0;
    for (unsigned i = 0; i < n_blobs; i++) {
        if (blobs[i].type == 6 && blobs[i].len == 64) {
            model->total_internal_size = orknn_fb_u32(blobs[i].data, 32);
            break;
        }
    }
    if (model->total_internal_size == 0)
        model->total_internal_size = 65536;

    free(blob_offsets);
    return 0;
}

/* ======================================================================
 * orknn_own_init — main model loading entry point
 * ====================================================================== */

int orknn_own_init(struct orknn_context *ctx, void *model_buf, uint32_t size,
                   uint32_t flag, rknn_init_extend *extend)
{
    (void)flag; (void)extend;
    struct orknn_model *m = &ctx->model;

    /* Copy model data */
    m->file_data = malloc(size);
    if (!m->file_data) return RKNN_ERR_MALLOC_FAIL;
    memcpy(m->file_data, model_buf, size);
    m->file_size = size;

    /* Validate RKNN magic */
    if (size < 0x50 || memcmp(m->file_data, "RKNN", 4) != 0) {
        orknn_log(0, "model: invalid RKNN magic or too small (%u bytes)", size);
        return RKNN_ERR_MODEL_INVALID;
    }

    /* Read header */
    m->version = orknn_fb_u64(m->file_data, 8);
    uint64_t export_data_size = orknn_fb_u64(m->file_data, 0x10);
    uint32_t config_start = (m->version > 1) ? 0x40 : 0x18;

    orknn_log(1, "model: version=%lu, export_data_size=%lu, config_start=0x%x",
              (unsigned long)m->version, (unsigned long)export_data_size,
              config_start);

    /* Parse JSON config section */
    uint32_t json_off = config_start + (uint32_t)export_data_size;
    if (json_off + 8 > size) {
        orknn_log(0, "model: JSON config offset out of bounds");
        return RKNN_ERR_MODEL_INVALID;
    }
    uint64_t config_size = orknn_fb_u64(m->file_data, json_off);
    const char *json_data = (const char *)(m->file_data + json_off + 8);

    if (json_off + 8 + config_size > size) {
        orknn_log(0, "model: JSON config extends past file end");
        return RKNN_ERR_MODEL_INVALID;
    }

    /* NUL-terminate JSON for string operations */
    char *json = calloc(1, config_size + 1);
    memcpy(json, json_data, config_size);

    orknn_log(2, "model: JSON config: %u bytes", (uint32_t)config_size);

    /* Extract metadata from JSON */
    int n_in = 0, n_out = 0;
    json_int(json, "input_num", &n_in);
    json_int(json, "output_num", &n_out);
    m->n_inputs = (n_in > 0) ? n_in : 1;
    m->n_outputs = (n_out > 0) ? n_out : 1;

    /* target_platform can be a string or array: "rk3588" or ["rk3588"] */
    char platform[64] = {0};
    if (json_str(json, "target_platform", platform, sizeof(platform)) <= 0) {
        /* Try as array: find first element */
        const char *tp = strstr(json, "\"target_platform\"");
        if (tp) {
            const char *bracket = strchr(tp, '[');
            if (bracket) {
                const char *q = strchr(bracket, '"');
                if (q) {
                    q++;
                    int i = 0;
                    while (*q && *q != '"' && i < 63)
                        platform[i++] = *q++;
                    platform[i] = '\0';
                }
            }
        }
    }
    m->target_platform = strdup(platform[0] ? platform : "rk3588");

    char name[256] = {0};
    json_str(json, "name", name, sizeof(name));
    m->model_name = strdup(name[0] ? name : "rknn model");

    orknn_log(1, "model: %s, platform=%s, inputs=%u, outputs=%u",
              m->model_name, m->target_platform, m->n_inputs, m->n_outputs);

    /* Parse tensor metadata from JSON.
     * RKNN JSON uses "norm_tensor" array with all tensors, and "connection"
     * to map which are inputs/outputs. Connection entries:
     *   {"left":"input", "left_tensor_id":0, "node_id":0,
     *    "right_tensor":{"tensor_id":0, "type":"norm_tensor"}}
     *   {"left":"output", ...}
     *
     * The norm_tensor entries have: url, dim_num, size[], dtype{qnt_type, vx_type}
     */

    /* Find input/output tensor IDs from connections */
    int input_tensor_ids[16], output_tensor_ids[16];
    int n_input_ids = 0, n_output_ids = 0;
    {
        const char *conn = strstr(json, "\"connection\"");
        if (conn) {
            const char *p = strchr(conn, '[');
            if (p) {
                p++;
                while (*p && *p != ']') {
                    const char *obj = strchr(p, '{');
                    if (!obj) break;
                    /* Find matching } */
                    int depth = 1;
                    const char *end = obj + 1;
                    while (*end && depth > 0) {
                        if (*end == '{') depth++;
                        else if (*end == '}') depth--;
                        end++;
                    }
                    /* Extract "left" and tensor_id from right_tensor */
                    char left[32] = {0};
                    json_str(obj, "left", left, sizeof(left));
                    /* Find right_tensor.tensor_id */
                    const char *rt = strstr(obj, "\"right_tensor\"");
                    int tid = -1;
                    if (rt) {
                        const char *rt_obj = strchr(rt, '{');
                        if (rt_obj && rt_obj < end)
                            json_int(rt_obj, "tensor_id", &tid);
                    }
                    if (tid >= 0) {
                        if (!strcmp(left, "input") && n_input_ids < 16)
                            input_tensor_ids[n_input_ids++] = tid;
                        else if (!strcmp(left, "output") && n_output_ids < 16)
                            output_tensor_ids[n_output_ids++] = tid;
                    }
                    p = end;
                }
            }
        }
    }

    /* Parse norm_tensor array */
    m->inputs = calloc(m->n_inputs, sizeof(*m->inputs));
    m->outputs = calloc(m->n_outputs, sizeof(*m->outputs));

    int norm_count = 0;
    json_int(json, "norm_tensor_num", &norm_count);

    orknn_log(2, "model: %d norm_tensors, %d input_ids, %d output_ids",
              norm_count, n_input_ids, n_output_ids);
    for (int i = 0; i < n_input_ids; i++)
        orknn_log(2, "  input[%d] -> tensor_id=%d", i, input_tensor_ids[i]);
    for (int i = 0; i < n_output_ids; i++)
        orknn_log(2, "  output[%d] -> tensor_id=%d", i, output_tensor_ids[i]);

    for (int ti = 0; ti < norm_count; ti++) {
        char *obj = json_array_obj_copy(json, "norm_tensor", ti);
        if (!obj) { orknn_log(2, "  norm_tensor[%d]: copy failed", ti); continue; }

        orknn_log(3, "  norm_tensor[%d] JSON: %.200s", ti, obj);

        struct orknn_tensor_info info;
        parse_tensor_json(obj, &info);

        int tensor_id = -1;
        json_int(obj, "tensor_id", &tensor_id);

        orknn_log(2, "  norm_tensor[%d]: id=%d name=%s dims=%ux%ux%ux%u type=%d",
                  ti, tensor_id, info.name,
                  info.dims[0], info.dims[1], info.dims[2], info.dims[3],
                  info.type);

        /* Match to input or output */
        for (int i = 0; i < n_input_ids && i < (int)m->n_inputs; i++) {
            if (input_tensor_ids[i] == tensor_id) {
                m->inputs[i] = info;
                orknn_log(2, "    -> assigned to input[%d]", i);
            }
        }
        for (int i = 0; i < n_output_ids && i < (int)m->n_outputs; i++) {
            if (output_tensor_ids[i] == tensor_id) {
                m->outputs[i] = info;
                orknn_log(2, "    -> assigned to output[%d]", i);
            }
        }
        free(obj);
    }

    free(json);

    /* Parse FlatBuffer for NPU data (weights, regcmd, tasks) */
    /* The FlatBuffer starts at config_start (same region as the model data).
     * The JSON config is within the export section; the FB root is at
     * config_start and weight_data is relative to that root. */
    const uint8_t *fb = m->file_data + config_start;
    uint32_t fb_size = size - config_start;

    int ret = extract_npu_data(fb, fb_size, m->version, m);
    if (ret != 0) {
        orknn_log(0, "model: NPU data extraction failed");
        return RKNN_ERR_MODEL_INVALID;
    }

    /* Also init via proxy for cross-validation if available */
    struct orknn_proxy *proxy = orknn_proxy_get();
    if (proxy) {
        ret = proxy->rknn_init(&ctx->real_ctx, model_buf, size, flag, extend);
        if (ret != RKNN_SUCC) {
            orknn_log(0, "model: proxy rknn_init failed: %d", ret);
            ctx->real_ctx = 0;
        } else {
            orknn_log(1, "model: proxy init OK (real_ctx=0x%lx)",
                      (unsigned long)ctx->real_ctx);

            /* Extract quantization params (scale/zp) from proxy.
             * These aren't in the JSON config — they come from FlatBuffer
             * quantization tables. Until we implement FB quant extraction,
             * pull them from the real library. */
            for (uint32_t i = 0; i < m->n_inputs; i++) {
                rknn_tensor_attr attr;
                memset(&attr, 0, sizeof(attr));
                attr.index = i;
                if (proxy->rknn_query(ctx->real_ctx, RKNN_QUERY_INPUT_ATTR,
                                      &attr, sizeof(attr)) == RKNN_SUCC) {
                    m->inputs[i].scale = attr.scale;
                    m->inputs[i].zp = attr.zp;
                    orknn_log(2, "model: input[%u] scale=%.6f zp=%d (from proxy)",
                              i, attr.scale, attr.zp);
                }
            }
            for (uint32_t i = 0; i < m->n_outputs; i++) {
                rknn_tensor_attr attr;
                memset(&attr, 0, sizeof(attr));
                attr.index = i;
                if (proxy->rknn_query(ctx->real_ctx, RKNN_QUERY_OUTPUT_ATTR,
                                      &attr, sizeof(attr)) == RKNN_SUCC) {
                    m->outputs[i].scale = attr.scale;
                    m->outputs[i].zp = attr.zp;
                    orknn_log(2, "model: output[%u] scale=%.6f zp=%d (from proxy)",
                              i, attr.scale, attr.zp);
                }
            }
        }
    }

    return RKNN_SUCC;
}
