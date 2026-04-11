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
#include <dirent.h>

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
static void parse_tensor_json(const char *obj, struct orknn_tensor_info *ti,
                              const uint8_t *file_data)
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

    /* Fallback for FP16 models: norm_tensor JSON may have empty vx_type.
     * The authoritative dtype lives in per-tensor hint dicts stored in
     * the RKNN file header (first ~2KB), e.g.:
     *   {"input.1": {"dtype": "float16", "layout": "NHWC"}}
     * Search for the tensor name in the hint area and extract dtype. */
    if (ti->type == RKNN_TENSOR_INT8 && !dtype_str[0] && ti->name[0] && file_data) {
        /* Search first 4KB of file for {"<name>": {"dtype": "..."}} */
        char needle[128];
        snprintf(needle, sizeof(needle), "\"%s\":", ti->name);
        size_t nlen = strlen(needle);
        const char *hit = NULL;
        for (uint32_t p = 0; p + nlen + 20 < 4096; p++) {
            if (memcmp(file_data + p, needle, nlen) == 0) {
                hit = (const char *)(file_data + p);
                break;
            }
        }
        if (hit) {
            const char *dp = strstr(hit, "\"dtype\"");
            if (dp && dp < hit + 200) {
                dp = strchr(dp + 7, '"');
                if (dp) {
                    dp++;
                    char htype[32] = {0};
                    int k = 0;
                    while (*dp && *dp != '"' && k < 31)
                        htype[k++] = *dp++;
                    htype[k] = '\0';
                    rknn_tensor_type ht = dtype_from_string(htype);
                    if (ht != RKNN_TENSOR_INT8) {
                        ti->type = ht;
                        orknn_log(2, "  tensor '%s': dtype '%s' from header hint",
                                  ti->name, htype);
                    }
                }
            }
        }
    }

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

    /* FP16 tensors: proxy reports QNT_AFFINE with scale=1.0 zp=0 (identity quant).
     * Set this automatically when dtype is FP16 and qnt is empty/NONE. */
    if ((ti->type == RKNN_TENSOR_FLOAT16 || ti->type == RKNN_TENSOR_BFLOAT16) &&
        ti->qnt_type == RKNN_TENSOR_QNT_NONE) {
        ti->qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
        ti->scale = 1.0f;
        ti->zp = 0;
    }

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
 * FlatBuffer tensor metadata extraction
 *
 * Subgraph field 0 = tensor descriptor vector (N entries)
 * Subgraph field 2 = input tensor index vector
 * Subgraph field 3 = output tensor index vector
 *
 * Per-tensor fields (from IDA RE):
 *   f[0]: packed byte (dtype at byte 0, flags)
 *   f[2]: layout byte (1=NHWC, 2=NCHW, 64=NC1HWC2)
 *   f[3]: native shape vector (5D for NC1HWC2)
 *   f[4]: user shape vector (NCHW order)
 *   f[5]: name string
 *   f[6]: qnt_method string ("layer", "channel")
 *   f[7]: dtype string ("int8", "uint8", "float16")
 *   f[10]: scale vector (float[])
 *   f[11]: zero_point vector (int32[] stored as float bits)
 *   f[12]: allocation size (uint32)
 *   f[13]: native layout code (64 = NC1HWC2)
 * ====================================================================== */

/* Layout byte to rknn_tensor_format */
static rknn_tensor_format layout_to_fmt(uint8_t layout)
{
    switch (layout) {
    case 1:  return RKNN_TENSOR_NHWC;
    case 2:  return RKNN_TENSOR_NCHW;
    case 64: return RKNN_TENSOR_NC1HWC2;
    default: return RKNN_TENSOR_UNDEFINED;
    }
}

/* Read a float from a FlatBuffer int32 vector entry.
 * The zp field stores int32 values in a vector that fb_bytes reads as raw. */
static int32_t fb_int32_vec_at(const uint8_t *fb, uint32_t fpos, uint32_t idx)
{
    uint32_t vec = orknn_fb_follow(fb, fpos);
    uint32_t count = orknn_fb_u32(fb, vec);
    if (idx >= count) return 0;
    return orknn_fb_i32(fb, vec + 4 + idx * 4);
}

static float fb_float_vec_at(const uint8_t *fb, uint32_t fpos, uint32_t idx)
{
    uint32_t vec = orknn_fb_follow(fb, fpos);
    uint32_t count = orknn_fb_u32(fb, vec);
    if (idx >= count) return 0.0f;
    uint32_t bits = orknn_fb_u32(fb, vec + 4 + idx * 4);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void extract_fb_tensor_info(const uint8_t *fb, uint32_t tensor_table_fpos,
                                   uint32_t tensor_idx,
                                   struct orknn_tensor_info *ti)
{
    uint32_t t = orknn_fb_vec_at(fb, tensor_table_fpos, tensor_idx);
    if (!t) return;

    /* f[5]: name */
    uint32_t f5 = orknn_fb_field(fb, t, 5);
    if (f5) orknn_fb_string(fb, f5, ti->name, sizeof(ti->name));

    /* f[7]: dtype string — override JSON dtype only if non-empty.
     * FP16 models (MobileSAM) may have empty f[7] in which case we
     * keep the dtype from the header hint parsed by parse_tensor_json. */
    uint32_t f7 = orknn_fb_field(fb, t, 7);
    if (f7) {
        char dtype_str[32] = {0};
        orknn_fb_string(fb, f7, dtype_str, sizeof(dtype_str));
        if (dtype_str[0])
            ti->type = dtype_from_string(dtype_str);
    }

    /* f[6]: qnt_method — skip if empty (FP16 models have empty qnt fields) */
    uint32_t f6 = orknn_fb_field(fb, t, 6);
    if (f6) {
        char qnt_str[32] = {0};
        orknn_fb_string(fb, f6, qnt_str, sizeof(qnt_str));
        if (!qnt_str[0]) {
            /* empty — keep JSON/header-derived qnt_type */
        } else if (strstr(qnt_str, "layer") || strstr(qnt_str, "channel"))
            ti->qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
        else if (strstr(qnt_str, "dfp"))
            ti->qnt_type = RKNN_TENSOR_QNT_DFP;
        else
            ti->qnt_type = RKNN_TENSOR_QNT_NONE;
    }

    /* f[10]: scale — only override if non-zero (FP16 models may store 0).
     * For FP16 tensors with scale already set to 1.0 (identity quant from
     * parse_tensor_json FP16 fallback), preserve the 1.0 if FB has 0. */
    uint32_t f10 = orknn_fb_field(fb, t, 10);
    if (f10) {
        float fb_scale = fb_float_vec_at(fb, f10, 0);
        if (fb_scale != 0.0f)
            ti->scale = fb_scale;
    }
    /* FP16 tensors: ensure scale=1.0 if still 0 (identity quantization) */
    if ((ti->type == RKNN_TENSOR_FLOAT16 || ti->type == RKNN_TENSOR_BFLOAT16) &&
        ti->scale == 0.0f) {
        ti->scale = 1.0f;
        ti->qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
    }

    /* f[11]: zero_point (stored as int32 in a vector) */
    uint32_t f11 = orknn_fb_field(fb, t, 11);
    if (f11) ti->zp = fb_int32_vec_at(fb, f11, 0);

    /* f[2]: layout byte → format.
     * For non-4D tensors, the proxy returns UNDEFINED regardless of layout byte. */
    uint32_t f2 = orknn_fb_field(fb, t, 2);
    if (f2) {
        if (ti->n_dims >= 4)
            ti->fmt = layout_to_fmt(fb[f2]);
        else
            ti->fmt = RKNN_TENSOR_UNDEFINED;
    }

    /* f[4]: user shape.
     * For NHWC tensors (layout=1): FB stores NCHW [N,C,H,W] → reorder to [N,H,W,C]
     * For NCHW tensors (layout=2): FB stores NHWC-order [N,H,W,C] already
     * For other/non-4D: copy as-is */
    uint32_t f4 = orknn_fb_field(fb, t, 4);
    if (f4) {
        uint32_t vec = orknn_fb_follow(fb, f4);
        uint32_t nd = orknn_fb_u32(fb, vec);
        ti->n_dims = nd;
        int raw[RKNN_MAX_DIMS];
        for (uint32_t i = 0; i < nd && i < RKNN_MAX_DIMS; i++)
            raw[i] = orknn_fb_i32(fb, vec + 4 + i * 4);

        uint8_t layout = 0;
        uint32_t f2_pos = orknn_fb_field(fb, t, 2);
        if (f2_pos) layout = fb[f2_pos];

        if (nd == 4 && layout == 1) {
            /* NHWC tensor: FB has [N,C,H,W] → reorder to [N,H,W,C] */
            ti->dims[0] = raw[0];
            ti->dims[1] = raw[2];
            ti->dims[2] = raw[3];
            ti->dims[3] = raw[1];
        } else {
            /* NCHW or non-4D: FB already in display order */
            for (uint32_t i = 0; i < nd; i++)
                ti->dims[i] = (uint32_t)raw[i];
        }

        ti->n_elems = 1;
        for (uint32_t i = 0; i < nd; i++)
            ti->n_elems *= ti->dims[i];
        ti->size = ti->n_elems * dtype_size(ti->type);
    }

    /* f[3]: native shape (NC1HWC2 for 5D) */
    uint32_t f3 = orknn_fb_field(fb, t, 3);
    if (f3) {
        uint32_t vec = orknn_fb_follow(fb, f3);
        uint32_t nd = orknn_fb_u32(fb, vec);
        ti->native_n_dims = nd;
        for (uint32_t i = 0; i < nd && i < RKNN_MAX_DIMS; i++)
            ti->native_dims[i] = orknn_fb_u32(fb, vec + 4 + i * 4);
    }

    /* f[12]: native allocation size */
    uint32_t f12 = orknn_fb_field(fb, t, 12);
    if (f12) ti->native_size = orknn_fb_u32(fb, f12);

    /* Do NOT apply a native_dims-based dim reorder here. The FB f[4]
     * field already stores the user-facing shape in the correct order
     * for each tensor (NHWC for YOLOv5/DeepLabv3 outputs, NCHW for
     * YOLOv8 outputs, etc.) and the vendor library reports them as-is.
     * An earlier heuristic that used JSON-derived native_dims fired
     * incorrectly on YOLOv5 because the native_dims was stale from
     * parse_tensor_json's c2=16 assumption. */

    /* w_stride: set for NHWC 4D tensors only. */
    if (ti->n_dims == 4 && ti->fmt == RKNN_TENSOR_NHWC)
        ti->w_stride = ti->dims[2]; /* W */
    else
        ti->w_stride = 0;

    /* size_with_stride: native allocation size for 4D, element size for others */
    ti->size_with_stride = (ti->n_dims >= 4) ? ti->native_size : ti->size;
}

/* ======================================================================
 * Per-operator graph extraction (phase 3)
 *
 * Subgraph field 1 is a vector of operator tables. Operator table fields
 * (discovered empirically via tests/probe_fb_schema.py across the 5
 * ground_truth models):
 *   f[1]: op type string ("Conv", "ConvRelu", "ConvClip", "ConvExSwish",
 *         "BatchNormalization", "InputOperator", ...)
 *   f[2]: op instance name string (may be empty)
 *   f[4]: input tensor indices (uint32[]); for Conv: [data, weight, bias]
 *   f[5]: output tensor indices (uint32[])
 *   f[8]: 16-element uint32 param vector (header/metadata)
 *   f[10]: 6-element op params (stride/dilation/etc.)
 *   f[11]: 6-element padding
 *   f[12]: 9-element sizes (inputs/outputs/scratch)
 *
 * Phase 3 extracts just f[1], f[4], f[5] into struct orknn_op_info. The
 * other fields are left for a follow-on once we know which ones drive the
 * CVT register values for op=1 (the phase-1 blocker).
 * ====================================================================== */
static void parse_fb_operators(const uint8_t *fb, struct orknn_model *m)
{
    uint32_t root = orknn_fb_u32(fb, 0);
    uint32_t f2_root = orknn_fb_field(fb, root, 2);
    if (!f2_root) return;
    uint32_t sg = orknn_fb_vec_at(fb, f2_root, 0);
    if (!sg) return;
    uint32_t ops_fpos = orknn_fb_field(fb, sg, 1);
    if (!ops_fpos) return;

    uint32_t n_ops = orknn_fb_vec_len(fb, ops_fpos);
    if (n_ops == 0 || n_ops > 4096) return;

    struct orknn_op_info *ops = calloc(n_ops, sizeof(*ops));
    if (!ops) return;

    for (uint32_t i = 0; i < n_ops; i++) {
        ops[i].implicit_wt_tidx = UINT32_MAX;
        ops[i].implicit_bs_tidx = UINT32_MAX;
        ops[i].softmax_rmax_tidx = UINT32_MAX;
        ops[i].softmax_rsum_tidx = UINT32_MAX;
    }

    for (uint32_t i = 0; i < n_ops; i++) {
        uint32_t op = orknn_fb_vec_at(fb, ops_fpos, i);
        if (!op) continue;

        uint32_t f1 = orknn_fb_field(fb, op, 1);
        if (f1) orknn_fb_string(fb, f1, ops[i].type, sizeof(ops[i].type));

        uint32_t f4 = orknn_fb_field(fb, op, 4);
        if (f4) {
            uint32_t vec = orknn_fb_follow(fb, f4);
            uint32_t count = orknn_fb_u32(fb, vec);
            if (count > ORKNN_MAX_OP_INPUTS) count = ORKNN_MAX_OP_INPUTS;
            ops[i].input_count = count;
            for (uint32_t k = 0; k < count; k++)
                ops[i].input_tensors[k] = orknn_fb_u32(fb, vec + 4 + k * 4);
        }

        uint32_t f5 = orknn_fb_field(fb, op, 5);
        if (f5) {
            uint32_t vec = orknn_fb_follow(fb, f5);
            uint32_t count = orknn_fb_u32(fb, vec);
            if (count > ORKNN_MAX_OP_OUTPUTS) count = ORKNN_MAX_OP_OUTPUTS;
            ops[i].output_count = count;
            for (uint32_t k = 0; k < count; k++)
                ops[i].output_tensors[k] = orknn_fb_u32(fb, vec + 4 + k * 4);
        }

        /* f[10] carries per-op task counts; see docs/segmentation_from_fb.md
         * for the full derivation. Slot 0 is the CONV-only task count, slot
         * 1 is the total including any activation-LUT pre-task, slot 2 is a
         * non-zero sub-count for "plain" ops and zero for activation-fused
         * ones — we use that to disambiguate. */
        uint32_t f10 = orknn_fb_field(fb, op, 10);
        if (f10) {
            uint32_t vec = orknn_fb_follow(fb, f10);
            uint32_t vlen = orknn_fb_u32(fb, vec);
            if (vlen >= 3) {
                uint32_t v0 = orknn_fb_u32(fb, vec + 4);
                uint32_t v1 = orknn_fb_u32(fb, vec + 8);
                uint32_t v2 = orknn_fb_u32(fb, vec + 12);
                ops[i].task_count = (v2 == 0) ? v1 : v0;
            } else if (vlen >= 1) {
                ops[i].task_count = orknn_fb_u32(fb, vec + 4);
            }
        }
    }

    /* Resolve implicit weight/bias tensors for ops like Resize that
     * reference compiler-generated blobs via name prefix matching (not
     * via input_tensors[]). See the comment on struct orknn_op_info's
     * implicit_wt_tidx / implicit_bs_tidx fields for the rationale. */
    uint32_t tensors_fpos = orknn_fb_field(fb, sg, 0);
    if (tensors_fpos) {
        uint32_t n_tensors = orknn_fb_vec_len(fb, tensors_fpos);
        for (uint32_t i = 0; i < n_ops; i++) {
            if (strcmp(ops[i].type, "Resize") != 0) continue;
            if (ops[i].input_count == 0) continue;
            uint32_t in0 = ops[i].input_tensors[0];
            if (in0 >= n_tensors) continue;
            uint32_t in0_tbl = orknn_fb_vec_at(fb, tensors_fpos, in0);
            if (!in0_tbl) continue;
            uint32_t in0_name_f = orknn_fb_field(fb, in0_tbl, 5);
            if (!in0_name_f) continue;
            char in0_name[128];
            orknn_fb_string(fb, in0_name_f, in0_name, sizeof(in0_name));
            size_t prefix_len = strlen(in0_name);
            if (prefix_len == 0) continue;
            for (uint32_t t = 0; t < n_tensors; t++) {
                uint32_t tbl = orknn_fb_vec_at(fb, tensors_fpos, t);
                if (!tbl) continue;
                uint32_t name_f = orknn_fb_field(fb, tbl, 5);
                if (!name_f) continue;
                char name[128];
                orknn_fb_string(fb, name_f, name, sizeof(name));
                if (strncmp(name, in0_name, prefix_len) != 0) continue;
                if (name[prefix_len] != '_') continue;
                const char *suffix = name + prefix_len + 1;
                /* Match `_weight...` and `_bias...` variants. The
                 * compiler appends dtype/shape tags (e.g.
                 * `_weight_1int8_64_1_1_1`). */
                if (strncmp(suffix, "weight", 6) == 0 &&
                    ops[i].implicit_wt_tidx == UINT32_MAX) {
                    ops[i].implicit_wt_tidx = t;
                } else if (strncmp(suffix, "bias", 4) == 0 &&
                           ops[i].implicit_bs_tidx == UINT32_MAX) {
                    ops[i].implicit_bs_tidx = t;
                }
                if (ops[i].implicit_wt_tidx != UINT32_MAX &&
                    ops[i].implicit_bs_tidx != UINT32_MAX)
                    break;
            }
            if (ops[i].implicit_wt_tidx != UINT32_MAX ||
                ops[i].implicit_bs_tidx != UINT32_MAX) {
                orknn_log(2, "model: op[%u] %s implicit wt=%u bs=%u "
                             "(matched `%s_*`)", i, ops[i].type,
                          ops[i].implicit_wt_tidx,
                          ops[i].implicit_bs_tidx, in0_name);
            }
        }

        /* exSoftmax13 compile-time weight tensors. MobileNet's softmax
         * lowering generates three em=0x0d tasks — ReduceMax, rescale,
         * and ReduceSum — each with its own CNA weight blob. The
         * rescale blob is in input_tensors[2] already; the ReduceMax
         * and ReduceSum blobs are top-level tensors whose names match
         *   `{output[0].name}_ReduceMax_output_weight*`
         *   `{output[0].name}_reducesum_output_weight*`
         * (the lowercase `reducesum` and uppercase `ReduceMax` are the
         * compiler's actual naming). */
        for (uint32_t i = 0; i < n_ops; i++) {
            if (strncmp(ops[i].type, "exSoftmax", 9) != 0) continue;
            if (ops[i].output_count == 0) continue;
            uint32_t out0 = ops[i].output_tensors[0];
            if (out0 >= n_tensors) continue;
            uint32_t out0_tbl = orknn_fb_vec_at(fb, tensors_fpos, out0);
            if (!out0_tbl) continue;
            uint32_t out0_name_f = orknn_fb_field(fb, out0_tbl, 5);
            if (!out0_name_f) continue;
            char out0_name[128];
            orknn_fb_string(fb, out0_name_f, out0_name, sizeof(out0_name));
            size_t prefix_len = strlen(out0_name);
            if (prefix_len == 0) continue;
            for (uint32_t t = 0; t < n_tensors; t++) {
                uint32_t tbl = orknn_fb_vec_at(fb, tensors_fpos, t);
                if (!tbl) continue;
                uint32_t name_f = orknn_fb_field(fb, tbl, 5);
                if (!name_f) continue;
                char name[128];
                orknn_fb_string(fb, name_f, name, sizeof(name));
                if (strncmp(name, out0_name, prefix_len) != 0) continue;
                if (name[prefix_len] != '_') continue;
                const char *suffix = name + prefix_len + 1;
                if (strstr(suffix, "ReduceMax_output_weight") &&
                    ops[i].softmax_rmax_tidx == UINT32_MAX) {
                    ops[i].softmax_rmax_tidx = t;
                } else if (strstr(suffix, "reducesum_output_weight") &&
                           ops[i].softmax_rsum_tidx == UINT32_MAX) {
                    ops[i].softmax_rsum_tidx = t;
                }
                if (ops[i].softmax_rmax_tidx != UINT32_MAX &&
                    ops[i].softmax_rsum_tidx != UINT32_MAX)
                    break;
            }
            if (ops[i].softmax_rmax_tidx != UINT32_MAX ||
                ops[i].softmax_rsum_tidx != UINT32_MAX) {
                orknn_log(2, "model: op[%u] %s softmax rmax=%u rsum=%u",
                          i, ops[i].type,
                          ops[i].softmax_rmax_tidx,
                          ops[i].softmax_rsum_tidx);
            }
        }
    }

    m->ops = ops;
    m->op_count = n_ops;

    orknn_log(1, "model: parsed %u operators from subgraph", n_ops);
    for (uint32_t i = 0; i < n_ops && i < 8; i++) {
        orknn_log(2, "model: op[%u] type=%s in=[%u,%u,%u] out=[%u]",
                  i, ops[i].type,
                  ops[i].input_count > 0 ? ops[i].input_tensors[0] : 0,
                  ops[i].input_count > 1 ? ops[i].input_tensors[1] : 0,
                  ops[i].input_count > 2 ? ops[i].input_tensors[2] : 0,
                  ops[i].output_count > 0 ? ops[i].output_tensors[0] : 0);
    }
}

/* ======================================================================
 * Phase 4b: parse the tensor memory plan from FB tensor.f[13].
 *
 * Each tensor that has a location in the activation BO stores its byte
 * offset in field 13 of the tensor table. Tensors without f[13] (weights,
 * constants, or input/output tensors not in the activation BO) have offset
 * 0. The memory plan is what the vendor runtime uses to resolve per-task
 * activation DMA addresses — specifically SRC_BASE, DST_BASE, and RDMA_ACT
 * register values of the form `act_base + tensor_offset + intra_tile_off`.
 *
 * Discovered via the librknnrt decompile: sub_2ABEF0 ("Tensor buffer
 * allocation info") extracts field 13 (`v4[15]` in the decompiled code,
 * which indexes past the 2-uint16 vtable header) — see
 * ~/projects/ida/librknnrt/docs/rknn-model-format.md §8.2 which lists the
 * function as "Tensor buffer allocation info".
 * ====================================================================== */
static void parse_fb_tensor_offsets(const uint8_t *fb, struct orknn_model *m)
{
    uint32_t root = orknn_fb_u32(fb, 0);
    uint32_t f2_root = orknn_fb_field(fb, root, 2);
    if (!f2_root) return;
    uint32_t sg = orknn_fb_vec_at(fb, f2_root, 0);
    if (!sg) return;
    uint32_t tensors_fpos = orknn_fb_field(fb, sg, 0);
    if (!tensors_fpos) return;
    uint32_t n_tensors = orknn_fb_vec_len(fb, tensors_fpos);
    if (n_tensors == 0 || n_tensors > 8192) return;

    uint32_t *offsets = calloc(n_tensors, sizeof(uint32_t));
    uint32_t *weight_blob = calloc(n_tensors, sizeof(uint32_t));
    if (!offsets || !weight_blob) {
        free(offsets);
        free(weight_blob);
        return;
    }
    for (uint32_t i = 0; i < n_tensors; i++)
        weight_blob[i] = 0xFFFFFFFFu; /* sentinel: not a weight */

    uint32_t nonzero_off = 0, have_blob = 0;
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint32_t t = orknn_fb_vec_at(fb, tensors_fpos, i);
        if (!t) continue;
        uint32_t f13 = orknn_fb_field(fb, t, 13);
        if (f13) {
            uint32_t v = orknn_fb_u32(fb, f13);
            offsets[i] = v;
            if (v) nonzero_off++;
        }
        /* f[18] = weight_data blob index for weight/bias tensors.
         * Present and non-zero only for tensors whose data is stored as
         * a blob in the weight section. Looked up via blob_offsets[] in
         * openrknn_run.c to translate into a byte offset in BO[1]. */
        uint32_t f18 = orknn_fb_field(fb, t, 18);
        if (f18) {
            uint32_t v = orknn_fb_u32(fb, f18);
            /* Index 0 is the task BO blob which is never a weight, so
             * treat 0 as "no weight" too. The sentinel UINT32_MAX lets
             * downstream code distinguish "not parsed" from "explicitly
             * zero". */
            if (v != 0) {
                weight_blob[i] = v;
                have_blob++;
            }
        }
    }

    m->tensor_offsets = offsets;
    m->tensor_weight_blob = weight_blob;
    m->tensor_count = n_tensors;

    /* Flag tensors that are subgraph outputs. These land in output BOs
     * rather than the activation BO, so REFORMAT tasks targeting them
     * use `out_base` rather than `act_base + tensor_off`. */
    uint32_t outputs_fpos = orknn_fb_field(fb, sg, 3);
    if (outputs_fpos) {
        uint32_t vec = orknn_fb_follow(fb, outputs_fpos);
        uint32_t n_out = orknn_fb_u32(fb, vec);
        uint8_t *is_out = calloc(n_tensors, 1);
        uint32_t *out_tidx = calloc(n_out > 0 ? n_out : 1, sizeof(uint32_t));
        if (is_out && out_tidx) {
            for (uint32_t i = 0; i < n_out; i++) {
                uint32_t tidx = orknn_fb_u32(fb, vec + 4 + i * 4);
                out_tidx[i] = tidx;
                if (tidx < n_tensors)
                    is_out[tidx] = 1;
            }
            m->tensor_is_sg_output = is_out;
            m->sg_output_tensor_idx = out_tidx;
        } else {
            free(is_out);
            free(out_tidx);
        }
    }

    /* Build wt_blob_offsets[] — per-weight_data-index byte offset into
     * BO[1]. We iterate the full weight_data vector in FB order, matching
     * extract_npu_data's traversal exactly (so layout is identical to the
     * bo1_data buffer we end up writing to the weight BO). */
    int wt_field = (m->version > 5) ? 20 : 4;
    uint32_t wt_fpos = orknn_fb_field(fb, root, wt_field);
    if (wt_fpos) {
        uint32_t n_wt = orknn_fb_vec_len(fb, wt_fpos);
        if (n_wt > 0 && n_wt < 8192) {
            uint32_t *wt_off = calloc(n_wt, sizeof(uint32_t));
            if (wt_off) {
                uint32_t bo1_off = 0;
                for (uint32_t i = 0; i < n_wt; i++) {
                    uint32_t entry = orknn_fb_vec_at(fb, wt_fpos, i);
                    uint32_t fo0 = entry ? orknn_fb_field(fb, entry, 0) : 0;
                    /* Even empty entries get an offset (the offset where
                     * the next entry would start); they contribute 0
                     * bytes to bo1. Matches how extract_npu_data skips
                     * empties — bo1_off only advances for non-empty. */
                    bo1_off = (bo1_off + 63u) & ~63u;
                    wt_off[i] = bo1_off;
                    if (fo0) {
                        uint32_t blen = 0;
                        orknn_fb_bytes(fb, fo0, &blen);
                        bo1_off += blen;
                    }
                }
                m->wt_blob_offsets = wt_off;
                m->wt_blob_count = n_wt;
            }
        }
    }

    orknn_log(1, "model: parsed memory plan: %u tensors (%u act offsets, "
              "%u weight blobs), %u wt_blob_offsets entries",
              n_tensors, nonzero_off, have_blob, m->wt_blob_count);
}

/* ======================================================================
 * Phase 3b: parse the header `attrs` Python-dict-like string.
 *
 * Each .rknn file contains a plain-text pre-processing config embedded in
 * the header (before the FlatBuffer root), shaped roughly like:
 *   {'attrs': {
 *       '<input name>': {
 *           'idx': 0, 'shape': [...], 'layout': 'nchw', 'range': [0, 1],
 *           'dtype': 'uint8' | 'int8' | 'float32',
 *           'mean': [m_r, m_g, m_b],
 *           'std':  [s_r, s_g, s_b],
 *           'rgb2bgr': False,
 *           ...
 *       },
 *       ...
 *   }, 'quant_tab': { ... }, ...}
 *
 * This is what the vendor RKNN runtime uses to compute the input CONV's
 * CNA_CVT_CON* registers (empirically validated on all 5 runtime models
 * in tests/ground_truth.json). The `attrs` dict is per-input but we only
 * parse the first input today.
 *
 * The parser is deliberately minimal — no full Python expression support.
 * We scan for the literal prefix "{'attrs':", then find the first input
 * object, then extract 'dtype', 'mean' and 'std' via simple substring
 * search. Good enough for Rockchip's compiler output format.
 * ====================================================================== */
static const char *find_key(const char *s, const char *end, const char *key)
{
    size_t kl = strlen(key);
    while (s + kl < end) {
        const char *hit = memchr(s, key[0], end - s - kl);
        if (!hit) return NULL;
        if (memcmp(hit, key, kl) == 0)
            return hit;
        s = hit + 1;
    }
    return NULL;
}

static int parse_float_list(const char *p, const char *end, float *out, int max_n)
{
    while (p < end && *p != '[') p++;
    if (p >= end) return 0;
    p++;
    int n = 0;
    while (p < end && *p != ']' && n < max_n) {
        while (p < end && (*p == ' ' || *p == ',')) p++;
        if (*p == ']') break;
        char *endp = NULL;
        float v = strtof(p, &endp);
        if (endp == p) break;
        out[n++] = v;
        p = endp;
    }
    return n;
}

static int parse_quoted_string(const char *p, const char *end,
                               char *out, int max_len)
{
    while (p < end && *p != '\'' && *p != '"') p++;
    if (p >= end) return 0;
    char q = *p++;
    int n = 0;
    while (p < end && *p != q && n < max_len - 1)
        out[n++] = *p++;
    out[n] = '\0';
    return n;
}

static void parse_fb_attrs(struct orknn_model *m)
{
    const char *data = (const char *)m->file_data;
    size_t data_len = m->file_size;
    const char *marker = "{'attrs':";
    const char *hit = NULL;
    /* Scan header (first 8KB is plenty). */
    size_t scan_len = data_len < 8192 ? data_len : 8192;
    for (size_t i = 0; i + strlen(marker) < scan_len; i++) {
        if (memcmp(data + i, marker, strlen(marker)) == 0) {
            hit = data + i;
            break;
        }
    }
    if (!hit) {
        orknn_log(2, "model: no attrs string in header — CVT will stay template");
        return;
    }

    /* End of attrs: stop at first non-printable/non-whitespace byte.
     * The attrs blob is pure ASCII in RKNN's output. */
    const char *end = hit;
    while (end < data + data_len &&
           ((*end >= 0x20 && *end < 0x7f) || *end == '\n' || *end == '\t' || *end == '\r'))
        end++;

    /* Within attrs, find the first nested dict (input tensor entry).
     * The structure is {'attrs': {'<name>': { ... }, ...}}, so we skip the
     * outer '{' pair, find the next "'<name>': {", and parse fields inside. */
    const char *p = hit + strlen(marker);
    /* Skip whitespace + outer '{' of the attrs dict */
    while (p < end && (*p == ' ' || *p == '{')) p++;
    /* Skip over the first key '<name>': */
    if (p < end && *p == '\'') {
        p++;
        while (p < end && *p != '\'') p++;
        if (p < end) p++;  /* closing quote */
        while (p < end && (*p == ' ' || *p == ':')) p++;
    }
    /* p now points at the opening '{' of the input object */
    if (p >= end || *p != '{') {
        orknn_log(1, "model: attrs parse: input object not found");
        return;
    }
    /* Find the matching closing brace */
    const char *obj_end = p + 1;
    int depth = 1;
    while (obj_end < end && depth > 0) {
        if (*obj_end == '{') depth++;
        else if (*obj_end == '}') depth--;
        obj_end++;
    }
    if (depth != 0) {
        orknn_log(1, "model: attrs parse: unterminated input object");
        return;
    }

    /* Now scan inside [p, obj_end) for 'dtype', 'mean', 'std'. */
    const char *dtype_key = find_key(p, obj_end, "'dtype'");
    if (dtype_key) {
        const char *colon = memchr(dtype_key, ':', obj_end - dtype_key);
        if (colon)
            parse_quoted_string(colon + 1, obj_end,
                                m->input_attr_dtype, sizeof(m->input_attr_dtype));
    }
    const char *mean_key = find_key(p, obj_end, "'mean'");
    if (mean_key) {
        const char *colon = memchr(mean_key, ':', obj_end - mean_key);
        if (colon)
            parse_float_list(colon + 1, obj_end, m->input_attr_mean, 4);
    }
    const char *std_key = find_key(p, obj_end, "'std'");
    if (std_key) {
        const char *colon = memchr(std_key, ':', obj_end - std_key);
        if (colon)
            parse_float_list(colon + 1, obj_end, m->input_attr_std, 4);
    }

    if (m->input_attr_dtype[0])
        m->input_attr_valid = 1;

    orknn_log(1, "model: attrs: dtype=%s mean=[%.3f,%.3f,%.3f] std=[%.3f,%.3f,%.3f]",
              m->input_attr_dtype,
              m->input_attr_mean[0], m->input_attr_mean[1], m->input_attr_mean[2],
              m->input_attr_std[0],  m->input_attr_std[1],  m->input_attr_std[2]);
}

/* Find which operator (by op_idx as stored in the task BO) reads the
 * subgraph's input tensor as its primary data input. Assumes parse_fb_operators
 * and extract_fb_tensors have already run. The return value is the operator's
 * index in model->ops[], which equals the op_idx the task BO uses. */
static void find_input_consuming_op(const uint8_t *fb, struct orknn_model *m)
{
    m->input_consuming_op_idx = 0;
    uint32_t root = orknn_fb_u32(fb, 0);
    uint32_t f2_root = orknn_fb_field(fb, root, 2);
    if (!f2_root) return;
    uint32_t sg = orknn_fb_vec_at(fb, f2_root, 0);
    if (!sg) return;
    uint32_t inputs_fpos = orknn_fb_field(fb, sg, 2);
    if (!inputs_fpos) return;
    uint32_t vec = orknn_fb_follow(fb, inputs_fpos);
    if (orknn_fb_u32(fb, vec) == 0) return;
    uint32_t sg_input = orknn_fb_u32(fb, vec + 4);

    for (uint32_t i = 0; i < m->op_count; i++) {
        if (m->ops[i].input_count > 0 &&
            m->ops[i].input_tensors[0] == sg_input) {
            m->input_consuming_op_idx = i;
            orknn_log(1, "model: input-consuming op is ops[%u] type=%s "
                      "(sg input tensor=%u)", i, m->ops[i].type, sg_input);
            return;
        }
    }
    orknn_log(1, "model: no op consumes subgraph input %u", sg_input);
}

/* Extract tensor metadata from FlatBuffer subgraph for input/output tensors */
static void extract_fb_tensors(const uint8_t *fb, uint32_t fb_size,
                               struct orknn_model *m)
{
    uint32_t root = orknn_fb_u32(fb, 0);

    /* Field 2 = subgraphs vector */
    uint32_t f2_root = orknn_fb_field(fb, root, 2);
    if (!f2_root) return;

    uint32_t sg = orknn_fb_vec_at(fb, f2_root, 0);
    if (!sg) return;

    /* Subgraph field 0 = tensor descriptors */
    uint32_t tensors_fpos = orknn_fb_field(fb, sg, 0);
    if (!tensors_fpos) return;

    /* Subgraph field 2 = input indices, field 3 = output indices */
    uint32_t inputs_fpos = orknn_fb_field(fb, sg, 2);
    uint32_t outputs_fpos = orknn_fb_field(fb, sg, 3);

    if (inputs_fpos) {
        uint32_t vec = orknn_fb_follow(fb, inputs_fpos);
        uint32_t n_in = orknn_fb_u32(fb, vec);
        for (uint32_t i = 0; i < n_in && i < m->n_inputs; i++) {
            uint32_t tidx = orknn_fb_u32(fb, vec + 4 + i * 4);
            extract_fb_tensor_info(fb, tensors_fpos, tidx, &m->inputs[i]);
            orknn_log(2, "model: FB input[%u] (tensor %u): scale=%.6f zp=%d "
                      "fmt=%d dims=%ux%ux%ux%u native_size=%u",
                      i, tidx, m->inputs[i].scale, m->inputs[i].zp,
                      m->inputs[i].fmt,
                      m->inputs[i].dims[0], m->inputs[i].dims[1],
                      m->inputs[i].dims[2], m->inputs[i].dims[3],
                      m->inputs[i].native_size);
        }
    }

    if (outputs_fpos) {
        uint32_t vec = orknn_fb_follow(fb, outputs_fpos);
        uint32_t n_out = orknn_fb_u32(fb, vec);
        for (uint32_t i = 0; i < n_out && i < m->n_outputs; i++) {
            uint32_t tidx = orknn_fb_u32(fb, vec + 4 + i * 4);
            extract_fb_tensor_info(fb, tensors_fpos, tidx, &m->outputs[i]);
            orknn_log(2, "model: FB output[%u] (tensor %u): scale=%.6f zp=%d "
                      "fmt=%d dims=%ux%ux%ux%u native_size=%u",
                      i, tidx, m->outputs[i].scale, m->outputs[i].zp,
                      m->outputs[i].fmt,
                      m->outputs[i].dims[0], m->outputs[i].dims[1],
                      m->outputs[i].dims[2], m->outputs[i].dims[3],
                      m->outputs[i].native_size);
        }
    }

    (void)fb_size;
}

/* ======================================================================
 * FB-derived segmentation — Task 9.3
 *
 * Replaces the /tmp/rknn_dump/submit_*.txt reader that extract_npu_data
 * used to call. See openrknn/docs/segmentation_from_fb.md for the full
 * rule derivation (including the f[10]-based task-count formula and the
 * LUT_OPS set). This function requires m->ops[] to be populated by
 * parse_fb_operators() and returns the same segments the vendor runtime
 * would emit, byte-exact against the intercept-dump ground truth for all
 * five runtime models.
 * ====================================================================== */

static int op_type_is_lut(const char *type)
{
    /* Activation-LUT ops get flags=0x1 submits (non-pingpong); everything
     * else gets flags=0x5 (PC | PINGPONG). Add new op types here when a
     * future model introduces ConvExMish / ConvExHardSwish / etc. */
    return (strcmp(type, "ConvSigmoid") == 0 ||
            strcmp(type, "ConvExSwish") == 0 ||
            strcmp(type, "exSoftmax13") == 0);
}

static int op_type_is_io(const char *type)
{
    return (strcmp(type, "InputOperator") == 0 ||
            strcmp(type, "OutputOperator") == 0);
}

static int fb_build_segments(struct orknn_model *m)
{
    if (!m->ops || m->op_count == 0) {
        orknn_log(0, "segments: m->ops not populated");
        return -1;
    }

    /* Count segment slots needed so we can allocate once. Plus one for
     * slop — ops might terminate a segment even though they don't start
     * a new one (IO ops at the end of the graph). */
    uint32_t max_segs = 1;
    {
        int cur_flags = -1;
        for (uint32_t i = 0; i < m->op_count; i++) {
            struct orknn_op_info *op = &m->ops[i];
            if (op_type_is_io(op->type) || op->task_count == 0) {
                if (cur_flags >= 0) cur_flags = -1;
                continue;
            }
            int flags = op_type_is_lut(op->type) ? 0x1 : 0x5;
            if (cur_flags != flags) {
                max_segs++;
                cur_flags = flags;
            }
        }
    }

    struct orknn_segment *segs = calloc(max_segs, sizeof(*segs));
    if (!segs) return -1;

    uint32_t n_segs = 0;
    uint32_t cur_task_idx = 0;
    int cur_flags = -1;
    uint32_t cur_start = 0;
    uint32_t cur_count = 0;

    for (uint32_t i = 0; i < m->op_count; i++) {
        struct orknn_op_info *op = &m->ops[i];
        uint32_t tc = op->task_count;

        if (op_type_is_io(op->type)) {
            /* IO ops reserve task-BO slots (DeepLabv3's input reformat
             * lives at task[0]) but are never submitted. Flush the
             * current accumulator and advance past their slots. */
            if (cur_count > 0) {
                segs[n_segs].flags = (uint32_t)cur_flags;
                segs[n_segs].sc_start = cur_start;
                segs[n_segs].sc_count = cur_count;
                segs[n_segs].task_number = cur_count;
                n_segs++;
                cur_count = 0;
                cur_flags = -1;
            }
            cur_task_idx += tc;
            continue;
        }

        if (tc == 0)
            continue;

        int flags = op_type_is_lut(op->type) ? 0x1 : 0x5;
        if (cur_flags != flags) {
            if (cur_count > 0) {
                segs[n_segs].flags = (uint32_t)cur_flags;
                segs[n_segs].sc_start = cur_start;
                segs[n_segs].sc_count = cur_count;
                segs[n_segs].task_number = cur_count;
                n_segs++;
            }
            cur_flags = flags;
            cur_start = cur_task_idx;
            cur_count = 0;
        }
        cur_count += tc;
        cur_task_idx += tc;
    }

    if (cur_count > 0) {
        segs[n_segs].flags = (uint32_t)cur_flags;
        segs[n_segs].sc_start = cur_start;
        segs[n_segs].sc_count = cur_count;
        segs[n_segs].task_number = cur_count;
        n_segs++;
    }

    m->segments = segs;
    m->segment_count = n_segs;

    /* task_count (the number of task-BO slots patch_regcmd_addresses
     * iterates over) must stay at whatever extract_npu_data set — the
     * raw total in the .rknn task blob, which includes cross-core
     * replicas. We only patch the regcmd slices belonging to this
     * specific build; the segments are the per-cycle single-core plan
     * that the RKNPU driver submits against the patched slices. */

    for (uint32_t s = 0; s < n_segs; s++) {
        orknn_log(1, "model: seg[%u] flags=0x%x sc=[%u..+%u] task_num=%u",
                  s, segs[s].flags, segs[s].sc_start,
                  segs[s].sc_count, segs[s].task_number);
    }
    orknn_log(1, "model: %u FB-derived segments, first-cycle task_idx=%u",
              n_segs, cur_task_idx);
    return 0;
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

    /* Find task BO within type=6 blobs (divisible by 40, valid enable_mask) */
    uint32_t rc_offset = 0, rc_size = 0;
    uint32_t tb_offset = 0, tb_size = 0;

    for (unsigned i = 0; i < n_blobs; i++) {
        if (blobs[i].type != 6) continue;
        if (blobs[i].len >= 40 && blobs[i].len % 40 == 0) {
            uint32_t em = orknn_fb_u32(bo1_data, blob_offsets[i] + 8);
            if (em == 0x1d || em == 0x18 || em == 0x60 || em == 0x0f) {
                tb_offset = blob_offsets[i];
                tb_size = blobs[i].len;
            }
        }
    }

    /* Find regcmd blob: use task BO's max regcmd_addr to identify which blob
     * contains the regcmd data. Tasks store offsets into the regcmd blob. */
    if (tb_size >= 40) {
        /* Find max regcmd_addr from tasks to determine regcmd size needed */
        uint32_t max_rc_end = 0;
        uint32_t total_tasks = tb_size / 40;
        for (uint32_t t = 0; t < total_tasks; t++) {
            uint64_t rc_addr = orknn_fb_u64(bo1_data, tb_offset + t * 40 + 32);
            uint32_t rc_amt = orknn_fb_u32(bo1_data, tb_offset + t * 40 + 24);
            uint32_t rc_end = (uint32_t)rc_addr + (rc_amt + 4) * 8;
            if (rc_end > max_rc_end) max_rc_end = rc_end;
        }
        orknn_log(2, "model: tasks need regcmd up to offset %u", max_rc_end);

        /* Find the smallest type=6 blob that covers the regcmd range.
         * The regcmd blob is distinguishable: its first 8 bytes decode as
         * a valid regcmd entry (register addr < 0x8000, target < 0x10). */
        uint32_t best_size = UINT32_MAX;
        for (unsigned i = 0; i < n_blobs; i++) {
            if (blobs[i].type != 6) continue;
            if (blob_offsets[i] == tb_offset) continue;
            if (blobs[i].len < max_rc_end) continue;
            /* Verify first entry looks like regcmd: reg addr in low 16 bits
             * should be a valid NPU register (0x0000-0x7FFF range) */
            uint64_t first = orknn_fb_u64(bo1_data, blob_offsets[i]);
            uint16_t reg = first & 0xFFFF;
            if (reg < 0x8000 && blobs[i].len < best_size) {
                rc_offset = blob_offsets[i];
                rc_size = blobs[i].len;
                best_size = blobs[i].len;
            }
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

    /* The .rknn task BO contains the per-op compile-time task records,
     * usually replicated 3× for triple-core submit layouts. We copy the
     * whole thing; fb_build_segments() (called from orknn_own_init after
     * parse_fb_operators) will set model->task_count to the first-cycle
     * slice it actually needs and leave the excess task_data bytes
     * unused. Segment derivation and per-op task counting all live in
     * fb_build_segments() — see docs/segmentation_from_fb.md. */
    uint32_t total_tasks = tb_size / 40;

#if 0  /* Legacy dump-reading segmentation path, replaced by fb_build_segments.
        * Kept temporarily as a reference for task 9.4 debugging; to be
        * deleted in task 9.8 along with every other /tmp/rknn_dump read. */
    /* Parse task BO — find single-core task count */
    uint32_t sc_tasks = total_tasks;

    /* Detect single-core task count.
     * The .rknn stores tasks for all 3 cores. Multi-core tasks share regcmd
     * offsets (same rc_addr). Find the minimal set that covers all unique
     * regcmd sections, then check for core-repetition patterns.
     *
     * RKNN submit uses task_number = total tasks in BO, sc_count = tasks
     * per pipeline stage (first CONV→REFORMAT cycle). */

    /* Find sc_count: first complete pipeline stage.
     *
     * Method 1: For multi-op models, find where a CONV task's op_idx
     * repeats (but skip the 0x60 continuation tasks which share op_idx).
     * Method 2: If all CONV tasks share the same op_idx (e.g., MBv1 NNBG),
     * use regcmd sharing detection instead.
     */
    uint32_t sc_count = total_tasks;
    {
        /* Count unique op_ids across ALL task types */
        uint32_t seen_ops[16];
        int n_unique = 0;
        for (uint32_t t = 0; t < total_tasks && n_unique < 16; t++) {
            uint32_t op = orknn_fb_u32(bo1_data, tb_offset + t * 40 + 4);
            int found = 0;
            for (int k = 0; k < n_unique; k++)
                if (seen_ops[k] == op) { found = 1; break; }
            if (!found) seen_ops[n_unique++] = op;
        }

        if (n_unique > 1) {
            /* Multi-op model: find first CONV op_idx repeat.
             * The 0x60 task between CONV repeats extends the stage. */
            int n_seen = 0;
            uint32_t conv_seen[16];
            for (uint32_t t = 0; t < total_tasks; t++) {
                uint32_t em = orknn_fb_u32(bo1_data, tb_offset + t * 40 + 8);
                if (em != 0x1d) continue;
                uint32_t op = orknn_fb_u32(bo1_data, tb_offset + t * 40 + 4);
                for (int k = 0; k < n_seen; k++) {
                    if (conv_seen[k] == op) {
                        sc_count = t;
                        /* Include any preceding 0x60 task */
                        if (t > 0) {
                            uint32_t prev_em = orknn_fb_u32(bo1_data, tb_offset + (t-1) * 40 + 8);
                            if (prev_em == 0x60) sc_count = t - 1;
                        }
                        orknn_log(1, "model: sc_count=%u (multi-op, op=%u repeat at %u)",
                                  sc_count, op, t);
                        goto sc_found;
                    }
                }
                if (n_seen < 16) conv_seen[n_seen++] = op;
            }
        } else {
            /* Single-op model (e.g., MBv1): use regcmd sharing */
            for (uint32_t t = 1; t < total_tasks; t++) {
                uint64_t rc = orknn_fb_u64(bo1_data, tb_offset + t * 40 + 32);
                for (uint32_t r = 0; r < t; r++) {
                    uint64_t rc0 = orknn_fb_u64(bo1_data, tb_offset + r * 40 + 32);
                    if (rc == rc0) {
                        sc_count = t;
                        orknn_log(1, "model: sc_count=%u (single-op, regcmd sharing at %u)",
                                  sc_count, t);
                        goto sc_found;
                    }
                }
            }
            /* No sharing → all tasks single-core */
            orknn_log(1, "model: sc_count=%u (no sharing, all single-core)", sc_count);
        }
    }
    sc_found:

    /* Detect multi-core: if task pattern repeats at sc_count intervals
     * with shared regcmd offsets, the total is multi-core × sc_count. */
    sc_tasks = total_tasks;
    if (total_tasks > sc_count) {
        /* Check if tasks at sc_count share regcmd with tasks at 0 */
        int repeats = 0;
        for (uint32_t t = sc_count; t < total_tasks; t++) {
            uint64_t rc = orknn_fb_u64(bo1_data, tb_offset + t * 40 + 32);
            /* Check if this rc_addr matches any task in [0..sc_count) */
            for (uint32_t r = 0; r < sc_count; r++) {
                uint64_t rc0 = orknn_fb_u64(bo1_data, tb_offset + r * 40 + 32);
                if (rc == rc0) { repeats++; break; }
            }
        }
        /* If most tasks beyond sc_count share regcmd → they're multi-core dups.
         * Use the point where non-shared tasks start as the total. */
        if (repeats > 0) {
            /* Find first task beyond sc_count with a NEW (non-shared) regcmd */
            uint32_t unique_end = sc_count;
            for (uint32_t t = sc_count; t < total_tasks; t++) {
                uint64_t rc = orknn_fb_u64(bo1_data, tb_offset + t * 40 + 32);
                int shared = 0;
                for (uint32_t r = 0; r < sc_count; r++) {
                    uint64_t rc0 = orknn_fb_u64(bo1_data, tb_offset + r * 40 + 32);
                    if (rc == rc0) { shared = 1; break; }
                }
                if (shared) {
                    /* Shared task = multi-core dup, skip. But include the tasks
                     * between unique_end and here (they may be needed). */
                    continue;
                }
                unique_end = t + 1;
            }
            /* Use tasks up to the last unique one, then add back the shared
             * tasks that follow for the multi-core pipeline. The total
             * task_number should cover all tasks that the submit needs. */
            sc_tasks = unique_end;
        }
    }

    /* Try to read proxy submit metadata from the intercept dump.
     * Models like YOLO issue MULTIPLE submits per rknn_run (one per
     * hardware stage). bench_rknn runs warmup + N iters and saves the
     * LAST iter's output. The proxy uses DIFFERENT task BO content for
     * each iteration (MBv1 iter 1 writes the final output to output
     * BO[0], warmup doesn't). We must use the LAST cycle's task data.
     *
     * Approach:
     *  1. Read all submit_*.txt files.
     *  2. Detect cycle length by the first sc_start repeat.
     *  3. Use the LAST complete cycle's submits as our segments
     *     (and remember their submit indices so we load the right
     *     sub<N>_bo_000 snapshot later). */
    #define MAX_SEGS 128
    uint32_t all_flags[256] = {0};
    uint32_t all_sc_start[256] = {0};
    uint32_t all_sc_count[256] = {0};
    uint32_t all_task_num[256] = {0};
    int n_all = 0;

    sc_tasks = total_tasks;
    sc_count = total_tasks;

    for (int sn = 1; sn <= 256 && n_all < 256; sn++) {
        char path[64];
        snprintf(path, sizeof(path), "/tmp/rknn_dump/submit_%d.txt", sn);
        FILE *sf = fopen(path, "r");
        if (!sf) break;

        uint32_t flags = 0, tasks_n = 0, scs = 0, scc = 0;
        char line[256];
        while (fgets(line, sizeof(line), sf)) {
            if (strncmp(line, "submit=", 7) == 0) {
                char *fp = strstr(line, "flags=0x");
                char *tp = strstr(line, "tasks=");
                char *ssp = strstr(line, "sc_start=");
                char *scp = strstr(line, "sc_count=");
                if (fp) flags = (uint32_t)strtoul(fp + 8, NULL, 16);
                if (tp) tasks_n = (uint32_t)strtoul(tp + 6, NULL, 10);
                if (ssp) scs = (uint32_t)strtoul(ssp + 9, NULL, 10);
                if (scp) scc = (uint32_t)strtoul(scp + 9, NULL, 10);
                break;
            }
        }
        fclose(sf);

        all_flags[n_all] = flags;
        all_sc_start[n_all] = scs;
        all_sc_count[n_all] = scc;
        all_task_num[n_all] = tasks_n;
        n_all++;
    }

    /* Detect cycle length: first index >0 where sc_start == all_sc_start[0]. */
    int cycle = n_all;
    for (int k = 1; k < n_all; k++) {
        if (all_sc_start[k] == all_sc_start[0]) { cycle = k; break; }
    }

    /* Last-cycle start index: largest multiple of cycle <= n_all - cycle. */
    int last_start = n_all - cycle;
    if (last_start < 0) last_start = 0;

    uint32_t seg_flags[MAX_SEGS] = {0};
    uint32_t seg_sc_start[MAX_SEGS] = {0};
    uint32_t seg_sc_count[MAX_SEGS] = {0};
    uint32_t seg_task_num[MAX_SEGS] = {0};
    int seg_submit_idx[MAX_SEGS] = {0}; /* 1-based submit index for dump path */
    int n_segs = 0;

    /* Segments themselves describe cycle layout from submit 1..cycle.
     * Per-cycle task BO snapshots are loaded separately below. */
    for (int k = 0; k < cycle && n_segs < MAX_SEGS && k < n_all; k++) {
        seg_flags[n_segs] = all_flags[k];
        seg_sc_start[n_segs] = all_sc_start[k];
        seg_sc_count[n_segs] = all_sc_count[k];
        seg_task_num[n_segs] = all_task_num[k];
        seg_submit_idx[n_segs] = k + 1;
        n_segs++;
    }
    (void)last_start;

    orknn_log(2, "model: read %d submits, cycle=%d, last_cycle starts at submit_%d",
              n_all, cycle, last_start + 1);

    if (n_segs == 0) {
        /* No dump available — single-segment fallback */
        sc_tasks = total_tasks;
        sc_count = total_tasks;
        n_segs = 1;
        seg_flags[0] = 0x5;
        seg_sc_start[0] = 0;
        seg_sc_count[0] = sc_count;
        seg_task_num[0] = sc_tasks;
    } else {
        /* sc_tasks = max task_number across all segments.
         * Segments share a task BO — each submit can reference the
         * full task array. */
        sc_tasks = 0;
        for (int s = 0; s < n_segs; s++) {
            if (seg_task_num[s] > sc_tasks) sc_tasks = seg_task_num[s];
        }
        /* Add max sc_start so we can reach all tasks the segments need */
        for (int s = 0; s < n_segs; s++) {
            uint32_t need = seg_sc_start[s] + seg_sc_count[s];
            /* Actual multi-core total = sc_count * cores, covered by task_number */
            (void)need;
        }
        sc_count = seg_sc_count[0];
    }

    if (sc_tasks > total_tasks) sc_tasks = total_tasks;

    orknn_log(1, "model: %u total tasks, %u segs, seg0 task_num=%u sc_count=%u",
              total_tasks, n_segs, sc_tasks, sc_count);

    /* Copy task data — only the tasks referenced by segment 0.
     * The FB task array has total_tasks entries, but for models like MBv1
     * only the first seg_task_num[0] tasks are valid — the rest have
     * uninitialized values that corrupt execution if copied. */
    uint32_t copy_tasks = seg_task_num[0] > 0 ? seg_task_num[0] : total_tasks;
    if (copy_tasks > total_tasks) copy_tasks = total_tasks;
    model->task_count = copy_tasks;
    model->task_data_size = copy_tasks * 40;
    model->task_data = malloc(model->task_data_size);
    memcpy(model->task_data, bo1_data + tb_offset, model->task_data_size);

    struct orknn_rknpu_task *tasks = (struct orknn_rknpu_task *)model->task_data;
    for (uint32_t t = 0; t < copy_tasks; t++)
        tasks[t].regcmd_addr += rc_offset;

    /* Store weight+regcmd combined BO data */
    model->wt_data = bo1_data;
    model->wt_size = bo1_size;
    model->regcmd_data = bo1_data + rc_offset;
    model->regcmd_size = rc_size;
    model->total_weight_size = bo1_size;

    /* Build submit segments from proxy dump info */
    model->segment_count = n_segs;
    model->segments = calloc(n_segs, sizeof(*model->segments));
    for (int s = 0; s < n_segs; s++) {
        model->segments[s].flags = seg_flags[s];
        model->segments[s].sc_start = seg_sc_start[s];
        model->segments[s].sc_count = seg_sc_count[s];
        model->segments[s].task_number = seg_task_num[s];
        orknn_log(2, "model: seg[%d] flags=0x%x sc_start=%u sc_count=%u task_number=%u",
                  s, seg_flags[s], seg_sc_start[s], seg_sc_count[s], seg_task_num[s]);

        /* Load per-cycle task BO snapshots.
         * Cycle c's submit index for this segment = s + 1 + c * cycle.
         * The proxy uses different task data per iteration, so we store
         * all available cycles and pick by run_count at submit time. */
        int total_cycles = cycle > 0 ? (n_all / cycle) : 1;
        if (total_cycles > ORKNN_MAX_CYCLES) total_cycles = ORKNN_MAX_CYCLES;
        model->segments[s].n_cycles = 0;
        for (int c = 0; c < total_cycles; c++) {
            int submit_idx = s + 1 + c * cycle;
            DIR *dd = opendir("/tmp/rknn_dump");
            if (!dd) break;
            char pfx[32];
            snprintf(pfx, sizeof(pfx), "sub%d_bo_000_", submit_idx);
            struct dirent *de;
            char tpath[256] = {0};
            while ((de = readdir(dd))) {
                if (strncmp(de->d_name, pfx, strlen(pfx)) == 0) {
                    snprintf(tpath, sizeof(tpath), "/tmp/rknn_dump/%s", de->d_name);
                    break;
                }
            }
            closedir(dd);
            if (!tpath[0]) continue;
            FILE *tf = fopen(tpath, "rb");
            if (!tf) continue;
            fseek(tf, 0, SEEK_END);
            long fsz = ftell(tf);
            fseek(tf, 0, SEEK_SET);
            if (fsz > 0) {
                uint32_t idx = model->segments[s].n_cycles;
                model->segments[s].task_bo_data[idx] = malloc(fsz);
                model->segments[s].task_bo_size[idx] = (uint32_t)fsz;
                fread(model->segments[s].task_bo_data[idx], 1, fsz, tf);
                model->segments[s].n_cycles++;
                orknn_log(2, "model: seg[%d] cycle[%u] loaded from %s (%ld bytes)",
                          s, idx, tpath, fsz);
            }
            fclose(tf);
        }
    }
#endif  /* end of legacy dump-reading segmentation path */

    /* Copy the complete task BO into model->task_data. fb_build_segments
     * will later shrink task_count to the first-cycle slice it needs;
     * we leave the tail bytes in place (they belong to cross-core
     * replicas the single-core submit path ignores). */
    model->task_count = total_tasks;
    model->task_data_size = total_tasks * 40;
    model->task_data = malloc(model->task_data_size);
    memcpy(model->task_data, bo1_data + tb_offset, model->task_data_size);

    struct orknn_rknpu_task *tasks = (struct orknn_rknpu_task *)model->task_data;
    for (uint32_t t = 0; t < total_tasks; t++)
        tasks[t].regcmd_addr += rc_offset;

    /* Store weight+regcmd combined BO data */
    model->wt_data = bo1_data;
    model->wt_size = bo1_size;
    model->regcmd_data = bo1_data + rc_offset;
    model->regcmd_size = rc_size;
    model->total_weight_size = bo1_size;

    /* Phase 2: total_internal_size is NOT reliably stored in the .rknn
     * FlatBuffer. The `type=6 len=64 u32@32` pattern we used to match
     * happens to hit blobs that hold unrelated data (0x10000 on ResNet50,
     * 0x01010101 on DeepLabv3, missing on MBv1/YOLO), so the extracted
     * value is always either wrong or coincidentally safe. The activation
     * BO sizing fallback in openrknn_memory.c derives its bound from a
     * regcmd scan plus a conservative multiplier instead; we leave this
     * field zero so the memory sizing code doesn't try to trust it. */
    model->total_internal_size = 0;

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
        parse_tensor_json(obj, &info, m->file_data);

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
        /* Complex models like MobileSAM store task data as many small
         * per-op blobs (not a single monolithic task BO), which our FB
         * parser can't assemble. Fall back to proxy dispatch for run —
         * init/query still work with our parsed tensor metadata. */
        orknn_log(1, "model: NPU data extraction failed — proxy-dispatched run path will be used");
    }

    /* Extract tensor metadata from FlatBuffer (scale, zp, format, native shape).
     * This overrides/augments the JSON-derived values with authoritative FB data. */
    extract_fb_tensors(fb, fb_size, m);

    /* Parse the operator graph: per-op type + input/output tensor indices.
     * Phase 3 foundation — unblocks future work that needs per-op metadata
     * (CVT computation, FC detection, per-channel ops). */
    parse_fb_operators(fb, m);

    /* Phase 9 / Task 9.3: build submit segments from the FB operator
     * graph now that m->ops[] is populated. This replaces the
     * /tmp/rknn_dump/submit_*.txt reader extract_npu_data used to call.
     * fb_build_segments also shrinks model->task_count from the raw
     * multi-core total to the first-cycle slice we actually submit,
     * leaving the tail bytes in task_data unused. */
    if (ret == 0) {
        if (fb_build_segments(m) != 0) {
            orknn_log(0, "model: fb_build_segments failed — run path unusable");
        }
    }

    /* Phase 3b: parse the header `attrs` string for input pre-processing
     * (mean/std/dtype), then find which op_idx consumes the model input.
     * Used by patch_regcmd_addresses to fix up the CVT registers for that
     * first conv, which the vendor runtime populates from the same data. */
    parse_fb_attrs(m);
    find_input_consuming_op(fb, m);

    /* Phase 4b: parse the per-tensor memory plan so patch_regcmd_addresses
     * can resolve activation-BO offsets for non-input-consuming tasks. */
    parse_fb_tensor_offsets(fb, m);

    /* Open RKNPU device and allocate DMA buffer objects */
    if (ctx->own_flags & (ORKNN_OWN_RUN | ORKNN_OWN_INPUTS_SET)) {
        ctx->npu_fd = orknn_drm_open();
        if (ctx->npu_fd < 0) {
            orknn_log(0, "model: failed to open RKNPU device");
            return RKNN_ERR_DEVICE_UNAVAILABLE;
        }
        ret = orknn_alloc_model_bos(ctx);
        if (ret != 0) {
            orknn_log(0, "model: failed to allocate DMA buffers");
            return RKNN_ERR_MALLOC_FAIL;
        }
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
        }
    }

    return RKNN_SUCC;
}
