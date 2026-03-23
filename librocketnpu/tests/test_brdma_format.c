/*
 * Analyze BRDMA data format dumped by intercept_swap.c DUMP_BRDMA mode.
 *
 * For each task's BRDMA dump, tries to determine the layout of bias (int32)
 * and MUL scale (int16) data that the RKNN runtime programs for per-channel
 * requantization via BS MUL with DMA source.
 *
 * Usage: ./test_brdma_format <model.tflite> <brdma_dir>
 *   model.tflite: the TFLite model (for known bias/scale values)
 *   brdma_dir: directory with task_NNN.bin and task_NNN.meta files
 *
 * Build: gcc -o test_brdma_format test_brdma_format.c ../src/rnpu_tflite.c -lm
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dirent.h>

#include "../src/rnpu_internal.h"
#include "../src/rnpu_tflite.h"

static int parse_meta(const char *path, unsigned *oc, unsigned *oc_pad,
                       unsigned *bs_cfg, unsigned *bs_mul_cfg,
                       unsigned *brdma_cfg, unsigned *cvt_scale, unsigned *cvt_shift)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        sscanf(line, "oc=%u", oc);
        sscanf(line, "oc_pad=%u", oc_pad);
        sscanf(line, "bs_cfg=0x%x", bs_cfg);
        sscanf(line, "bs_mul_cfg=0x%x", bs_mul_cfg);
        sscanf(line, "brdma_cfg=0x%x", brdma_cfg);
        sscanf(line, "out_cvt_scale=%u", cvt_scale);
        sscanf(line, "out_cvt_shift=%u", cvt_shift);
    }
    fclose(f);
    return 0;
}

/* Compute bias correction (sum of w*izp for each output channel) */
static int32_t calc_bias_correction(const struct rnpu_tfl_model *tfl,
                                     const struct rnpu_tfl_op *op,
                                     unsigned oc)
{
    int wt_idx = op->inputs[1];
    const struct rnpu_tfl_tensor *wt = &tfl->tensors[wt_idx];
    const struct rnpu_tfl_buffer *buf = &tfl->buffers[wt->buffer_index];
    unsigned ic = tfl->tensors[op->inputs[0]].shape[3];
    int izp = (uint8_t)tfl->tensors[op->inputs[0]].quant.zero_point;
    unsigned ww = wt->shape[1], wh = wt->shape[2];
    int wzp = (uint8_t)wt->quant.zero_point;
    const uint8_t *w = buf->data;

    int32_t corr = 0;
    for (unsigned x = 0; x < ww; x++)
        for (unsigned y = 0; y < wh; y++)
            for (unsigned i = 0; i < ic; i++) {
                unsigned flat = oc * ww * wh * ic + x * wh * ic + y * ic + i;
                corr += (w[flat] - wzp) * (izp - 0x80);
            }
    return corr;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.tflite> <brdma_dir>\n", argv[0]);
        return 1;
    }

    struct rnpu_tfl_model tfl = {0};
    if (rnpu_tflite_parse(argv[1], &tfl) < 0) {
        fprintf(stderr, "Failed to parse %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "Parsed model: %u ops, %u tensors\n", tfl.op_count, tfl.tensor_count);

    /* Find first CONV_2D with per-axis quantization */
    const struct rnpu_tfl_op *first_conv = NULL;
    for (unsigned i = 0; i < tfl.op_count; i++) {
        if (tfl.ops[i].builtin_code == TFLITE_OP_CONV_2D ||
            tfl.ops[i].builtin_code == TFLITE_OP_DEPTHWISE_CONV_2D) {
            const struct rnpu_tfl_tensor *wt = &tfl.tensors[tfl.ops[i].inputs[1]];
            if (wt->quant.scales && wt->quant.num_scales > 1) {
                first_conv = &tfl.ops[i];
                fprintf(stderr, "First per-axis CONV: op %u, OC=%u\n",
                        i, tfl.tensors[first_conv->outputs[0]].shape[3]);
                break;
            }
        }
    }
    if (!first_conv) {
        fprintf(stderr, "No per-axis CONV found\n");
        return 1;
    }

    unsigned full_oc = tfl.tensors[first_conv->outputs[0]].shape[3];
    const struct rnpu_tfl_tensor *wt = &tfl.tensors[first_conv->inputs[1]];
    int bias_idx = first_conv->inputs[2];
    const struct rnpu_tfl_buffer *bias_buf = &tfl.buffers[tfl.tensors[bias_idx].buffer_index];
    const int32_t *tfl_biases = (const int32_t *)bias_buf->data;

    /* Compute expected biases (with correction applied) */
    int32_t *expected_bias = malloc(full_oc * sizeof(int32_t));
    for (unsigned c = 0; c < full_oc; c++) {
        int32_t corr = calc_bias_correction(&tfl, first_conv, c);
        expected_bias[c] = tfl_biases[c] - corr;
    }

    /* Compute max weight scale for uniform OUT_CVT */
    float max_ws = wt->quant.scales[0];
    for (unsigned i = 1; i < wt->quant.num_scales; i++)
        if (wt->quant.scales[i] > max_ws) max_ws = wt->quant.scales[i];

    fprintf(stderr, "Expected biases[0..3]: %d %d %d %d\n",
            expected_bias[0], expected_bias[1], expected_bias[2], expected_bias[3]);
    fprintf(stderr, "Weight scales[0..3]: %e %e %e %e (max=%e)\n",
            wt->quant.scales[0], wt->quant.scales[1],
            wt->quant.scales[2], wt->quant.scales[3], max_ws);

    /* Scan BRDMA dump directory for first task with data */
    DIR *dir = opendir(argv[2]);
    if (!dir) {
        fprintf(stderr, "Cannot open dir %s\n", argv[2]);
        return 1;
    }

    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        if (strncmp(de->d_name, "task_", 5) != 0) continue;
        if (!strstr(de->d_name, ".bin")) continue;

        char bin_path[512], meta_path[512];
        snprintf(bin_path, sizeof(bin_path), "%s/%s", argv[2], de->d_name);
        /* Build meta path from bin path */
        strcpy(meta_path, bin_path);
        char *dot = strrchr(meta_path, '.');
        if (dot) strcpy(dot, ".meta");

        unsigned oc = 0, oc_pad = 0, bs_cfg = 0, bs_mul_cfg = 0;
        unsigned brdma_cfg = 0, cvt_scale = 0, cvt_shift = 0;
        if (parse_meta(meta_path, &oc, &oc_pad, &bs_cfg, &bs_mul_cfg,
                        &brdma_cfg, &cvt_scale, &cvt_shift) < 0)
            continue;

        /* Only analyze tasks matching our first CONV's OC */
        if (oc != full_oc) continue;

        FILE *f = fopen(bin_path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *data = malloc(sz);
        fread(data, 1, sz, f);
        fclose(f);

        printf("\n=== %s (OC=%u, oc_pad=%u, %ld bytes) ===\n",
               de->d_name, oc, oc_pad, sz);
        printf("BS_CFG=0x%x BS_MUL_CFG=0x%x BRDMA_CFG=0x%x\n",
               bs_cfg, bs_mul_cfg, brdma_cfg);
        printf("OUT_CVT_SCALE=%u OUT_CVT_SHIFT=%u\n", cvt_scale, cvt_shift);

        /* Try Layout A: [bias[0..oc_pad-1] as int32] [mul[0..oc_pad-1] as int16, pad to 4B] */
        printf("\n--- Layout A: bias[int32] then mul[int16] ---\n");
        unsigned bias_bytes = oc_pad * 4;
        unsigned mul_offset_a = bias_bytes;
        if ((long)mul_offset_a + oc_pad * 2 <= sz) {
            const int32_t *bias_a = (const int32_t *)data;
            const int16_t *mul_a = (const int16_t *)(data + mul_offset_a);

            int bias_match = 1;
            for (unsigned c = 0; c < oc && c < 8; c++) {
                printf("  bias[%u]: dump=%d expected=%d %s\n",
                       c, bias_a[c], expected_bias[c],
                       bias_a[c] == expected_bias[c] ? "OK" : "MISMATCH");
                if (bias_a[c] != expected_bias[c]) bias_match = 0;
            }

            /* Check MUL values: expected = round(ws[c]/max_ws * (1<<14)) */
            unsigned mul_shift = (bs_mul_cfg >> 8) & 0x3f;
            printf("  mul_shift from BS_MUL_CFG = %u\n", mul_shift);
            for (unsigned c = 0; c < oc && c < 8; c++) {
                float expected_mul = wt->quant.scales[c] / max_ws * (1 << mul_shift);
                printf("  mul[%u]: dump=%d expected=%.1f ratio=%.4f\n",
                       c, mul_a[c], expected_mul,
                       mul_a[c] != 0 ? expected_mul / mul_a[c] : 0.0);
            }
            if (bias_match) printf("  ** BIAS MATCHES! Layout A bias section confirmed **\n");
        }

        /* Try Layout B: interleaved [bias(int32), mul(int16), pad(int16)] */
        printf("\n--- Layout B: interleaved [bias32, mul16, pad16] ---\n");
        if (sz >= (long)oc * 8) {
            int bias_match = 1;
            for (unsigned c = 0; c < oc && c < 8; c++) {
                int32_t bias_b = *(int32_t *)(data + c * 8);
                int16_t mul_b = *(int16_t *)(data + c * 8 + 4);
                int16_t pad_b = *(int16_t *)(data + c * 8 + 6);
                printf("  [%u]: bias=%d mul=%d pad=%d (expected bias=%d)\n",
                       c, bias_b, mul_b, pad_b, expected_bias[c]);
                if (bias_b != expected_bias[c]) bias_match = 0;
            }
            if (bias_match) printf("  ** BIAS MATCHES! Layout B confirmed **\n");
        }

        /* Try Layout C: [bias(int32), mul(int32)] interleaved */
        printf("\n--- Layout C: interleaved [bias32, mul32] ---\n");
        if (sz >= (long)oc * 8) {
            int bias_match = 1;
            for (unsigned c = 0; c < oc && c < 8; c++) {
                int32_t bias_c = *(int32_t *)(data + c * 8);
                int32_t mul_c = *(int32_t *)(data + c * 8 + 4);
                printf("  [%u]: bias=%d mul=%d (expected bias=%d)\n",
                       c, bias_c, mul_c, expected_bias[c]);
                if (bias_c != expected_bias[c]) bias_match = 0;
            }
            if (bias_match) printf("  ** BIAS MATCHES! Layout C confirmed **\n");
        }

        /* Raw hex dump of first 128 bytes */
        printf("\nRaw hex (first %ld bytes):\n", sz < 128 ? sz : (long)128);
        for (long i = 0; i < sz && i < 128; i++) {
            printf("%02x ", data[i]);
            if ((i + 1) % 16 == 0) printf("\n");
        }
        printf("\n");

        free(data);
        break; /* Only analyze first matching task */
    }

    closedir(dir);
    free(expected_bias);
    rnpu_tflite_free(&tfl);
    return 0;
}
