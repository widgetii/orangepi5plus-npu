/*
 * .rknn model parser — extract pre-computed BRDMA data and OUT_CVT params
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_RKNN_H
#define RNPU_RKNN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Per-op BRDMA + OUT_CVT data extracted from .rknn file */
struct rnpu_rknn_op {
   unsigned output_channels;   /* from DPU_DATA_CUBE_CHANNEL */
   int32_t out_cvt_offset;
   uint32_t out_cvt_scale;
   uint32_t out_cvt_shift;
   uint8_t *brdma_data;        /* 64 × ceil(oc_pad/8) bytes */
   uint32_t brdma_size;
   int32_t *biases;            /* first bias per 8-ch group, for matching */
   unsigned bias_count;        /* number of valid bias entries */
   bool matched;               /* set when matched to a TFLite op */
};

struct rnpu_rknn_model {
   uint8_t *file_data;
   size_t file_size;
   struct rnpu_rknn_op *ops;
   unsigned op_count;
};

/* Parse .rknn file and extract per-op BRDMA data.
 * Returns 0 on success, -1 on error. */
int rnpu_rknn_parse(const char *rknn_path, struct rnpu_rknn_model *out);

/* Free parsed .rknn model data. */
void rnpu_rknn_free(struct rnpu_rknn_model *m);

/* Find the .rknn op that matches a given set of biases and output channel count.
 * Returns index into m->ops, or -1 if not found. */
int rnpu_rknn_match_op(const struct rnpu_rknn_model *m,
                       const int32_t *biases, unsigned oc);

#endif /* RNPU_RKNN_H */
