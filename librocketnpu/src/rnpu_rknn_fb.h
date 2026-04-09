/*
 * Direct .rknn model file parser — FlatBuffer-based
 * Extracts pre-compiled NPU data (weights, regcmd, tasks) from .rknn files
 * without requiring runtime capture/intercept.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_RKNN_FB_H
#define RNPU_RKNN_FB_H

#include <stdint.h>
#include <stdbool.h>

/* Parsed .rknn model data — everything needed for NPU execution */
struct rknn_parsed {
   /* Combined weight + regcmd blob (maps to RKNN's BO[1]) */
   uint8_t *wt_rc_data;
   uint32_t wt_rc_size;
   uint32_t rc_start_offset;  /* Where regcmd starts within wt_rc blob */

   /* Blob offsets within wt_rc for DMA patching */
   uint32_t n_blob_offsets;
   uint32_t blob_offsets[64];
   uint32_t blob_sizes[64];
   uint8_t  blob_types[64];

   /* Task metadata */
   uint32_t task_count;
   struct {
      uint32_t rc_offset;     /* Regcmd offset within wt_rc blob */
      uint32_t rc_amount;     /* Number of regcmd entries */
      uint32_t enable_mask;   /* 0x1d=CONV, 0x18=REFORMAT */
      uint32_t op_idx;
   } *tasks;

   /* Raw task BO data (rknpu_task structs, 40 bytes each) */
   uint8_t *task_bo_data;
   uint32_t task_bo_size;

   /* BO allocation table */
   uint32_t bo_count;
   struct {
      uint64_t dma_addr;      /* Template DMA address */
      uint64_t size;
   } *bos;

   /* Submit segments */
   uint32_t segment_count;
   struct {
      uint32_t flags;
      uint32_t sc_start;
      uint32_t sc_count;
      uint32_t task_number;
   } *segments;

   /* I/O tensor info */
   uint32_t input_size;       /* Input tensor bytes (H×W×C) */
   uint32_t output_count;
   struct {
      uint32_t size;          /* Output tensor bytes */
      float scale;            /* Quant scale */
      int32_t zero_point;     /* Quant zero point */
   } outputs[8];
};

/* Parse a .rknn file and extract all NPU execution data.
 * Returns 0 on success, -1 on error.
 * Caller must free with rknn_parsed_free(). */
int rknn_parse_model(const char *path, struct rknn_parsed *out);

/* Free parsed model data */
void rknn_parsed_free(struct rknn_parsed *p);

#endif /* RNPU_RKNN_FB_H */
