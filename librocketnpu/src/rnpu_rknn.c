/*
 * .rknn model parser — extract pre-computed BRDMA blobs
 *
 * The .rknn file contains BRDMA data blobs (bias int32 + MUL int16,
 * 64-byte-per-8-channel layout) for per-axis CONV ops. We find them
 * by pattern scanning: each 64-byte chunk has bytes[32..47] all zeros
 * and MUL values (at bytes[48..63]) in the range [100..16384].
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#include "rnpu_rknn.h"

/* Check if a 64-byte chunk looks like a BRDMA chunk:
 * - bytes[32..47] all zeros (padding gap)
 * - at least 4 non-zero MUL values (int16 at bytes[48..63])
 * - MUL values in reasonable range [1..16384] */
static int is_brdma_chunk(const uint8_t *chunk)
{
   for (int k = 32; k < 48; k++)
      if (chunk[k] != 0) return 0;

   const int16_t *mul = (const int16_t *)(chunk + 48);
   int non_zero = 0;
   for (int k = 0; k < 8; k++) {
      if (mul[k] < 0 || mul[k] > 16384) return 0;
      if (mul[k] > 0) non_zero++;
   }
   return non_zero >= 4; /* need strong MUL signal, not just padding */
}

int rnpu_rknn_parse(const char *rknn_path, struct rnpu_rknn_model *out)
{
   memset(out, 0, sizeof(*out));

   FILE *f = fopen(rknn_path, "rb");
   if (!f) {
      fprintf(stderr, "rknn: cannot open %s\n", rknn_path);
      return -1;
   }

   fseek(f, 0, SEEK_END);
   size_t fsize = ftell(f);
   fseek(f, 0, SEEK_SET);

   uint8_t *data = malloc(fsize);
   if (!data || fread(data, 1, fsize, f) != fsize) {
      fprintf(stderr, "rknn: failed to read %s\n", rknn_path);
      fclose(f);
      free(data);
      return -1;
   }
   fclose(f);

   if (fsize < 0x50 || memcmp(data, "RKNN", 4) != 0) {
      fprintf(stderr, "rknn: invalid magic in %s\n", rknn_path);
      free(data);
      return -1;
   }

   out->file_data = data;
   out->file_size = fsize;

   fprintf(stderr, "rknn: parsing %s (%zu bytes)\n", rknn_path, fsize);

   /* Find all BRDMA candidate chunks */
   unsigned max_chunks = 4096;
   uint32_t *chunk_offsets = malloc(max_chunks * sizeof(uint32_t));
   unsigned chunk_count = 0;

   for (size_t off = 0; off + 64 <= fsize; off += 64) {
      if (is_brdma_chunk(data + off)) {
         if (chunk_count < max_chunks)
            chunk_offsets[chunk_count++] = (uint32_t)off;
      }
   }

   fprintf(stderr, "rknn: found %u BRDMA chunk candidates\n", chunk_count);

   /* Group consecutive chunks into per-op blobs */
   unsigned max_groups = 256;
   struct {
      uint32_t file_offset;
      uint32_t size;
   } *groups = malloc(max_groups * sizeof(*groups));
   unsigned group_count = 0;

   unsigned gi = 0;
   while (gi < chunk_count && group_count < max_groups) {
      uint32_t start = chunk_offsets[gi];
      uint32_t end = start + 64;
      unsigned gj = gi + 1;
      while (gj < chunk_count && chunk_offsets[gj] == end) {
         end += 64;
         gj++;
      }
      groups[group_count].file_offset = start;
      groups[group_count].size = end - start;
      group_count++;
      gi = gj;
   }

   free(chunk_offsets);

   fprintf(stderr, "rknn: %u BRDMA blob groups\n", group_count);

   /* Build op array from groups */
   out->ops = calloc(group_count, sizeof(struct rnpu_rknn_op));
   out->op_count = group_count;

   /* Filter: only keep groups with oc_pad >= 32 (real BRDMA ops).
    * Smaller groups are likely false positives from weight data. */
   unsigned valid_count = 0;
   for (unsigned i = 0; i < group_count; i++) {
      unsigned nchunks = groups[i].size / 64;
      unsigned oc_pad = nchunks * 8;
      if (oc_pad < 32) continue;

      out->ops[valid_count].brdma_data = malloc(groups[i].size);
      memcpy(out->ops[valid_count].brdma_data, data + groups[i].file_offset, groups[i].size);
      out->ops[valid_count].brdma_size = groups[i].size;
      out->ops[valid_count].output_channels = oc_pad;

      /* Extract biases from first few chunks for matching */
      out->ops[valid_count].biases = malloc(nchunks * 8 * sizeof(int32_t));
      out->ops[valid_count].bias_count = 0;
      for (unsigned c = 0; c < nchunks; c++) {
         const int32_t *b = (const int32_t *)(out->ops[valid_count].brdma_data + c * 64);
         for (unsigned k = 0; k < 8; k++)
            out->ops[valid_count].biases[out->ops[valid_count].bias_count++] = b[k];
      }

      valid_count++;
   }
   out->op_count = valid_count;

   free(groups);

   fprintf(stderr, "rknn: extracted %u BRDMA blobs\n", out->op_count);
   return 0;
}

void rnpu_rknn_free(struct rnpu_rknn_model *m)
{
   if (!m) return;
   for (unsigned i = 0; i < m->op_count; i++) {
      free(m->ops[i].brdma_data);
      free(m->ops[i].biases);
   }
   free(m->ops);
   free(m->file_data);
   memset(m, 0, sizeof(*m));
}

int rnpu_rknn_match_op(const struct rnpu_rknn_model *m,
                       const int32_t *biases, unsigned oc)
{
   /* Match by bias values. RKNN BRDMA biases are in a different format
    * (they include bias correction with RKNN's own izp calculation),
    * so exact match may not work. Use ratio-based matching:
    * compare signs and relative magnitudes. */

   int best = -1;
   int best_score = -1;

   for (unsigned i = 0; i < m->op_count; i++) {
      if (m->ops[i].matched) continue;
      if (!m->ops[i].brdma_data) continue;

      /* RKNN pads OC to multiples of 8. Our oc should be <= rknn's oc. */
      unsigned rknn_oc = m->ops[i].output_channels;
      /* rknn_oc is oc_pad (padded to 8), our oc is real OC.
       * Compute our expected oc_pad and compare. */
      unsigned our_oc_pad = oc;
      if (our_oc_pad < 32) our_oc_pad = 32;
      our_oc_pad = ((our_oc_pad + 15) / 16) * 16;
      if (rknn_oc != our_oc_pad) continue;

      /* Compare bias signs as a fingerprint */
      const int32_t *rknn_biases = m->ops[i].biases;
      unsigned check = (oc < m->ops[i].bias_count) ? oc : m->ops[i].bias_count;
      if (check < 2) continue;

      int score = 0;
      for (unsigned j = 0; j < check && j < 32; j++) {
         /* Match if same sign and similar magnitude (within 2x) */
         if (biases[j] == 0 && rknn_biases[j] == 0) { score += 2; continue; }
         if ((biases[j] > 0) == (rknn_biases[j] > 0)) score++;
         int64_t a = (int64_t)biases[j], b = (int64_t)rknn_biases[j];
         if (a != 0 && b != 0) {
            double ratio = (double)a / (double)b;
            if (ratio > 0.5 && ratio < 2.0) score++;
         }
      }

      if (score > best_score) {
         best_score = score;
         best = (int)i;
      }
   }

   /* Require decent match */
   if (best >= 0 && best_score >= 4)
      return best;

   /* Fallback: first unmatched with matching oc_pad */
   unsigned our_oc_pad = oc;
   if (our_oc_pad < 32) our_oc_pad = 32;
   our_oc_pad = ((our_oc_pad + 15) / 16) * 16;
   for (unsigned i = 0; i < m->op_count; i++) {
      if (m->ops[i].matched) continue;
      if (!m->ops[i].brdma_data) continue;
      if (m->ops[i].output_channels == our_oc_pad)
         return (int)i;
   }

   return -1;
}
