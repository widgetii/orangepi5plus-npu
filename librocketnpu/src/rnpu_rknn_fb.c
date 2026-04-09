/*
 * Direct .rknn model file parser
 * Extracts pre-compiled NPU data from RKNN FlatBuffer format.
 *
 * SPDX-License-Identifier: MIT
 */

#include "rnpu_rknn_fb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * Minimal FlatBuffer reader
 * ====================================================================== */

static inline uint16_t fb_u16(const uint8_t *b, uint32_t p) {
   return b[p] | ((uint16_t)b[p+1] << 8);
}
static inline uint32_t fb_u32(const uint8_t *b, uint32_t p) {
   return b[p] | ((uint32_t)b[p+1]<<8) | ((uint32_t)b[p+2]<<16) | ((uint32_t)b[p+3]<<24);
}
static inline int32_t fb_i32(const uint8_t *b, uint32_t p) {
   return (int32_t)fb_u32(b, p);
}
static inline uint64_t fb_u64(const uint8_t *b, uint32_t p) {
   return (uint64_t)fb_u32(b,p) | ((uint64_t)fb_u32(b,p+4) << 32);
}
static inline uint32_t fb_follow(const uint8_t *b, uint32_t p) {
   return p + fb_u32(b, p);
}

/* Field offset within a table. Returns 0 if absent. */
static uint32_t fb_field(const uint8_t *b, uint32_t table, int field) {
   uint32_t vt = table - fb_i32(b, table);
   uint16_t vt_size = fb_u16(b, vt);
   uint32_t fo = 4 + field * 2;
   if (fo + 2 > vt_size) return 0;
   uint16_t off = fb_u16(b, vt + fo);
   return off ? table + off : 0;
}

/* Read byte vector: pointer + length */
static const uint8_t *fb_bytes(const uint8_t *b, uint32_t fpos, uint32_t *len) {
   uint32_t vec = fb_follow(b, fpos);
   *len = fb_u32(b, vec);
   return b + vec + 4;
}

/* ======================================================================
 * RKNN model parser
 * ====================================================================== */

#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

int rknn_parse_model(const char *path, struct rknn_parsed *out)
{
   memset(out, 0, sizeof(*out));

   FILE *f = fopen(path, "rb");
   if (!f) { fprintf(stderr, "rknn_parse: cannot open %s\n", path); return -1; }
   fseek(f, 0, SEEK_END);
   long file_size = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint8_t *file_data = malloc(file_size);
   if (!file_data || fread(file_data, 1, file_size, f) != (size_t)file_size) {
      free(file_data); fclose(f); return -1;
   }
   fclose(f);

   if (file_size < 0x50 || memcmp(file_data, "RKNN", 4) != 0) {
      fprintf(stderr, "rknn_parse: invalid RKNN magic\n");
      free(file_data); return -1;
   }

   uint64_t version = fb_u64(file_data, 8);
   uint32_t fb_off = (version > 1) ? 0x40 : 0x18;
   const uint8_t *fb = file_data + fb_off;
   uint32_t fb_size = file_size - fb_off;
   uint32_t root = fb_u32(fb, 0);

   int wt_field = (version > 5) ? 20 : 4;
   uint32_t wt_fpos = fb_field(fb, root, wt_field);
   if (!wt_fpos) {
      fprintf(stderr, "rknn_parse: no weight_data field\n");
      free(file_data); return -1;
   }

   uint32_t n_entries = fb_u32(fb, fb_follow(fb, wt_fpos));
   uint32_t wt_first = fb_follow(fb, wt_fpos) + 4;

   /* ================================================================
    * Pass 1: Collect all data blobs from weight_data entries.
    * Skip entry[0] (type=1, empty root). Entries 1..N-1 contain
    * the actual data that gets concatenated into BO[1].
    * ================================================================ */
   struct blob {
      const uint8_t *data;
      uint32_t len;
      uint8_t type;
      int entry_idx;
   } blobs[256];
   unsigned n_blobs = 0;

   for (unsigned i = 0; i < n_entries && i < 256; i++) {
      uint32_t entry = fb_follow(fb, wt_first + i * 4);
      if (entry + 70 > fb_size) continue;
      uint8_t type_byte = fb[entry + 66];

      uint32_t fo0 = fb_field(fb, entry, 0);
      if (!fo0) continue;

      uint32_t data_len;
      const uint8_t *data_ptr = fb_bytes(fb, fo0, &data_len);
      if (!data_ptr || data_len == 0) continue;

      blobs[n_blobs].data = data_ptr;
      blobs[n_blobs].len = data_len;
      blobs[n_blobs].type = type_byte;
      blobs[n_blobs].entry_idx = i;
      n_blobs++;
   }

   /* ================================================================
    * Pass 2: Assemble BO[1] (weight+regcmd+metadata+taskbo).
    * RKNN concatenates all blobs in order with 64-byte alignment.
    * Identify regcmd (largest type=6) and task BO (size%40==0 with
    * valid enable_mask) by content after assembly.
    * ================================================================ */
   /* First compute total size with alignment */
   uint32_t bo1_size = 0;
   uint32_t *blob_offsets = calloc(n_blobs, sizeof(uint32_t));
   for (unsigned i = 0; i < n_blobs; i++) {
      /* Align to 64 bytes (matching RKNN's observed alignment) */
      bo1_size = ALIGN_UP(bo1_size, 64);
      blob_offsets[i] = bo1_size;
      bo1_size += blobs[i].len;
   }
   bo1_size = ALIGN_UP(bo1_size, 4096); /* Page-align total */

   uint8_t *bo1_data = calloc(1, bo1_size);
   for (unsigned i = 0; i < n_blobs; i++)
      memcpy(bo1_data + blob_offsets[i], blobs[i].data, blobs[i].len);

   /* Find regcmd and task BO within assembled data */
   uint32_t rc_offset = 0, rc_size = 0;
   uint32_t tb_offset = 0, tb_size = 0;

   for (unsigned i = 0; i < n_blobs; i++) {
      if (blobs[i].type != 6) continue;

      /* Task BO: size divisible by 40, first entry has valid enable_mask */
      if (blobs[i].len >= 40 && blobs[i].len % 40 == 0) {
         uint32_t em = fb_u32(bo1_data, blob_offsets[i] + 8);
         if (em == 0x1d || em == 0x18 || em == 0x60) {
            tb_offset = blob_offsets[i];
            tb_size = blobs[i].len;
         }
      }
      /* Regcmd: largest type=6 (excluding task BO) */
      if (blobs[i].len > rc_size && blobs[i].len != tb_size) {
         rc_offset = blob_offsets[i];
         rc_size = blobs[i].len;
      }
   }

   if (!rc_size || !tb_size) {
      fprintf(stderr, "rknn_parse: regcmd(%u) or taskbo(%u) not found\n",
              rc_size, tb_size);
      free(bo1_data); free(blob_offsets); free(file_data);
      return -1;
   }

   out->wt_rc_data = bo1_data;
   out->wt_rc_size = bo1_size;
   out->rc_start_offset = rc_offset;

   /* Store blob offsets for DMA patching */
   out->n_blob_offsets = n_blobs;
   for (unsigned i = 0; i < n_blobs && i < 64; i++) {
      out->blob_offsets[i] = blob_offsets[i];
      out->blob_sizes[i] = blobs[i].len;
      out->blob_types[i] = blobs[i].type;
   }

   /* ================================================================
    * Parse task BO — extract first N tasks for single-core mode.
    * The .rknn stores tasks for all cores; single-core uses a subset.
    * Heuristic: stop when task pattern repeats (multi-core duplicate).
    * ================================================================ */
   uint32_t total_tasks = tb_size / 40;

   /* Find single-core task count: detect when task sequence repeats.
    * RKNN stores [core0_tasks][core1_tasks][core2_tasks].
    * For 3 cores: total = 3 * single_core_count (approximately). */
   uint32_t sc_tasks = total_tasks;
   if (total_tasks > 3) {
      /* Try total/3, total/2, check if tasks repeat */
      for (int div = 3; div >= 2; div--) {
         if (total_tasks % div != 0) continue;
         uint32_t chunk = total_tasks / div;
         /* Check if chunk 0 and chunk 1 have same (op, enable, amt) pattern */
         int match = 1;
         for (uint32_t t = 0; t < chunk && match; t++) {
            uint32_t *a = (uint32_t *)(bo1_data + tb_offset + t * 40);
            uint32_t *b = (uint32_t *)(bo1_data + tb_offset + (chunk + t) * 40);
            /* Compare op_idx[1] and enable_mask[2] */
            if (a[1] != b[1] || a[2] != b[2]) match = 0;
         }
         if (match) {
            /* But also verify they DON'T share regcmd offsets —
             * true multi-core tasks have different rc_addr values */
            uint64_t rc0 = fb_u64(bo1_data, tb_offset + 32);
            uint64_t rc1 = fb_u64(bo1_data, tb_offset + chunk * 40 + 32);
            if (rc0 != rc1) {
               sc_tasks = chunk;
               break;
            }
         }
      }
   }

   /* For this simple model: if we can't determine, try N/2 then N */
   /* Actually check the submit from the .rknn metadata entry[3] (type=4) */
   /* For now: use a simpler heuristic — count unique regcmd offsets */
   {
      uint32_t unique_rc = 0;
      uint64_t seen_rc[256];
      for (uint32_t t = 0; t < total_tasks && t < 256; t++) {
         uint64_t rc = fb_u64(bo1_data, tb_offset + t * 40 + 32);
         int found = 0;
         for (uint32_t j = 0; j < unique_rc; j++)
            if (seen_rc[j] == rc) { found = 1; break; }
         if (!found && unique_rc < 256) seen_rc[unique_rc++] = rc;
      }
      /* If fewer unique offsets than tasks, multi-core tasks share regcmd.
       * The actual task count per submit ≈ tasks that reference unique regcmd
       * plus those that share. For single-core: use all with first-seen offsets. */
   }

   /* Compute sc_count and single-core task count BEFORE copying task data.
    * sc_count: first CONV→REFORMAT→CONV block.
    * single_core_tasks: tasks before multi-core duplicates. */
   uint32_t sc_count = 0;
   {
      /* Temp task metadata for analysis */
      int seen_reformat = 0;
      for (uint32_t t = 0; t < total_tasks; t++) {
         uint32_t em = fb_u32(bo1_data, tb_offset + t * 40 + 8);
         if (em == 0x18) seen_reformat = 1;
         else if (seen_reformat && em == 0x1d) { sc_count = t + 1; break; }
      }
      if (sc_count == 0) sc_count = total_tasks;
   }

   uint32_t single_core_tasks = total_tasks;
   if (total_tasks > sc_count * 2) {
      uint32_t max_core0_rc = 0;
      for (uint32_t t = 0; t < sc_count; t++) {
         uint64_t rc = fb_u64(bo1_data, tb_offset + t * 40 + 32) + rc_offset;
         if ((uint32_t)rc > max_core0_rc) max_core0_rc = (uint32_t)rc;
      }
      single_core_tasks = 0;
      for (uint32_t t = 0; t < total_tasks; t++) {
         uint64_t rc = fb_u64(bo1_data, tb_offset + t * 40 + 32) + rc_offset;
         if ((uint32_t)rc > max_core0_rc + 4096) break;
         single_core_tasks = t + 1;
      }
   }

   out->task_count = single_core_tasks;
   out->task_bo_data = malloc(single_core_tasks * 40);
   out->task_bo_size = single_core_tasks * 40;
   memcpy(out->task_bo_data, bo1_data + tb_offset, single_core_tasks * 40);

   /* Fix task rc_addr: template has offsets relative to regcmd blob start.
    * Add the regcmd position within BO[1] so rc_addr = BO offset. */
   struct __attribute__((packed)) task_entry {
      uint32_t f[8];
      uint64_t regcmd_addr;
   };
   struct task_entry *tasks = (struct task_entry *)out->task_bo_data;
   for (uint32_t t = 0; t < single_core_tasks; t++)
      tasks[t].regcmd_addr += rc_offset;

   /* Build task metadata array */
   out->tasks = calloc(single_core_tasks, sizeof(*out->tasks));
   for (uint32_t t = 0; t < single_core_tasks; t++) {
      uint32_t *fld = (uint32_t *)(out->task_bo_data + t * 40);
      out->tasks[t].rc_offset = (uint32_t)tasks[t].regcmd_addr;
      out->tasks[t].rc_amount = fld[6];
      out->tasks[t].enable_mask = fld[2];
      out->tasks[t].op_idx = fld[1];
   }

   /* ================================================================
    * Build submit segments.
    * RKNN uses a single submit per NPU subgraph with PINGPONG flag.
    * sc_count = first CONV block (tasks 0..first_reformat_end).
    * The HW chains through ALL tasks via PC_BASE_ADDRESS pointers.
    * ================================================================ */
   out->segments = calloc(1, sizeof(*out->segments));
   out->segment_count = 1;

   /* sc_count and single_core_tasks already computed above */
   out->segments[0].flags = 0x5; /* PC + PINGPONG */
   out->segments[0].sc_start = 0;
   out->segments[0].sc_count = sc_count;
   out->segments[0].task_number = single_core_tasks;

   /* ================================================================
    * Extract BO allocation sizes.
    * BO[0] = task BO, BO[1] = wt+rc, BO[2] = activation,
    * BO[3] = input, BO[4+] = output.
    * Sizes come from type=4 metadata entry.
    * ================================================================ */
   for (unsigned i = 0; i < n_blobs; i++) {
      if (blobs[i].type == 4 && blobs[i].len >= 16) {
         /* Metadata entry: input_size at offset 8 (uint32) */
         out->input_size = fb_u32(blobs[i].data, 8);
      }
      if (blobs[i].type == 6 && blobs[i].len == 16 && blobs[i].entry_idx > 0) {
         /* Output info: output_count at +8 */
         uint32_t n_out = fb_u32(blobs[i].data, 8);
         if (n_out > 0 && n_out <= 8)
            out->output_count = n_out;
      }
   }

   /* Activation BO size: from entry[5] type=6 (memory layout metadata).
    * The value at offset +32 is the activation buffer size. */
   uint32_t act_bo_size = 0;
   for (unsigned i = 0; i < n_blobs; i++) {
      if (blobs[i].type == 6 && blobs[i].len == 64) {
         act_bo_size = fb_u32(blobs[i].data, 32);
         break;
      }
   }
   if (act_bo_size == 0) act_bo_size = 65536; /* fallback */

   out->bo_count = 5;
   out->bos = calloc(out->bo_count, sizeof(*out->bos));
   out->bos[0].size = ALIGN_UP(total_tasks * 40, 4096);
   out->bos[1].size = bo1_size;
   out->bos[2].size = ALIGN_UP(act_bo_size + 0x10000, 4096); /* extra padding for safety */
   out->bos[3].size = ALIGN_UP(out->input_size, 4096);
   out->bos[4].size = 4096;

   fprintf(stderr, "rknn_parse: %u tasks (of %u total), %u bytes BO[1], "
           "%u regcmd, %u segments, input=%u\n",
           sc_tasks, total_tasks, bo1_size, rc_size,
           out->segment_count, out->input_size);

   free(blob_offsets);
   free(file_data);
   return 0;
}

void rknn_parsed_free(struct rknn_parsed *p)
{
   free(p->wt_rc_data);
   free(p->task_bo_data);
   free(p->tasks);
   free(p->bos);
   free(p->segments);
   memset(p, 0, sizeof(*p));
}
