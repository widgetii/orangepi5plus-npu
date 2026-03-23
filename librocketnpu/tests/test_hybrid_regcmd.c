/*
 * Binary search tool for per-channel regcmd on RKNPU.
 * Tests register groups to find which cause NPU hangs.
 *
 * Usage: ./test_hybrid_regcmd <model.tflite> [mask_hex] [golden.bin]
 *   mask_hex: hex bitmask of per-channel register overrides (default: 0 = standard)
 *   golden.bin: golden output to compare against
 *
 * Environment: RNPU_HYBRID_MASK=0xHHHH overrides mask_hex argument.
 *
 * Predefined groups (from rnpu_regcmd.h):
 *   A=0x0842 (DPU output)  B=0x062C (bias path)
 *   C=0x0191 (post-proc)   D=0x7000 (input/misc)
 *   E=0x38000 (extra)      ALL=0x3FFFF
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/rocketnpu.h"
#include "../src/rnpu_regcmd.h"

static double now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const char *bit_names[] = {
   [0]  = "DPU_DATA_FORMAT",
   [1]  = "DPU_DATA_CUBE_CHANNEL",
   [2]  = "DPU_BS_CFG",
   [3]  = "DPU_BS_ALU_CFG",
   [4]  = "DPU_BS_RELUX_CMP",
   [5]  = "DPU_BS_OW_OP",
   [6]  = "DPU_WDMA_SIZE_0",
   [7]  = "DPU_BN_CFG",
   [8]  = "DPU_EW_CFG",
   [9]  = "BRDMA_CFG",
   [10] = "BS_BASE_ADDR",
   [11] = "DPU_SURFACE_ADD",
   [12] = "CORE_CLIP_TRUNCATE",
   [13] = "CNA_CVT_CON5",
   [14] = "CNA_PAD_CON1",
   [15] = "RDMA_DATA_CUBE_CH",
   [16] = "CNA_CONV_CON2",
   [17] = "RDMA_WEIGHT",
};

static void print_mask(uint32_t mask)
{
   printf("mask=0x%05x [", mask);
   int first = 1;
   for (int i = 0; i < 18; i++) {
      if (mask & (1u << i)) {
         if (!first) printf(",");
         printf("%s", bit_names[i]);
         first = 0;
      }
   }
   printf("]\n");
}

static int compare_output(const uint8_t *a, const uint8_t *b, size_t n,
                           const char *label)
{
   int max_diff = 0;
   int diff_count = 0;
   long sum_diff = 0;
   for (size_t i = 0; i < n; i++) {
      int d = abs((int)a[i] - (int)b[i]);
      if (d > max_diff) max_diff = d;
      if (d > 0) { diff_count++; sum_diff += d; }
   }
   printf("  %s: max_diff=%d, diffs=%d/%zu (%.1f%%), mean_diff=%.2f\n",
          label, max_diff, diff_count, n,
          100.0 * diff_count / n,
          diff_count ? (double)sum_diff / diff_count : 0.0);
   return max_diff;
}

int main(int argc, char **argv)
{
   const char *model_path = argc > 1 ? argv[1] : "model.tflite";
   uint32_t mask = 0;
   const char *golden_path = NULL;

   /* Parse mask from arg or env */
   const char *env_mask = getenv("RNPU_HYBRID_MASK");
   if (env_mask) {
      mask = strtoul(env_mask, NULL, 0);
   } else if (argc > 2) {
      mask = strtoul(argv[2], NULL, 0);
   }
   if (argc > 3) golden_path = argv[3];

   /* Special mode: "scan" — test each bit individually */
   int scan_mode = 0;
   if (argc > 2 && strcmp(argv[2], "scan") == 0) {
      scan_mode = 1;
      printf("SCAN MODE: testing each bit individually\n");
   }
   /* Group scan: test each group */
   int group_mode = 0;
   if (argc > 2 && strcmp(argv[2], "groups") == 0) {
      group_mode = 1;
      printf("GROUP MODE: testing each register group\n");
   }

   int fd = rnpu_open(NULL);
   if (fd < 0) return 1;

   if (scan_mode || group_mode) {
      /* Load model once */
      uint32_t test_masks[32];
      const char *test_names[32];
      int test_count = 0;

      if (group_mode) {
         test_masks[test_count] = HYBRID_GROUP_A; test_names[test_count] = "GROUP_A (DPU output)"; test_count++;
         test_masks[test_count] = HYBRID_GROUP_B; test_names[test_count] = "GROUP_B (bias path)"; test_count++;
         test_masks[test_count] = HYBRID_GROUP_C; test_names[test_count] = "GROUP_C (post-proc)"; test_count++;
         test_masks[test_count] = HYBRID_GROUP_D; test_names[test_count] = "GROUP_D (input/misc)"; test_count++;
         test_masks[test_count] = HYBRID_GROUP_E; test_names[test_count] = "GROUP_E (extra)"; test_count++;
         test_masks[test_count] = HYBRID_ALL;     test_names[test_count] = "ALL";                 test_count++;
      } else {
         for (int i = 0; i < 18; i++) {
            test_masks[test_count] = 1u << i;
            test_names[test_count] = bit_names[i];
            test_count++;
         }
      }

      /* First run: standard (mask=0) as baseline */
      printf("\n=== BASELINE (standard, mask=0x0) ===\n");
      rnpu_hybrid_mask = UINT32_MAX; /* standard path */
      rnpu_model_t *m = rnpu_model_load(fd, model_path);
      if (!m) { rnpu_close(fd); return 1; }

      int w, h, c;
      rnpu_get_input_dims(m, &w, &h, &c);
      size_t isz = w * h * c;
      uint8_t *input = malloc(isz);
      memset(input, 128, isz); /* all-128 = zero after ZP subtraction */

      rnpu_invoke(m, input, isz);

      /* Save baseline output */
      int n_out = rnpu_output_count(m);
      uint8_t **baseline = malloc(n_out * sizeof(uint8_t *));
      size_t *out_sizes = malloc(n_out * sizeof(size_t));
      for (int i = 0; i < n_out; i++) {
         int ow, oh, oc;
         rnpu_get_output_dims(m, i, &ow, &oh, &oc);
         out_sizes[i] = ow * oh * oc;
         baseline[i] = malloc(out_sizes[i]);
         rnpu_get_output(m, i, baseline[i], out_sizes[i]);
         printf("  Output %d: %dx%dx%d (%zu bytes)\n", i, ow, oh, oc, out_sizes[i]);
      }
      rnpu_model_free(m);

      /* Load golden if provided */
      uint8_t *golden = NULL;
      if (golden_path) {
         FILE *gf = fopen(golden_path, "rb");
         if (gf) {
            golden = malloc(out_sizes[0]);
            fread(golden, 1, out_sizes[0], gf);
            fclose(gf);
            printf("Loaded golden: %s\n", golden_path);
            compare_output(baseline[0], golden, out_sizes[0], "baseline vs golden");
         }
      }

      /* Test each mask */
      for (int t = 0; t < test_count; t++) {
         printf("\n=== TEST: %s (mask=0x%05x) ===\n", test_names[t], test_masks[t]);
         rnpu_hybrid_mask = test_masks[t];
         m = rnpu_model_load(fd, model_path);
         if (!m) {
            printf("  FAILED: model load error\n");
            continue;
         }

         double t0 = now_ms();
         int ret = rnpu_invoke(m, input, isz);
         double dt = now_ms() - t0;

         if (ret) {
            printf("  FAILED: invoke returned %d (%.1f ms) — HANG?\n", ret, dt);
            rnpu_model_free(m);
            continue;
         }

         printf("  OK (%.1f ms)\n", dt);
         for (int i = 0; i < n_out; i++) {
            uint8_t *out = malloc(out_sizes[i]);
            rnpu_get_output(m, i, out, out_sizes[i]);

            char label[64];
            snprintf(label, sizeof(label), "out%d vs baseline", i);
            compare_output(out, baseline[i], out_sizes[i], label);

            if (golden && i == 0) {
               compare_output(out, golden, out_sizes[i], "out0 vs golden");
            }
            free(out);
         }
         rnpu_model_free(m);
      }

      /* Cleanup */
      for (int i = 0; i < n_out; i++) free(baseline[i]);
      free(baseline);
      free(out_sizes);
      free(golden);
      free(input);
   } else {
      /* Single-mask mode */
      printf("Testing with ");
      if (mask == 0) printf("STANDARD regcmd (mask=0)\n");
      else { printf("HYBRID regcmd "); print_mask(mask); }

      rnpu_hybrid_mask = (mask == 0) ? UINT32_MAX : mask;
      rnpu_model_t *m = rnpu_model_load(fd, model_path);
      if (!m) { rnpu_close(fd); return 1; }

      int w, h, c;
      rnpu_get_input_dims(m, &w, &h, &c);
      size_t isz = w * h * c;
      uint8_t *input = malloc(isz);
      memset(input, 128, isz);

      printf("Invoking...\n");
      double t0 = now_ms();
      int ret = rnpu_invoke(m, input, isz);
      double dt = now_ms() - t0;

      if (ret) {
         printf("FAILED: invoke returned %d (%.1f ms)\n", ret, dt);
      } else {
         printf("OK (%.1f ms)\n", dt);
         int n_out = rnpu_output_count(m);
         for (int i = 0; i < n_out; i++) {
            int ow, oh, oc;
            rnpu_get_output_dims(m, i, &ow, &oh, &oc);
            size_t osz = ow * oh * oc;
            uint8_t *out = malloc(osz);
            rnpu_get_output(m, i, out, osz);

            /* Stats */
            int mn = 255, mx = 0, unique[256] = {0};
            for (size_t j = 0; j < osz; j++) {
               if (out[j] < mn) mn = out[j];
               if (out[j] > mx) mx = out[j];
               unique[out[j]]++;
            }
            int ucnt = 0;
            for (int j = 0; j < 256; j++) if (unique[j]) ucnt++;
            printf("Output %d: %dx%dx%d, min=%d max=%d unique=%d\n",
                   i, ow, oh, oc, mn, mx, ucnt);

            /* Save output */
            char fname[256];
            snprintf(fname, sizeof(fname), "hybrid_out_%d_0x%05x.bin", i, mask);
            FILE *f = fopen(fname, "wb");
            if (f) { fwrite(out, 1, osz, f); fclose(f); }
            printf("  Saved to %s\n", fname);

            /* Compare with golden if provided */
            if (golden_path && i == 0) {
               FILE *gf = fopen(golden_path, "rb");
               if (gf) {
                  uint8_t *golden = malloc(osz);
                  fread(golden, 1, osz, gf);
                  fclose(gf);
                  compare_output(out, golden, osz, "vs golden");
                  free(golden);
               }
            }
            free(out);
         }
      }

      free(input);
      rnpu_model_free(m);
   }

   rnpu_close(fd);
   return 0;
}
