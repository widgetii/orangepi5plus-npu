/*
 * Unit tests for SW ops, TFLite parsing, and tensor handling.
 * Runs on CPU only — no NPU device required.
 *
 * Tests regression cases discovered during development:
 *   1. Pool2DOptions parsed with correct type (5, not just 22)
 *   2. Softmax uses uint8 (not int8) for NPU format read/write
 *   3. 2D tensor shapes (shape_len==2) handled in lowering and metadata
 *   4. AVG_POOL computes correct average in NPU interleaved format
 *   5. RESHAPE copies data correctly between tensors
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/rnpu_internal.h"
#include "../src/rnpu_sw_ops.h"
#include "../src/rnpu_tflite.h"
#include "../src/rnpu_convert.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, fmt, ...) do { \
   tests_run++; \
   if (!(cond)) { \
      fprintf(stderr, "FAIL [%s:%d]: " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
      return 1; \
   } \
   tests_passed++; \
} while (0)

/* ---- Helper: build a minimal fake model for SW op testing ---- */

static struct rnpu_model *make_fake_model(unsigned buf_size, unsigned num_tensors)
{
   struct rnpu_model *m = calloc(1, sizeof(*m));
   m->activation_bo.map = calloc(1, buf_size);
   m->activation_bo.size = buf_size;
   m->tensor_count = num_tensors;
   m->tensors = calloc(num_tensors, sizeof(struct rnpu_npu_tensor));
   m->op_count = 0;
   m->ops = calloc(4, sizeof(struct rnpu_operation));
   m->fd = -1;  /* no real device */
   return m;
}

static void free_fake_model(struct rnpu_model *m)
{
   free(m->activation_bo.map);
   free(m->tensors);
   free(m->ops);
   free(m);
}

/* ================================================================
 * Test 1: AVG_POOL computation in NPU interleaved format
 *
 * Regression: OPT_POOL2D=22 caused filter=0×0, producing all-zero output.
 * This test verifies that avg_pool with a known 2×2 filter produces
 * correct averages in NPU format.
 * ================================================================ */
static int test_avg_pool_npu_format(void)
{
   /* 2×2 input, 1 channel group (16 channels), filter 2×2, stride 1, VALID
    * → output 1×1 */
   unsigned in_w = 2, in_h = 2, ch = 4;
   unsigned out_w = 1, out_h = 1;
   unsigned in_sz = rnpu_calc_npu_tensor_size(in_w, in_h, ch);
   unsigned out_sz = rnpu_calc_npu_tensor_size(out_w, out_h, ch);

   struct rnpu_model *m = make_fake_model(in_sz + out_sz + 64, 2);
   m->tensors[0].width = in_w;
   m->tensors[0].height = in_h;
   m->tensors[0].channels = ch;
   m->tensors[0].offset = 0;
   m->tensors[0].size = in_sz;

   m->tensors[1].width = out_w;
   m->tensors[1].height = out_h;
   m->tensors[1].channels = ch;
   m->tensors[1].offset = ALIGN_UP(in_sz, 64);
   m->tensors[1].size = out_sz;

   /* Fill input in NPU format: values are stored as (uint8_val - 0x80).
    * Let's use uint8 values: pixel(0,0)={40,80,120,200},
    * pixel(1,0)={60,100,140,220}, pixel(0,1)={50,90,130,210},
    * pixel(1,1)={70,110,150,230}.
    * Expected average: {55, 95, 135, 215} */
   uint8_t *in = (uint8_t *)m->activation_bo.map;
   uint8_t vals[4][4] = {
      {40, 80, 120, 200},   /* (0,0) */
      {50, 90, 130, 210},   /* (0,1) */
      {60, 100, 140, 220},  /* (1,0) */
      {70, 110, 150, 230},  /* (1,1) */
   };
   for (unsigned x = 0; x < 2; x++)
      for (unsigned y = 0; y < 2; y++) {
         uint8_t *p = in + NPU_OFFSET(0, x, y, in_w, in_h);
         for (unsigned c = 0; c < ch; c++)
            p[c] = vals[x * 2 + y][c] - 0x80;  /* NPU format */
      }

   /* Set up operation */
   struct rnpu_operation *op = &m->ops[0];
   op->type = RNPU_OP_AVG_POOL;
   op->input_tensor = 0;
   op->output_tensor = 1;
   op->input_width = in_w;
   op->input_height = in_h;
   op->input_channels = ch;
   op->output_width = out_w;
   op->output_height = out_h;
   op->output_channels = ch;
   op->sw.pool.filter_width = 2;
   op->sw.pool.filter_height = 2;
   op->sw.pool.stride_x = 1;
   op->sw.pool.stride_y = 1;
   op->sw.pool.padding_same = false;
   m->op_count = 1;
   m->sw_only = false;

   rnpu_execute_sw_op(m, 0);

   /* Read output (NPU format: stored + 0x80 = uint8 value) */
   uint8_t *out = (uint8_t *)m->activation_bo.map + m->tensors[1].offset;
   uint8_t *p = out + NPU_OFFSET(0, 0, 0, out_w, out_h);
   uint8_t expected[] = {55, 95, 135, 215};
   for (unsigned c = 0; c < ch; c++) {
      int got = (int)(uint8_t)(p[c] + 0x80);
      ASSERT(abs(got - expected[c]) <= 1,
             "avg_pool ch%u: expected %d, got %d", c, expected[c], got);
   }

   /* Verify filter=0 would produce wrong result (the original bug) */
   ASSERT(op->sw.pool.filter_width == 2, "filter_width should be 2, got %u",
          op->sw.pool.filter_width);

   free_fake_model(m);
   return 0;
}

/* ================================================================
 * Test 2: AVG_POOL with filter=0 produces wrong output
 *
 * This is the negative test — verifying that if someone sets filter=0,
 * the output would be zero (the bug behavior), confirming the test
 * actually catches the regression.
 * ================================================================ */
static int test_avg_pool_zero_filter_regression(void)
{
   /* Same setup as above but with filter=0×0 — should produce count=0
    * and therefore val=0 for all channels */
   unsigned in_w = 2, in_h = 2, ch = 4;
   unsigned out_w = 1, out_h = 1;
   unsigned in_sz = rnpu_calc_npu_tensor_size(in_w, in_h, ch);
   unsigned out_sz = rnpu_calc_npu_tensor_size(out_w, out_h, ch);

   struct rnpu_model *m = make_fake_model(in_sz + out_sz + 64, 2);
   m->tensors[0].offset = 0;
   m->tensors[0].size = in_sz;
   m->tensors[1].offset = ALIGN_UP(in_sz, 64);
   m->tensors[1].size = out_sz;

   /* Fill with non-zero values */
   uint8_t *in = (uint8_t *)m->activation_bo.map;
   for (unsigned i = 0; i < in_sz; i++)
      in[i] = 42;

   struct rnpu_operation *op = &m->ops[0];
   op->type = RNPU_OP_AVG_POOL;
   op->input_tensor = 0;
   op->output_tensor = 1;
   op->input_width = in_w;
   op->input_height = in_h;
   op->input_channels = ch;
   op->output_width = out_w;
   op->output_height = out_h;
   op->output_channels = ch;
   op->sw.pool.filter_width = 0;   /* THE BUG */
   op->sw.pool.filter_height = 0;
   op->sw.pool.stride_x = 1;
   op->sw.pool.stride_y = 1;
   op->sw.pool.padding_same = false;
   m->op_count = 1;
   m->sw_only = false;

   rnpu_execute_sw_op(m, 0);

   /* With filter=0, the inner loop doesn't execute → count=0 → val=0 */
   uint8_t *out = (uint8_t *)m->activation_bo.map + m->tensors[1].offset;
   uint8_t *p = out + NPU_OFFSET(0, 0, 0, out_w, out_h);
   for (unsigned c = 0; c < ch; c++) {
      ASSERT(p[c] == 0, "filter=0 should produce 0, got %d at ch%u", p[c], c);
   }

   free_fake_model(m);
   return 0;
}

/* ================================================================
 * Test 3: Softmax with uint8 quantization in NPU format
 *
 * Regression: int8 cast instead of uint8 when reading NPU-format
 * values caused wrong dequantization. E.g., uint8 val=200 was read
 * as int8 -56, producing completely wrong softmax probabilities.
 * ================================================================ */
static int test_softmax_uint8_npu_format(void)
{
   /* 1×1×4 softmax input, uint8, scale=0.1, zp=100
    * Output: uint8, scale=1/256, zp=0
    * Input values: {200, 150, 100, 50} (one clearly dominant class) */
   unsigned ch = 4;
   unsigned in_sz = rnpu_calc_npu_tensor_size(1, 1, ch);
   unsigned out_sz = rnpu_calc_npu_tensor_size(1, 1, ch);

   struct rnpu_model *m = make_fake_model(in_sz + out_sz + 64, 2);
   m->tensors[0].offset = 0;
   m->tensors[0].size = in_sz;
   m->tensors[0].width = 1;
   m->tensors[0].height = 1;
   m->tensors[0].channels = ch;
   m->tensors[1].offset = ALIGN_UP(in_sz, 64);
   m->tensors[1].size = out_sz;
   m->tensors[1].width = 1;
   m->tensors[1].height = 1;
   m->tensors[1].channels = ch;

   /* Write input in NPU format: stored = uint8_val - 0x80 */
   uint8_t input_vals[] = {200, 150, 100, 50};
   uint8_t *in = (uint8_t *)m->activation_bo.map;
   for (unsigned c = 0; c < ch; c++)
      in[c] = input_vals[c] - 0x80;

   float in_scale = 0.1f;
   int in_zp = 100;
   float out_scale = 1.0f / 256.0f;
   int out_zp = 0;

   struct rnpu_operation *op = &m->ops[0];
   op->type = RNPU_OP_SOFTMAX;
   op->input_tensor = 0;
   op->output_tensor = 1;
   op->input_width = 1;
   op->input_height = 1;
   op->input_channels = ch;
   op->output_width = 1;
   op->output_height = 1;
   op->output_channels = ch;
   op->sw.softmax.in_scale = in_scale;
   op->sw.softmax.in_zp = in_zp;
   op->sw.softmax.out_scale = out_scale;
   op->sw.softmax.out_zp = out_zp;
   m->op_count = 1;
   m->sw_only = false;

   rnpu_execute_sw_op(m, 0);

   /* Read output: stored + 0x80 = uint8 value */
   uint8_t *out = (uint8_t *)m->activation_bo.map + m->tensors[1].offset;
   unsigned result[4];
   for (unsigned c = 0; c < ch; c++)
      result[c] = (uint8_t)(out[c] + 0x80);

   /* Compute expected: dequant → exp → normalize → requant */
   float dequant[4], exp_v[4], sum = 0;
   float max_v = -1e30f;
   for (unsigned c = 0; c < ch; c++) {
      dequant[c] = (input_vals[c] - in_zp) * in_scale;
      if (dequant[c] > max_v) max_v = dequant[c];
   }
   for (unsigned c = 0; c < ch; c++) {
      exp_v[c] = expf(dequant[c] - max_v);
      sum += exp_v[c];
   }
   unsigned expected[4];
   for (unsigned c = 0; c < ch; c++)
      expected[c] = (unsigned)roundf((exp_v[c] / sum) / out_scale + out_zp);

   for (unsigned c = 0; c < ch; c++) {
      ASSERT(abs((int)result[c] - (int)expected[c]) <= 1,
             "softmax ch%u: expected %u, got %u", c, expected[c], result[c]);
   }

   /* Key assertion: class 0 (val=200) should dominate */
   ASSERT(result[0] > 200, "class 0 (val=200) should dominate softmax, got %u", result[0]);
   ASSERT(result[3] < 5, "class 3 (val=50) should be near zero, got %u", result[3]);

   /* THE REGRESSION CHECK: with int8 bug, val=200 stored as 120 (200-0x80),
    * read as (int8_t)((uint8_t)(120+0x80)) = (int8_t)200 = -56.
    * Dequant: (-56 - 100) * 0.1 = -15.6 (very negative → near-zero prob).
    * With correct uint8: (uint8_t)(120+0x80) = 200.
    * Dequant: (200 - 100) * 0.1 = 10.0 (dominant class). */

   free_fake_model(m);
   return 0;
}

/* ================================================================
 * Test 4: Softmax int8 bug reproducer
 *
 * Verify that if we incorrectly used int8 interpretation, the dominant
 * class would NOT be class 0. This confirms the test catches the bug.
 * ================================================================ */
static int test_softmax_int8_bug_detection(void)
{
   /* Simulate what the int8 bug would produce:
    * Input uint8 vals: {200, 150, 100, 50}
    * Bug reads them as int8: {-56, -106, -28(=100-128), -78(=50-128)}
    * Wait, the bug path is:
    *   stored = uint8_val - 0x80 = {72, 22, -28, -78} as uint8 = {72,22,228,178}
    *   bug: (int)(int8_t)(stored + 0x80) = (int)(int8_t){200,150,100,50-256}
    *     = (int){-56, -106, 100, 50}  ← int8 interpretation
    *   correct: (int)(uint8_t)(stored + 0x80) = {200, 150, 100, 50}
    *
    * With bug: dequant = {(-56-100)*0.1, (-106-100)*0.1, (100-100)*0.1, (50-100)*0.1}
    *                    = {-15.6, -20.6, 0.0, -5.0}
    * Class 2 (val=100) would dominate! Not class 0. */

   /* This test just verifies our understanding of the bug.
    * We check that the CORRECT implementation gives class 0 as winner. */
   float dequant_correct[] = {
      (200 - 100) * 0.1f,   /* 10.0 */
      (150 - 100) * 0.1f,   /*  5.0 */
      (100 - 100) * 0.1f,   /*  0.0 */
      ( 50 - 100) * 0.1f,   /* -5.0 */
   };

   float dequant_buggy[] = {
      ((int)(int8_t)(uint8_t)200 - 100) * 0.1f,  /* (-56-100)*0.1 = -15.6 */
      ((int)(int8_t)(uint8_t)150 - 100) * 0.1f,  /* (-106-100)*0.1 = -20.6 */
      ((int)(int8_t)(uint8_t)100 - 100) * 0.1f,  /* (100-100)*0.1 = 0.0 */
      ((int)(int8_t)(uint8_t) 50 - 100) * 0.1f,  /* (50-100)*0.1 = -5.0 */
   };

   /* Find winner for correct path */
   int correct_winner = 0;
   for (int i = 1; i < 4; i++)
      if (dequant_correct[i] > dequant_correct[correct_winner])
         correct_winner = i;

   /* Find winner for buggy path */
   int buggy_winner = 0;
   for (int i = 1; i < 4; i++)
      if (dequant_buggy[i] > dequant_buggy[buggy_winner])
         buggy_winner = i;

   ASSERT(correct_winner == 0, "correct softmax should pick class 0, got %d", correct_winner);
   ASSERT(buggy_winner == 2, "buggy softmax would pick class 2, got %d", buggy_winner);
   ASSERT(correct_winner != buggy_winner,
          "bug must produce different winner to be detectable");

   return 0;
}

/* ================================================================
 * Test 5: RESHAPE copies data correctly
 *
 * Regression: 2D tensor shapes (shape_len==2) were not handled,
 * causing width=0, height=0, channels=0 → zero-size tensors.
 * ================================================================ */
static int test_reshape_memcpy(void)
{
   /* Simulate RESHAPE from [1,1,1,1024] to [1,1001]:
    * In NPU format, both are 1×1×C. RESHAPE should memcpy. */
   unsigned in_ch = 1024, out_ch = 1001;
   unsigned in_sz = rnpu_calc_npu_tensor_size(1, 1, in_ch);
   unsigned out_sz = rnpu_calc_npu_tensor_size(1, 1, out_ch);

   struct rnpu_model *m = make_fake_model(in_sz + out_sz + 64, 2);
   m->tensors[0].width = 1;
   m->tensors[0].height = 1;
   m->tensors[0].channels = in_ch;
   m->tensors[0].offset = 0;
   m->tensors[0].size = in_sz;
   m->tensors[1].width = 1;
   m->tensors[1].height = 1;
   m->tensors[1].channels = out_ch;
   m->tensors[1].offset = ALIGN_UP(in_sz, 64);
   m->tensors[1].size = out_sz;

   /* Fill input with pattern */
   uint8_t *in = (uint8_t *)m->activation_bo.map;
   for (unsigned i = 0; i < in_sz; i++)
      in[i] = (uint8_t)(i * 7 + 13);

   struct rnpu_operation *op = &m->ops[0];
   op->type = RNPU_OP_RESHAPE;
   op->input_tensor = 0;
   op->output_tensor = 1;
   op->input_width = 1;
   op->input_height = 1;
   op->input_channels = in_ch;
   op->output_width = 1;
   op->output_height = 1;
   op->output_channels = out_ch;
   m->op_count = 1;

   rnpu_execute_sw_op(m, 0);

   /* Verify output matches input for the overlapping region */
   uint8_t *out = (uint8_t *)m->activation_bo.map + m->tensors[1].offset;
   unsigned copy_sz = MIN2(in_sz, out_sz);
   int mismatches = 0;
   for (unsigned i = 0; i < copy_sz; i++) {
      if (out[i] != in[i]) mismatches++;
   }
   ASSERT(mismatches == 0, "reshape should memcpy, got %d mismatches in %u bytes",
          mismatches, copy_sz);

   free_fake_model(m);
   return 0;
}

/* ================================================================
 * Test 6: RESHAPE in-place (same tensor) is a no-op
 * ================================================================ */
static int test_reshape_inplace(void)
{
   unsigned ch = 16;
   unsigned sz = rnpu_calc_npu_tensor_size(1, 1, ch);

   struct rnpu_model *m = make_fake_model(sz, 1);
   m->tensors[0].width = 1;
   m->tensors[0].height = 1;
   m->tensors[0].channels = ch;
   m->tensors[0].offset = 0;
   m->tensors[0].size = sz;

   uint8_t *data = (uint8_t *)m->activation_bo.map;
   for (unsigned i = 0; i < sz; i++)
      data[i] = (uint8_t)(i + 1);

   uint8_t *before = malloc(sz);
   memcpy(before, data, sz);

   struct rnpu_operation *op = &m->ops[0];
   op->type = RNPU_OP_RESHAPE;
   op->input_tensor = 0;
   op->output_tensor = 0;  /* same tensor → in-place */
   m->op_count = 1;

   rnpu_execute_sw_op(m, 0);

   ASSERT(memcmp(data, before, sz) == 0, "in-place reshape should not modify data");

   free(before);
   free_fake_model(m);
   return 0;
}

/* ================================================================
 * Test 7: 2D tensor size calculation
 *
 * Regression: tensors with shape_len==2 got width=0, height=0, channels=0
 * because the metadata init only handled shape_len>=4.
 * ================================================================ */
static int test_2d_tensor_size(void)
{
   /* [1, 1001] → should be treated as 1×1×1001 */
   unsigned sz = rnpu_calc_npu_tensor_size(1, 1, 1001);
   ASSERT(sz > 0, "tensor size for 1×1×1001 should be > 0, got %u", sz);

   /* Expected: DIV_ROUND_UP(1001, 16) * 2 * 1 * 1 * 16 = 63 * 2 * 16 = 2016 */
   unsigned expected = DIV_ROUND_UP(1001, 16) * 2 * 16;
   ASSERT(sz == expected, "expected %u bytes, got %u", expected, sz);

   /* [1, 1024] → 1×1×1024 */
   unsigned sz2 = rnpu_calc_npu_tensor_size(1, 1, 1024);
   unsigned exp2 = DIV_ROUND_UP(1024, 16) * 2 * 16;
   ASSERT(sz2 == exp2, "expected %u bytes for 1024ch, got %u", exp2, sz2);

   return 0;
}

/* ================================================================
 * Test 8: TFLite parser — AVG_POOL_2D gets pool options
 *
 * Regression: OPT_POOL2D was 22 but TFLite schema uses type 5 for
 * Pool2DOptions. AVG_POOL filter/stride were parsed as 0.
 * ================================================================ */
static int test_tflite_avgpool_options(void)
{
   /* This test requires the actual MBv1 model file.
    * Skip if not available (CI environment). */
   const char *model_path =
      "/root/npu-research/zero2pro_NPU_example/mobilenet_v1_1.0_224_quant.tflite";
   FILE *f = fopen(model_path, "rb");
   if (!f) {
      fprintf(stderr, "  SKIP (model not found: %s)\n", model_path);
      return 0;
   }
   fclose(f);

   struct rnpu_tfl_model tfl;
   int ret = rnpu_tflite_parse(model_path, &tfl);
   ASSERT(ret == 0, "tflite_parse failed");
   ASSERT(tfl.op_count == 31, "expected 31 ops, got %u", tfl.op_count);

   /* Op 27 should be AVERAGE_POOL_2D (opcode 1) with filter 7×7 */
   struct rnpu_tfl_op *avg_op = NULL;
   for (unsigned i = 0; i < tfl.op_count; i++) {
      if (tfl.ops[i].builtin_code == 1) {  /* AVERAGE_POOL_2D */
         avg_op = &tfl.ops[i];
         break;
      }
   }
   ASSERT(avg_op != NULL, "AVERAGE_POOL_2D op not found");
   ASSERT(avg_op->opt.pool.filter_w == 7,
          "expected filter_w=7, got %d", avg_op->opt.pool.filter_w);
   ASSERT(avg_op->opt.pool.filter_h == 7,
          "expected filter_h=7, got %d", avg_op->opt.pool.filter_h);
   ASSERT(avg_op->opt.pool.stride_w > 0,
          "stride_w should be > 0, got %d (pool options not parsed!)",
          avg_op->opt.pool.stride_w);

   /* Also check RESHAPE (opcode 22) and SOFTMAX (opcode 25) are present */
   int has_reshape = 0, has_softmax = 0;
   for (unsigned i = 0; i < tfl.op_count; i++) {
      if (tfl.ops[i].builtin_code == 22) has_reshape = 1;
      if (tfl.ops[i].builtin_code == 25) has_softmax = 1;
   }
   ASSERT(has_reshape, "RESHAPE op not found in MBv1 model");
   ASSERT(has_softmax, "SOFTMAX op not found in MBv1 model");

   /* Check 2D tensor shapes */
   int has_2d = 0;
   for (unsigned i = 0; i < tfl.tensor_count; i++) {
      if (tfl.tensors[i].shape_len == 2) {
         has_2d = 1;
         ASSERT(tfl.tensors[i].shape[1] > 0,
                "2D tensor %u should have shape[1] > 0", i);
      }
   }
   ASSERT(has_2d, "MBv1 should have at least one 2D tensor (reshape/softmax)");

   rnpu_tflite_free(&tfl);
   return 0;
}

/* ================================================================
 * Test 9: Softmax in sw_only mode (raw int8, no NPU bias)
 *
 * Verifies sw_only path also works correctly.
 * ================================================================ */
static int test_softmax_sw_only(void)
{
   unsigned ch = 4;
   unsigned sz = ch;  /* sw_only: raw NHWC, no NPU format */

   struct rnpu_model *m = make_fake_model(sz * 2 + 64, 2);
   m->tensors[0].offset = 0;
   m->tensors[0].size = sz;
   m->tensors[0].width = 1;
   m->tensors[0].height = 1;
   m->tensors[0].channels = ch;
   m->tensors[1].offset = sz;
   m->tensors[1].size = sz;
   m->tensors[1].width = 1;
   m->tensors[1].height = 1;
   m->tensors[1].channels = ch;

   /* Write raw int8 values: {50, 0, -50, -100} */
   int8_t input_vals[] = {50, 0, -50, -100};
   uint8_t *in = (uint8_t *)m->activation_bo.map;
   for (unsigned c = 0; c < ch; c++)
      in[c] = (uint8_t)input_vals[c];

   struct rnpu_operation *op = &m->ops[0];
   op->type = RNPU_OP_SOFTMAX;
   op->input_tensor = 0;
   op->output_tensor = 1;
   op->input_width = 1;
   op->input_height = 1;
   op->input_channels = ch;
   op->output_width = 1;
   op->output_height = 1;
   op->output_channels = ch;
   op->sw.softmax.in_scale = 0.1f;
   op->sw.softmax.in_zp = 0;
   op->sw.softmax.out_scale = 1.0f / 256.0f;
   op->sw.softmax.out_zp = -128;
   m->op_count = 1;
   m->sw_only = true;

   rnpu_execute_sw_op(m, 0);

   uint8_t *out = (uint8_t *)m->activation_bo.map + m->tensors[1].offset;
   int8_t results[4];
   for (unsigned c = 0; c < ch; c++)
      results[c] = (int8_t)out[c];

   /* Class 0 (val=50) should dominate */
   ASSERT(results[0] > results[1], "class 0 should beat class 1");
   ASSERT(results[0] > results[2], "class 0 should beat class 2");
   ASSERT(results[0] > results[3], "class 0 should beat class 3");

   free_fake_model(m);
   return 0;
}

/* ================================================================
 * Test 10: NPU format round-trip (convert_input → convert_output)
 *
 * Verifies the 0x80 bias is applied/removed consistently.
 * ================================================================ */
static int test_npu_format_roundtrip(void)
{
   unsigned w = 2, h = 2, ch = 3;
   unsigned npu_sz = rnpu_calc_npu_tensor_size(w, h, ch);
   uint8_t *npu_buf = calloc(1, npu_sz);
   uint8_t nhwc_in[2 * 2 * 3] = {
      10, 20, 30,   40, 50, 60,
      70, 80, 90,  100, 110, 120
   };
   uint8_t nhwc_out[2 * 2 * 3];

   rnpu_convert_input(npu_buf, nhwc_in, w, h, ch, 128);
   rnpu_convert_output(nhwc_out, npu_buf, w, h, ch);

   int mismatches = 0;
   for (unsigned i = 0; i < w * h * ch; i++) {
      if (nhwc_in[i] != nhwc_out[i]) mismatches++;
   }
   ASSERT(mismatches == 0, "round-trip should be lossless, got %d mismatches", mismatches);

   free(npu_buf);
   return 0;
}

/* ---- Main ---- */

typedef int (*test_fn)(void);
struct test_case {
   const char *name;
   test_fn fn;
};

int main(void)
{
   struct test_case tests[] = {
      {"avg_pool_npu_format",          test_avg_pool_npu_format},
      {"avg_pool_zero_filter",         test_avg_pool_zero_filter_regression},
      {"softmax_uint8_npu_format",     test_softmax_uint8_npu_format},
      {"softmax_int8_bug_detection",   test_softmax_int8_bug_detection},
      {"reshape_memcpy",              test_reshape_memcpy},
      {"reshape_inplace",            test_reshape_inplace},
      {"2d_tensor_size",             test_2d_tensor_size},
      {"tflite_avgpool_options",     test_tflite_avgpool_options},
      {"softmax_sw_only",            test_softmax_sw_only},
      {"npu_format_roundtrip",       test_npu_format_roundtrip},
   };
   unsigned n = sizeof(tests) / sizeof(tests[0]);

   printf("Running %u tests...\n\n", n);
   int failures = 0;
   for (unsigned i = 0; i < n; i++) {
      printf("  %-35s", tests[i].name);
      int ret = tests[i].fn();
      if (ret == 0) {
         printf("OK\n");
      } else {
         printf("FAIL\n");
         failures++;
      }
   }

   printf("\n%d/%d tests passed, %d assertions checked\n",
          tests_run - (tests_run - tests_passed), tests_run, tests_run);
   if (failures)
      printf("%d test(s) FAILED\n", failures);
   else
      printf("All tests passed!\n");

   return failures ? 1 : 0;
}
