/*
 * Capture RKNN vendor runtime golden output for comparison.
 * Links against librknnrt.so (vendor runtime on Orange Pi vendor kernel).
 *
 * Usage: ./capture_rknn_golden <model.rknn> [output_prefix]
 *   Saves each output tensor as <prefix>_out_N.bin
 *
 * Build: gcc -o capture_rknn_golden capture_rknn_golden.c -lrknnrt -lm
 *
 * Note: must run on vendor kernel with RKNN runtime installed.
 * The model must be in .rknn format (not .tflite — convert with rknn-toolkit2).
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

/* RKNN API types — minimal subset needed for inference */
typedef uint64_t rknn_context;

typedef struct {
   uint32_t id;
   uint32_t size;
   uint32_t n_dims;
   uint32_t dims[16];
   char name[256];
   uint32_t n_elems;
   uint32_t type;
   uint32_t qnt_type;
   int8_t fl;
   uint32_t zp;
   float scale;
   uint32_t w_stride;
   uint32_t size_with_stride;
   uint8_t pass_through;
   uint8_t h_stride;
} rknn_tensor_attr;

typedef struct {
   uint8_t *buf;
   uint32_t size;
   int32_t  index;
   uint32_t type;
   uint32_t fmt;
   uint8_t  pass_through;
} rknn_output;

typedef struct {
   void *buf;
   uint32_t size;
   uint8_t  pass_through;
   uint32_t type;
   uint32_t fmt;
   uint32_t index;
} rknn_input;

typedef struct {
   uint32_t model_channel;
   uint32_t model_width;
   uint32_t model_height;
   uint32_t n_input;
   uint32_t n_output;
} rknn_input_output_num;

/* RKNN API function pointers */
typedef int (*rknn_init_fn)(rknn_context *, void *, uint32_t, uint32_t, void *);
typedef int (*rknn_query_fn)(rknn_context, int, void *, uint32_t);
typedef int (*rknn_inputs_set_fn)(rknn_context, uint32_t, rknn_input *);
typedef int (*rknn_run_fn)(rknn_context, void *);
typedef int (*rknn_outputs_get_fn)(rknn_context, uint32_t, rknn_output *, void *);
typedef int (*rknn_outputs_release_fn)(rknn_context, uint32_t, rknn_output *);
typedef int (*rknn_destroy_fn)(rknn_context);

#define RKNN_QUERY_IN_OUT_NUM     0
#define RKNN_QUERY_INPUT_ATTR     1
#define RKNN_QUERY_OUTPUT_ATTR    2
#define RKNN_TENSOR_UINT8         2
#define RKNN_TENSOR_INT8          3
#define RKNN_TENSOR_NHWC          0
#define RKNN_FLAG_PRIOR_HIGH      (1 << 0)

int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "Usage: %s <model.rknn> [output_prefix]\n", argv[0]);
      return 1;
   }
   const char *model_path = argv[1];
   const char *prefix = argc > 2 ? argv[2] : "rknn_golden";

   /* Load RKNN runtime dynamically */
   void *lib = dlopen("librknnrt.so", RTLD_NOW);
   if (!lib) {
      fprintf(stderr, "Cannot load librknnrt.so: %s\n", dlerror());
      fprintf(stderr, "Make sure you're on the vendor kernel with RKNN installed.\n");
      return 1;
   }

   rknn_init_fn p_init = dlsym(lib, "rknn_init");
   rknn_query_fn p_query = dlsym(lib, "rknn_query");
   rknn_inputs_set_fn p_inputs_set = dlsym(lib, "rknn_inputs_set");
   rknn_run_fn p_run = dlsym(lib, "rknn_run");
   rknn_outputs_get_fn p_outputs_get = dlsym(lib, "rknn_outputs_get");
   rknn_outputs_release_fn p_outputs_release = dlsym(lib, "rknn_outputs_release");
   rknn_destroy_fn p_destroy = dlsym(lib, "rknn_destroy");

   if (!p_init || !p_query || !p_inputs_set || !p_run || !p_outputs_get) {
      fprintf(stderr, "Missing RKNN API symbols\n");
      dlclose(lib);
      return 1;
   }

   /* Load model file */
   FILE *mf = fopen(model_path, "rb");
   if (!mf) { perror("fopen model"); return 1; }
   fseek(mf, 0, SEEK_END);
   size_t msz = ftell(mf);
   fseek(mf, 0, SEEK_SET);
   void *mdata = malloc(msz);
   fread(mdata, 1, msz, mf);
   fclose(mf);

   /* Init RKNN context */
   rknn_context ctx = 0;
   int ret = p_init(&ctx, mdata, msz, RKNN_FLAG_PRIOR_HIGH, NULL);
   free(mdata);
   if (ret != 0) {
      fprintf(stderr, "rknn_init failed: %d\n", ret);
      dlclose(lib);
      return 1;
   }

   /* Query I/O */
   rknn_input_output_num io_num;
   p_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
   printf("Model: %u inputs, %u outputs\n", io_num.n_input, io_num.n_output);

   /* Query input attrs */
   for (uint32_t i = 0; i < io_num.n_input; i++) {
      rknn_tensor_attr attr = { .id = i };
      p_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
      printf("Input %u: %s, dims=[", i, attr.name);
      for (uint32_t d = 0; d < attr.n_dims; d++)
         printf("%s%u", d ? "," : "", attr.dims[d]);
      printf("], type=%u, zp=%u, scale=%.6f\n", attr.type, attr.zp, attr.scale);
   }

   /* Query output attrs */
   rknn_tensor_attr *out_attrs = calloc(io_num.n_output, sizeof(rknn_tensor_attr));
   for (uint32_t i = 0; i < io_num.n_output; i++) {
      out_attrs[i].id = i;
      p_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
      printf("Output %u: %s, dims=[", i, out_attrs[i].name);
      for (uint32_t d = 0; d < out_attrs[i].n_dims; d++)
         printf("%s%u", d ? "," : "", out_attrs[i].dims[d]);
      printf("], type=%u, zp=%u, scale=%.6f, size=%u\n",
             out_attrs[i].type, out_attrs[i].zp, out_attrs[i].scale,
             out_attrs[i].size);
   }

   /* Set input: all-128 (matches zero-point for uint8, zero for int8+128) */
   rknn_tensor_attr in_attr = { .index = 0 };
   p_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

   uint32_t in_size = in_attr.n_elems;
   uint8_t *input = malloc(in_size);
   memset(input, 128, in_size);

   rknn_input inputs[1] = {{
      .buf = input,
      .size = in_size,
      .pass_through = 0,
      .type = RKNN_TENSOR_UINT8,
      .fmt = RKNN_TENSOR_NHWC,
      .index = 0,
   }};
   ret = p_inputs_set(ctx, 1, inputs);
   if (ret != 0) {
      fprintf(stderr, "rknn_inputs_set failed: %d\n", ret);
      goto cleanup;
   }

   /* Run inference */
   ret = p_run(ctx, NULL);
   if (ret != 0) {
      fprintf(stderr, "rknn_run failed: %d\n", ret);
      goto cleanup;
   }
   printf("Inference completed successfully.\n");

   /* Get outputs (want_float=0 to get raw int8/uint8) */
   rknn_output *outputs = calloc(io_num.n_output, sizeof(rknn_output));
   for (uint32_t i = 0; i < io_num.n_output; i++)
      outputs[i].index = i;
   /* pass_through=1 means return raw NPU output without dequantization */

   ret = p_outputs_get(ctx, io_num.n_output, outputs, NULL);
   if (ret != 0) {
      fprintf(stderr, "rknn_outputs_get failed: %d\n", ret);
      goto cleanup;
   }

   /* Save outputs */
   for (uint32_t i = 0; i < io_num.n_output; i++) {
      char fname[512];
      snprintf(fname, sizeof(fname), "%s_out_%u.bin", prefix, i);
      FILE *f = fopen(fname, "wb");
      if (f) {
         fwrite(outputs[i].buf, 1, outputs[i].size, f);
         fclose(f);
      }
      printf("Output %u: %u bytes → %s\n", i, outputs[i].size, fname);

      /* Stats */
      uint8_t *d = outputs[i].buf;
      int mn = 255, mx = 0, unique[256] = {0};
      for (uint32_t j = 0; j < outputs[i].size; j++) {
         if (d[j] < mn) mn = d[j];
         if (d[j] > mx) mx = d[j];
         unique[d[j]]++;
      }
      int ucnt = 0;
      for (int j = 0; j < 256; j++) if (unique[j]) ucnt++;
      printf("  min=%d max=%d unique=%d\n", mn, mx, ucnt);
   }

   p_outputs_release(ctx, io_num.n_output, outputs);
   free(outputs);

cleanup:
   free(input);
   free(out_attrs);
   p_destroy(ctx);
   dlclose(lib);
   return ret;
}
