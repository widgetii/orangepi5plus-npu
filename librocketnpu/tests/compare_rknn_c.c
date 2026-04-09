/*
 * Byte-identical comparison: RKNN C API vs librocketnpu native cache.
 * Uses want_float=0 to get RKNN's native int8 output (no float roundtrip).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include "rocketnpu.h"

/* RKNN API types (from rknn_api.h) */
typedef uint64_t rknn_context;
typedef struct { uint32_t api_ver; uint32_t drv_ver; char pad[256]; } rknn_sdk_version;
typedef struct {
    uint32_t index; void *buf; uint32_t size;
    uint8_t pass_through; uint8_t type; uint8_t fmt; uint8_t pad;
} rknn_input;
typedef struct {
    uint8_t want_float; uint8_t is_prealloc; uint32_t index;
    void *buf; uint32_t size;
} rknn_output;

/* RKNN function pointers */
typedef int (*fn_rknn_init)(rknn_context*, void*, uint32_t, uint32_t, void*);
typedef int (*fn_rknn_destroy)(rknn_context);
typedef int (*fn_rknn_inputs_set)(rknn_context, uint32_t, rknn_input*);
typedef int (*fn_rknn_run)(rknn_context, void*);
typedef int (*fn_rknn_outputs_get)(rknn_context, uint32_t, rknn_output*, void*);
typedef int (*fn_rknn_outputs_release)(rknn_context, uint32_t, rknn_output*);

int main(int argc, char **argv) {
    const char *rknn_path = "/root/npu-research/conv_int8.rknn";
    const char *tflite_path = "/root/npu-research/conv_int8.tflite";
    if (argc > 1) rknn_path = argv[1];
    if (argc > 2) tflite_path = argv[2];

    /* Generate deterministic input (same as np.random.seed(42)) */
    uint8_t input[3072];
    uint32_t s = 42;
    for (int i = 0; i < 3072; i++) {
        /* numpy's RandomState uses MT19937, not LCG. Use fixed pattern instead. */
        s = s * 1103515245 + 12345;
        input[i] = (s >> 16) & 0xFF;
    }

    /* === RKNN C API === */
    void *rknn_lib = dlopen("/usr/lib/librknnrt.so", RTLD_NOW);
    if (!rknn_lib) { fprintf(stderr, "dlopen librknnrt failed: %s\n", dlerror()); return 1; }

    fn_rknn_init p_init = dlsym(rknn_lib, "rknn_init");
    fn_rknn_destroy p_destroy = dlsym(rknn_lib, "rknn_destroy");
    fn_rknn_inputs_set p_inputs_set = dlsym(rknn_lib, "rknn_inputs_set");
    fn_rknn_run p_run = dlsym(rknn_lib, "rknn_run");
    fn_rknn_outputs_get p_outputs_get = dlsym(rknn_lib, "rknn_outputs_get");
    fn_rknn_outputs_release p_outputs_release = dlsym(rknn_lib, "rknn_outputs_release");

    if (!p_init || !p_run || !p_outputs_get) {
        fprintf(stderr, "dlsym failed\n"); return 1;
    }

    /* Load RKNN model */
    FILE *mf = fopen(rknn_path, "rb");
    if (!mf) { fprintf(stderr, "cannot open %s\n", rknn_path); return 1; }
    fseek(mf, 0, SEEK_END);
    long msz = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    void *model_data = malloc(msz);
    fread(model_data, 1, msz, mf);
    fclose(mf);

    rknn_context ctx = 0;
    int ret = p_init(&ctx, model_data, msz, 0, NULL);
    free(model_data);
    if (ret != 0) { fprintf(stderr, "rknn_init failed: %d\n", ret); return 1; }

    /* Convert uint8 to float32 (RKNN expects float input for this model) */
    float input_f32[3072];
    for (int i = 0; i < 3072; i++)
        input_f32[i] = input[i] / 255.0f;

    /* Set input as float32 NHWC */
    rknn_input inp = {
        .index = 0, .buf = input_f32, .size = 3072 * 4,
        .pass_through = 0, .type = 1 /* FLOAT32 */, .fmt = 1 /* NHWC */
    };
    ret = p_inputs_set(ctx, 1, &inp);
    if (ret != 0) {
        /* Try uint8 if float fails */
        inp.buf = input; inp.size = 3072;
        inp.type = 3; /* UINT8 */
        ret = p_inputs_set(ctx, 1, &inp);
    }
    if (ret != 0) { fprintf(stderr, "rknn_inputs_set failed: %d\n", ret); return 1; }

    /* Run */
    ret = p_run(ctx, NULL);
    if (ret != 0) { fprintf(stderr, "rknn_run failed: %d\n", ret); return 1; }

    /* Get output with want_float=0 (native int8) */
    rknn_output out_rknn = { .want_float = 0, .is_prealloc = 0, .index = 0 };
    ret = p_outputs_get(ctx, 1, &out_rknn, NULL);
    if (ret != 0) { fprintf(stderr, "rknn_outputs_get failed: %d\n", ret); return 1; }

    int8_t *rknn_i8 = (int8_t *)out_rknn.buf;
    int rknn_size = out_rknn.size;
    printf("RKNN: size=%d first8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
           rknn_size, rknn_i8[0], rknn_i8[1], rknn_i8[2], rknn_i8[3],
           rknn_i8[4], rknn_i8[5], rknn_i8[6], rknn_i8[7]);

    /* Save RKNN output */
    FILE *rf = fopen("/tmp/rknn_native_i8.bin", "wb");
    if (rf) { fwrite(rknn_i8, 1, rknn_size, rf); fclose(rf); }

    p_outputs_release(ctx, 1, &out_rknn);
    p_destroy(ctx);
    dlclose(rknn_lib);

    /* === librocketnpu === */
    /* Hide .rknn to force native cache */
    char bak_path[256];
    snprintf(bak_path, sizeof(bak_path), "%s.hidden", rknn_path);
    rename(rknn_path, bak_path);

    int fd = rnpu_open(NULL);
    rnpu_model_t *m = rnpu_model_load(fd, tflite_path);
    if (!m) {
        fprintf(stderr, "rnpu_model_load failed\n");
        rename(bak_path, rknn_path);
        return 1;
    }

    rnpu_invoke(m, input, 3072);

    /* Get raw output BO */
    uint8_t raw[65536];
    int raw_size = rnpu_get_output_raw(m, 0, raw, sizeof(raw));
    int8_t *our_i8 = (int8_t *)raw;
    printf("Ours: size=%d first8=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
           raw_size, our_i8[0], our_i8[1], our_i8[2], our_i8[3],
           our_i8[4], our_i8[5], our_i8[6], our_i8[7]);

    /* Save our output */
    rf = fopen("/tmp/ours_native_raw.bin", "wb");
    if (rf) { fwrite(raw, 1, raw_size, rf); fclose(rf); }

    rename(bak_path, rknn_path);

    /* === Compare === */
    /* Note: RKNN returns NHWC int8 (detiled + dequantized to model output quant).
     * Our raw BO is in NC1HWC2 format with NPU's internal quantization.
     * These are DIFFERENT domains — we need to understand the conversion. */
    printf("\nDirect byte comparison (raw BO vs RKNN native int8):\n");
    int min_size = rknn_size < raw_size ? rknn_size : raw_size;
    int exact = 0;
    for (int i = 0; i < min_size; i++)
        if (rknn_i8[i] == our_i8[i]) exact++;
    printf("  exact=%d/%d (%.1f%%)\n", exact, min_size, 100.0*exact/min_size);

    return 0;
}
