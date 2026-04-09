/*
 * Test: load model via proxy, query tensor info, run one inference.
 * Validates that the proxy layer is transparent.
 *
 * Build: make test_proxy
 * Run:   ./test_proxy model.rknn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.rknn [num_runs]\n", argv[0]);
        return 1;
    }
    int num_runs = argc > 2 ? atoi(argv[2]) : 5;

    /* Load model file */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    uint32_t model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *model_data = malloc(model_size);
    if (!model_data || fread(model_data, 1, model_size, f) != model_size) {
        fprintf(stderr, "Failed to read model\n");
        return 1;
    }
    fclose(f);

    printf("Model: %s (%u bytes)\n", argv[1], model_size);

    /* Init */
    rknn_context ctx;
    double t0 = now_ms();
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    printf("rknn_init: %d (%.1f ms)\n", ret, now_ms() - t0);
    if (ret != 0) return 1;

    /* Query I/O num */
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("rknn_query(IN_OUT_NUM): %d -> inputs=%u outputs=%u\n",
           ret, io_num.n_input, io_num.n_output);

    /* Query SDK version */
    rknn_sdk_version ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("rknn_query(SDK_VERSION): %d -> api=%s drv=%s\n",
           ret, ver.api_version, ver.drv_version);

    /* Query input attrs */
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        printf("Input[%u]: %s %ux%ux%ux%u type=%s fmt=%s qnt=%s "
               "scale=%.6f zp=%d size=%u\n",
               i, attr.name,
               attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
               get_type_string(attr.type), get_format_string(attr.fmt),
               get_qnt_type_string(attr.qnt_type),
               attr.scale, attr.zp, attr.size);
    }

    /* Query output attrs */
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        printf("Output[%u]: %s %ux%ux%ux%u type=%s fmt=%s qnt=%s "
               "scale=%.6f zp=%d size=%u\n",
               i, attr.name,
               attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
               get_type_string(attr.type), get_format_string(attr.fmt),
               get_qnt_type_string(attr.qnt_type),
               attr.scale, attr.zp, attr.size);
    }

    /* Query native output attrs */
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attr, sizeof(attr));
        printf("NativeOut[%u]: %ux%ux%ux%u type=%s fmt=%s size=%u\n",
               i, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
               get_type_string(attr.type), get_format_string(attr.fmt),
               attr.size);
    }

    /* Prepare input (zeros) */
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

    uint32_t input_size = in_attr.size;
    uint8_t *input_data = calloc(1, input_size);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;
    inputs[0].size = input_size;
    inputs[0].pass_through = 0;

    /* Warmup */
    rknn_inputs_set(ctx, 1, inputs);
    rknn_run(ctx, NULL);

    rknn_output outputs[16];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < io_num.n_output && i < 16; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 0;
    }
    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    /* Benchmark */
    printf("\nRunning %d iterations...\n", num_runs);
    double total = 0, min_ms = 1e9;
    for (int i = 0; i < num_runs; i++) {
        rknn_inputs_set(ctx, 1, inputs);

        t0 = now_ms();
        rknn_run(ctx, NULL);
        double dt = now_ms() - t0;

        memset(outputs, 0, sizeof(outputs));
        for (uint32_t j = 0; j < io_num.n_output && j < 16; j++) {
            outputs[j].index = j;
            outputs[j].want_float = 0;
        }
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        /* Print first few output bytes on last run */
        if (i == num_runs - 1) {
            for (uint32_t j = 0; j < io_num.n_output && j < 16; j++) {
                printf("Output[%u] first 16 bytes:", j);
                uint8_t *p = outputs[j].buf;
                for (int k = 0; k < 16 && k < (int)outputs[j].size; k++)
                    printf(" %02x", p[k]);
                printf(" (size=%u)\n", outputs[j].size);
            }
        }

        rknn_outputs_release(ctx, io_num.n_output, outputs);
        total += dt;
        if (dt < min_ms) min_ms = dt;
    }
    printf("Latency: avg=%.2f ms, min=%.2f ms\n", total / num_runs, min_ms);

    free(input_data);
    free(model_data);
    rknn_destroy(ctx);
    printf("Done.\n");
    return 0;
}
