/*
 * Test: compare our patched regcmd against the proxy's actual regcmd.
 *
 * Step 1: Init via proxy, run once, then read the proxy's regcmd from
 *         the intercept_swap BO dump.
 * Step 2: Init via own, patch regcmd, dump to file.
 * Step 3: Compare.
 *
 * For now, this just dumps our patched regcmd. Compare manually with
 * intercept_swap.c DUMP_REGCMD output.
 *
 * Usage: ORKNN_OWN=init,query,input,run ORKNN_NO_SUBMIT=1 \
 *        ORKNN_DUMP_REGCMD=/tmp/our_regcmd.txt \
 *        ./test_regcmd_diff model.rknn
 *
 * Then compare with:
 *   DUMP_REGCMD=1 LD_PRELOAD=./intercept_swap.so ./bench_rknn model.rknn 1
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_api.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.rknn\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    uint32_t model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, f);
    fclose(f);

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    printf("rknn_init: %d\n", ret);
    if (ret != 0) return 1;

    /* Query input info */
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("inputs=%u outputs=%u\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

    /* Set input (zeros) */
    uint8_t *input_data = calloc(1, in_attr.size);
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;
    inputs[0].size = in_attr.size;

    ret = rknn_inputs_set(ctx, 1, inputs);
    printf("rknn_inputs_set: %d\n", ret);

    /* Run (will be skipped if ORKNN_NO_SUBMIT is set) */
    ret = rknn_run(ctx, NULL);
    printf("rknn_run: %d\n", ret);

    free(input_data);
    free(model_data);
    rknn_destroy(ctx);
    printf("Done. Check ORKNN_DUMP_REGCMD output.\n");
    return 0;
}
