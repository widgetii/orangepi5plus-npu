#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "rocketnpu.h"

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "/root/npu-research/conv_int8.tflite";
    const char *input_file = argc > 2 ? argv[2] : NULL;

    int fd = rnpu_open(NULL);
    rnpu_model_t *m = rnpu_model_load(fd, model);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }

    /* Input: from file or deterministic PRNG */
    int w, h, c;
    rnpu_get_input_dims(m, &w, &h, &c);
    int inp_size = w * h * c;
    uint8_t *inp = malloc(inp_size);

    if (input_file) {
        FILE *f = fopen(input_file, "rb");
        if (f) { fread(inp, 1, inp_size, f); fclose(f); }
    } else {
        /* LCG PRNG seed=42 */
        unsigned s = 42;
        for (int i = 0; i < inp_size; i++) {
            s = s * 1103515245 + 12345;
            inp[i] = (s >> 16) & 0xFF;
        }
    }

    int ret = rnpu_invoke(m, inp, inp_size);
    if (ret != 0) { fprintf(stderr, "invoke failed: %d\n", ret); return 1; }

    /* Get converted output (applies deinterleave) */
    int ow, oh, oc;
    rnpu_get_output_dims(m, 0, &ow, &oh, &oc);
    int out_size = ow * oh * oc;
    uint8_t *out = malloc(out_size);
    rnpu_get_output(m, 0, out, out_size);

    /* Save converted output */
    FILE *f = fopen("/tmp/conv_converted.bin", "wb");
    if (f) { fwrite(out, 1, out_size, f); fclose(f); }

    /* Dump RAW output BO (no deinterleave) */
    uint8_t *raw = malloc(65536);
    int raw_size = rnpu_get_output_raw(m, 0, raw, 65536);
    f = fopen("/tmp/conv_raw_bo.bin", "wb");
    if (f) { fwrite(raw, 1, raw_size, f); fclose(f); }
    printf("Raw BO: %d bytes, nonzero=%d\n", raw_size,
           ({ int nz=0; for(int i=0;i<raw_size;i++) if(raw[i]) nz++; nz; }));
    printf("  raw[0:16]: ");
    for (int i = 0; i < 16 && i < raw_size; i++) printf("%02x ", raw[i]);
    printf("\n");

    free(raw);
    free(inp);
    free(out);
    return 0;
}
