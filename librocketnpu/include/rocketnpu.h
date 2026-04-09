/*
 * librocketnpu - Standalone NPU driver for RK3588
 * Copyright (c) 2024-2026 Orange Pi 5 Plus NPU Research
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCKETNPU_H
#define ROCKETNPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rnpu_model rnpu_model_t;

/* Open the NPU device. Returns fd >= 0 on success, -1 on error. */
int rnpu_open(const char *device);

/* Load a quantized INT8 TFLite model and compile it for the NPU.
 * All BOs are pre-allocated here — no allocations during invoke. */
rnpu_model_t *rnpu_model_load(int fd, const char *tflite_path);

/* Run inference. Input is NHWC uint8 (TFLite quantized format).
 * Blocks until NPU completes. */
int rnpu_invoke(rnpu_model_t *m, const void *input, size_t input_size);

/* Copy output tensor idx to caller buffer. Returns bytes written, -1 on error. */
int rnpu_get_output(rnpu_model_t *m, int idx, void *out, size_t max_size);

/* Get output tensor dimensions: sets *w, *h, *c. */
int rnpu_get_output_dims(rnpu_model_t *m, int idx, int *w, int *h, int *c);

/* Get input tensor dimensions. */
int rnpu_get_input_dims(rnpu_model_t *m, int *w, int *h, int *c);

/* Get number of output tensors. */
int rnpu_output_count(rnpu_model_t *m);

/* Read any intermediate tensor by index (for debugging).
 * Converts from NPU interleaved format to NHWC uint8.
 * Returns bytes written, -1 on error, 0 if tensor not found/empty. */
int rnpu_get_tensor(rnpu_model_t *m, int tensor_idx, void *out, size_t max_size,
                    int *w, int *h, int *c);

/* Free all resources. */
void rnpu_model_free(rnpu_model_t *m);

/* Close the NPU device. */
void rnpu_close(int fd);

#ifdef __cplusplus
}
#endif

#endif /* ROCKETNPU_H */
