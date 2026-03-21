/*
 * TFLite FlatBuffer parser (no TF dependency)
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_TFLITE_H
#define RNPU_TFLITE_H

#include "rnpu_internal.h"

/* Parse a .tflite file into rnpu_tfl_model. Returns 0 on success. */
int rnpu_tflite_parse(const char *path, struct rnpu_tfl_model *model);

/* Free parsed model data. */
void rnpu_tflite_free(struct rnpu_tfl_model *model);

#endif /* RNPU_TFLITE_H */
