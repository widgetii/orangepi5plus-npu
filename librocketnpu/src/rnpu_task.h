/*
 * CBUF task splitting
 * SPDX-License-Identifier: MIT
 */

#ifndef RNPU_TASK_H
#define RNPU_TASK_H

#include "rnpu_internal.h"

/* Split a CONV operation into CBUF-sized tasks. Allocates op->tasks. */
void rnpu_split_tasks(struct rnpu_operation *op);

#endif /* RNPU_TASK_H */
