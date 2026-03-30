# RKNN Byte-Identical Output: Findings and Plan

## Executive Summary

We built an instrumented RKNPU kernel driver, captured full register-level
traces of both RKNN and librocketnpu, and performed register-by-register
comparison on a minimal single-CONV INT8 model. The analysis reveals that
byte-identical RKNN output is achievable but requires restructuring how
librocketnpu handles per-axis quantization and tensor layout.

## Infrastructure Built

### 1. Instrumented RKNPU kernel module (`rknpu-driver/`)

- Forked from Armbian vendor driver (`rk-6.1-rkr5` branch)
- Builds as out-of-tree module (`rknpu_trace.ko`)
- Traces every `REG_WRITE` via ftrace's `trace_printk`
- Toggled at runtime: `echo 1 > /sys/module/rknpu_trace/parameters/rknpu_trace_enabled`
- **Load procedure**: Boot with `initcall_blacklist=rknpu_init` in `/boot/armbianEnv.txt`,
  then `insmod rknpu_trace.ko`
- **WARNING**: Never `rmmod` — crashes kernel (IOMMU teardown)

### 2. Kprobes tracing (no module swap needed)

Works with the stock built-in driver:
```bash
echo 'p:rknpu_sub __rknpu_submit_ioctl flags=+0(%x1):u32 task_start=+8(%x1):u32 task_num=+12(%x1):u32 core_mask=+56(%x1):u32' > /sys/kernel/debug/tracing/kprobe_events
```

### 3. LD_PRELOAD intercept (`librocketnpu/tests/intercept_swap.c`)

Captures full regcmd content per task via `DUMP_FULL=1 DUMP_REGCMD=1`.

## RKNN Architecture (Reverse-Engineered)

### Task pipeline for a single per-axis CONV (32x32x3 → 32x32x16)

RKNN submits 9 tasks in one `RKNPU_SUBMIT` call:

| Task | Type | BS_CFG | DFMT | Function |
|------|------|--------|------|----------|
| 0 | BRDMA Conv | 0x20140 | 0xe0 | First spatial tile, per-channel requant |
| 1 | Reformat | 0x53 | 0x24000001 | Reshape rows 0-15: NPU grouped → output format |
| 2 | Reformat | 0x53 | 0x24000001 | Reshape rows 16-31 |
| 3 | BRDMA Conv | 0x20140 | 0xe0 | Second spatial tile |
| 4 | Continuation | 0x0 | 0x0 | Sticky DPU state for tile 2 |
| 5 | Reformat | 0x53 | 0x24000001 | Reshape tile 2, rows 0-15 |
| 6 | Reformat | 0x53 | 0x24000001 | Reshape tile 2, rows 16-31 |
| 7 | BRDMA Conv | 0x20140 | 0xe0 | Third spatial tile |
| 8 | Continuation | 0x0 | 0x0 | Sticky state |

### Reformat tasks (DFMT=0x24000001)

Pure DPU+RDMA operations — NO convolution (no CNA/CORE registers):
- Read from conv output via RDMA (`SRC_BASE_ADDR` = conv output buffer)
- Write to final output via DPU (`DST_BASE_ADDR` = output tensor)
- `FLYING_MODE=1`, `SURF_LEN=512` — special DMA routing mode
- All SDP bypass (`BS_CFG=0x53, BN_CFG=0x53`), pass-through CVT (`scale=1, shift=0`)
- **Purpose**: Hardware tensor layout conversion (NPU grouped → flat NHWC)
- librocketnpu does this in software via `rnpu_convert_output()`

### Continuation tasks (BS_CFG=0x0)

Between the reformat tasks and the next conv tile:
- The reformat tasks' `BS_CFG=0x53` (all-bypass) preserves the BRDMA DPU
  state from the preceding conv task
- When the next conv's continuation task arrives (BS_CFG=0x0, no DPU
  registers), the DPU retains the BRDMA configuration

## Register Comparison: RKNN vs librocketnpu

104 of 120 registers match. The 16 differences:

### Input format (architectural — not a bug)

| Register | Ours | RKNN | Meaning |
|----------|------|------|---------|
| CONV_CON1 (0x100c) | 0x0 | 0x6000a000 | NONALIGN_DMA + GROUP_LINE_OFF + ARGB_IN=10 |
| CVT_CON1-3 (input conv) | 0x10000 | 0x4000ff80 | HW uint8→int8 (scale=16384, offset=-128) |

**RKNN stores input tensors in ARGB_IN=10 format** with hardware uint8→int8
conversion. librocketnpu uses standard NPU grouped format with software
conversion. Setting ARGB_IN=10 on our data produces wrong results because
the data layout doesn't match.

### Quantization (affects accuracy)

| Register | Ours | RKNN | Meaning |
|----------|------|------|---------|
| OUT_CVT_SCALE | 21794 | 23517 | Different conv_scale (different grouping) |
| OUT_CVT_SHIFT | 23 | 22 | Different shift |
| BS_OW_OP | 0x1f | 0x0 | We set OC-1, RKNN sets 0 |
| BS_OW_CFG | 0x124 | 0x125 | One bit difference |

The scale/shift difference comes from different channel grouping:
- **RKNN**: All 16 OC in one BRDMA task. `conv_scale = max(all 16 ws)`.
- **Ours**: 2 requant groups of 8 OC. `conv_scale = max(group ws)`.

### Weight packing

| Register | Ours | RKNN | Meaning |
|----------|------|------|---------|
| WT_SIZE0/2 | 32 kernels | 16 kernels | We pad to 32, RKNN uses actual 16 |
| CBUF_CON1 | 8 | 1024 | Different CBUF entry allocation |
| LINE/SURF_STRIDE | differs | differs | Different stride convention |

### Unknown

| Register | Ours | RKNN | Meaning |
|----------|------|------|---------|
| REG_1180 | 0x0 | 0xfff | Unknown register, RKNN sets all bits |

## BRDMA Data Extraction Bugs Found

1. **oc_pad < 32 filter** rejected models with 16 output channels.
   Fix: change to `oc_pad < 16`.

2. **non_zero >= 4 threshold** for MUL values was too low — caught false
   positives from weight data. Fix: change to `non_zero >= 6`.

3. **requant_group_count <= 1 condition** prevented RKNN BRDMA override
   for per-axis models that split into multiple groups.

4. **Bias matching failure**: RKNN re-quantizes biases during model build,
   so TFLite bias values don't match RKNN's BRDMA biases. Need a
   different matching strategy (e.g., channel count + position-based).

## Plan: Achieving Byte-Identical Output

### Phase A: Fix BRDMA data extraction (quick wins)

1. Lower `oc_pad` threshold from 32 to 16
2. Raise `non_zero` MUL threshold from 4 to 6
3. Remove `requant_group_count <= 1` restriction for RKNN override
4. Implement position-based RKNN blob matching (sequential, not bias-based)

### Phase B: Single-group BRDMA for per-axis ops

Currently librocketnpu splits per-axis convolutions into multiple requant
groups (each with 16 output channels and its own `conv_scale`). RKNN
processes ALL channels in one task with per-channel MUL correction via BRDMA.

Changes needed:
- When RKNN BRDMA data is available, use **1 requant group** for all OC
- Set `conv_scale = max(all weight_scales) * input_scale / output_scale`
- BRDMA MUL values handle per-channel correction: `MUL[c] = ws[c] / max_ws * 16384`
- This eliminates the need for scale-sorting and multi-group decomposition

Files: `rnpu_model.c` (group creation), `rnpu_coefs.c` (scale computation),
`rnpu_task.c` (task splitting)

### Phase C: Match remaining registers

1. **BS_OW_OP = 0** (currently OC-1). Simple one-line fix.
2. **BS_OW_CFG bit**: 0x124 → 0x125. One bit in the SDP output write config.
3. **REG_1180 = 0xfff**: Set this register in all regcmd functions.
4. **Weight kernel padding**: Use actual OC (16) instead of padded (32).
   This affects weight packing and CBUF allocation.

Files: `rnpu_regcmd.c`, `rnpu_coefs.c`

### Phase D: Match conv_scale exactly

Compute `conv_scale` identically to RKNN:
- Use the maximum weight scale across ALL output channels (not per-group)
- Match the shift selection algorithm (find highest shift where scale < 2^15)
- Verify BRDMA MUL values match RKNN's blob byte-for-byte

### Phase E: Hardware input conversion (optional)

RKNN uses hardware uint8→int8 conversion via CVT_CON registers:
- `CVT_CON0 = 0xe38e0` (enable conversion with specific params)
- `CVT_CON1-3 = 0x4000ff80` (scale=16384, offset=-128)
- `CONV_CON1 = 0x6000a000` (ARGB_IN=10 input format)

This requires changing `rnpu_convert_input()` to write data in ARGB_IN=10
format AND setting the corresponding CNA registers. Non-trivial but would
eliminate the software uint8→int8 subtraction.

### Phase F: Verify on progressively larger models

1. Single CONV (32x32x3 → 32x32x16) — current test model
2. Two-layer CONV (add depthwise + pointwise)
3. MobileNetV1 (31 ops)
4. YOLOv5s (91 ops)

At each stage: `diff npu_output.bin rknn_output.bin` must show 0 bytes different.

## Key Architectural Insights

### Why the reformat tasks exist

RKNN separates computation (CONV) from data layout (REFORMAT). The conv
writes to an intermediate buffer in NPU grouped format. The reformat task
reads it back and writes to the output tensor in the target layout. This
allows the conv and reformat to use different surface strides, enabling
multi-group tiled ops where the conv's surface stride is per-tile but the
output's surface stride is per-tensor.

librocketnpu combines these into one step — the conv writes directly to the
output tensor, and `rnpu_convert_output()` does the layout conversion in
software. For byte-identical output, we need the same intermediate values
(from the conv), even though our layout conversion is software-based.

### Why RKNN re-quantizes biases

During `rknn.build()`, RKNN's compiler re-computes bias values based on its
chosen `conv_scale`. Since RKNN uses `max(all_ws)` for conv_scale while
TFLite computed biases with individual `ws[c]` values, the bias values differ.
The BRDMA blob in the .rknn file contains RKNN's re-computed biases, which
is why we must use them directly rather than computing from TFLite biases.

### Why continuation works with reformat in between

The reformat tasks set `BS_CFG = 0x53` (all SDP bypass). This doesn't
CLEAR the DPU's BRDMA state — it just bypasses it for the current task.
When the next conv's continuation task arrives with `BS_CFG = 0x0` (sticky
state), the BRDMA configuration from the previous conv task is still intact
in the hardware registers. The reformat tasks act as "transparent" pipeline
stages that don't disturb the BRDMA state.
