# RK3588 Rocket NPU — Hardware Quirks and QEMU Correctness Spec

Findings from reverse-engineering the Rocket NPU via register intercepts,
Mesa/RKNN driver analysis, and hardware testing on Orange Pi 5 Plus.

## 1. Core Architecture

- 3 independent NPU cores at `0xFDAB0000`, `0xFDAC0000`, `0xFDAD0000`
- Each core: PC(+0x0), CNA(+0x1000), CORE(+0x3000), DPU(+0x4000), DPU_RDMA(+0x5000)
- NVDLA-derived, register-programmed (no instruction set)
- Frontend (PC) is a DMA engine: reads packed 64-bit "regcmd" entries from DRAM,
  writes them as register values to the convolution pipeline
- Each core has its own IOMMU group and IRQ

## 2. Regcmd Format (64-bit packed entries)

```
bits[63:48] = target (which HW block: 0x0101=PC, 0x0201=CNA, 0x0801=CORE,
                       0x1001=DPU, 0x2001=DPU_RDMA)
bits[47:16] = value (32-bit register value)
bits[15:0]  = register offset within the block
```

Special entries (must be handled, not parsed as register writes):
- `target == 0x0041`: interrupt/sync marker — skip
- `target == 0x0081`: PC_OPERATION_ENABLE trigger — skip (kernel handles via MMIO)
- `entry == 0x0`: null entry (chain termination for last task)

## 3. Kernel MMIO Sequence (per job)

The kernel programs the PC DMA engine in this exact order:

```
1. PC_BASE_ADDRESS    = 0x1  (PC_SEL=1, initialize DMA mode)
2. CNA_S_POINTER      = PP_EN(1) | EXECUTER_PP_EN(1) | PP_MODE(1) | extra_bit
3. CORE_S_POINTER     = PP_EN(1) | EXECUTER_PP_EN(1) | PP_MODE(1) | extra_bit
4. PC_BASE_ADDRESS    = task[0].regcmd  (DMA address of first regcmd block)
5. PC_REGISTER_AMOUNTS = (regcmd_count + 1) / 2 - 1
6. PC_INTERRUPT_MASK   = DPU_0 | DPU_1
7. PC_INTERRUPT_CLEAR  = DPU_0 | DPU_1
8. PC_TASK_CON         = RESERVED_0(1) | TASK_COUNT_CLEAR(1) |
                         TASK_NUMBER(job.task_count) | TASK_PP_EN(1)
9. PC_TASK_DMA_BASE_ADDR = 0x0
10. PC_OPERATION_ENABLE = OP_EN(1)  ← triggers execution
```

**Critical**: `extra_bit = 0x10000000 * core_index` in the S_POINTER writes.
Core 0 = 0, Core 1 = 0x10000000, Core 2 = 0x20000000.

## 4. Multi-Task Chaining

A single job can contain multiple tasks (CBUF-split convolutions).

- Kernel programs ONLY task[0] into PC registers
- TASK_CON.TASK_NUMBER = total task count for the job
- Tasks are chained via regcmd entries at the END of each task's block:
  - Entry `[count-4]`: `PC_BASE_ADDRESS` with next task's DMA address
  - Entry `[count-3]`: `PC_REGISTER_AMOUNTS` with next task's entry count
  - Entry `[count-2]`: `0x0041000000000000` (sync marker)
  - Entry `[count-1]`: `PC_OPERATION_ENABLE` (re-trigger)
- Last task: entry `[count-4]` = `0x0` (null = no chain)

**QEMU implication**: after parsing task 0's regcmd, check if there's a chain
pointer. If yes, fetch and parse the next block. Repeat until null chain.
Each chained block triggers a separate "task completion" (tracked by TASK_NUMBER).

## 5. REGISTER_AMOUNTS Encoding

```
MMIO write:  PC_DATA_AMOUNT = (regcmd_count + 1) / 2 - 1
Decode:      actual_entries = (PC_DATA_AMOUNT + 1) * 2
```

For 130 entries: `(130+1)/2 - 1 = 64` → `(64+1)*2 = 130` entries.
Each entry is 8 bytes. PC reads `130 * 8 = 1040` bytes from the DMA address.

## 6. IRQ Handling

- Interrupt fires after ALL tasks in a job complete (not per-task)
- Kernel checks `PC_INTERRUPT_RAW_STATUS` for bits:
  - `DPU_0` (0x100) or `DPU_1` (0x200)
- If either bit set → `IRQ_WAKE_THREAD`
- Kernel clears mask: `PC_INTERRUPT_MASK = 0x0`
- Kernel clears status: `PC_INTERRUPT_CLEAR = 0x1FFFF`

**IRQ mask semantics**: mask=1 means ENABLED (not masked).
`status = raw_status & mask` — only fire IRQ if masked bit is set.

## 7. Register Encoding Quirks

### CNA registers: raw values (NOT count-1)
- `DATA_SIZE0` = `WIDTH[26:16] | HEIGHT[10:0]` — actual dimensions
- `DATA_SIZE1` = `CHANNEL[15:0]` (aligned) | `CHANNEL_REAL[29:16]` (count-1 for real)
- `DATA_SIZE2` = `DATAOUT_WIDTH` — actual width
- `DATA_SIZE3` = `DATAOUT_ATOMICS` — actual count
- `WEIGHT_SIZE2` = `WIDTH[28:24] | HEIGHT[20:16] | KERNELS[13:0]` — actual values

### CORE/DPU registers: count-1
- `DATAOUT_SIZE_0` = `WIDTH[15:0] | HEIGHT[31:16]` — both are (value - 1)
- `DATAOUT_SIZE_1` = `CHANNEL[12:0]` — (value - 1)
- `DPU_DATA_CUBE_WIDTH/HEIGHT/CHANNEL` — all count-1

### DPU output quantization
- `OUT_CVT_SCALE`: 16-bit scale value in bits[15:0]
- `OUT_CVT_SHIFT`: shift amount in bits[4:0]
- `OUT_CVT_OFFSET`: signed 8-bit offset (output_zero_point - 0x80)

Scale extraction from float conv_scale:
```c
uint32_t sb = float_as_uint32(conv_scale);
unsigned shift = 127 + 31 - 32 - (sb >> 23) + 16;
if (truncate_bits > 0) shift--;
unsigned scale = ((sb >> 9) & 0x7fff) + 1;
if (scale < (1 << 14)) scale |= (1 << 14);
```

## 8. QEMU Register Alignment Status

The following register address mismatches were fixed in the QEMU NPU model:

| Register | Old (wrong) | Fixed | Notes |
|----------|------------|-------|-------|
| `CNA_PAD_CON1` | 0x106c | 0x1184 | pad_value was always 0 for non-zero zp |
| `DPU_DATA_FORMAT` | 0x4014 | 0x4010 | was parsing DPU_OFFSET_PEND instead |
| `DPU_SURFACE_ADD` | 0x40b0 | 0x40c0 | surface_add at wrong offset |
| `DPU_FEAT_MODE_CFG` | 0x4010 | 0x400c | define was at DATA_FORMAT's offset |

Newly parsed registers (stored in RocketConvTask, available for computation):
- `DPU_FEATURE_MODE_CFG` (0x400c), `DPU_BS_OW_CFG` (0x4050)
- `DPU_WDMA_SIZE_0/1` (0x4058/0x405c), `DPU_EW_RELUX_CMP` (0x407c)
- `RDMA_SRC_BASE_ADDR` (0x5018), `RDMA_NRDMA_CFG` (0x5028), `RDMA_BN_BASE_ADDR` (0x502c)
- `RDMA_FEATURE_MODE_CFG` (0x5044), `RDMA_WEIGHT` (0x5068)

Functional additions:
- **Output surface stride**: when `output_surface_stride > 0` and multiple output groups,
  DMA writes use per-group stride (supports sparse layouts from per-channel decomposition)
- **EW ReLUx**: EW stage now supports ReLUx (clamp to `ew_relux_cmp`) via bit 10 of `ew_cfg`

## 9. Depthwise Convolution

Signaled by:
- `CNA_CONV_CON1.CONV_MODE = 3`
- `CORE_MISC_CFG.DW_EN = 1`
- `DPU_FEATURE_MODE_CFG.CONV_MODE = 3`

Differences from standard conv:
- `weight_kernels = 1` (one kernel per input channel)
- Weight tensor shape: `[1][KH][KW][OC]`
- `BS_OW_CFG.SIZE_E = 3` (instead of 1 for standard)
- Output channels align to 64 instead of 32 for small channel counts
- `surfaces_per_row *= 2` for depthwise

## 10. Per-Channel Quantization (RKNN-style)

When `DPU_DATA_CUBE_CHANNEL = 0` (single output channel per task):
- Each task has its own `OUT_CVT_SCALE` and `OUT_CVT_SHIFT`
- `BS_CFG = 0x13F` (BS bypassed, but RELU active in BN)
- Bias loaded via `BS_ALU_CFG` as scalar (not DMA)
- `BRDMA_CFG = 0x1F` (all DMA channels disabled)
- `BN_CFG = 0x12` (bypass=0, alu_bypass=1, mul_bypass=1, relu_bypass=0)
- `DPU_DATA_FORMAT = 0xE0` (BS_MUL_SHIFT_VALUE_NEG=7)
- `NRDMA_CFG = 0`, `BN_BASE_ADDR = 0` (no per-channel DMA data)

This is how RKNN achieves perfect per-axis quantization: decompose each
CONV into one task per output channel, each with its own requantization scale.

## 11. Data Formats

### NPU Interleaved Tensor Format (x-major)
```
offset(group, x, y, W, H) = group * W * H * 16 + x * H * 16 + y * 16
```
- Groups of 16 channels (FEATURE_ATOMIC_SIZE = 16)
- x-major within each group (width varies fastest after the 16-byte atomic)
- Output buffer allocated at 2x groups for HW alignment

### Weight Format
- Standard: `[OC][KH][KW][IC]` packed into 32-byte atoms (WEIGHT_ATOMIC_SIZE)
- Depthwise: `[1][KH][KW][OC]` with kernels=1
- Minimum 32 output channels padded (align to 32)
- int8 signed (TFLite uint8 values - 0x80)

### Bias Format
- 32-bit per output channel (int32)
- Computed as: `sum(input_zp * weight_value) * -1` per channel
- Truncate bits applied based on max bias magnitude

## 12. CBUF (Convolution Buffer) Constants

```
CBUF_BANK_SIZE        = 32768 bytes
CBUF_BANKS            = 12
CBUF_ENTRIES_PER_BANK = 256
CBUF_ENTRY_SIZE       = 128 bytes
FEATURE_ATOMIC_SIZE   = 16 bytes
WEIGHT_ATOMIC_SIZE    = 32 bytes
```

When input doesn't fit in CBUF, operations are split into multiple tasks:
- Each task processes a horizontal slice of the input
- Adjacent tasks overlap by `(kernel_height - 1)` rows
- First task gets `pad_top`, last gets `pad_bottom`

## 13. Kernel Driver Quirks (for QEMU correctness)

### Job submission
- Mesa batches ~27 jobs in a single SUBMIT ioctl
- Single-job-per-SUBMIT causes NPU timeouts on real hardware
  (DRM scheduler distributes to different cores; individual submits
  may have fencing issues with shared activation BOs)
- QEMU should work with any job_count since it's synchronous

### BO lifecycle
- `FINI_BO` flushes CPU cache → NPU-visible (must be called before SUBMIT)
- `PREP_BO` syncs NPU → CPU-visible (blocks until job completes)
- Without PREP before GEM_CLOSE, the DRM scheduler's fence tracking
  gets corrupted on real hardware

### IOMMU
- Each fd gets its own IOMMU domain
- All BOs created on an fd are mapped in that domain permanently
- Domain attached to core before job runs; cached across jobs
- Upstream driver bug: `iommu_detach_group(NULL, ...)` — our patch fixes this

### Timeout behavior
- Kernel timeout = 500ms per task
- Timeout handler resets the core and signals the fence with error
- Multiple timeouts corrupt IOMMU state → kernel memory corruption
  (with `panic_on_oops=1`, this triggers clean reboot)

## 14. Specific Register Values to Validate

### Convolution pipeline stages (standard conv, no addition)
```
BS_CFG         = BS_ALU_ALGO(2) | BS_ALU_SRC(1) | BS_RELU_BYPASS(1) | BS_MUL_BYPASS(1)
BS_OW_OP       = 0x80 - weight_zero_point
BN_CFG         = BN_RELU_BYPASS(1) | BN_MUL_BYPASS(1) | BN_ALU_BYPASS(1) | BN_BYPASS(1)
EW_CFG         = EW_RELU_BYPASS(1) | EW_OP_CVT_BYPASS(1) | EW_LUT_BYPASS(1) |
                 EW_OP_BYPASS(1) | EW_BYPASS(1)
BRDMA_CFG      = BRDMA_DATA_USE(1)
ERDMA_CFG      = ERDMA_DISABLE(1)
RDMA_WEIGHT    = E_WEIGHT(1) | N_WEIGHT(1) | B_WEIGHT(1) | M_WEIGHT(1)
FEATURE_MODE   = BURST_LEN(15) | OUTPUT_MODE(2)
```

### Addition (residual connection)
When `add_tensor != -1`:
```
EW_CFG         = EW_CVT_TYPE(1) | EW_DATA_MODE(1) | EDATA_SIZE(1) | EW_ALU_ALGO(2) |
                 EW_RELU_BYPASS(1) | EW_LUT_BYPASS(1) | EW_OP_SRC(1)
ERDMA_CFG      = ERDMA_DATA_MODE(1) | ERDMA_DATA_SIZE(1)
RDMA_FEATURE   = ... | COMB_USE(5)
```

## 15. Known Hardware Limitations

- INT16 output precision (`OUT_PRECISION=1`) NOT SUPPORTED — causes NPU timeout
- Small spatial sizes (8x8) produce incorrect results even with Mesa
- Per-axis quantization requires per-channel task decomposition (no HW support for
  per-channel scale in a single task)
- 3-channel input (RGB) requires special CNA config:
  `NONALIGN_DMA(1) | GROUP_LINE_OFF(1) | ARGB_IN(8)` in CONV_CON1,
  and custom CVT_CON0-5 truncate/scale values

## 16. Test Vectors

MobileNetV1 INT8 (224x224x3 input):
- 28 CONV operations (14 standard + 14 depthwise)
- 41 total tasks (some ops split into 2-3 CBUF tasks)
- 130 regcmd entries per task (fixed for this model)
- 171 BOs in Mesa, 4 BOs in librocketnpu
- Expected output: bit-exact match with Mesa (max_diff=0 for per-tensor weights)

YOLOv5s-relu INT8 (640x640x3 input):
- 61 CONV operations + 5 SW ops (concat, maxpool, pad, resize, logistic)
- Per-axis weights: 643 groups with scale-sorted grouping
- Mixed HW/SW execution segments
