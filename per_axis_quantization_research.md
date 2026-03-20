# Per-Axis Quantization on Rockchip NPU: Research Findings

## Problem Statement

YOLO models use per-axis (per-channel) weight quantization, but the Rockchip
NPU's DPU output conversion (`OUT_CVT`) applies a **single requantization scale
to all output channels**. With up to 27x scale ratio between channels, this
causes catastrophic int8 clipping for channels with small weight_scale, producing
completely wrong detection output.

## RKNN Vendor Driver Analysis

**Method**: Built an LD_PRELOAD intercept library (`intercept.so`) that hooks the
RKNN ioctl calls on the vendor kernel (6.1.115-vendor-rk35xx). Captured 227
register command (regcmd) dumps from YOLOv5s-640x640 inference.

**Key Finding: BN Stage is NOT Used for Per-Channel Multiply**

Contrary to initial hypothesis, the RKNN driver does NOT use the DPU's BN
(Batch Normalize) stage for per-channel requantization:
- `BN_BASE_ADDR = 0` in ALL 227 tasks (no per-channel DMA data)
- `NRDMA_CFG = 0` in ALL tasks (NRDMA disabled)
- BN is used only for ReLU activation (`BN_CFG = 0x12`: bypass=0, alu_bypass=1,
  mul_bypass=1, relu_bypass=0)

**Key Finding: Per-Channel Task Decomposition**

RKNN handles per-axis quantization by decomposing every CONV into
**per-output-channel tasks**:
- ALL 227 tasks have `DPU_DATA_CUBE_CHANNEL = 0` (1 output channel)
- Each task has its own `OUT_CVT_SCALE` and `OUT_CVT_SHIFT`
- Example: regcmd0 has scale=16448, shift=22; regcmd19 has scale=26037, shift=24

**Other RKNN Register Differences**:
- `BS_CFG = 0x13F` (BS stage bypassed, but ReLU active within BS)
- `DPU_DATA_FORMAT = 0xE0` (BS_MUL_SHIFT_VALUE_NEG=7)
- 591 tasks total (197 per core × 3 cores) for YOLOv5s

## Other Key Findings

### INT16 Output Not Supported
Setting `OUT_PRECISION=1` (INT16) in `DPU_DATA_FORMAT` causes NPU timeout.
The Rockchip NPU variant does not support INT16 output mode, despite the
register field being present in the NVDLA-derived register spec.

### Bias Rescaling Has Negligible Effect
The existing `bias * ws[oc]/max_ws` rescaling is mathematically incorrect
(introduces extra `ws[oc]/max_ws` factor on bias term), but the effect on
int8 output is negligible because conv_scale values are very small.

### Small Spatial Size Limitation
Models with 8×8 spatial size produce wrong output even on the system mesa
(max_diff=95-127 for per-tensor models). The Rocket driver was tuned
specifically for MobileNetV1 (224×224). This is unrelated to per-axis.

## Implementation: Per-Group CONV Decomposition

Implemented a per-group approach (groups of 16 channels = FEATURE_ATOMIC_SIZE):

### Architecture
1. Each per-axis CONV is split into `ceil(OC/16)` operations
2. Each group gets its own weight BO, bias BO, and `conv_scale`
3. `conv_scale = max(weight_scale[g*16 .. g*16+15])` within the group
4. Within-group `per_axis_correction` handles residual ratio
5. Output tensor uses sparse 2x layout (32-channel HW minimum)
6. CPU compaction step moves data from sparse to contiguous after NPU

### Files Changed
| File | Location | Changes |
|------|----------|---------|
| `rkt_ml.h` | local src/ | Added `output_tensor_channels`, `per_channel_group_offset` fields |
| `rkt_ml.c` | local src/ | Per-group lowering, compaction, adjusted correction |
| `rkt_coefs.c` | board | Added `rkt_fill_weights_group`, `rkt_fill_biases_group` |
| `rkt_coefs.h` | board | Declared new functions |
| `rkt_regcmd.c` | board | Added `per_channel_group_offset` to DST_BASE_ADDR |
| `rkt_task.c` | board | Fixed `weights_kernels` to match output_channels alignment |

### Results
- **MobileNetV1**: Still bit-exact vs system mesa (max_diff=0)
- **YOLO latency**: 292ms (was 1120ms — 3.8x faster due to smaller per-group CONVs)
- **YOLO accuracy**: max_diff=217-255, mean=17-48 (similar to original max_scale approach)

### Current Limitation
Within-group scale ratios remain up to 18x because channels are grouped by
position, not by scale similarity. This means the per-group max_scale approach
still suffers from significant int8 clipping within each group.

## Next Steps for Full Fix

### Option A: Scale-Sorted Grouping
Sort channels by `weight_scale` before grouping. The first group gets the 16
channels with most similar scales, etc. After NPU CONV, un-sort the output.
- Minimizes within-group ratios (approaching 1.0 for smooth scale distributions)
- Requires weight/bias reordering and output scatter
- Moderate implementation effort

### Option B: True Per-Channel Processing (RKNN-Style)
Process each output channel individually (`output_channels=1`).
- Eliminates within-group ratio entirely (ratio always 1.0)
- Requires discovering how RKNN sets HW registers for ch=1
  (our 32-channel minimum may need relaxation)
- Each channel's output is 16 bytes per pixel (1 group), needs scatter to
  correct position in full tensor
- ~8000 tasks for YOLO (may have overhead concerns)

### Option C: Hybrid Per-Channel/Per-Group
Use per-channel for layers with high scale ratios (>4x), per-group for others.
Combines benefits of both approaches.

## Intercept Tool Reference
- **Build**: `gcc -shared -fPIC -o intercept.so intercept.c -I... -ldl`
- **Run**: `LD_PRELOAD=intercept.so python3 run_rknn.py model.rknn`
- **Parse**: `python3 parse_regcmd.py regcmd*.bin`
- **MEM_CREATE ioctl**: `0xc0306442` (struct size 48, differs from header's 40)
- **Captured data**: `/tmp/rknn_intercept/` on board (vendor kernel)
