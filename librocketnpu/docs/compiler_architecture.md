# Open-Source NPU Compiler Architecture

Research document for building an open-source compiler for the Rockchip RKNPU,
as an alternative to the proprietary rknn-toolkit2/librknnc.so stack.

## 1. Framework Comparison

### ONNC (Open Neural Network Compiler)

Skymizer's open-source compiler framework designed for DLA accelerators.

- **Status**: Dormant — last commit July 2020, v1.3.0
- **License**: BSD-3-Clause
- **Architecture**: 5-phase pipeline: Translation → Optimization → Scheduling → Memory → Emission
- **Backend**: Only NVDLA fully implemented; generates NVDLA Loadable binaries
- **Build**: Complex C++ (autotools/CMake), requires "onnc-umbrella" wrapper
- **Relevance**: Rockchip's librknnc.so is derived from ONNC (same ONNX IR design,
  similar pass structure), but is a proprietary fork with undocumented modifications.

**Verdict**: Architecture is instructive but codebase is too stale to fork. The
5-phase pipeline model is the right mental model for our compiler.

### Apache TVM

Active ML compiler framework with BYOC (Bring Your Own Codegen) for custom accelerators.

- **Status**: Active — regular releases, large community
- **License**: Apache 2.0
- **Architecture**: Relay IR → TE (Tensor Expression) → target code
- **BYOC**: Graph-level codegen dispatch to custom backends
- **VTA**: Open tensor accelerator design with TVM integration
- **Build**: ~1M LOC, Python + C++, heavy dependency chain

**Verdict**: Best framework if targeting many model architectures and wanting
auto-tuning. Overkill for our focused CONV-only NPU target. BYOC integration
would require wrapping our regcmd generation as a TVM external codegen — possible
but adds massive dependency for little gain over standalone approach.

### ONNX-MLIR

MLIR-based compiler for ONNX models, with multi-level IR lowering.

- **Status**: Active (IBM-led)
- **Architecture**: ONNX dialect → Krnl dialect → Affine → LLVM
- **Strength**: Polyhedral optimization, loop fusion, vectorization
- **Build**: Requires LLVM/MLIR build (very heavy)

**Verdict**: Most modern approach but highest complexity. MLIR's pass-centric design
is excellent but the LLVM dependency makes it impractical for embedded deployment.

### Standalone (evolve librocketnpu)

Extend our existing ~3,500-line compiler backend with an ONNX frontend.

- **Status**: Working now — produces correct inference on hardware
- **Dependencies**: libdrm only (+ libprotobuf-c for ONNX parsing)
- **Architecture**: Already implements scheduling, memory allocation, code emission
- **Build**: Single `make` command, ~10 seconds

**Verdict**: Recommended. We already have 80% of a compiler. Adding an ONNX frontend
completes the picture with minimal new dependencies.


## 2. Current librocketnpu Pipeline — Mapped to Compiler Phases

```
ONNC Phase          │ librocketnpu Implementation
════════════════════╪══════════════════════════════════════════════
                    │
1. TRANSLATION      │ rnpu_tflite.c (362 lines)
   (IR building)    │   Parse TFLite FlatBuffer → rnpu_operation[]
                    │   Extract: op type, tensor shapes, quant params
                    │   Hand-written FlatBuffer decoder (zero deps)
                    │
2. OPTIMIZATION     │ rnpu_model.c:960-1080
   (graph passes)   │   Per-channel scale-sorted grouping
                    │   Greedy partitioning with configurable ratio
                    │   Channel reordering for min within-group variance
                    │
3. SCHEDULING       │ rnpu_task.c (313 lines)
   (task ordering)  │   CBUF bank allocation (12 banks: weight vs input)
                    │   Spatial tiling for large inputs
                    │   Weight reuse optimization across frames
                    │
4. MEMORY ALLOC     │ rnpu_model.c:1100-1250
   (buffer layout)  │   BO allocation: weights, biases, BRDMA, regcmd, I/O
                    │   Offset computation within shared BOs
                    │   DMA address assignment per task
                    │
5. CODE EMISSION    │ rnpu_regcmd.c (1420 lines)
   (machine code)   │   Register command generation per task
                    │   4 paths: standard, per-channel GS=1, BRDMA, hybrid
                    │   ~130 register commands per task across 5 HW units
                    │   Quantization math: float→fixed scale/shift
                    │
   RUNTIME          │ rnpu_drm.c (421 lines)
   (execution)      │   DRM IOCTL submission (CREATE_BO, SUBMIT)
                    │   Multi-core job dispatch
                    │   Fence/sync handling
```

**Total backend**: ~3,500 lines (excluding auto-generated registers.h)

### Register Command Format

Each hardware instruction is a packed 64-bit value:
```
[63..48]  target    — hardware unit (PC, CNA, CORE, DPU, DPU_RDMA, ...)
[47..16]  value     — 32-bit register value
[15..0]   register  — register offset within unit
```

The NPU is not instruction-programmed — it's register-programmed. The "program counter"
(PC) unit is a DMA engine that reads register command buffers from memory and writes
the values to the convolution engine (CNA), data processing unit (DPU), etc.


## 3. Proposed ONNX Frontend Design

### Why ONNX?

RKNN-toolkit2 converts all input formats (TFLite, PyTorch, Caffe) to ONNX internally.
The optimized ONNX graph (`check0_base_optimize.onnx`) preserves:
- QuantizeLinear/DequantizeLinear nodes with per-channel scale/zero_point
- Conv nodes with proper weight tensor references
- Graph topology after basic optimization passes

By consuming ONNX directly, we can:
1. Accept output from any ONNX-producing tool (PyTorch export, tf2onnx, etc.)
2. Consume RKNN-toolkit's optimized graph for apples-to-apples comparison
3. Support quantized models from any framework (QAT or PTQ)

### Parsing Approach: protobuf-c

ONNX uses Protocol Buffers (proto3). Options for C parsing:

| Approach | Dependencies | Code size | Dev effort |
|----------|-------------|-----------|------------|
| **protobuf-c** (code-gen) | libprotobuf-c (~500KB .so) | ~3K gen lines | Low |
| Hand-written parser | None | ~2-5K lines | High |
| nanopb | nanopb library | ~1K gen lines | Medium |

**Chosen: protobuf-c** — mature, well-tested, one-time code generation from onnx.proto3.

### ONNX Graph Structure for Quantized Models

```
QuantizeLinear(input, in_scale, in_zp)          → quantized_input
DequantizeLinear(quantized_input, in_scale, in_zp) → dequant_input
DequantizeLinear(weights, w_scale, w_zp)        → dequant_weights
Conv(dequant_input, dequant_weights, bias)      → conv_output
QuantizeLinear(conv_output, out_scale, out_zp)  → quantized_output
```

For per-channel quantization, `w_scale` is a 1D tensor with one value per output channel.

### Frontend Implementation (`rnpu_onnx.c`)

```c
// Core data structure — mirrors what rnpu_tflite.c produces
struct rnpu_onnx_model {
    // Graph topology
    unsigned n_conv_ops;
    struct rnpu_onnx_conv {
        unsigned index;                  // sequential conv index
        char name[64];                   // ONNX node name
        // Tensor shapes
        unsigned ic, oc, kh, kw;         // conv parameters
        unsigned ih, iw, oh, ow;         // spatial dimensions
        unsigned stride_h, stride_w;
        unsigned pad_top, pad_bottom, pad_left, pad_right;
        unsigned group;                  // 1=standard, oc=depthwise
        // Quantization
        float input_scale, output_scale;
        int32_t input_zp, output_zp;
        float *weight_scales;            // per-channel (length=oc)
        int32_t *weight_zps;             // per-channel
        // Weight data
        int8_t *weights;                 // raw int8 weight tensor
        int32_t *biases;                 // int32 bias tensor (may be NULL)
    } *conv_ops;
};

// API
struct rnpu_onnx_model *rnpu_onnx_load(const char *path);
void rnpu_onnx_free(struct rnpu_onnx_model *m);
```

Parsing algorithm:
1. Read file, call `onnx__model_proto__unpack()`
2. Build name→TensorProto map from `graph->initializer[]`
3. Walk `graph->node[]` topologically:
   - For `DequantizeLinear`: record (tensor_name → scale, zp) mapping
   - For `Conv`: look up input/weight/output quant params from DequantizeLinear map
   - Extract weight data from initializer TensorProto's `raw_data`
   - Parse attributes: `kernel_shape`, `strides`, `pads`, `group`
4. Build `rnpu_onnx_conv[]` array
5. Free protobuf tree

### Integration with Existing Pipeline

```
rnpu_onnx_load("model.onnx")
    │
    ▼
rnpu_onnx_conv[]  ──────────►  rnpu_operation[]  (same struct as TFLite path)
                                    │
                    rnpu_model_create_from_onnx()
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              rnpu_task.c     rnpu_coefs.c    rnpu_regcmd.c
              (splitting)     (weights)       (emission)
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                              rnpu_drm.c
                              (submit)
```

The key integration point is `rnpu_model.c` — we add a new entry point
`rnpu_model_load_onnx()` that populates the same `rnpu_model` struct as
`rnpu_model_load_tflite()`, then the rest of the pipeline is unchanged.


## 4. Custom Binary Format Proposal

For "compile once, run many" deployment without runtime model parsing:

### Format: RNPU (Rocket NPU Binary)

```
Offset  Size    Field
──────  ──────  ─────────────────────────────
0x00    4       Magic: "RNPU"
0x04    4       Version: 1
0x08    4       Number of operations
0x0C    4       Total weight data size
0x10    4       Total BRDMA data size
0x14    4       Total regcmd size
0x18    4       Input tensor size
0x1C    4       Output tensor count
0x20    var     Operation table (N × 64 bytes each):
                  - weight_offset, weight_size
                  - bias_offset, bias_size
                  - brdma_offset, brdma_size
                  - regcmd_offset, regcmd_count
                  - input/output tensor indices
                  - quantization params (scale, zp)
...     var     Weight data blob (64-byte aligned)
...     var     BRDMA data blob (64-byte aligned)
...     var     Regcmd data blob (64-byte aligned)
```

**Not Rockchip-compatible** — this is our own format, simpler and fully documented.
Loading skips all compilation: mmap file, create BOs from pre-computed data, submit.

### Implementation Priority

This is a Phase 2 deliverable. The ONNX frontend (Phase 1) provides immediate value
by enabling direct model consumption. Binary output is an optimization for deployment.


## 5. Roadmap

### Phase 1: ONNX Frontend (this PR)
- protobuf-c code generation ✓
- rnpu_onnx.c parser
- test_onnx_parse validation tool
- **Effort**: ~500 lines new code, 1-2 sessions

### Phase 2: ONNX-to-Inference Pipeline
- Wire rnpu_onnx into rnpu_model (create operations from ONNX data)
- Handle tensor dimension differences (ONNX NCHW vs TFLite NHWC)
- Run inference on check0_base_optimize.onnx and compare vs TFLite path
- **Effort**: ~300 lines, 1 session

### Phase 3: Custom Binary Output (optional)
- RNPU binary format serializer
- RNPU binary loader (skip compilation on load)
- **Effort**: ~400 lines, 1 session

### Phase 4: More Op Support (ongoing)
- ONNX nodes beyond Conv: Add, Relu, MaxPool, Resize, Concat, Sigmoid, etc.
- Map to existing SW ops in rnpu_sw_ops.c
- **Effort**: ~100 lines per op type

### Phase 5: TVM BYOC Integration (future, optional)
- Only if targeting diverse model architectures beyond YOLO/MobileNet
- Wrap rnpu_regcmd as TVM external codegen
- **Effort**: Large — TVM build integration, relay patterns, etc.


## References

- [ONNC GitHub](https://github.com/ONNC/onnc) — BSD-3, dormant since 2020
- [ONNC Backend Porting Guide](https://github.com/ONNC/onnc/blob/master/docs/ONNC-Backend-Porting-Guide.md)
- [Apache TVM](https://github.com/apache/tvm) — BYOC framework
- [ONNX proto3 schema](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)
- [protobuf-c](https://github.com/protobuf-c/protobuf-c) — C protobuf implementation
- [wuhanstudio/onnx-parser](https://github.com/wuhanstudio/onnx-parser) — minimal C ONNX parser example
