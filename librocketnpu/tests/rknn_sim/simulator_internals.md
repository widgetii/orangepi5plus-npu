# RKNN Simulator Internals — Reverse Engineering Notes

## 1. Simulator Architecture — How Parts Interact

```
Python API (rknn.py)
  └─ RKNNBase (rknn_base.cpython-310.so) — orchestrator
       ├─ .ir : IRGraph — ONNX-based intermediate representation
       │    └─ .graph : onnx.GraphProto — 154 nodes for YOLO
       │    └─ .nodes : dict{name→index} — node registry
       │    └─ get_quant_ann(tensor, attr) — per-tensor quant params
       │    └─ get_const(name) — weight/bias data access
       │
       ├─ .simulator : Simulator (simulator.cpython-310.so)
       │    ├─ .inference() — main entry: preprocess → session.run → fake_quant → output
       │    ├─ .accuracy_analysis() — per-layer FP32 vs quantized comparison
       │    └─ Functions:
       │         ├─ quant_tensor(f_tensor, s_par, d_dtype, d_layout) — FP32→INT8
       │         ├─ dequant_tensor(q_tensor, s_par, d_dtype, ori_dtype) — INT8→FP32
       │         ├─ fake_tensor(tensor, par, node, ir, a, can_dump) — fake quantize (round-trip)
       │         └─ simu_symmetric(q_tensor, ir, par, sess) — symmetric quant simulation
       │
       ├─ Session (session.cpython-310.so) — ONNX Runtime wrapper
       │    ├─ .run(input_dict, inters=None, quant_tab={}) — executes via onnxruntime
       │    │    └─ Returns FP32 outputs (range [0,1] for sigmoid)
       │    │    └─ inters= parameter: request intermediate node outputs
       │    └─ Uses onnxruntime InferenceSession under the hood
       │
       ├─ .rknn_runtime : RKNNRuntime (rknn_runtime.cpython-310.so)
       │    └─ C API wrapper using ctypes for librknn_api.so
       │    └─ Defines RKNNInput, RKNNOutput, RKNNTensorAttr structs
       │    └─ Used ONLY for on-device inference (target != None)
       │
       └─ Native libs:
            ├─ librknnc.so (31MB) — compiler backend (ONNC-based)
            │    ├─ RKNNCompiler_simulator() — builds simulator graph
            │    ├─ RKNNCompiler_build() — builds hardware graph
            │    ├─ rknn::RKNNComputeVisitor — graph execution visitor
            │    ├─ rknn::RKNNRegisterEmitVisitor — register command emitter
            │    └─ rknn::RKNNPerChannelPrep pass — per-channel decomposition
            │
            └─ librknn_api.so (225KB) — standard RKNN C API
                 ├─ rknn_init/rknn_run/rknn_destroy — full C API
                 ├─ __qnt_f32_to_none / __qnt_none_to_f32 — quant helpers
                 └─ __neon_convert_u8c3_nhwc — NEON layout conversion
```

**Critical insight**: The simulator does NOT use librknnc.so for inference. It uses:
1. **ONNX Runtime** for FP32 graph execution (Session.run)
2. **Python/Cython quant functions** for INT8 simulation (quant_tensor/fake_tensor)
3. **librknnc.so only for compilation** (graph optimization, op fusion, building the ONNX IR)

### Data Flow During Simulator Inference

```
input image (uint8 NHWC)
  → preprocess (normalize, layout convert → float32 NCHW)
  → Session.run(input_dict)          # ONNX Runtime FP32 execution
  → for each output tensor:
       fake_tensor(fp32_output, quant_params)   # FP32 → INT8 → FP32 round-trip
  → post-process (layout convert back to NHWC if needed)
  → return quantized outputs
```

The simulator computes the entire graph in FP32, then applies quantization only at
output boundaries. This means **intermediate activations are NOT quantized** in the
default simulator path — only the final outputs undergo fake quantization.

This is fundamentally different from hardware execution where every layer's output is
quantized to INT8 before feeding the next layer (accumulating quantization error).

### Graph Optimization Passes (librknnc.so)

The RKNN compiler applies these passes (visible from exported ONNX checkpoints):

| Checkpoint file | Pass | Effect |
|---|---|---|
| `check0_base_optimize.onnx` | Base optimization | Basic graph cleanup |
| `check1_fold_constant.onnx` | Constant folding | Pre-compute constant subgraphs |
| `check2_correct_ops.onnx` | Op correction | Fix op semantics for target HW |
| `check3_fuse_ops.onnx` | Op fusion | Fuse conv+bn+relu, etc. |

After these passes, the optimized ONNX graph is what Session.run executes. The graph
structure may differ from the original TFLite model due to fusion and optimization.


## 2. dlopen/dlsym Analysis — What's Accessible

### librknn_api.so — Full C API

**Exported symbols** (accessible via dlopen/ctypes):
- `rknn_init`, `rknn_run`, `rknn_inputs_set`, `rknn_outputs_get`, `rknn_query`, `rknn_destroy`
- `rknn_create_mem`, `rknn_destroy_mem`, `rknn_set_io_mem`, `rknn_set_weight_mem`
- `rknn_get_perf_detail`, `rknn_run_async`
- `__qnt_f32_to_none` / `__qnt_none_to_f32` — quantization helpers
- `__neon_convert_u8c3_nhwc` — NEON layout conversion

**Limitation**: Only works for on-device mode (needs `target` set and RKNPU hardware).
Simulator mode through this API is blocked for load_rknn path.

### librknnc.so — Compiler Backend

**Exported C functions:**
- `RKNNCompiler_simulator()` / `RKNNCompiler_build()` — accessible via ctypes
- `RKNNCompiler_destroy()`, `RKNNCompiler_create()`

**Internal C++ classes** (mangled, not directly callable):
- `rknn::RKNNCompiler::run()`, `::build()`, `::simulator()`
- `rknn::RKNNComputeVisitor` — graph execution visitor
- `rknn::RKNNRegisterEmitVisitor` — register command emitter
- `rknn::RKNNPerChannelPrep` — per-channel quantization decomposition pass

**Useful but opaque**: No documented way to intercept per-layer computation.

### simulator.cpython-310.so — Python-Visible Functions

These are the core INT8 simulation functions, callable directly from Python:

```python
from rknn.api.simulator import quant_tensor, dequant_tensor, fake_tensor, simu_symmetric

# Quantize FP32 tensor to INT8
quant_tensor(f_tensor, s_par, d_dtype, d_layout=None)

# Dequantize INT8 tensor back to FP32
dequant_tensor(q_tensor, s_par, d_dtype, ori_dtype)

# Round-trip: FP32 → INT8 → FP32 (fake quantization)
fake_tensor(tensor, par, node=None, ir=None, a=1.0, can_dump=False)

# Symmetric quantization path
simu_symmetric(q_tensor, ir, par, sess)

# Internal helpers
_snapshot_tensor()   # accuracy analysis snapshots
_calc_err()          # error calculation
```

**These ARE the simulator's INT8 logic** — they define exactly how RKNN rounds and
clips values during quantization. Calling them directly lets us compare rounding
behavior against our librocketnpu implementation.

### session.cpython-310.so — ONNX Runtime Bridge

```python
from rknn.api.session import Session

sess = Session(ir)                    # Creates onnxruntime.InferenceSession from IRGraph
outputs = sess.run(input_dict)        # Returns FP32 outputs
outputs = sess.run(input_dict, inters=['node_name1', 'node_name2'])  # Intermediate outputs
```

- Input must be **NCHW float32**, keyed by tensor name
- `inters=` parameter requests intermediate node outputs (needs matching session build)
- Returns numpy arrays

### ir_graph.cpython-310.so — Graph & Quant Params

```python
# Access quantization parameters
scale, zp = ir.get_quant_ann(tensor_name, 'scale'), ir.get_quant_ann(tensor_name, 'zero_point')

# Access weight/bias data
weights = ir.get_const(weight_tensor_name)

# ONNX graph access
graph = ir.graph          # onnx.GraphProto
nodes = ir.nodes          # dict{name → index}

# Export optimized graph
ir.save_onnx('debug.onnx')
```


## 3. Action Points for Achieving RKNN Bit-Match Results

### Action 1: Extract Per-Layer Quantization Parameters — DONE

**Status**: COMPLETED. Script: `extract_quant_params.py`, output: `quant_params.json`

**Key finding**: `ir.get_quant_ann()` returns EMPTY STRINGS for QAT models — the
post-fusion RKNN IR (154 nodes) has no quant annotations. Instead, quant params must
be extracted from `check0_base_optimize.onnx` which preserves QuantizeLinear/DequantizeLinear
nodes from the TFLite→ONNX conversion.

```python
# CORRECT approach — use pre-fusion ONNX graph:
import onnx
from onnx.numpy_helper import to_array
model = onnx.load('check0_base_optimize.onnx')
# 444 nodes: 199 DequantizeLinear, 87 QuantizeLinear, 61 Conv, etc.
# Conv inputs come from DequantizeLinear nodes that carry scale/zp as initializers

# WRONG approach — ir.get_quant_ann returns empty strings:
# ir = rknn.rknn_base.ir  # (note: rknn_base, not rknnbase)
# ir.get_quant_ann(tensor_name, 'scale')  # → '' (empty string)
```

**Results** (61 conv ops matched between RKNN and TFLite):
- Weight scales: **22/61 EXACT**, 6 SIZE_MISMATCH, 33 differ
- Output scales: **26/61 EXACT**, 35 differ
- Output ZP: **56/61 EXACT**, 5 differ
- Early convs (0-25) mostly match; divergence in later layers from RKNN op fusion
- SIZE_MISMATCH in detection heads: RKNN reorders/reshapes YOLO heads differently
- Sorted grouping consistently improves worst-case ratio (0.08→0.24 at worst)

**Priority**: HIGH — this is the foundation for all other comparisons.

### Action 2: Run Session.run for Per-Layer FP32 Reference — DONE

**Status**: COMPLETED. Script: `per_layer_reference.py`, output: `per_layer_ref/`

**Key findings on Session API:**
- `sess.run(input, inters=[...])` does NOT return intermediates (silently ignored)
- `Session(ir)` creates 154 per-node sub-sessions in `sess_list[]`, but each is per-node
  (not cumulative) — requires chaining outputs manually (impractical)
- **Working approach**: Modify ONNX graph to add Conv outputs as extra graph outputs,
  then pass modified model to `Session(ir, model=modified_onnx_model)`:

```python
import onnx
from onnx import helper
ir.save_onnx('/tmp/base.onnx')
model = onnx.load('/tmp/base.onnx')
for name in conv_out_names:
    model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
sess = Session(ir, model=model)  # model= takes ONNX model object, NOT path string
result = sess.run({input_name: input_nchw})
# result[:N_original] = final outputs, result[N_original:] = conv intermediates
```

- Input: FP32 NCHW (raw uint8 values transposed, no normalization — RKNN handles it)
- Output: FP32 NCHW per Conv node, quantized to INT8 using check0 QuantizeLinear params
- Cannot use vanilla onnxruntime — RKNN graph has custom `exDataConvert` ops
- 61 Conv + 3 final outputs saved to `per_layer_ref/` (FP32 + INT8 binaries)

**Priority**: HIGH — pinpoints where our computation diverges.

### Action 3: Understand fake_tensor Quantization Scheme — DONE

**Status**: COMPLETED. No new scripts — findings documented here.

**Rounding mode**: Round-half-to-even (banker's rounding), same as `np.rint()`.
```
Input:  [ 0.5  1.5  2.5  3.5 -0.5 -1.5 -2.5 -3.5]
Output: [ 0    2    2    4    0   -2   -2   -4  ]
```

**Saturation**: Clips to [-128, 127]. -128.5 → -128 (round-to-even then clip).

**Exact formula**:
```
int8_val = clip(rint(fp32_val / scale) + zero_point, -128, 127)
```
NumPy equivalent: `np.clip(np.rint(fp32 / scale) + zp, -128, 127).astype(np.int8)`

**Manual match**: 100% exact — random 1x32x10x10 tensor matches `quant_tensor()` perfectly.

**Per-channel scales**: Applied along axis=1 (NCHW channel dim), each channel independent.

**`fake_tensor`**: NOT usable standalone — needs `qmethod` key in params dict (internal
RKNN pipeline state). Use manual `np.rint` or `quant_tensor` + `dequant_tensor` instead.

**`quant_tensor` param dict** (note: `dtype` key is REQUIRED):
```python
params = {'scale': np.array([...], np.float32), 'zero_point': np.array([...], np.int32), 'dtype': 'int8'}
quant_tensor(fp32_tensor_nchw, params, 'int8')  # returns int8 ndarray
dequant_tensor(int8_tensor, params, 'float32', 'int8')  # returns float32
```

**Priority**: HIGH — rounding differences cause ~1 LSB error per layer, accumulating.

### Action 4: Use accuracy_analysis for Automated Per-Layer Comparison

**Goal**: Get RKNN's own per-layer FP32 vs quantized comparison.

```python
rknn = RKNN()
rknn.config(target_platform='rk3588', optimization_level=2)
rknn.load_tflite('yolov5s_relu_int8.tflite')
rknn.build(do_quantization=False)  # QAT model, no re-quantization

# This saves per-layer snapshots to disk
rknn.accuracy_analysis(inputs=['input0.bin'], target=None)
```

Note: Failed previously with QAT models — retry with `optimization_level=2`
(RKNN warned: "QAT model with optimization_level=3 may affect accuracy").

**Priority**: MEDIUM — automated but may not work for QAT models.

### Action 5: Test optimization_level=2

**Goal**: Check if simulator output changes with different optimization level.

RKNN warned: "QAT model loaded but optimization_level=3, some passes will affect
accuracy." Re-run simulator with `optimization_level=2` and compare outputs against
current golden (which used default optimization_level=3).

```python
rknn.config(target_platform='rk3588', optimization_level=2)
```

If sim output changes, the diff tells us which optimization passes affect accuracy.

**Priority**: MEDIUM — may reduce sim-vs-hardware gap.

### Action 6: Build Per-Layer Comparison Tool

**Goal**: Script that identifies exact layer where librocketnpu diverges from RKNN.

Pipeline:
1. Load TFLite → build RKNN IR
2. Run Session.run for FP32 reference (all intermediates)
3. Apply quant_tensor per layer to get RKNN's quantized reference
4. Compare with librocketnpu per-layer dumps (`RNPU_TRACE_OPS=1`)
5. Report first layer with significant divergence

Requires matching layer names between RKNN's ONNX graph and our TFLite op indices.

**Priority**: HIGH — the key debugging tool.

### Action 7: Investigate Graph Optimization Passes

**Goal**: Understand if RKNN's op fusion changes computation vs our TFLite-based decomposition.

```python
# Export optimized ONNX graph
ir.save_onnx('optimized_graph.onnx')

# Compare node count and types vs TFLite
import onnx
model = onnx.load('optimized_graph.onnx')
for node in model.graph.node:
    print(f"{node.op_type}: {node.input} → {node.output}")
```

Check files already exported:
- `check0_base_optimize.onnx` through `check3_fuse_ops.onnx`

Key concern: If RKNN fuses conv+bn+relu differently, the intermediate precision may
differ from our separate-operation approach.

**Priority**: LOW — likely not the main source of divergence for QAT model.

### Action 8: Fix Output 2 (20x20) Being All -128

**Goal**: Fix the hard bug where librocketnpu Output 2 is constant -128.

This is a separate bug from quantization accuracy — likely a spatial dimension or
channel count issue in the last YOLO head. The 20x20 head processes the smallest
spatial size with the most channels.

Debugging approach:
1. Check if the output tensor dimensions match expectations (1x255x20x20)
2. Verify the last conv feeding this head produces non-constant values
3. Check for off-by-one in spatial split or channel offset for small feature maps

**Priority**: HIGHEST — this is a correctness bug, not a precision issue. Fix first
before worrying about quantization accuracy.


## Key Files Reference

| File | Role |
|------|------|
| `rknn/api/rknn.py` | Readable Python entry point (426 lines) |
| `rknn/api/simulator.cpython-310.so` | INT8 sim logic: quant_tensor, fake_tensor |
| `rknn/api/session.cpython-310.so` | ONNX Runtime wrapper: Session.run |
| `rknn/api/rknn_base.cpython-310.so` | Orchestrator: .ir, .simulator, .rknn_runtime |
| `rknn/api/ir_graph.cpython-310.so` | IRGraph: graph nodes, quant annotations, weights |
| `rknn/api/lib/linux-x86_64/librknnc.so` | Compiler backend (opaque C++) |
| `rknn/api/lib/linux-x86_64/librknn_api.so` | C API (on-device only) |
| `run_yolo_sim.py` | Our simulator runner script (in this directory) |
| `run_yolo_sim_rknn.py` | RKNN model runner script |
| `compare_outputs.py` | Output comparison tool |
| `extract_quant_params.py` | RKNN vs TFLite quant param extraction & comparison |
| `per_layer_reference.py` | Per-layer FP32/INT8 reference extraction via ONNX Runtime |
| `quant_params.json` | Extracted quant params for all 61 CONV ops |
| `per_layer_ref/` | 61 Conv FP32+INT8 references + layer_mapping.json |
| `check0_base_optimize.onnx` | Pre-fusion ONNX with QuantizeLinear/DequantizeLinear nodes |

## Verification Commands

```bash
# Confirm quant_tensor is callable from Docker
docker run --rm -v "$(pwd):/data" rknn-sim python3 -c "
from rknn.api.simulator import quant_tensor, dequant_tensor
import numpy as np
t = np.array([0.5, -0.3, 1.0, 0.0], dtype=np.float32)
print('quant_tensor callable:', callable(quant_tensor))
print('dequant_tensor callable:', callable(dequant_tensor))
"

# List all exported symbols from librknnc.so
docker run --rm rknn-sim nm -D /usr/lib/python3/dist-packages/rknn/api/lib/linux-x86_64/librknnc.so | grep -i 'rknn'

# Export optimized ONNX graph for inspection
docker run --rm -v "$(pwd):/data" rknn-sim python3 -c "
from rknn.api import RKNN
rknn = RKNN()
ret = rknn.load_rknn('/data/yolov5s_relu_int8.rknn')
if ret == 0:
    ir = rknn.rknnbase.ir
    ir.save_onnx('/data/optimized.onnx')
    print(f'Graph nodes: {len(ir.graph.node)}')
"
```
