#!/usr/bin/env python3
"""Extract per-layer FP32 and INT8 reference outputs from RKNN's ONNX Runtime.

Runs inside the rknn-sim Docker container:
  docker run --rm -v "$(pwd):/data" -w /data rknn-sim python3 /data/per_layer_reference.py

Approach:
  1. Build RKNN IR from TFLite (post-fusion ONNX graph, 154 nodes)
  2. Modify ONNX graph to expose all Conv node outputs as extra graph outputs
  3. Run Session with modified model → get FP32 NCHW intermediates for all 61 Convs
  4. Quantize FP32 → INT8 using per-tensor output scale/zp from check0_base_optimize.onnx
  5. Save per-layer INT8 tensors + mapping JSON for comparison with librocketnpu
"""

import json
import os
import sys
import numpy as np


TFLITE_MODEL = 'yolov5s_relu_int8.tflite'
INPUT_FILE = 'input0.bin'
OUTPUT_DIR = 'per_layer_ref'
QUANT_PARAMS_FILE = 'quant_params.json'


def build_rknn_ir(tflite_path):
    """Build RKNN IR from TFLite model."""
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588',
                quantized_dtype='asymmetric_quantized-8',
                quantized_algorithm='normal',
                quantized_method='channel')

    ret = rknn.load_tflite(model=tflite_path)
    if ret != 0:
        print(f'load_tflite failed: {ret}', file=sys.stderr)
        sys.exit(1)

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f'build failed: {ret}', file=sys.stderr)
        sys.exit(1)

    return rknn


def extract_output_quant_params(onnx_path='check0_base_optimize.onnx'):
    """Extract per-Conv output scale/zp from pre-fusion ONNX graph.

    Returns list of (scale, zp) tuples, one per Conv node in topological order.
    """
    import onnx
    from onnx.numpy_helper import to_array

    model = onnx.load(onnx_path)
    graph = model.graph

    # Build initializer lookup
    inits = {}
    for init in graph.initializer:
        try:
            inits[init.name] = to_array(init)
        except Exception:
            pass

    # Build consumer map
    input_consumers = {}
    for n in graph.node:
        for inp in n.input:
            input_consumers.setdefault(inp, []).append(n)

    conv_quant = []
    for node in graph.node:
        if node.op_type != 'Conv':
            continue

        scale, zp = None, None
        conv_output = node.output[0]

        # Look for QuantizeLinear consuming this conv's output (directly or via Relu)
        for consumer in input_consumers.get(conv_output, []):
            if consumer.op_type == 'QuantizeLinear':
                s = inits.get(consumer.input[1])
                z = inits.get(consumer.input[2]) if len(consumer.input) > 2 else None
                scale = float(s) if s is not None else None
                zp = int(z) if z is not None else None
                break
            if consumer.op_type == 'Relu':
                for c2 in input_consumers.get(consumer.output[0], []):
                    if c2.op_type == 'QuantizeLinear':
                        s = inits.get(c2.input[1])
                        z = inits.get(c2.input[2]) if len(c2.input) > 2 else None
                        scale = float(s) if s is not None else None
                        zp = int(z) if z is not None else None
                        break

        conv_quant.append((scale, zp))

    return conv_quant


def run_with_intermediates(rknn, input_path):
    """Run inference and return FP32 intermediates for all Conv nodes."""
    from rknn.api.session import Session
    import onnx
    from onnx import helper

    ir = rknn.rknn_base.ir

    # Save and modify ONNX graph to expose conv outputs
    ir.save_onnx('/tmp/base_model.onnx')
    model = onnx.load('/tmp/base_model.onnx')

    conv_nodes = [(i, n) for i, n in enumerate(model.graph.node) if n.op_type == 'Conv']
    conv_out_names = [n.output[0] for _, n in conv_nodes]

    for name in conv_out_names:
        model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    # Create session with modified model
    sess = Session(ir, model=model)

    # Prepare input: FP32 NCHW (raw uint8 values, matching RKNN's preprocessing)
    input_nhwc = np.fromfile(input_path, dtype=np.uint8).reshape(1, 640, 640, 3)
    input_nchw = np.transpose(input_nhwc.astype(np.float32), (0, 3, 1, 2))
    input_name = ir.graph.input[0].name

    print(f'Running inference with {len(conv_out_names)} intermediate outputs...')
    result = sess.run({input_name: input_nchw})

    # Result: first N are original outputs, then conv intermediates
    n_original = len([o for o in ir.graph.output])
    final_outputs = result[:n_original]
    conv_outputs = result[n_original:]

    if len(conv_outputs) != len(conv_out_names):
        print(f'WARNING: expected {len(conv_out_names)} conv outputs, got {len(conv_outputs)}')

    return conv_nodes, conv_outputs, final_outputs


def quantize_fp32_to_int8(fp32_tensor, scale, zp):
    """Quantize FP32 tensor to INT8 using scale and zero_point.

    int8_val = clip(round(fp32 / scale) + zp, -128, 127)
    """
    if scale is None or scale == 0:
        return None
    quantized = np.round(fp32_tensor / scale) + zp
    return np.clip(quantized, -128, 127).astype(np.int8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Build RKNN IR
    print(f'Building RKNN IR from {TFLITE_MODEL}...')
    rknn = build_rknn_ir(TFLITE_MODEL)

    # Step 2: Get output quant params from pre-fusion graph
    print('Extracting output quantization parameters...')
    conv_quant = extract_output_quant_params()
    print(f'  Got quant params for {len(conv_quant)} Conv nodes')

    # Step 3: Run with intermediates
    conv_nodes, conv_outputs, final_outputs = run_with_intermediates(rknn, INPUT_FILE)
    print(f'  Got {len(conv_outputs)} Conv intermediate outputs')

    if len(conv_quant) != len(conv_outputs):
        print(f'WARNING: quant params ({len(conv_quant)}) != conv outputs ({len(conv_outputs)})')

    # Step 4: Save per-layer references
    layers = []
    n_saved = 0
    for i, ((node_idx, node), fp32_arr) in enumerate(zip(conv_nodes, conv_outputs)):
        fp32_arr = np.array(fp32_arr)
        scale, zp = conv_quant[i] if i < len(conv_quant) else (None, None)

        # Save FP32 (NCHW layout, as produced by ONNX Runtime)
        fp32_path = os.path.join(OUTPUT_DIR, f'conv_{i:02d}_fp32.bin')
        fp32_arr.astype(np.float32).tofile(fp32_path)

        # Quantize and save INT8
        int8_arr = None
        if scale is not None:
            int8_arr = quantize_fp32_to_int8(fp32_arr, scale, zp)
            int8_path = os.path.join(OUTPUT_DIR, f'conv_{i:02d}_int8.bin')
            int8_arr.tofile(int8_path)
            n_saved += 1

        # Compute stats
        shape = list(fp32_arr.shape)
        oc = shape[1]  # NCHW

        layer_info = {
            'conv_index': i,
            'rknn_node_index': node_idx,
            'name': node.name[:80],
            'shape_nchw': shape,
            'output_channels': oc,
            'fp32_range': [float(fp32_arr.min()), float(fp32_arr.max())],
            'fp32_mean': float(fp32_arr.mean()),
            'output_scale': scale,
            'output_zp': zp,
        }

        if int8_arr is not None:
            layer_info['int8_range'] = [int(int8_arr.min()), int(int8_arr.max())]
            unique = len(np.unique(int8_arr))
            layer_info['int8_unique'] = unique

        layers.append(layer_info)

        status = f'scale={scale:.6f} zp={zp}' if scale else 'NO QUANT'
        print(f'  Conv {i:2d}: {shape} {status} '
              f'fp32=[{fp32_arr.min():.2f}, {fp32_arr.max():.2f}]')

    # Save final outputs
    for i, arr in enumerate(final_outputs):
        arr = np.array(arr)
        arr.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, f'final_{i}_fp32.bin'))
        print(f'  Final output {i}: shape={arr.shape}')

    # Step 5: Save mapping JSON
    mapping = {
        'model': TFLITE_MODEL,
        'input': INPUT_FILE,
        'layout': 'NCHW',
        'dtype_fp32': 'float32',
        'dtype_int8': 'int8',
        'n_conv_layers': len(layers),
        'n_final_outputs': len(final_outputs),
        'layers': layers,
    }

    mapping_path = os.path.join(OUTPUT_DIR, 'layer_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f'\nSaved {n_saved} INT8 + {len(layers)} FP32 references to {OUTPUT_DIR}/')
    print(f'Mapping: {mapping_path}')

    rknn.release()


if __name__ == '__main__':
    main()
