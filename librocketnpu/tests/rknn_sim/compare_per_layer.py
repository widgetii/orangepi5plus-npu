#!/usr/bin/env python3
"""Compare librocketnpu per-layer outputs against RKNN references.

Usage:
  python3 compare_per_layer.py [--ref per_layer_ref] [--dump op_dump]

IMPORTANT: The per_layer_ref INT8 files are quantized from FP32 ONNX Runtime output.
These represent "ideal" quantization of a full-precision computation. Real NPU output
differs because each layer's int8 output feeds the next layer, accumulating quantization
error. The FP32 reference is useful for understanding quantization bounds but NOT for
direct comparison with NPU output (which operates in int8 throughout).

For actual accuracy comparison, use compare_outputs.py which compares librocketnpu's
final outputs against RKNN hardware golden (rknn_yolo_i8_*.bin).

Format notes:
  - librocketnpu dumps: uint8 NHWC with +0x80 offset (NPU int8 + 128)
  - RKNN reference: int8 NCHW from FP32 quantization (NOT from real int8 inference)
  - Op indices include SW ops; matched to CONV indices by tensor size
"""

import argparse
import json
import os
import sys
import numpy as np


def build_op_mapping(dump_dir, layers):
    """Map CONV indices to dump op indices by matching expected tensor sizes.

    librocketnpu dumps ALL ops (CONV + SW). We match by scanning dump files
    in order and checking if their size matches the next expected CONV output.
    """
    # Get all dump files sorted by op index
    dump_files = {}
    for f in os.listdir(dump_dir):
        if f.startswith('op_') and f.endswith('.bin'):
            idx = int(f[3:-4])
            dump_files[idx] = os.path.getsize(os.path.join(dump_dir, f))

    max_op = max(dump_files.keys()) if dump_files else 0

    # Match: walk through ops in order, assign CONV indices to ops whose
    # size matches expected H*W*C
    conv_to_op = {}
    conv_idx = 0
    for op_idx in range(max_op + 1):
        if op_idx not in dump_files:
            continue
        if conv_idx >= len(layers):
            break

        layer = layers[conv_idx]
        _, c, h, w = layer['shape_nchw']
        expected_size = h * w * c

        if dump_files[op_idx] == expected_size:
            conv_to_op[conv_idx] = op_idx
            conv_idx += 1

    return conv_to_op


def load_ref_int8(ref_dir, conv_idx, shape_nchw):
    """Load RKNN reference INT8 tensor (NCHW) and transpose to NHWC."""
    path = os.path.join(ref_dir, f'conv_{conv_idx:02d}_int8.bin')
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.int8).reshape(shape_nchw)
    # NCHW -> NHWC: (1,C,H,W) -> (H,W,C)
    return data[0].transpose(1, 2, 0)  # (H, W, C)


def load_dump_int8(dump_dir, op_idx, h, w, c):
    """Load librocketnpu dump (uint8 NHWC) and convert to int8."""
    path = os.path.join(dump_dir, f'op_{op_idx:02d}.bin')
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    expected = h * w * c
    if len(data) != expected:
        return None
    nhwc = data.reshape(h, w, c)
    # Convert uint8 (+0x80 offset) to int8
    return nhwc.astype(np.int16) - 0x80


def compare_layer(ref_nhwc, dump_nhwc):
    """Compare two int16 tensors."""
    diff = np.abs(ref_nhwc.astype(np.int32) - dump_nhwc.astype(np.int32))
    n = diff.size
    return {
        'exact': np.sum(diff == 0) / n * 100,
        'le1': np.sum(diff <= 1) / n * 100,
        'le5': np.sum(diff <= 5) / n * 100,
        'mean': float(diff.mean()),
        'max': int(diff.max()),
        'ge10_pct': np.sum(diff >= 10) / n * 100,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare per-layer NPU outputs vs RKNN reference')
    parser.add_argument('--ref', default='per_layer_ref', help='Reference directory')
    parser.add_argument('--dump', default='op_dump', help='librocketnpu dump directory')
    args = parser.parse_args()

    mapping_path = os.path.join(args.ref, 'layer_mapping.json')
    if not os.path.exists(mapping_path):
        print(f'ERROR: {mapping_path} not found. Run per_layer_reference.py first.')
        sys.exit(1)

    with open(mapping_path) as f:
        mapping = json.load(f)

    layers = mapping['layers']
    print(f'Reference: {len(layers)} Conv layers from {args.ref}/')
    print(f'Dump:      {args.dump}/')

    # Build CONV→op index mapping
    conv_to_op = build_op_mapping(args.dump, layers)
    print(f'Matched:   {len(conv_to_op)} CONV ops to dump files')
    print()

    # Header
    print(f'{"Conv":>4s} {"Op":>3s} {"Shape":>15s} {"exact%":>7s} {"<=1%":>6s} {"<=5%":>6s} '
          f'{"mean":>6s} {"max":>4s} {">=10%":>6s}  {"Status"}')
    print('-' * 85)

    first_diverge = None
    results = []

    for layer in layers:
        ci = layer['conv_index']
        shape = layer['shape_nchw']  # [1, C, H, W]
        _, c, h, w = shape

        if ci not in conv_to_op:
            print(f'{ci:4d} {"":>3s} {f"{h}x{w}x{c}":>15s}  -- no matching dump --')
            continue

        op_idx = conv_to_op[ci]
        ref = load_ref_int8(args.ref, ci, shape)
        dump = load_dump_int8(args.dump, op_idx, h, w, c)

        if ref is None:
            print(f'{ci:4d} {op_idx:3d} {f"{h}x{w}x{c}":>15s}  -- no reference --')
            continue
        if dump is None:
            print(f'{ci:4d} {op_idx:3d} {f"{h}x{w}x{c}":>15s}  -- dump load failed --')
            continue

        stats = compare_layer(ref, dump)
        results.append({'conv_index': ci, 'op_index': op_idx, **stats})

        if stats['mean'] < 0.5:
            status = 'OK'
        elif stats['mean'] < 2.0:
            status = 'minor'
        elif stats['mean'] < 10.0:
            status = 'DIVERGE'
        else:
            status = 'BAD'

        if first_diverge is None and stats['mean'] >= 2.0:
            first_diverge = ci

        print(f'{ci:4d} {op_idx:3d} {f"{h}x{w}x{c}":>15s} {stats["exact"]:7.1f} {stats["le1"]:6.1f} '
              f'{stats["le5"]:6.1f} {stats["mean"]:6.2f} {stats["max"]:4d} '
              f'{stats["ge10_pct"]:6.2f}  {status}')

    # Summary
    print()
    if results:
        avg_mean = np.mean([r['mean'] for r in results])
        avg_exact = np.mean([r['exact'] for r in results])
        print(f'Summary: {len(results)} layers compared, avg_mean_diff={avg_mean:.2f}, '
              f'avg_exact={avg_exact:.1f}%')

    if first_diverge is not None:
        print(f'First diverging layer (mean >= 2.0): Conv {first_diverge}')
    else:
        print('No significant divergence found (all layers mean < 2.0)')


if __name__ == '__main__':
    main()
