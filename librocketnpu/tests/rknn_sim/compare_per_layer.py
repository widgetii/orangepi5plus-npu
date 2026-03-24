#!/usr/bin/env python3
"""Compare librocketnpu per-layer outputs against RKNN FP32/INT8 references.

Usage:
  python3 compare_per_layer.py [--ref per_layer_ref] [--dump op_dump]

Expects:
  - per_layer_ref/  : from per_layer_reference.py (conv_NN_int8.bin in NCHW int8)
  - op_dump/        : from RNPU_DUMP_OPS on board (op_NN.bin in NHWC uint8)

Format conversion:
  librocketnpu outputs uint8 NHWC with +0x80 offset (NPU stores as int8 internally).
  RKNN reference is int8 NCHW from ONNX Runtime quantization.
  Compare as int8: librocketnpu_int8 = librocketnpu_uint8 - 0x80
"""

import argparse
import json
import os
import sys
import numpy as np


def load_ref_int8(ref_dir, conv_idx, shape_nchw):
    """Load RKNN reference INT8 tensor (NCHW) and transpose to NHWC."""
    path = os.path.join(ref_dir, f'conv_{conv_idx:02d}_int8.bin')
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.int8).reshape(shape_nchw)
    # NCHW -> NHWC: (1,C,H,W) -> (H,W,C)
    return data[0].transpose(1, 2, 0)  # (H, W, C)


def load_dump_uint8(dump_dir, op_idx, h, w, c):
    """Load librocketnpu dump (uint8 NHWC) and convert to int8."""
    path = os.path.join(dump_dir, f'op_{op_idx:02d}.bin')
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    expected = h * w * c
    if len(data) != expected:
        print(f'  WARNING: op_{op_idx:02d}.bin size={len(data)}, expected={expected}')
        return None
    nhwc = data.reshape(h, w, c)
    # Convert uint8 (+0x80 offset) to int8
    return nhwc.astype(np.int16) - 0x80


def compare_layer(ref_nhwc, dump_nhwc):
    """Compare two int16 tensors (to avoid int8 overflow in diff)."""
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
    print()

    # Header
    print(f'{"Conv":>4s} {"Shape":>15s} {"exact%":>7s} {"<=1%":>6s} {"<=5%":>6s} '
          f'{"mean":>6s} {"max":>4s} {">=10%":>6s}  {"Status"}')
    print('-' * 80)

    first_diverge = None
    results = []

    for layer in layers:
        ci = layer['conv_index']
        shape = layer['shape_nchw']  # [1, C, H, W]
        _, c, h, w = shape

        ref = load_ref_int8(args.ref, ci, shape)
        dump = load_dump_uint8(args.dump, ci, h, w, c)

        if ref is None:
            print(f'{ci:4d} {f"{h}x{w}x{c}":>15s}  -- no reference --')
            continue
        if dump is None:
            print(f'{ci:4d} {f"{h}x{w}x{c}":>15s}  -- no dump --')
            continue

        stats = compare_layer(ref, dump)
        results.append({'conv_index': ci, **stats})

        # Status classification
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

        print(f'{ci:4d} {f"{h}x{w}x{c}":>15s} {stats["exact"]:7.1f} {stats["le1"]:6.1f} '
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
