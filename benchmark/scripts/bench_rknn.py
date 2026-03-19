#!/usr/bin/env python3
"""Benchmark RKNN NPU inference (vendor kernel + rknn_toolkit_lite2)."""
import argparse
import json
import os
import time

import numpy as np


def benchmark(model_path, num_warmup=10, num_runs=100, core_mask=None):
    """Run RKNN benchmark with detailed statistics."""
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"Failed to load RKNN model: {ret}")
        return None, None

    # Core mask: 1=core0, 2=core1, 4=core2, 7=all three
    mask = core_mask if core_mask is not None else RKNNLite.NPU_CORE_AUTO
    ret = rknn.init_runtime(core_mask=mask)
    if ret != 0:
        print(f"Failed to init RKNN runtime: {ret}")
        return None, None

    # Get input shape from SDK
    input_attrs = rknn.get_sdk_version()

    # Generate random input (assume 224x224x3 uint8 for classification)
    # Actual shape should match model — override as needed
    input_data = np.random.randint(0, 255, size=(1, 224, 224, 3)).astype(np.uint8)

    # Warmup
    for _ in range(num_warmup):
        rknn.inference(inputs=[input_data])

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = rknn.inference(inputs=[input_data])
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    results = {
        'avg_ms': float(np.mean(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'std_ms': float(np.std(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times)),
    }

    if core_mask is not None:
        results['core_mask'] = core_mask

    rknn.release()
    return results, outputs[0] if outputs else None


def main():
    parser = argparse.ArgumentParser(description='RKNN NPU Benchmark')
    parser.add_argument('model', help='Path to .rknn model')
    parser.add_argument('--runs', '-n', type=int, default=100)
    parser.add_argument('--warmup', '-w', type=int, default=10)
    parser.add_argument('--cores', '-c', type=int, default=None,
                        choices=[1, 2, 4, 3, 7],
                        help='Core mask: 1=core0, 2=core1, 4=core2, 7=all')
    parser.add_argument('--output-json', '-o', default=None)
    parser.add_argument('--label', '-l', default=None)
    args = parser.parse_args()

    model_name = os.path.basename(args.model).replace('.rknn', '')
    cores_desc = f"mask={args.cores}" if args.cores else "auto"

    print(f"Model:   {model_name}")
    print(f"Backend: rknn ({cores_desc})")
    print(f"Runs:    {args.warmup} warmup + {args.runs} measured")
    print()

    results, output = benchmark(args.model, args.warmup, args.runs, args.cores)
    if results is None:
        sys.exit(1)

    print(f"  avg:    {results['avg_ms']:.2f} ms")
    print(f"  min:    {results['min_ms']:.2f} ms")
    print(f"  median: {results['median_ms']:.2f} ms")
    print(f"  p95:    {results['p95_ms']:.2f} ms")
    print(f"  p99:    {results['p99_ms']:.2f} ms")
    print(f"  max:    {results['max_ms']:.2f} ms")
    print(f"  std:    {results['std_ms']:.2f} ms")
    print(f"  FPS:    {results['fps']:.1f}")

    record = {
        'model': model_name,
        'backend': 'rknn',
        'warmup': args.warmup,
        'runs': args.runs,
        **results,
    }
    if args.label:
        record['label'] = args.label

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        existing = []
        if os.path.exists(args.output_json):
            with open(args.output_json) as f:
                existing = json.load(f)
        existing.append(record)
        with open(args.output_json, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nResults appended to {args.output_json}")


if __name__ == '__main__':
    import sys
    main()
