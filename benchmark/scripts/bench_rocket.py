#!/usr/bin/env python3
"""Benchmark Teflon/Rocket NPU vs CPU inference with detailed metrics."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import ai_edge_litert.interpreter as tflite


def load_input(model_path, input_details, image_path=None):
    """Load input tensor - random data or from image file."""
    shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']

    if image_path and os.path.exists(image_path):
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        h, w = shape[1], shape[2]
        img = img.resize((w, h))
        data = np.array(img, dtype=dtype)
        if len(shape) == 4:
            data = np.expand_dims(data, axis=0)
        return data

    return np.random.randint(0, 255, size=shape).astype(dtype)


def count_ioctls(model_path, delegate_path):
    """Run two inferences under strace, report IOCTLs from second invoke."""
    script = f"""
import ai_edge_litert.interpreter as tflite
import numpy as np
d = [tflite.load_delegate("{delegate_path}")]
i = tflite.Interpreter(model_path="{model_path}", experimental_delegates=d)
i.allocate_tensors()
inp = i.get_input_details()
data = np.random.randint(0, 255, size=inp[0]['shape']).astype(inp[0]['dtype'])
i.set_tensor(inp[0]['index'], data)
i.invoke()
i.invoke()
"""
    try:
        result = subprocess.run(
            ['strace', '-e', 'ioctl', '-e', 'raw=ioctl', '-c',
             sys.executable, '-c', script],
            capture_output=True, text=True, timeout=60
        )
        lines = result.stderr.strip().split('\n')
        ioctl_count = 0
        for line in lines:
            if 'ioctl' in line and '%' in line:
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        ioctl_count = int(p)
                        break
        return ioctl_count
    except Exception as e:
        print(f"  strace failed: {e}", file=sys.stderr)
        return -1


def benchmark(model_path, delegate_path=None, num_warmup=10, num_runs=100,
              image_path=None, validate_cpu=False):
    """Run benchmark with detailed statistics."""
    ext_delegate = None
    if delegate_path:
        ext_delegate = [tflite.load_delegate(delegate_path)]

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=ext_delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = load_input(model_path, input_details, image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Warmup
    for _ in range(num_warmup):
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.invoke()
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

    # Get output tensor for correctness checking
    out_idx = output_details[0]['index']
    output_data = interpreter.get_tensor(out_idx).copy()

    # CPU reference for correctness validation
    if validate_cpu and delegate_path:
        cpu_interp = tflite.Interpreter(model_path=model_path)
        cpu_interp.allocate_tensors()
        cpu_interp.set_tensor(input_details[0]['index'], input_data)
        cpu_interp.invoke()
        cpu_output = cpu_interp.get_tensor(cpu_interp.get_output_details()[0]['index'])
        max_diff = int(np.max(np.abs(output_data.astype(int) - cpu_output.astype(int))))
        results['max_quant_diff'] = max_diff
        results['correctness'] = 'PASS' if max_diff <= 1 else 'FAIL'

    return results, output_data


def main():
    parser = argparse.ArgumentParser(description='Rocket NPU Benchmark')
    parser.add_argument('model', help='Path to .tflite model')
    parser.add_argument('--delegate', '-d', default=None,
                        help='Path to Teflon delegate .so')
    parser.add_argument('--runs', '-n', type=int, default=100,
                        help='Number of measurement runs (default: 100)')
    parser.add_argument('--warmup', '-w', type=int, default=10,
                        help='Number of warmup runs (default: 10)')
    parser.add_argument('--image', '-i', default=None,
                        help='Input image path (optional)')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate against CPU reference')
    parser.add_argument('--count-ioctls', action='store_true',
                        help='Count IOCTLs via strace (separate run)')
    parser.add_argument('--output-json', '-o', default=None,
                        help='Write results to JSON file')
    parser.add_argument('--label', '-l', default=None,
                        help='Label for this run (e.g., phase name)')
    args = parser.parse_args()

    model_name = os.path.basename(args.model).replace('.tflite', '')
    backend = 'rocket' if args.delegate else 'cpu'

    print(f"Model:   {model_name}")
    print(f"Backend: {backend}")
    print(f"Runs:    {args.warmup} warmup + {args.runs} measured")
    print()

    results, output = benchmark(
        args.model, args.delegate, args.warmup, args.runs,
        args.image, args.validate
    )

    # Print results
    print(f"  avg:    {results['avg_ms']:.2f} ms")
    print(f"  min:    {results['min_ms']:.2f} ms")
    print(f"  median: {results['median_ms']:.2f} ms")
    print(f"  p95:    {results['p95_ms']:.2f} ms")
    print(f"  p99:    {results['p99_ms']:.2f} ms")
    print(f"  max:    {results['max_ms']:.2f} ms")
    print(f"  std:    {results['std_ms']:.2f} ms")
    print(f"  FPS:    {results['fps']:.1f}")

    if 'correctness' in results:
        print(f"\n  Correctness: {results['correctness']} "
              f"(max quant diff: {results['max_quant_diff']})")

    # Count IOCTLs if requested
    ioctl_count = -1
    if args.count_ioctls and args.delegate:
        print("\nCounting IOCTLs via strace...")
        ioctl_count = count_ioctls(args.model, args.delegate)
        results['ioctl_count'] = ioctl_count
        if ioctl_count >= 0:
            print(f"  IOCTLs per inference: {ioctl_count}")

    # Build output record
    record = {
        'model': model_name,
        'backend': backend,
        'delegate': args.delegate,
        'warmup': args.warmup,
        'runs': args.runs,
        **results,
    }
    if args.label:
        record['label'] = args.label

    # Write JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        # Append to JSON array if file exists
        existing = []
        if os.path.exists(args.output_json):
            with open(args.output_json) as f:
                existing = json.load(f)
        existing.append(record)
        with open(args.output_json, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nResults appended to {args.output_json}")


if __name__ == '__main__':
    main()
