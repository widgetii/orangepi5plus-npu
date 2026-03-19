#!/usr/bin/env python3
"""CPU-only TFLite baseline benchmark."""
import argparse
import json
import os
import sys
import time

import numpy as np
import ai_edge_litert.interpreter as tflite


def benchmark(model_path, num_warmup=10, num_runs=100, num_threads=None):
    """Run CPU-only TFLite benchmark."""
    interpreter = tflite.Interpreter(
        model_path=model_path,
        num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']
    input_data = np.random.randint(0, 255, size=shape).astype(dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    for _ in range(num_warmup):
        interpreter.invoke()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    return {
        'avg_ms': float(np.mean(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'std_ms': float(np.std(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times)),
        'num_threads': num_threads or os.cpu_count(),
    }


def main():
    parser = argparse.ArgumentParser(description='CPU-only TFLite Benchmark')
    parser.add_argument('model', help='Path to .tflite model')
    parser.add_argument('--runs', '-n', type=int, default=100)
    parser.add_argument('--warmup', '-w', type=int, default=10)
    parser.add_argument('--threads', '-t', type=int, default=None,
                        help='Number of CPU threads (default: all cores)')
    parser.add_argument('--output-json', '-o', default=None)
    parser.add_argument('--label', '-l', default=None)
    args = parser.parse_args()

    model_name = os.path.basename(args.model).replace('.tflite', '')
    threads = args.threads or os.cpu_count()

    print(f"Model:   {model_name}")
    print(f"Backend: cpu ({threads} threads)")
    print(f"Runs:    {args.warmup} warmup + {args.runs} measured")
    print()

    results = benchmark(args.model, args.warmup, args.runs, args.threads)

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
        'backend': 'cpu',
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
    main()
