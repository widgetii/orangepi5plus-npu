#!/usr/bin/env python3
"""Benchmark Teflon/Rocket NPU vs CPU inference."""
import argparse
import time
import numpy as np
import ai_edge_litert.interpreter as tflite

def benchmark(model_path, delegate_path=None, num_warmup=5, num_runs=20):
    ext_delegate = None
    if delegate_path:
        ext_delegate = [tflite.load_delegate(delegate_path)]

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=ext_delegate)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate random input matching model's expected shape/type
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_data = np.random.randint(0, 255, size=input_shape).astype(input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Warmup
    for _ in range(num_warmup):
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = np.mean(times)
    std = np.std(times)
    mn = np.min(times)
    print(f"  Avg: {avg:.2f}ms  Std: {std:.2f}ms  Min: {mn:.2f}ms  FPS: {1000/avg:.1f}")
    return avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to .tflite model')
    parser.add_argument('--delegate', '-e', default=None, help='Path to delegate .so')
    parser.add_argument('--runs', '-n', type=int, default=20)
    args = parser.parse_args()

    mode = "NPU (Teflon/Rocket)" if args.delegate else "CPU"
    print(f"Benchmarking: {args.model}")
    print(f"Mode: {mode}")
    benchmark(args.model, args.delegate, num_runs=args.runs)
