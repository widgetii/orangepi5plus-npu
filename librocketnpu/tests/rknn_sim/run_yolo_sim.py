#!/usr/bin/env python3
"""Run YOLOv5s INT8 through the RKNN x86 simulator and save outputs."""

import sys
import numpy as np
from rknn.api import RKNN

TFLITE_MODEL = 'yolov5s_relu_int8.tflite'
INPUT = 'input0.bin'

def main():
    rknn = RKNN(verbose=True)

    # Configure for RK3588
    print('Configuring...')
    rknn.config(target_platform='rk3588',
                quantized_dtype='asymmetric_quantized-8',
                quantized_algorithm='normal',
                quantized_method='channel')

    print(f'Loading {TFLITE_MODEL}...')
    ret = rknn.load_tflite(model=TFLITE_MODEL)
    if ret != 0:
        print(f'load_tflite failed: {ret}')
        sys.exit(1)

    print('Building model (this may take a while)...')
    ret = rknn.build(do_quantization=False)  # already quantized INT8
    if ret != 0:
        print(f'build failed: {ret}')
        sys.exit(1)

    print('Initializing simulator runtime...')
    ret = rknn.init_runtime(target=None)
    if ret != 0:
        print(f'init_runtime failed: {ret}')
        sys.exit(1)

    # Load input (640x640x3 uint8)
    input_data = np.fromfile(INPUT, dtype=np.uint8).reshape(1, 640, 640, 3)
    print(f'Input: shape={input_data.shape}, dtype={input_data.dtype}')

    # Query SDK for output info
    print('\n--- SDK input/output info ---')
    try:
        in_attrs = rknn.get_sdk_version()
        print(f'SDK version: {in_attrs}')
    except Exception as e:
        print(f'get_sdk_version: {e}')

    # Run inference twice to check determinism
    print('\nRunning inference (run 1)...')
    outputs1 = rknn.inference(inputs=[input_data])

    print('Running inference (run 2)...')
    outputs2 = rknn.inference(inputs=[input_data])

    # Check determinism
    deterministic = True
    for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
        a1, a2 = np.array(o1), np.array(o2)
        if not np.array_equal(a1, a2):
            print(f'  Output {i}: NOT deterministic! max_diff={np.max(np.abs(a1 - a2))}')
            deterministic = False
    print(f'Deterministic: {deterministic}')

    # Save and analyze outputs
    for i, out in enumerate(outputs1):
        arr = np.array(out)
        print(f'\nOutput {i}: shape={arr.shape}, dtype={arr.dtype}, '
              f'min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}')

        # Save float output
        arr.astype(np.float32).tofile(f'sim_output_{i}_float.bin')
        print(f'  Saved sim_output_{i}_float.bin ({arr.astype(np.float32).nbytes} bytes)')

        # If already int8, save directly
        if arr.dtype == np.int8:
            arr.tofile(f'sim_output_{i}_int8.bin')
            print(f'  Saved sim_output_{i}_int8.bin ({arr.nbytes} bytes)')

    # Try to get int8 outputs via want_float=False
    print('\n--- Trying int8 output mode (want_float=False) ---')
    try:
        outputs_q = rknn.inference(inputs=[input_data], data_format='nhwc')
        for i, out in enumerate(outputs_q):
            arr = np.array(out)
            print(f'  Output {i}: shape={arr.shape}, dtype={arr.dtype}')
    except Exception as e:
        print(f'  Failed: {e}')

    # Export the built rknn model (for comparison with the one from Docker build)
    print('\nExporting built .rknn model...')
    ret = rknn.export_rknn('yolov5s_relu_int8_sim.rknn')
    if ret != 0:
        print(f'export_rknn failed: {ret}')

    # Try accuracy analysis
    print('\n--- Attempting accuracy analysis ---')
    try:
        rknn.accuracy_analysis(inputs=[INPUT],
                                output_dir='accuracy_results',
                                target=None)
        print('Accuracy analysis complete — see accuracy_results/')
    except Exception as e:
        print(f'Accuracy analysis failed: {e}')

    rknn.release()
    print('\nDone.')

if __name__ == '__main__':
    main()
