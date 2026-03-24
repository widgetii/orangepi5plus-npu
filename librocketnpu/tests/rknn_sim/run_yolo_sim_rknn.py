#!/usr/bin/env python3
"""Run YOLOv5s INT8 through RKNN x86 simulator using pre-converted .rknn file."""

import sys
import numpy as np
from rknn.api import RKNN

RKNN_MODEL = 'yolov5s_relu_int8.rknn'
INPUT = 'input0.bin'

def main():
    rknn = RKNN(verbose=True)

    print(f'Loading {RKNN_MODEL}...')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print(f'load_rknn failed: {ret}')
        sys.exit(1)

    # Per docs: init_runtime() with no args = simulator mode
    print('Initializing simulator runtime (no target)...')
    ret = rknn.init_runtime()
    if ret != 0:
        print(f'init_runtime() failed: {ret}')
        # Try with explicit target=None
        print('Trying init_runtime(target=None)...')
        ret = rknn.init_runtime(target=None)
        if ret != 0:
            print(f'init_runtime(target=None) also failed: {ret}')
            sys.exit(1)

    # Load input (640x640x3 uint8)
    input_data = np.fromfile(INPUT, dtype=np.uint8).reshape(1, 640, 640, 3)
    print(f'Input: shape={input_data.shape}, dtype={input_data.dtype}')

    print('Running inference...')
    outputs = rknn.inference(inputs=[input_data])

    for i, out in enumerate(outputs):
        arr = np.array(out)
        print(f'\nOutput {i}: shape={arr.shape}, dtype={arr.dtype}, '
              f'min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}')
        # Save
        if arr.dtype == np.int8:
            arr.tofile(f'sim_rknn_output_{i}_int8.bin')
            print(f'  Saved sim_rknn_output_{i}_int8.bin')
        arr.astype(np.float32).tofile(f'sim_rknn_output_{i}_float.bin')
        print(f'  Saved sim_rknn_output_{i}_float.bin')

    # Check if rknn-loaded sim output matches tflite-loaded sim output
    print('\n--- Comparing rknn-loaded vs tflite-loaded sim outputs ---')
    for i in range(3):
        rknn_path = f'sim_rknn_output_{i}_int8.bin'
        tfl_path = f'sim_output_{i}_int8.bin'
        try:
            a = np.fromfile(rknn_path, dtype=np.int8)
            b = np.fromfile(tfl_path, dtype=np.int8)
            if len(a) == len(b):
                diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
                exact = np.sum(diff == 0) / len(diff) * 100
                print(f'  Output {i}: exact={exact:.1f}%, mean_diff={diff.mean():.3f}, max_diff={diff.max()}')
            else:
                print(f'  Output {i}: SIZE MISMATCH rknn={len(a)} tfl={len(b)}')
        except FileNotFoundError:
            print(f'  Output {i}: missing file(s)')

    rknn.release()
    print('\nDone.')

if __name__ == '__main__':
    main()
