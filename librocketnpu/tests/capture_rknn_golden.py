#!/usr/bin/env python3
"""
Capture RKNN golden output for comparison.
Uses rknn-toolkit-lite2 (available on ARM boards).

Usage: python3 capture_rknn_golden.py <model.rknn> [output_prefix]

Note: The .tflite model must be converted to .rknn first using
rknn-toolkit2 on x86. This script only runs inference.
"""

import sys
import numpy as np
from rknnlite.api import RKNNLite

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.rknn> [output_prefix]")
        sys.exit(1)

    model_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "rknn_golden"

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"Failed to load RKNN model: {ret}")
        sys.exit(1)

    ret = rknn.init_runtime()
    if ret != 0:
        print(f"Failed to init runtime: {ret}")
        sys.exit(1)

    # Get input shape
    sdk_version = rknn.get_sdk_version()
    print(f"SDK version: {sdk_version}")

    # Create all-128 input (zero after zero-point subtraction for uint8)
    # Assume NHWC format, get shape from model
    input_data = np.full((1, 416, 416, 3), 128, dtype=np.uint8)

    # Run inference with passthrough (raw int8/uint8 output)
    outputs = rknn.inference(inputs=[input_data])

    if outputs is None:
        print("Inference failed!")
        sys.exit(1)

    for i, out in enumerate(outputs):
        fname = f"{prefix}_out_{i}.bin"
        out_np = np.array(out)
        print(f"Output {i}: shape={out_np.shape}, dtype={out_np.dtype}, "
              f"min={out_np.min()}, max={out_np.max()}")

        # Save raw bytes
        out_flat = out_np.flatten().astype(np.uint8)
        out_flat.tofile(fname)
        print(f"  Saved {len(out_flat)} bytes to {fname}")

        # Stats
        unique = np.unique(out_flat)
        print(f"  Unique values: {len(unique)}")

    rknn.release()
    print("Done.")

if __name__ == "__main__":
    main()
