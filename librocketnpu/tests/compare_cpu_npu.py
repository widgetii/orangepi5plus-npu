#!/usr/bin/env python3
"""
Compare librocketnpu NPU output vs TFLite CPU reference.
Uses the same input for both paths.

Usage: python3 compare_cpu_npu.py <model.tflite> [npu_binary] [hybrid_mask]
"""

import sys
import os
import struct
import subprocess
import numpy as np
from ai_edge_litert.interpreter import Interpreter

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.tflite"
    npu_bin = sys.argv[2] if len(sys.argv) > 2 else "./test_hybrid_regcmd"
    mask = sys.argv[3] if len(sys.argv) > 3 else "0"

    # TFLite CPU inference
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    inp_shape = inp_det["shape"]
    inp_dtype = inp_det["dtype"]
    inp_zp = int(inp_det["quantization_parameters"]["zero_points"][0])

    # Generate random input with fixed seed
    np.random.seed(42)
    if inp_dtype == np.uint8:
        input_data = np.random.randint(0, 256, size=inp_shape, dtype=np.uint8)
    else:  # int8
        input_data = np.random.randint(-128, 128, size=inp_shape, dtype=np.int8)

    # Save input for NPU
    input_flat = input_data.flatten()
    if inp_dtype == np.int8:
        # Convert int8 to uint8 for NPU (add 128)
        npu_input = (input_flat.astype(np.int16) + 128).astype(np.uint8)
    else:
        npu_input = input_flat.astype(np.uint8)
    npu_input.tofile("/tmp/compare_input.bin")

    # Run TFLite CPU
    interp.set_tensor(inp_det["index"], input_data)
    interp.invoke()

    cpu_outputs = []
    for i, out_det in enumerate(interp.get_output_details()):
        result = interp.get_tensor(out_det["index"]).flatten()
        out_zp = int(out_det["quantization_parameters"]["zero_points"][0])
        out_dtype = out_det["dtype"]
        # Convert to uint8 for comparison (NPU returns uint8)
        if out_dtype == np.int8:
            result_u8 = (result.astype(np.int16) + 128).astype(np.uint8)
        else:
            result_u8 = result.astype(np.uint8)
        cpu_outputs.append(result_u8)
        fname = f"/tmp/cpu_golden_{i}.bin"
        result_u8.tofile(fname)
        print(f"CPU Out {i}: shape={interp.get_output_details()[i]['shape']} "
              f"zp={out_zp} min={int(result_u8.min())} max={int(result_u8.max())} "
              f"unique={len(np.unique(result_u8))}")

    # Run NPU via test_hybrid_regcmd
    # First, we need to modify the test to accept input file
    # For now, use test_yolo which accepts input file
    print(f"\nRunning NPU with mask={mask}...")
    env = os.environ.copy()
    if mask != "0":
        env["RNPU_HYBRID_MASK"] = mask

    # Use test_yolo which supports input file
    result = subprocess.run(
        [npu_bin, model_path, "/tmp/compare_input.bin"],
        capture_output=True, text=True, env=env, cwd=os.path.dirname(npu_bin) or ".",
        timeout=120
    )
    print(result.stderr[:500] if result.stderr else "")
    print(result.stdout[:500] if result.stdout else "")

    # Read NPU outputs
    for i, cpu_out in enumerate(cpu_outputs):
        npu_fname = f"yolo_output_{i}.bin"
        npu_path = os.path.join(os.path.dirname(npu_bin) or ".", npu_fname)
        if not os.path.exists(npu_path):
            print(f"NPU output {npu_fname} not found")
            continue
        npu_out = np.fromfile(npu_path, dtype=np.uint8)
        if len(npu_out) != len(cpu_out):
            print(f"Size mismatch: NPU={len(npu_out)} CPU={len(cpu_out)}")
            continue

        diff = np.abs(npu_out.astype(np.int16) - cpu_out.astype(np.int16))
        max_diff = int(diff.max())
        diff_count = int(np.sum(diff > 0))
        mean_diff = float(diff[diff > 0].mean()) if diff_count > 0 else 0
        print(f"Out {i}: max_diff={max_diff} diffs={diff_count}/{len(diff)} "
              f"({100*diff_count/len(diff):.1f}%) mean_diff={mean_diff:.2f}")

if __name__ == "__main__":
    main()
