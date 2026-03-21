#!/usr/bin/env python3
"""Compare librocketnpu output against CPU inference (TFLite)."""

import sys
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.tflite> [npu_output_0.bin ...]")
        return

    model_path = sys.argv[1]
    npu_files = sys.argv[2:] if len(sys.argv) > 2 else ["output_0.bin"]

    # CPU inference
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    shape = inp["shape"]
    # Use same test input as test_mobilenet.c (all 128)
    data = np.full(shape, 128, dtype=np.uint8)
    interp.set_tensor(inp["index"], data)
    interp.invoke()

    outputs = interp.get_output_details()
    for i, out in enumerate(outputs):
        cpu = interp.get_tensor(out["index"]).flatten()

        if i < len(npu_files):
            npu = np.fromfile(npu_files[i], dtype=np.uint8)
        else:
            print(f"Output {i}: no NPU file provided")
            continue

        if len(cpu) != len(npu):
            print(f"Output {i}: size mismatch (CPU={len(cpu)}, NPU={len(npu)})")
            continue

        diff = np.abs(cpu.astype(int) - npu.astype(int))
        max_diff = diff.max()
        mean_diff = diff.mean()
        exact = np.sum(diff == 0)

        print(f"Output {i}: max_diff={max_diff}, mean_diff={mean_diff:.2f}, "
              f"exact={exact}/{len(cpu)} ({100*exact/len(cpu):.1f}%)")

        if max_diff > 0:
            worst = np.argsort(diff)[-10:]
            for idx in worst:
                print(f"  [{idx}] cpu={cpu[idx]} npu={npu[idx]} diff={diff[idx]}")


if __name__ == "__main__":
    main()
