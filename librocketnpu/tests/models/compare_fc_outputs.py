#!/usr/bin/env python3
"""Compare FC model outputs across three execution paths.

Usage: python compare_fc_outputs.py [cpu_golden] [rknn_output] [npu_output]

Default file names:
  fc_cpu_golden.bin    - TFLite CPU reference
  fc_rknn_out_0.bin    - RKNN vendor runtime (real HW)
  fc_npu_output_0.bin  - librocketnpu (real HW or QEMU)
"""

import sys
import numpy as np

def load_int8(path):
    try:
        return np.fromfile(path, dtype=np.int8)
    except FileNotFoundError:
        return None

def compare(name_a, a, name_b, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    exact = int(np.sum(diff == 0))
    total = len(a)
    print(f"  {name_a} vs {name_b}: exact={exact}/{total} "
          f"max_diff={int(diff.max())} mean_diff={diff.mean():.3f} "
          f"{'BIT-EXACT' if diff.max() == 0 else ''}")

cpu_path = sys.argv[1] if len(sys.argv) > 1 else 'fc_cpu_golden.bin'
rknn_path = sys.argv[2] if len(sys.argv) > 2 else 'fc_rknn_out_0.bin'
npu_path = sys.argv[3] if len(sys.argv) > 3 else 'fc_npu_output_0.bin'

cpu = load_int8(cpu_path)
rknn = load_int8(rknn_path)
npu = load_int8(npu_path)

print("=== FC Model Output Comparison ===\n")

outputs = {'CPU(TFLite)': cpu, 'RKNN(vendor)': rknn, 'librocketnpu': npu}
for name, data in outputs.items():
    if data is not None:
        print(f"{name}: {data.tolist()}")
    else:
        print(f"{name}: NOT FOUND")

print()

pairs = [
    ('CPU(TFLite)', cpu, 'RKNN(vendor)', rknn),
    ('CPU(TFLite)', cpu, 'librocketnpu', npu),
    ('RKNN(vendor)', rknn, 'librocketnpu', npu),
]

for name_a, a, name_b, b in pairs:
    if a is not None and b is not None:
        if len(a) != len(b):
            print(f"  {name_a} vs {name_b}: SIZE MISMATCH ({len(a)} vs {len(b)})")
        else:
            compare(name_a, a, name_b, b)
    else:
        missing = name_a if a is None else name_b
        print(f"  {name_a} vs {name_b}: SKIPPED ({missing} missing)")
