#!/usr/bin/env python3
"""Three-way comparison: RKNN simulator vs RKNN hardware vs librocketnpu."""

import os
import numpy as np

# Output head names and shapes from the simulator
HEADS = {
    0: '40x40 (H1)',   # 408000 bytes
    1: '80x80 (H0)',   # 1632000 bytes
    2: '20x20 (H2)',   # 102000 bytes
}

def load_int8(path):
    return np.fromfile(path, dtype=np.int8)

def compare(name, a, b):
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    n = len(diff)
    exact = np.sum(diff == 0) / n * 100
    le1 = np.sum(diff <= 1) / n * 100
    le5 = np.sum(diff <= 5) / n * 100
    mean = diff.mean()
    maxd = diff.max()
    ge10 = np.sum(diff >= 10) / n * 100
    ge100 = np.sum(diff >= 100) / n * 100
    print(f'  {name:20s}: exact={exact:5.1f}% <=1={le1:5.1f}% <=5={le5:5.1f}% '
          f'mean={mean:6.2f} max={maxd:3d} >=10={ge10:.2f}% >=100={ge100:.2f}%')

def main():
    print('=== Three-Way YOLO Output Comparison ===\n')

    for i in range(3):
        hw_path = f'rknn_yolo_i8_{i}.bin'
        npu_path = f'yolo_output_{i}.bin'
        sim_path = f'sim_output_{i}_int8.bin'

        print(f'--- Output {i} {HEADS.get(i, "")} ---')

        hw = load_int8(hw_path) if os.path.exists(hw_path) else None
        npu = load_int8(npu_path) if os.path.exists(npu_path) else None
        sim = load_int8(sim_path) if os.path.exists(sim_path) else None

        if hw is not None:
            print(f'  hw_golden: {len(hw)} values, range=[{hw.min()}, {hw.max()}], mean={hw.mean():.2f}')
        if npu is not None:
            print(f'  librknpu:  {len(npu)} values, range=[{npu.min()}, {npu.max()}], mean={npu.mean():.2f}')
        if sim is not None:
            print(f'  simulator: {len(sim)} values, range=[{sim.min()}, {sim.max()}], mean={sim.mean():.2f}')

        if sim is not None and hw is not None and len(sim) == len(hw):
            compare('sim vs hw_golden', sim, hw)
        if sim is not None and npu is not None and len(sim) == len(npu):
            compare('sim vs librocketnpu', sim, npu)
        if hw is not None and npu is not None and len(hw) == len(npu):
            compare('hw_golden vs npu', hw, npu)

        print()

    # Summary
    print('=== Interpretation ===')
    print('sim vs hw_golden:    How much RKNN hardware diverges from ideal computation')
    print('sim vs librocketnpu: Total gap between our driver and ideal')
    print('hw_golden vs npu:    Our gap vs RKNN hardware (existing baseline)')

if __name__ == '__main__':
    main()
