#!/usr/bin/env python3
"""Test each new software op against CPU baseline."""
import sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

DELEGATE_PATH = '/usr/local/lib/aarch64-linux-gnu/libteflon.so'

MODELS = [
    ('test_concat.tflite',   'CONCATENATION', 2),
    ('test_maxpool.tflite',  'MAX_POOL_2D',   2),
    ('test_pad.tflite',      'PAD',           2),
    ('test_resize.tflite',   'RESIZE_NN',     2),
    ('test_logistic.tflite', 'LOGISTIC',      2),
]

def run_test(model_path, name, max_allowed_diff):
    inp = np.random.randint(0, 255, (1, 8, 8, 3), dtype=np.uint8)

    # CPU baseline
    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]['index'], inp)
    interp.invoke()
    cpu_out = interp.get_tensor(interp.get_output_details()[0]['index']).copy()

    # NPU delegate
    try:
        delegate = load_delegate(DELEGATE_PATH)
        interp2 = Interpreter(model_path=model_path, experimental_delegates=[delegate])
        interp2.allocate_tensors()
        interp2.set_tensor(interp2.get_input_details()[0]['index'], inp)
        interp2.invoke()
        npu_out = interp2.get_tensor(interp2.get_output_details()[0]['index']).copy()
    except Exception as e:
        print(f'  {name}: CRASH - {e}')
        return False

    diff = np.abs(cpu_out.astype(int) - npu_out.astype(int))
    max_diff = diff.max()
    mean_diff = diff.mean()
    passed = max_diff <= max_allowed_diff

    status = 'PASS' if passed else 'FAIL'
    print(f'  {name}: max_diff={max_diff}, mean_diff={mean_diff:.4f} [{status}]')
    if not passed:
        # Show where differences are largest
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f'    worst at {idx}: cpu={cpu_out[idx]}, npu={npu_out[idx]}')
    return passed

if __name__ == '__main__':
    base_dir = '/root/npu-research'
    all_pass = True
    print('Software op correctness tests:')
    for model_file, name, max_diff in MODELS:
        path = f'{base_dir}/{model_file}'
        if not run_test(path, name, max_diff):
            all_pass = False

    print()
    if all_pass:
        print('ALL TESTS PASSED')
    else:
        print('SOME TESTS FAILED')
        sys.exit(1)
