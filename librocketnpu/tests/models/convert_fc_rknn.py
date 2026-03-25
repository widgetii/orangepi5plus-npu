#!/usr/bin/env python3
"""Convert fc_model_int8.tflite to .rknn for vendor kernel testing.

Run inside Docker container with rknn-toolkit2:
  docker run --rm -v $PWD:/data rknn-sim python /data/convert_fc_rknn.py
"""

import sys
from rknn.api import RKNN

MODEL = 'fc_model_int8.tflite'
OUTPUT = 'fc_model_int8.rknn'

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')

print(f'Loading {MODEL}...')
ret = rknn.load_tflite(model=MODEL)
if ret != 0:
    print(f'load_tflite failed: {ret}')
    sys.exit(1)

print('Building (no re-quantization)...')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f'build failed: {ret}')
    sys.exit(1)

print(f'Exporting {OUTPUT}...')
ret = rknn.export_rknn(OUTPUT)
if ret != 0:
    print(f'export_rknn failed: {ret}')
    sys.exit(1)

rknn.release()
print('Done.')
