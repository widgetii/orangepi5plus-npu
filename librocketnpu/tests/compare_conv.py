#!/usr/bin/env python3
"""Compare RKNN vs librocketnpu native cache for conv_int8 model.

Runs 10 random inputs through both paths, compares raw output BO bytes.
RKNN float output is requantized to int8 for comparison with our raw BO.
"""
import numpy as np
import ctypes
import os
import subprocess
import sys

N_INPUTS = 10

# Generate deterministic inputs
np.random.seed(42)
inputs = [np.random.randint(0, 256, (1, 32, 32, 3), dtype=np.uint8) for _ in range(N_INPUTS)]

# === RKNN path ===
sys.path.insert(0, "/root/npu-research/venv/lib/python3.12/site-packages")
from rknnlite.api import RKNNLite

r = RKNNLite()
r.load_rknn("/root/npu-research/conv_int8.rknn")
r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

rknn_outs = []
for inp in inputs:
    out = r.inference(inputs=[inp])
    rknn_outs.append(out[0].flatten().astype(np.float32))
r.release()

# === librocketnpu native cache path ===
# Temporarily hide .rknn to force native cache (use safe symlink trick)
_rknn = "/root/npu-research/conv_int8.rknn"
_bak = _rknn + ".hidden"
os.rename(_rknn, _bak)

os.chdir("/root/npu-research/librocketnpu")
lib = ctypes.CDLL("./librocketnpu.so")
lib.rnpu_open.restype = ctypes.c_int
lib.rnpu_model_load.restype = ctypes.c_void_p
lib.rnpu_invoke.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.rnpu_invoke.restype = ctypes.c_int
lib.rnpu_get_output_raw.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.rnpu_get_output_raw.restype = ctypes.c_int

fd = lib.rnpu_open(None)
m = lib.rnpu_model_load(fd, b"/root/npu-research/conv_int8.tflite")

our_raw_bos = []
for inp in inputs:
    ret = lib.rnpu_invoke(m, inp.flatten().ctypes.data, 3072)
    raw = np.zeros(16384, dtype=np.uint8)
    if ret == 0:
        lib.rnpu_get_output_raw(m, 0, raw.ctypes.data, 16384)
    our_raw_bos.append((ret, raw.copy()))

os.rename(_bak, _rknn)

# === Compare ===
# RKNN output quant: scale=0.0091667455, zp=7
# RKNN returns float = (int8 - zp) * scale
# So RKNN's internal int8 = round(float / scale) + zp
#
# Our raw BO is in NPU NC1HWC2 format. We need to deinterleave and compare.
# NC1HWC2: c2=16 for this model (16 output channels = 1 group)
# pixel(y,x) at offset: x * H * c2 + y * c2 + c
# where H=32, c2=16

W, H, C = 32, 32, 16
rknn_scale = 0.0091667455
rknn_zp = 7

print("=== CONV INT8: RKNN vs librocketnpu native cache ===\n")

all_exact = 0
all_total = 0
all_le1 = 0

for i in range(N_INPUTS):
    ret, raw = our_raw_bos[i]
    if ret != 0:
        print(f"  [{i:2d}] INVOKE FAILED (ret={ret})")
        continue

    # Standard NC1HWC2 detiling: [c1][h][w][c2], c2=8 for int8
    c2 = 8
    c1 = (C + c2 - 1) // c2
    our_nhwc_i8 = np.zeros(H * W * C, dtype=np.int8)
    for y in range(H):
        for x in range(W):
            for ch in range(C):
                tile = ch // c2
                within = ch % c2
                src_off = tile * (H * W * c2) + y * (W * c2) + x * c2 + within
                if src_off < len(raw):
                    our_nhwc_i8[y * W * C + x * C + ch] = np.int8(raw[src_off])

    # Dequantize our int8 to float: float = (int8 - zp) * scale
    our_float = (our_nhwc_i8.astype(np.float32) - rknn_zp) * rknn_scale

    # Compare float with RKNN float
    diff_f = np.abs(our_float - rknn_outs[i])
    # Compare requantized int8
    our_req = np.clip(np.round(our_float / rknn_scale + rknn_zp), -128, 127).astype(np.int16)
    rknn_req = np.clip(np.round(rknn_outs[i] / rknn_scale + rknn_zp), -128, 127).astype(np.int16)
    diff = np.abs(our_req - rknn_req)

    exact = int(np.sum(diff == 0))
    le1 = int(np.sum(diff <= 1))
    total = len(diff)
    all_exact += exact
    all_total += total
    all_le1 += le1

    tag = "MATCH" if exact == total else f"exact={exact}/{total} ({100*exact/total:.1f}%)"
    print(f"  [{i:2d}] {tag}  le1={le1}/{total} ({100*le1/total:.1f}%)  max_fdiff={diff_f.max():.4f}")
    if exact != total and i == 0:
        idx = np.where(diff > 0)[0][:5]
        for j in idx:
            y = j // (W * C)
            x = (j % (W * C)) // C
            ch = j % C
            print(f"        pixel({y},{x}) ch{ch}: ours={our_nhwc_i8[j]} rknn_req={int(rknn_req[j])} ours_f={our_float[j]:.4f} rknn_f={rknn_outs[i][j]:.4f}")

print(f"\nOverall: exact={all_exact}/{all_total} ({100*all_exact/all_total:.1f}%)  "
      f"le1={all_le1}/{all_total} ({100*all_le1/all_total:.1f}%)")
