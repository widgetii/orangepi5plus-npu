#!/usr/bin/env python3
"""Brute-force NC1HWC2 layout detection for CONV INT8 output BO."""
import ctypes, numpy as np, os, sys

os.chdir("/root/npu-research/librocketnpu")
lib = ctypes.CDLL("./librocketnpu.so")
lib.rnpu_open.restype = ctypes.c_int
lib.rnpu_model_load.restype = ctypes.c_void_p
lib.rnpu_invoke.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.rnpu_invoke.restype = ctypes.c_int
lib.rnpu_get_output_raw.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.rnpu_get_output_raw.restype = ctypes.c_int

np.random.seed(42)
inp = np.random.randint(0, 256, 3072, dtype=np.uint8)
fd = lib.rnpu_open(None)
m = lib.rnpu_model_load(fd, b"/root/npu-research/conv_int8.tflite")
lib.rnpu_invoke(m, inp.ctypes.data, 3072)
raw = np.zeros(16384, dtype=np.uint8)
lib.rnpu_get_output_raw(m, 0, raw.ctypes.data, 16384)
raw_i8 = raw.view(np.int8)

# RKNN pixel(0,0) and pixel(1,0) from RKNN Python API (requantized to int8)
rknn_00 = np.array([17, -24, 83, -5, 49, -1, 11, 1, -56, 7, 12, 13, -17, 1, -37, 36], dtype=np.int8)
rknn_10 = np.array([3, -47, 18, 25, 53, 19, -31, -6, -39, 23, 17, -9, -40, 16, -31, 28], dtype=np.int8)

H, W, C = 32, 32, 16

def try_layout(name, fn):
    vals_00 = []
    vals_10 = []
    for ch in range(16):
        off = fn(0, 0, ch)
        vals_00.append(int(raw_i8[off]) if 0 <= off < len(raw_i8) else 999)
        off = fn(1, 0, ch)
        vals_10.append(int(raw_i8[off]) if 0 <= off < len(raw_i8) else 999)
    m00 = sum(1 for a, b in zip(vals_00, rknn_00) if abs(a - b) <= 1)
    m10 = sum(1 for a, b in zip(vals_10, rknn_10) if abs(a - b) <= 1)
    if m00 + m10 > 4:
        print(f"{name:40s}: p00={m00}/16 p10={m10}/16")
        print(f"  ours_00={vals_00}")
        print(f"  rknn_00={list(rknn_00)}")

# Standard NC1HWC2 [c1][h][w][c2] with various c2
for c2 in [8, 16]:
    try_layout(f"c1_h_w_c2 c2={c2}",
               lambda y, x, ch, c2=c2: (ch//c2)*(H*W*c2) + y*(W*c2) + x*c2 + (ch%c2))
    # Swapped H/W
    try_layout(f"c1_w_h_c2 c2={c2}",
               lambda y, x, ch, c2=c2: (ch//c2)*(H*W*c2) + x*(H*c2) + y*c2 + (ch%c2))

# From REFORMAT registers: DST_STRIDE=16, SURFACE_STRIDE=256
try_layout("y*256+x*16+ch",
           lambda y, x, ch: y*256 + x*16 + ch)
# With offset for second channel group
try_layout("y*256+x*16+ch (split 8K)",
           lambda y, x, ch: (ch//8)*8192 + y*256 + x*16 + (ch%8) if ch < 8 else (ch//8)*8192 + y*256 + x*16 + (ch%8))
# Actually: T1 writes ch0-7 at BO+0, T2 writes ch8-15 at BO+8192
# Each pixel: 8 bytes at DST_STRIDE=16? No, DST_STRIDE=16 means stride between consecutive x pixels
# So for ch0-7: off = y * SURFACE_STRIDE + x * DST_STRIDE + ci
# But DST_STRIDE=16 for 8 channels? That leaves 8 bytes gap.
# Maybe DST_STRIDE is the full channel width (16), written by both T1 and T2
try_layout("T1: y*256+x*16+ch%8, T2: +8",
           lambda y, x, ch: y*256 + x*16 + ch)
# 256 = 32 * 8. Maybe: y * (W * c2) + x * c2 + ci, then T2 at offset +c2
# This is just standard [h][w][c2] with c2=8, and T2 adds 8 to ch offset
# Let's try: the full output is [h][w][16] written as two passes of [h][w][8]
# Pass 1 at BO+0: y*(W*8)+x*8+ch for ch<8
# Pass 2 at BO+8192: y*(W*8)+x*8+(ch-8) for ch>=8
try_layout("two-pass [h][w][8] split at 8192",
           lambda y, x, ch: (ch//8)*8192 + y*(W*8) + x*8 + (ch%8))
# Or maybe DST_STRIDE=16 means pixels are 16 bytes apart, but only 8 used per T
# So output is [h][w][16] with T1 writing bytes 0-7 and T2 writing bytes 8-15
try_layout("contiguous [h][w][16]",
           lambda y, x, ch: y*(W*16) + x*16 + ch)
# Plain NHWC
try_layout("NHWC flat",
           lambda y, x, ch: y*(W*C) + x*C + ch)
