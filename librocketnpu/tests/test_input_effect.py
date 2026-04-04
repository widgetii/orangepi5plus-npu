#!/usr/bin/env python3
"""Check if input data actually affects NPU output."""
import ctypes, numpy as np, os

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

raw = np.zeros(16384, dtype=np.uint8)

# Run with random input
np.random.seed(42)
inp1 = np.random.randint(0, 256, 3072, dtype=np.uint8)
lib.rnpu_invoke(m, inp1.ctypes.data, 3072)
lib.rnpu_get_output_raw(m, 0, raw.ctypes.data, 16384)
out1 = raw.copy().view(np.int8)

# Run with zeros
inp2 = np.zeros(3072, dtype=np.uint8)
lib.rnpu_invoke(m, inp2.ctypes.data, 3072)
lib.rnpu_get_output_raw(m, 0, raw.ctypes.data, 16384)
out2 = raw.copy().view(np.int8)

# Run with all-255
inp3 = np.full(3072, 255, dtype=np.uint8)
lib.rnpu_invoke(m, inp3.ctypes.data, 3072)
lib.rnpu_get_output_raw(m, 0, raw.ctypes.data, 16384)
out3 = raw.copy().view(np.int8)

print("Random first 8:", list(out1[:8]))
print("Zeros  first 8:", list(out2[:8]))
print("All255 first 8:", list(out3[:8]))
print()
print("Random == Zeros:", np.array_equal(out1, out2))
print("Random == All255:", np.array_equal(out1, out3))
print("Zeros == All255:", np.array_equal(out2, out3))
diff_rz = np.abs(out1.astype(int) - out2.astype(int))
diff_ra = np.abs(out1.astype(int) - out3.astype(int))
print(f"Random vs Zeros: max={diff_rz.max()} mean={diff_rz.mean():.1f}")
print(f"Random vs All255: max={diff_ra.max()} mean={diff_ra.mean():.1f}")
