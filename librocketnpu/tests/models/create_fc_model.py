#!/usr/bin/env python3
"""Create a simple Flatten+FC model as quantized INT8 TFLite.

Produces:
  fc_model_int8.tflite  - quantized INT8 TFLite model
  fc_test_input.bin     - deterministic test input (3072 bytes, uint8)
  fc_cpu_golden.bin     - TFLite CPU reference output (10 bytes, int8)

Equivalent PyTorch model:
  class OneLayerModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.flatten = nn.Flatten()
          self.layer = nn.Linear(3072, 10)
      def forward(self, x):
          x = self.flatten(x)
          return self.layer(x)
"""

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow required. Install: pip install tensorflow")
    raise SystemExit(1)

np.random.seed(42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
])

# Deterministic weights
w = np.random.randn(3072, 10).astype(np.float32) * 0.01
b = np.random.randn(10).astype(np.float32) * 0.01
model.layers[1].set_weights([w, b])

model.summary()

# Representative dataset for quantization calibration
def representative_dataset():
    for _ in range(100):
        yield [np.random.randint(0, 256, (1, 32, 32, 3)).astype(np.float32) / 255.0]

# Convert to INT8 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open('fc_model_int8.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"Saved fc_model_int8.tflite ({len(tflite_model)} bytes)")

# Deterministic test input (uint8 for NHWC image-like data)
np.random.seed(123)
test_input = np.random.randint(0, 256, (1, 32, 32, 3)).astype(np.uint8)
test_input.tofile('fc_test_input.bin')
print(f"Saved fc_test_input.bin ({test_input.size} bytes)")

# Run TFLite CPU inference for golden reference
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print(f"Input detail: dtype={inp_detail['dtype']}, shape={inp_detail['shape']}, "
      f"quant={inp_detail.get('quantization_parameters', {})}")
print(f"Output detail: dtype={out_detail['dtype']}, shape={out_detail['shape']}, "
      f"quant={out_detail.get('quantization_parameters', {})}")

# INT8 model expects int8 input: convert uint8 → int8 by subtracting 128
input_int8 = (test_input.astype(np.int16) - 128).astype(np.int8)
interp.set_tensor(inp_detail['index'], input_int8)
interp.invoke()
cpu_output = interp.get_tensor(out_detail['index'])

cpu_output.tofile('fc_cpu_golden.bin')
print(f"Saved fc_cpu_golden.bin ({cpu_output.size} bytes)")
print(f"CPU golden output: {cpu_output.flatten().tolist()}")
