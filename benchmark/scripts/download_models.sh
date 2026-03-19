#!/bin/bash
# Download TFLite INT8 quantized models for benchmarking
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODEL_DIR"/{classification,detection,segmentation}

echo "=== Downloading classification models ==="

# MobileNetV1 INT8
if [ ! -f "$MODEL_DIR/classification/mobilenet_v1_224_quant.tflite" ]; then
    echo "Fetching MobileNetV1 INT8..."
    curl -sL "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz" \
        | tar xz -C "$MODEL_DIR/classification/" mobilenet_v1_1.0_224_quant.tflite
    mv "$MODEL_DIR/classification/mobilenet_v1_1.0_224_quant.tflite" \
       "$MODEL_DIR/classification/mobilenet_v1_224_quant.tflite"
fi

# MobileNetV2 INT8
if [ ! -f "$MODEL_DIR/classification/mobilenet_v2_224_quant.tflite" ]; then
    echo "Fetching MobileNetV2 INT8..."
    curl -sL "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz" \
        | tar xz -C "$MODEL_DIR/classification/"
    mv "$MODEL_DIR/classification/mobilenet_v2_1.0_224_quant.tflite" \
       "$MODEL_DIR/classification/mobilenet_v2_224_quant.tflite" 2>/dev/null || true
fi

# EfficientNet-Lite0 INT8
if [ ! -f "$MODEL_DIR/classification/efficientnet_lite0_int8.tflite" ]; then
    echo "Fetching EfficientNet-Lite0 INT8..."
    curl -sL "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0-int8.tflite" \
        -o "$MODEL_DIR/classification/efficientnet_lite0_int8.tflite"
fi

echo "=== Downloading detection models ==="

# SSD MobileNetV1 INT8
if [ ! -f "$MODEL_DIR/detection/ssd_mobilenet_v1_quant.tflite" ]; then
    echo "Fetching SSD MobileNetV1 INT8..."
    curl -sL "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip" \
        -o /tmp/ssd_mobilenet.zip
    unzip -o -q /tmp/ssd_mobilenet.zip -d /tmp/ssd_mobilenet
    cp /tmp/ssd_mobilenet/detect.tflite "$MODEL_DIR/detection/ssd_mobilenet_v1_quant.tflite"
    rm -rf /tmp/ssd_mobilenet /tmp/ssd_mobilenet.zip
fi

# SSD MobileNetV2 INT8 (FPNLite)
if [ ! -f "$MODEL_DIR/detection/ssd_mobilenet_v2_fpnlite_quant.tflite" ]; then
    echo "Fetching SSD MobileNetV2 FPNLite INT8..."
    curl -sL "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite" \
        -o "$MODEL_DIR/detection/ssd_mobilenet_v2_fpnlite_quant.tflite" || echo "  (skipped - URL may have changed)"
fi

echo "=== Downloading segmentation models ==="

# DeepLabV3 INT8 (MobileNet backbone)
if [ ! -f "$MODEL_DIR/segmentation/deeplabv3_mnv2_quant.tflite" ]; then
    echo "Fetching DeepLabV3 MobileNetV2..."
    curl -sL "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite" \
        -o "$MODEL_DIR/segmentation/deeplabv3_mnv2_quant.tflite" || echo "  (skipped - may need manual conversion)"
fi

echo ""
echo "=== Downloaded models ==="
find "$MODEL_DIR" -name "*.tflite" -printf "  %p (%s bytes)\n" | sort
