FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Pin versions to avoid onnx/ml_dtypes conflicts
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    onnx==1.16.2 \
    ml-dtypes==0.2.0 \
    tensorflow==2.14.0 \
    rknn-toolkit2

WORKDIR /data
