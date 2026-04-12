FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# Pin versions known to work with rknn-toolkit2 (matching rknn_sim image)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    onnx==1.16.2 \
    ml-dtypes==0.2.0 \
    tensorflow==2.14.0 \
    rknn-toolkit2

# SmolVLM sharding pipeline deps (from upstream requirements_convert.txt)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    transformers==4.46.3 \
    pillow \
    huggingface_hub

WORKDIR /work
