# Use official NVIDIA CUDA runtime base
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && ln -sf python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Make sure appformer is a package
RUN touch appformer/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir \
    addict future numpy opencv-python Pillow pyyaml runpod requests \
    scikit-image scipy tb-nightly torch>=1.7.1 torchvision tqdm yapf lpips gdown

# Declare volume for persistent storage
VOLUME ["/runpod-volume"]

# Run handler by default
CMD ["python3", "handler.py"]
