# Use official Python base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git curl libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && ln -sf python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create working directory
WORKDIR /app

# Copy your script
COPY handler.py .

# Install Python dependencies
RUN pip install --no-cache-dir \
    addict \
    future \
    lmdb \
    numpy \
    opencv-python \
    Pillow \
    pyyaml \
    requests \
    scikit-image\
    scipy \
    tb-nightly \
    torch>=1.7.1\
    torchvision\
    tqdm \
    yapf \
    lpips \
    gdown \

# Volume for caching models
VOLUME ["/runpod-volume"]

# Run handler
CMD ["python3", "handler.py"]
