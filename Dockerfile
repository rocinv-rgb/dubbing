# AI Dubbing Pipeline — RunPod
# Base: CUDA 12.8 + cuDNN + Python 3.10 (podporuje Blackwell sm_120)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL maintainer="dubbing-pipeline" \
      version="3.0.0" \
      description="AI dubbing: Whisper + Qwen3 + XTTS v2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/workspace/models

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    ffmpeg sox libsox-dev \
    git curl wget \
    libsndfile1 libgomp1 \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip setuptools wheel

# --- Krok 1: PyTorch cu128 nightly (Blackwell) ---
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# --- Krok 2: Ostatne zavislosti ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Krok 3: Coqui TTS (XTTS v2) ---
# Musí byť po torch — TTS si stiahne závislosti bez toho aby prepísal torch
RUN pip install --no-cache-dir TTS>=0.22.0

# --- App súbory ---
COPY pipeline.py handler.py ./

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

CMD ["python", "-u", "handler.py"]
