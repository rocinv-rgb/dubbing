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
    MODEL_DIR=/workspace/models \
    CACHE_DIR=/workspace/cache \
    COQUI_TTS_AGREED_TO_CPML=1

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

# --- Krok 4: Fix torch.load weights_only pre XTTS (PyTorch 2.6+ breaking change) ---
# XTTS checkpoint je pickle — torch 2.6 zmenil default weights_only=True, XTTS to neocakava
RUN sed -i 's/torch\.load(f, map_location=map_location, \*\*kwargs)/torch.load(f, map_location=map_location, weights_only=False)/' \
    /usr/local/lib/python3.10/dist-packages/TTS/utils/io.py

# OpenVoice V2 — pkg-config + libav needed for MeloTTS (PyAV dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/myshell-ai/OpenVoice /opt/openvoice && \
    pip install -e /opt/openvoice && \
    pip install git+https://github.com/myshell-ai/MeloTTS.git

ENV OPENVOICE_CHECKPOINT_URL="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

# --- App súbory ---
COPY pipeline.py handler.py test_input.json ./

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

CMD ["python", "-u", "handler.py"]
