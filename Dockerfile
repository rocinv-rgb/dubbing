# AI Dubbing Pipeline — RunPod A100 80GB
# Base: CUDA 12.1 + cuDNN 8 + Python 3.10
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="dubbing-pipeline" \
      version="1.1.0" \
      description="AI dubbing: Demucs + Whisper + Qwen3 + CosyVoice2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/workspace/models

WORKDIR /app

# System deps: FFmpeg, sox, git-lfs
# python3.10-dev vynechané — nie je potrebné pre runtime
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    ffmpeg sox libsox-dev \
    git git-lfs curl wget \
    libsndfile1 libgomp1 \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# --- PyTorch CUDA 12.1 (musí byť pred requirements.txt!) ---
# Explicitná CUDA verzia — pip defaultne stiahne CPU build, čo na A100 nefunguje
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

# --- Ostatné Python závislosti (torch už je, requirements ho preskočí) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- CosyVoice2 ---
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice && \
    cd /app/CosyVoice && pip install --no-cache-dir -r requirements.txt || true
ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# --- Handler ---
COPY pipeline.py handler.py ./

# Models sa načítajú z RunPod Network Volume (/workspace/models) za runtime
# Ak volume nie je pripojený, handler ich stiahne pri prvom spustení

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

CMD ["python", "-u", "handler.py"]
