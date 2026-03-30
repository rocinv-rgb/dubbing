# AI Dubbing Pipeline — RunPod
# Base: CUDA 12.8 + cuDNN + Python 3.10 (podporuje Blackwell sm_120)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL maintainer="dubbing-pipeline" \
      version="3.1.0" \
      description="AI dubbing: Whisper + Helsinki-NLP + XTTS v2 + OpenVoice V2"

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
    pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python alias + build tools
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip setuptools wheel Cython

# --- Krok 1: PyTorch cu128 nightly (Blackwell) ---
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# --- Krok 2: Ostatne zavislosti ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Krok 3: Coqui TTS (XTTS v2) ---
RUN pip install --no-cache-dir "TTS>=0.22.0"

# --- Krok 4: Fix numpy po TTS (TTS downgraduje numpy, pyannote potrebuje >=2.0) ---
RUN pip install --no-cache-dir "numpy>=2.0" "scipy>=1.15.1" "pandas>=2.2.3" "matplotlib>=3.10.0"

# --- Krok 5: Fix torch.load weights_only pre XTTS (PyTorch 2.6+ breaking change) ---
RUN sed -i 's/torch\.load(f, map_location=map_location, \*\*kwargs)/torch.load(f, map_location=map_location, weights_only=False)/' \
    /usr/local/lib/python3.10/dist-packages/TTS/utils/io.py

# --- Krok 6: OpenVoice V2 ---
# --no-deps: OpenVoice requirements obsahuju av (cez faster-whisper) a numpy==1.22 ktore by
# rozbili pyannote. Instalujeme len OpenVoice samotny + jeho skutocne runtime deps pre TCC.
RUN git clone https://github.com/myshell-ai/OpenVoice /opt/openvoice && \
    pip install --no-cache-dir --no-deps -e /opt/openvoice && \
    pip install --no-cache-dir librosa pydub wavmark inflect unidecode pypinyin cn2an jieba langid

ENV OPENVOICE_CHECKPOINT_URL="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
ENV MELOTTS_INSTALL_ON_STARTUP="true"

# --- App subory ---
COPY handler.py h_v1.py test_input.json ./
COPY pipeline/ ./pipeline/

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

CMD ["python", "-u", "h_v1.py"]
