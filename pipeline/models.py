"""
pipeline/models.py - Globálne singleton modely (lazy load)
"""

import os
import logging
import torch
from .config import MODEL_DIR, DEVICE

logger = logging.getLogger(__name__)

_whisper_model    = None
_qwen_pipe        = None
_qwen_model       = None
_qwen_tokenizer   = None
_xtts_model       = None
_diarize_pipeline = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper large-v3...")
        _whisper_model = whisper.load_model(
            "large-v3", device=DEVICE,
            download_root=str(MODEL_DIR / "whisper"),
        )
    return _whisper_model


def get_translator():
    global _qwen_pipe
    if _qwen_pipe is None:
        from transformers import pipeline
        logger.info("Loading Helsinki-NLP translation model...")
        _qwen_pipe = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-cs",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _qwen_pipe


def get_qwen():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = str(MODEL_DIR / "qwen3-14B")
        logger.info(f"Loading Qwen3-14B from {model_path}...")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto"
        )
        logger.info("Qwen3-14B loaded.")
    return _qwen_model, _qwen_tokenizer


def get_xtts():
    global _xtts_model
    if _xtts_model is None:
        import functools
        _orig_torch_load = torch.load
        torch.load = functools.partial(_orig_torch_load, weights_only=False)
        try:
            from TTS.api import TTS as CoquiTTS
            logger.info("Loading XTTS v2...")
            tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts.to(DEVICE)
            _xtts_model = tts
            logger.info("XTTS v2 loaded.")
        finally:
            torch.load = _orig_torch_load
    return _xtts_model


def get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        from pyannote.audio import Pipeline as PyPipeline
        token = os.environ.get("HF_API_TOKEN") or os.environ.get("RUNPOD_SECRET_HF_API_TOKEN", "")
        if not token:
            raise RuntimeError("HF_API_TOKEN env var nie je nastaveny — potrebny pre pyannote diarizaciu")
        logger.info("Loading pyannote speaker-diarization-3.1...")
        _diarize_pipeline = PyPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token
        )
        _diarize_pipeline.to(torch.device(DEVICE))
        logger.info("Diarization pipeline loaded.")
    return _diarize_pipeline


_openvoice_converter = None

def get_openvoice():
    global _openvoice_converter
    if _openvoice_converter is None:
        from .config import OPENVOICE_CHECKPOINT_DIR
        ckpt_path = str(OPENVOICE_CHECKPOINT_DIR / "converter")
        # Stiahni checkpointy ak neexistuju
        import urllib.request, zipfile, io as _io
        ckpt_file = OPENVOICE_CHECKPOINT_DIR / "converter" / "checkpoint.pth"
        if not ckpt_file.exists():
            url = os.environ.get("OPENVOICE_CHECKPOINT_URL", "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip")
            logger.info(f"Stahuju OpenVoice V2 checkpointy...")
            OPENVOICE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as r:
                with zipfile.ZipFile(_io.BytesIO(r.read())) as z:
                    # ZIP obsahuje checkpoints_v2_0417/ — extrahujeme a premiestnime na OPENVOICE_CHECKPOINT_DIR
                    z.extractall(str(MODEL_DIR))
            # Presun checkpoints_v2_0417/* -> OPENVOICE_CHECKPOINT_DIR
            import shutil as _shutil
            extracted = MODEL_DIR / 'checkpoints_v2_0417'
            if extracted.exists() and not OPENVOICE_CHECKPOINT_DIR.exists():
                _shutil.move(str(extracted), str(OPENVOICE_CHECKPOINT_DIR))
            logger.info("OpenVoice V2 checkpointy stiahnute.")
        from openvoice.api import ToneColorConverter
        converter = ToneColorConverter(f"{ckpt_path}/config.json", device=DEVICE, enable_watermark=False)
        converter.load_ckpt(f"{ckpt_path}/checkpoint.pth")
        _openvoice_converter = converter
        logger.info("OpenVoice V2 TCC loaded.")
    return _openvoice_converter
