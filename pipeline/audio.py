"""
pipeline/audio.py - Extrakcia, separácia a príprava audia
"""

import os
import logging
import shutil
import subprocess
from .config import MODEL_DIR

logger = logging.getLogger(__name__)

SEPARATOR_VENV      = "/venv-separator/bin/python"
SEPARATOR_MODEL     = "MDX23C-8KFFT-InstVoc_HQ.ckpt"
SEPARATOR_MODEL_DIR = str(MODEL_DIR / "mdx23c")


def _ffmpeg(cmd: list[str], timeout: int = 120, step: str = "ffmpeg") -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout ({timeout}s) in step '{step}'.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise RuntimeError(f"FFmpeg failed in step '{step}': {stderr}")


def step_extract_audio(video_path: str, workdir: str) -> str:
    out = os.path.join(workdir, "audio_raw.wav")
    _ffmpeg(["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", out],
            timeout=120, step="extract_audio")
    return out


def step_separate_audio(audio_path: str, workdir: str) -> tuple[str, str]:
    """
    MDX23C separacia cez /venv-separator (audio-separator).
    Vracia (vocals_path, accompaniment_path).
    Fallback: ak venv neexistuje, vrati povodne audio pre oba vystupy.
    """
    if not os.path.exists(SEPARATOR_VENV):
        logger.warning("audio-separator venv nenajdeny — preskakujem separaciu, generujem tiche accompaniment")
        vocals = os.path.join(workdir, "vocals.wav")
        shutil.copy(audio_path, vocals)
        # Ticha stopa ako accompaniment — zabrani mixovaniu povodneho hlasu do vystupu
        silence = os.path.join(workdir, "silence.wav")
        _ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-t", "300", silence], timeout=30, step="gen_silence")
        return vocals, silence

    vocals_out = os.path.join(workdir, "vocals.wav")
    accompaniment_out = os.path.join(workdir, "accompaniment.wav")

    script = f"""
import sys
sys.path.append('/venv-separator/lib/python3.10/site-packages')
from audio_separator.separator import Separator
import torch
print('CUDA:', torch.cuda.is_available(), file=sys.stderr)
sep = Separator(
    model_file_dir='{SEPARATOR_MODEL_DIR}',
    output_dir='{workdir}',
    output_format='wav',
    mdx_params={{'hop_length': 1024, 'segment_size': 256, 'overlap': 0.25, 'batch_size': 1}},
)
sep.load_model('{SEPARATOR_MODEL}')
out = sep.separate('{audio_path}')
print('|'.join(out))
"""
    result = subprocess.run(
        ["python", "-c", script],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        logger.error(f"audio-separator failed: {result.stderr[-500:]}")
        logger.warning("Fallback: pouzivam povodne audio bez separacie")
        shutil.copy(audio_path, vocals_out)
        return vocals_out, audio_path

    # Zisti ktory subor je vocals a ktory accompaniment
    outputs = result.stdout.strip().split('|')
    logger.info(f"Separator outputs: {outputs}")
    vocals_path = None
    accomp_path = None
    for p in outputs:
        p = p.strip()
        if not p:
            continue
        # Ak nie je absolutna cesta, pridaj workdir
        if not os.path.isabs(p):
            p = os.path.join(workdir, p)
        pl = os.path.basename(p).lower()
        if 'instrumental' in pl or 'accomp' in pl or 'no_vocals' in pl:
            accomp_path = p
        else:
            vocals_path = p

    if not vocals_path or not os.path.exists(vocals_path):
        logger.warning(f"Separator: vocals subor nenajdeny ({vocals_path}) — fallback")
        shutil.copy(audio_path, vocals_out)
        return vocals_out, audio_path

    if not accomp_path or not os.path.exists(accomp_path):
        logger.warning(f"Separator: accompaniment subor nenajdeny ({accomp_path}) — pouzivam povodne audio")
        accomp_path = audio_path

    # Resample vocals na 16k mono pre Whisper/pyannote
    vocals_16k = os.path.join(workdir, "vocals_16k.wav")
    _ffmpeg(["ffmpeg", "-y", "-i", vocals_path, "-ar", "16000", "-ac", "1", vocals_16k],
            timeout=120, step="resample_vocals")

    logger.info(f"Separacia hotova: vocals={vocals_16k}, accompaniment={accomp_path}")
    return vocals_16k, accomp_path


def step_prepare_audio(audio_path: str, workdir: str) -> tuple[str, str]:
    vocals_16k = os.path.join(workdir, "vocals_16k.wav")
    _ffmpeg(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", vocals_16k],
            timeout=120, step="resample_vocals")
    return vocals_16k, audio_path
