"""
pipeline.py - AI Dubbing Pipeline
Kroky: FFmpeg -> Whisper -> Qwen3-14B (preklad) -> XTTS v2 (TTS+klonovanie) -> FFmpeg mix

Verzia 1.2:
- Demucs odstraneny (padal na nvcc) — pouziva cele audio
- CosyVoice2 nahradeny za Coqui XTTS v2 (nevyzaduje nvcc/deepspeed)
- Oprava: source_lang="auto" -> None pre Whisper API
- Oprava: _xtts_model global (bolo _cosyvoice_model)
- Oprava: cache adresare presunute na /workspace (perzistentny Volume), nie do docasneho workdir
"""

import os
import json
import re
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path

# Presmeruj Coqui TTS cache na perzistentny Volume — model sa nestiahne pri kazdom reštarte podu
os.environ.setdefault("COQUI_TTS_HOME", "/workspace/models")

import torch
import soundfile as sf
import numpy as np


def _ffmpeg(cmd: list[str], timeout: int = 120, step: str = "ffmpeg") -> None:
    """Spusti FFmpeg s timeoutom. Pri chybe vypise stderr pre debugovanie."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"FFmpeg timeout ({timeout}s) in step '{step}'. "
            "Video moze byt poskodene alebo je pod pretazeny."
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise RuntimeError(f"FFmpeg failed in step '{step}': {stderr}")


logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/workspace/cache"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded globals (nacitaju sa raz pri prvom jobu = warm start)
_whisper_model = None
_qwen_pipe = None
_xtts_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper large-v3...")
        _whisper_model = whisper.load_model(
            "large-v3",
            device=DEVICE,
            download_root=str(MODEL_DIR / "whisper"),
        )
    return _whisper_model


def get_qwen():
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


def get_xtts():
    """Lazy-load XTTS v2 (Coqui TTS). Nevyzaduje nvcc ani deepspeed."""
    global _xtts_model
    if _xtts_model is None:
        import functools
        import torchaudio
        # Nastav torchaudio backend na soundfile — torchcodec nie je nainštalovany
        try:
            torchaudio.set_audio_backend("soundfile")
            logger.info("torchaudio backend: soundfile")
        except Exception as e:
            logger.warning(f"torchaudio backend set failed: {e}")
        # PyTorch 2.6+ zmenil default weights_only=True, coz rozbije XTTS checkpoint.
        # Monkey-patch torch.load aby pouzival weights_only=False (XTTS je trusted source).
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
            torch.load = _orig_torch_load  # vrat povodny torch.load
    return _xtts_model


# --- Pipeline kroky ---

def step_extract_audio(video_path: str, workdir: str) -> str:
    """FFmpeg: extrahuje audio z videa ako WAV 16kHz mono."""
    out = os.path.join(workdir, "audio_raw.wav")
    _ffmpeg(
        ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", out],
        timeout=120, step="extract_audio"
    )
    logger.info(f"Audio extracted: {out}")
    return out


def step_prepare_audio(audio_path: str, workdir: str) -> tuple[str, str]:
    """
    Bez Demucs separacie — pouziva cele audio pre Whisper aj ako sprievod.
    Vracia (vocals_path, accompaniment_path).
    """
    vocals_16k = os.path.join(workdir, "vocals_16k.wav")
    _ffmpeg(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", vocals_16k],
        timeout=120, step="resample_vocals"
    )
    logger.info(f"Audio prepared — vocals={vocals_16k}, accompaniment={audio_path}")
    return vocals_16k, audio_path


def step_transcribe(vocals_path: str, source_lang: str | None = None) -> list[dict]:
    """
    Whisper large-v3: transkripcia s timestampmi.
    source_lang=None -> auto-detect jazyka (spanielcina, cistina, arabcina...).
    POZOR: Whisper ocakava None pre auto-detect, nie retazec "auto".
    """
    model = get_whisper()
    # Normalizacia: "auto" alebo prazdny retazec -> None
    if source_lang and source_lang.lower() in ("auto", ""):
        source_lang = None

    logger.info(f"Transcribing... (language={source_lang or 'auto-detect'})")
    result = model.transcribe(
        vocals_path,
        language=source_lang,
        word_timestamps=True,
        verbose=False,
    )
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result["segments"]
    ]
    logger.info(f"Transcribed {len(segments)} segments")
    return segments


def _parse_translation_json(raw: str, batch_size: int) -> dict[int, str]:
    """
    Robustny parser pre JSON vystup z Qwen3.
    Zvlada: markdown fences, verbose prefix, multiline JSON.
    [\s\S]* namiesto .*? — spolahlive pre multiline.
    """
    # Odstráň <think>...</think> blok (Qwen3 thinking mode)
    clean = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()
    match = re.search(r"(\[[\s\S]*\])", clean)
    if match:
        clean = match.group(1)

    try:
        data = json.loads(clean)
        if isinstance(data, list):
            return {
                item["id"]: item["text"]
                for item in data
                if isinstance(item, dict) and "id" in item and "text" in item
            }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Translation JSON parse failed: {e} | raw[:300]={raw[:300]}")

    return {}


def step_translate(segments: list[dict], target_lang: str = "cs") -> list[dict]:
    """
    Helsinki-NLP opus-mt: rýchly preklad bez thinking mode.
    Batch po 50, priamy translation pipeline.
    """
    pipe = get_qwen()
    translated = []

    batch_size = 50
    texts = [seg["text"] for seg in segments]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        results = pipe(batch_texts, max_length=512)
        for j, result in enumerate(results):
            translated.append({
                **segments[i + j],
                "translated": result["translation_text"],
            })
        logger.info(f"Translated batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    logger.info(f"Translated {len(translated)} segments")
    return translated


def step_tts_clone(
    segments: list[dict],
    reference_audio_path: str,
    workdir: str,
    target_lang: str = "cs",
) -> str:
    """
    XTTS v2 (Coqui TTS) zero-shot voice cloning.
    Sample rate vystupu: 24000 Hz.
    """
    XTTS_LANG_MAP = {
        "sk": "sk", "cs": "cs", "de": "de", "fr": "fr",
        "es": "es", "it": "it", "pl": "pl", "hu": "hu",
        "uk": "uk", "ru": "ru",
    }
    xtts_lang = XTTS_LANG_MAP.get(target_lang, "en")
    model = get_xtts()
    tts_sample_rate = 24000

    # Validate reference audio — XTTS needs at least 3s of clean audio
    ref_data, ref_sr = sf.read(reference_audio_path, dtype="float32")
    ref_duration = len(ref_data) / ref_sr
    logger.info(f"Reference audio: {ref_duration:.1f}s @ {ref_sr}Hz")
    if ref_duration < 3.0:
        logger.warning(f"Reference audio too short ({ref_duration:.1f}s < 3s) — voice cloning may be poor")

    all_audio_chunks: list[np.ndarray] = []
    prev_end = 0.0

    for i, seg in enumerate(segments):
        text = seg.get("translated", seg.get("text", "")).strip()
        if not text:
            continue

        gap = seg["start"] - prev_end
        if gap > 0.05:
            all_audio_chunks.append(np.zeros(int(gap * tts_sample_rate), dtype=np.float32))

        try:
            tmp_out = os.path.join(workdir, f"seg_{i:04d}.wav")
            model.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language=xtts_lang,
                file_path=tmp_out,
            )
            if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) < 100:
                raise RuntimeError(f"TTS output missing or empty: {tmp_out}")
            audio_data, sr = sf.read(tmp_out, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            if len(audio_data) == 0:
                raise RuntimeError("TTS returned zero-length audio")
            # Normalize to avoid clipping / silence from near-zero output
            peak = np.abs(audio_data).max()
            if peak > 0:
                audio_data = audio_data / peak * 0.9
            all_audio_chunks.append(audio_data)
            logger.info(f"Segment {i}: '{text[:50]}' -> {len(audio_data)/tts_sample_rate:.2f}s (peak={peak:.3f})")
        except Exception as e:
            logger.warning(f"TTS failed for segment {i} '{text[:60]}': {e}")
            dur = seg["end"] - seg["start"]
            all_audio_chunks.append(np.zeros(int(dur * tts_sample_rate), dtype=np.float32))

        prev_end = seg["end"]

    final_audio = (
        np.concatenate(all_audio_chunks)
        if all_audio_chunks
        else np.zeros(tts_sample_rate, dtype=np.float32)
    )
    out_path = os.path.join(workdir, "dubbed_voice.wav")
    sf.write(out_path, final_audio, tts_sample_rate)
    logger.info(f"TTS complete: {out_path} ({len(final_audio)/tts_sample_rate:.1f}s)")
    return out_path


def step_mix_final(
    original_video: str,
    dubbed_voice: str,
    accompaniment: str,
    workdir: str,
    output_path: str,
) -> str:
    """
    FFmpeg: zmiesaj novy hlas (0dB) + sprievod (0.7x) + zachovaj video.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", dubbed_voice,
        "-i", accompaniment,
        "-filter_complex",
        "[1:a]volume=1.0[voice];[2:a]volume=0.7[music];[voice][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    _ffmpeg(cmd, timeout=600, step="final_mix")
    logger.info(f"Final video: {output_path}")
    return output_path


# --- Hlavna funkcia ---

def run_dubbing_pipeline(
    video_path: str,
    reference_audio_path: str | None,
    target_lang: str,
    source_lang: str | None,
    output_path: str,
    job_id: str = "local",
) -> dict:
    """
    Spusti cely pipeline.
    Cache (transcription, translation, dubbed_voice) sa uklada do CACHE_DIR/<job_id>/
    na perzistentnom Volume — prezije restart podu.
    """
    # Cache adresár na Volume (perzistentny), nie v docasnom workdir
    job_cache = CACHE_DIR / job_id
    job_cache.mkdir(parents=True, exist_ok=True)

    transcription_cache = job_cache / "transcription.json"
    translation_cache   = job_cache / "translation.json"
    dubbed_voice_cache  = job_cache / "dubbed_voice.wav"

    with tempfile.TemporaryDirectory(prefix=f"dubbing_{job_id}_") as workdir:
        # 1. Extrakcia audia
        raw_audio = step_extract_audio(video_path, workdir)

        # 2. Priprava audia (bez Demucs)
        vocals, accompaniment = step_prepare_audio(raw_audio, workdir)

        # Referencia pre klonovanie
        ref_audio = reference_audio_path or vocals

        # 3. Transkripcia (cache)
        if transcription_cache.exists():
            logger.info(f"Loading cached transcription: {transcription_cache}")
            segments = json.loads(transcription_cache.read_text())
        else:
            segments = step_transcribe(vocals, source_lang)
            transcription_cache.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2)
            )
            logger.info(f"Transcription cached: {transcription_cache}")

        # 4. Preklad (cache)
        if translation_cache.exists():
            logger.info(f"Loading cached translation: {translation_cache}")
            segments = json.loads(translation_cache.read_text())
        else:
            segments = step_translate(segments, target_lang)
            translation_cache.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2)
            )
            logger.info(f"Translation cached: {translation_cache}")

        # 5. TTS (cache)
        if dubbed_voice_cache.exists():
            logger.info(f"Loading cached dubbed voice: {dubbed_voice_cache}")
            dubbed_voice = str(dubbed_voice_cache)
        else:
            dubbed_voice_tmp = step_tts_clone(segments, ref_audio, workdir, target_lang)
            shutil.copy(dubbed_voice_tmp, dubbed_voice_cache)
            dubbed_voice = str(dubbed_voice_cache)
            logger.info(f"Dubbed voice cached: {dubbed_voice_cache}")

        # 6. Finalny mix
        step_mix_final(video_path, dubbed_voice, accompaniment, workdir, output_path)

        total_seconds = segments[-1]["end"] if segments else 0
        return {
            "output_path": output_path,
            "duration_seconds": round(total_seconds, 1),
            "segments_count": len(segments),
            "target_lang": target_lang,
        }
