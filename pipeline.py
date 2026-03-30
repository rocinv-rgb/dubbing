"""
pipeline.py - AI Dubbing Pipeline
Kroky: FFmpeg -> Whisper -> Helsinki-NLP (preklad) -> XTTS v2 (TTS+klonovanie) -> FFmpeg mix

Verzia 1.3:
- Canvas model: kazdy TTS clip overlay-ovany na presny timestamp (ziadny kumulativny drift)
- ffmpeg atempo time-stretch: TTS clip sa zmesti do available_duration slotu
- enrich_segments_with_available_duration(): O(n) algoritmus, vyuziva pauzy medzi segmentmi
- merge_speaker_blocks(): zlucuje po sebe iduce segmenty toho isteho speakera
  (rovnaky speaker_id, pauza < MAX_MERGE_PAUSE_S, blok max MAX_MERGE_BLOCK_S)
- PAUSE_MARKER: konstanta (None/"..."/"," atd), prepisatelna z job inputu

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

# --- Konfigurácia zlučovania segmentov ---
# Prepisatelne z job inputu (pozri run_dubbing_pipeline parameter pause_marker)
PAUSE_MARKER: str | None = None   # None = ignoruj pauzy, "..." = vloz marker medzi vety
MAX_MERGE_PAUSE_S: float = 1.0    # max pauza medzi segmentmi na zlucenie (s)
MAX_MERGE_BLOCK_S: float = 7.0    # max dlzka zluceneho bloku (s)

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
        from TTS.api import TTS as CoquiTTS
        logger.info("Loading XTTS v2...")
        tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(DEVICE)
        _xtts_model = tts
        logger.info("XTTS v2 loaded.")
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


def enrich_segments_with_available_duration(segments: list[dict], video_end: float) -> list[dict]:
    """
    O(n) algoritmus: pre kazdy segment spocita available_duration =
    slot (end-start) + pauza do dalsiho segmentu.
    Vysledok sa pouziva v step_tts_clone namiesto caseho slotu — TTS moze
    vyuzit ticho pred dalsim segmentom a nemusite ho toho zrychlovat.
    """
    for i, seg in enumerate(segments):
        slot = seg["end"] - seg["start"]
        if i + 1 < len(segments):
            pause = segments[i + 1]["start"] - seg["end"]
        else:
            pause = video_end - seg["end"]
        seg["available_duration"] = slot + max(0.0, pause)
    return segments


def merge_speaker_blocks(
    segments: list[dict],
    max_pause_s: float = MAX_MERGE_PAUSE_S,
    max_block_s: float = MAX_MERGE_BLOCK_S,
    pause_marker: str | None = PAUSE_MARKER,
) -> list[dict]:
    """
    Zlucuje po sebe iduce segmenty toho isteho speakera do blokov.
    Podmienky pre zlucenie:
      - rovnaky speaker_id (ak chyba, povazuje sa za rovnakeho)
      - pauza medzi segmentmi < max_pause_s
      - celkova dlzka bloku (end posledneho - start prveho) <= max_block_s

    Zluceny blok ma:
      - start = start prveho segmentu
      - end   = end posledneho segmentu
      - available_duration = suma available_duration vsetkych zlucených
      - text / translated  = spojene pause_marker-om (alebo medzerou)
      - speaker_id         = speaker_id prveho segmentu

    Segmenty bez speaker_id sa nikdy nezlucuju s inym.
    """
    if not segments:
        return segments

    merged: list[dict] = []
    current = dict(segments[0])


    for i in range(1, len(segments)):
        seg = segments[i]
        seg_speaker = seg.get("speaker_id") or seg.get("speaker") or ""
        cur_speaker = current.get("speaker_id") or current.get("speaker") or ""

        pause = seg["start"] - current["end"]
        block_end = seg["end"]
        block_duration = block_end - current["start"]

        can_merge = (
            cur_speaker != ""
            and cur_speaker == seg_speaker
            and pause < max_pause_s
            and block_duration <= max_block_s
        )

        if can_merge:
            # Zluc text/translated
            joiner = pause_marker if pause_marker is not None else " "
            if "translated" in current and "translated" in seg:
                current["translated"] = current["translated"].rstrip() + joiner + seg["translated"].lstrip()
            if "text" in current and "text" in seg:
                current["text"] = current["text"].rstrip() + joiner + seg["text"].lstrip()
            current["end"] = seg["end"]
            # available_duration: suma (pauzy sa uz spotrebuju v bloku)
            current["available_duration"] = (
                current.get("available_duration", current["end"] - current["start"])
                + seg.get("available_duration", seg["end"] - seg["start"])
            )
        else:
            merged.append(current)
            current = dict(seg)

    merged.append(current)
    logger.info(f"merge_speaker_blocks: {len(segments)} → {len(merged)} blokov")
    return merged


def _stretch_audio_ffmpeg(audio_data: np.ndarray, sample_rate: int, speed_factor: float) -> np.ndarray:
    """
    Time-stretch audio pomocou FFmpeg atempo filtra.
    speed_factor > 1.0 = zrychlenie (TTS je dlhsi ako slot)
    speed_factor < 1.0 = spomalenie (TTS je kratsi ako slot)
    atempo je obmedzeny na 0.5-2.0, pre extremne hodnoty retiazime filtre.
    """
    # Clamp: nema zmysel stretovat o viac ako 3x/0.33x
    speed_factor = max(0.33, min(3.0, speed_factor))
    if abs(speed_factor - 1.0) < 0.02:
        return audio_data  # zanedbatelna zmena, nerobit nic

    # Serialize audio_data -> WAV bytes -> pipe do ffmpeg
    import io
    buf_in = io.BytesIO()
    sf.write(buf_in, audio_data, sample_rate, format="wav")
    buf_in.seek(0)

    # Pre atempo mimo 0.5-2.0 retiazime: napr. 0.33 = atempo=0.5,atempo=0.66
    if speed_factor < 0.5:
        # dva kroky: sqrt(speed_factor) aplikovany 2x
        import math
        step = math.sqrt(speed_factor)
        atempo_filter = f"atempo={step:.4f},atempo={step:.4f}"
    elif speed_factor > 2.0:
        import math
        step = math.sqrt(speed_factor)
        atempo_filter = f"atempo={step:.4f},atempo={step:.4f}"
    else:
        atempo_filter = f"atempo={speed_factor:.4f}"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-filter:a", atempo_filter,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=buf_in.read(), capture_output=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg atempo failed: {proc.stderr.decode()[-300:]}")

    buf_out = io.BytesIO(proc.stdout)
    stretched, _ = sf.read(buf_out, dtype="float32")
    if stretched.ndim > 1:
        stretched = stretched.mean(axis=1)
    return stretched


def step_tts_clone(
    segments: list[dict],
    reference_audio_path: str,
    workdir: str,
    target_lang: str = "sk",
) -> str:
    """
    XTTS v2 (Coqui TTS) zero-shot voice cloning — CANVAS MODEL.

    Namiesto linearneho buffera pouzivame canvas (ticha stopa dlzky videa),
    do ktoreho overlay-ujeme kazdy TTS clip na presny start_ms.
    Tym eliminujeme kumulativny drift — kazdy segment je vzdy na spravnom mieste.

    Ak TTS clip je dlhsi ako segment slot → time-stretch cez ffmpeg atempo.
    Max stretch ratio: 3.0x (ak prelozena veta je 3x dlhsia = problem prekladu).
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

    # Zistíme celkovú dĺžku videa z posledného segmentu (+1s buffer)
    total_duration = (segments[-1]["end"] if segments else 10.0) + 1.0
    canvas_samples = int(total_duration * tts_sample_rate)
    canvas = np.zeros(canvas_samples, dtype=np.float32)

    for i, seg in enumerate(segments):
        text = seg.get("translated", seg.get("text", "")).strip()
        if not text:
            continue

        seg_start_s = seg["start"]
        seg_end_s = seg["end"]
        slot_duration_s = seg_end_s - seg_start_s
        slot_samples = max(1, int(slot_duration_s * tts_sample_rate))
        canvas_offset = int(seg_start_s * tts_sample_rate)

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

            # --- TIME-STRETCH ak TTS clip nezapadá do available_duration ---
            # available_duration = slot + pauza do dalsiho (predpocitane v enrich_segments)
            # Fallback na slot_duration ak enrich nebol volany (napr. stara cache)
            available_s = seg.get("available_duration", slot_duration_s)
            tts_duration_s = len(audio_data) / tts_sample_rate
            if available_s > 0.1:
                speed_factor = tts_duration_s / available_s
                if abs(speed_factor - 1.0) > 0.05:  # >5% odchylka -> stretch
                    logger.info(
                        f"Seg {i}: available={available_s:.2f}s TTS={tts_duration_s:.2f}s "
                        f"→ stretch x{speed_factor:.2f}"
                    )
                    audio_data = _stretch_audio_ffmpeg(audio_data, tts_sample_rate, speed_factor)

            # Normalize
            peak = np.abs(audio_data).max()
            if peak > 0:
                audio_data = audio_data / peak * 0.9

            # Overlay do canvasu na presnu pozíciu (nie append!)
            end_idx = min(canvas_offset + len(audio_data), canvas_samples)
            write_len = end_idx - canvas_offset
            if write_len > 0:
                canvas[canvas_offset:end_idx] += audio_data[:write_len]

            logger.info(
                f"Segment {i}: '{text[:50]}' → @{seg_start_s:.2f}s "
                f"slot={slot_duration_s:.2f}s tts={len(audio_data)/tts_sample_rate:.2f}s"
            )
        except Exception as e:
            logger.warning(f"TTS failed for segment {i} '{text[:60]}': {e}")
            # Pri chybe len preskočíme segment — canvas zostane tichý na tej pozícii

    # Clamp canvas na [-1, 1] pre prípad overlappingu (viac speakerov naraz)
    max_val = np.abs(canvas).max()
    if max_val > 1.0:
        canvas = canvas / max_val * 0.95

    out_path = os.path.join(workdir, "dubbed_voice.wav")
    sf.write(out_path, canvas, tts_sample_rate)
    logger.info(f"TTS complete (canvas): {out_path} ({len(canvas)/tts_sample_rate:.1f}s)")
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
    pause_marker: str | None = PAUSE_MARKER,
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

        # 4b. Enrich + merge (vždy — rýchly algoritmus, bez cache)
        video_end = segments[-1]["end"] + 1.0 if segments else 10.0
        segments = enrich_segments_with_available_duration(segments, video_end)
        segments = merge_speaker_blocks(
            segments,
            max_pause_s=MAX_MERGE_PAUSE_S,
            max_block_s=MAX_MERGE_BLOCK_S,
            pause_marker=pause_marker,
        )

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
