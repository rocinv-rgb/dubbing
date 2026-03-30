"""
pipeline/tts.py - TTS voice cloning (XTTS v2), time-stretching a merging segmentov
"""

import os
import io
import logging
import subprocess
import numpy as np
import soundfile as sf
from .config import PAUSE_MARKER, MAX_MERGE_PAUSE_S, MAX_MERGE_BLOCK_S, SEGMENT_GAP_MS
from .models import get_xtts
from .translate import _normalize_text

logger = logging.getLogger(__name__)


def enrich_segments_with_available_duration(
    segments: list[dict],
    video_end: float,
    gap_ms: float = SEGMENT_GAP_MS,
) -> list[dict]:
    """
    O(n) algoritmus: pre kazdy segment spocita available_duration =
    slot (end-start) + pauza do dalsiho segmentu - gap_ms buffer.

    gap_ms zabezpeci ze TTS skonci aspon X ms pred zaciatkom dalsej vety.
    Ak je pauza kratsia ako gap_ms -> available_duration = len slot (bez rozsierovania).
    """
    gap_s = gap_ms / 1000.0
    for i, seg in enumerate(segments):
        slot = seg["end"] - seg["start"]
        if i + 1 < len(segments):
            pause = segments[i + 1]["start"] - seg["end"] - gap_s
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


def _ensure_czech_base_speaker() -> str:
    from .config import CZECH_BASE_SPEAKER
    if CZECH_BASE_SPEAKER.exists():
        return str(CZECH_BASE_SPEAKER)
    logger.info("Stahujem cesku base vzorku z Mozilla Common Voice...")
    CZECH_BASE_SPEAKER.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/resolve/main/audio/cs/train/common_voice_cs_24004527.mp3"
    import urllib.request
    mp3_path = str(CZECH_BASE_SPEAKER).replace('.wav', '.mp3')
    urllib.request.urlretrieve(url, mp3_path)
    from .audio import _ffmpeg
    _ffmpeg(["ffmpeg", "-y", "-i", mp3_path, "-ar", "22050", "-ac", "1", str(CZECH_BASE_SPEAKER)],
            timeout=30, step="czech_base_convert")
    import os; os.remove(mp3_path)
    logger.info(f"Czech base speaker: {CZECH_BASE_SPEAKER}")
    return str(CZECH_BASE_SPEAKER)


def step_tone_convert(
    tts_audio_path: str,
    source_ref_wav: str,
    target_ref_wav: str,
    workdir: str,
    output_path: str,
) -> str:
    """
    OpenVoice V2 TCC — pouziva converter.extract_se() priamo (nie se_extractor.get_se)
    aby sme sa vyhli zavislosti na faster_whisper/av ktore sa nedaju buildovat.
    """
    from .models import get_openvoice
    import torch
    converter = get_openvoice()
    # extract_se priamo z wav suborov — nepotrebuje Whisper ani VAD
    source_se = converter.extract_se([source_ref_wav])
    target_se = converter.extract_se([target_ref_wav])
    converter.convert(
        audio_src_path=tts_audio_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell",
    )
    return output_path


def step_tts_clone(
    segments: list[dict],
    reference_audio_path,  # str alebo dict {speaker_id: path}
    workdir: str,
    target_lang: str = "sk",
    use_openvoice: bool = False,
) -> str:
    """
    XTTS v2 (Coqui TTS) zero-shot voice cloning — CANVAS MODEL.

    Namiesto linearneho buffera pouzivame canvas (ticha stopa dlzky videa),
    do ktoreho overlay-ujeme kazdy TTS clip na presny start_ms.
    Tym eliminujeme kumulativny drift — kazdy segment je vzdy na spravnom mieste.

    Ak TTS clip je dlhsi ako segment slot → time-stretch cez ffmpeg atempo.
    Max stretch ratio: 3.0x (ak prelozena veta je 3x dlhsia = problem prekladu).
    Sample rate vystupu: 24000 Hz.

    reference_audio_path moze byt:
    - str: jedna referencna stopa pre vsetkych speakerov
    - dict {speaker_id: path}: per-speaker referencie

    use_openvoice=True: dvojkrokovy process — XTTS s ceskou base vzorkou + OpenVoice V2 TCC prefarbenie
    """
    XTTS_LANG_MAP = {
        "sk": "sk", "cs": "cs", "de": "de", "fr": "fr",
        "es": "es", "it": "it", "pl": "pl", "hu": "hu",
        "uk": "uk", "ru": "ru",
    }
    xtts_lang = XTTS_LANG_MAP.get(target_lang, "en")
    model = get_xtts()
    tts_sample_rate = 24000

    czech_base = _ensure_czech_base_speaker() if use_openvoice else None

    # Normalize reference_audio_path na dict
    if isinstance(reference_audio_path, dict):
        speaker_refs = reference_audio_path
        default_ref = next(iter(speaker_refs.values()))
    else:
        speaker_refs = {}
        default_ref = reference_audio_path

    # Validate default reference audio
    ref_data, ref_sr = sf.read(default_ref, dtype="float32")
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
        text = _normalize_text(text, xtts_lang)
        if not text:
            continue

        seg_start_s = seg["start"]
        seg_end_s = seg["end"]
        slot_duration_s = seg_end_s - seg_start_s
        slot_samples = max(1, int(slot_duration_s * tts_sample_rate))
        canvas_offset = int(seg_start_s * tts_sample_rate)

        try:
            # Per-speaker ref audio — fallback na default ak speaker nema vlastny
            speaker_id = seg.get("speaker", "SPEAKER_00")
            ref_wav = speaker_refs.get(speaker_id, default_ref)
            tmp_out = os.path.join(workdir, f"seg_{i:04d}.wav")
            tts_ref = czech_base if use_openvoice else ref_wav
            model.tts_to_file(
                text=text,
                speaker_wav=tts_ref,
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

            # OpenVoice TCC — prefarbenie na hlas speakera
            if use_openvoice and czech_base:
                try:
                    tmp_ov = os.path.join(workdir, f"seg_{i:04d}_ov.wav")
                    step_tone_convert(tmp_out, czech_base, ref_wav, workdir, tmp_ov)
                    audio_data, sr = sf.read(tmp_ov, dtype="float32")
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                except Exception as ov_err:
                    logger.warning(f"OpenVoice TCC failed for seg {i}: {ov_err} — pouzivam XTTS output")

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
