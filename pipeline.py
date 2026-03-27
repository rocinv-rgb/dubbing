"""
pipeline.py - AI Dubbing Pipeline
Kroky: FFmpeg -> Demucs -> Whisper -> Qwen3 (preklad) -> CosyVoice2 (TTS+klonovanie) -> FFmpeg mix

Opravy v1.1:
- Demucs: odstranene --mp3, vystup je WAV + ffmpeg resample na 16kHz
- Qwen parser: JSON mode s few-shot promptom, odolny voci verbose prefixom
- CosyVoice2: explicitny .float() cast pred inferenciou
"""

import os
import sys
import json
import re
import subprocess
import tempfile
import logging
from pathlib import Path

import torch
import soundfile as sf
import numpy as np


# Timeout pre FFmpeg — zabrani zaseknutiu warm podu pri korumpovanych suboroch.
# 120s pre kratke operacie (extract/resample), 600s pre finalny mix dlheho videa.
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Lazy-loaded globals (nacitaju sa raz pri prvom jobu = warm start) ---
_whisper_model = None
_qwen_pipe = None
_cosyvoice_model = None


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
        model_id = str(MODEL_DIR / "qwen3-14b")
        if not (MODEL_DIR / "qwen3-14B").exists():
            model_id = "Qwen/Qwen3-14B"  # HF download fallback
        logger.info(f"Loading Qwen3-14B from {model_id}...")
        _qwen_pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_new_tokens=4096,
        )
    return _qwen_pipe


def get_cosyvoice():
    global _cosyvoice_model
    if _cosyvoice_model is None:
        sys.path.insert(0, "/app/CosyVoice")
        sys.path.insert(0, "/app/CosyVoice/third_party/Matcha-TTS")
        from cosyvoice.cli.cosyvoice import CosyVoice2

        model_path = MODEL_DIR / "CosyVoice2-0.5B"
        if not model_path.exists():
            from huggingface_hub import snapshot_download
            logger.info("Downloading CosyVoice2-0.5B...")
            snapshot_download(
                "FunAudioLLM/CosyVoice2-0.5B",
                local_dir=str(model_path),
            )
        logger.info("Loading CosyVoice2-0.5B...")
        _cosyvoice_model = CosyVoice2(str(model_path), load_jit=False, load_trt=False)
    return _cosyvoice_model


# --- Pipeline kroky ---

def step_extract_audio(video_path: str, workdir: str) -> str:
    """FFmpeg: extrahuje audio z videa ako WAV 16kHz mono."""
    out = os.path.join(workdir, "audio_raw.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-vn",
        out,
    ]
    _ffmpeg(cmd, timeout=120, step="extract_audio")
    logger.info(f"Audio extracted: {out}")
    return out


def step_separate_audio(audio_path: str, workdir: str) -> tuple[str, str]:
    """
    Demucs htdemucs_ft: oddeli hlas (vocals) od hudby/zvukov (no_vocals).
    Vracia (vocals_16k_path, accompaniment_path).

    OPRAVA: --mp3 flag je odstraneny. torchaudio MP3 backend nie je
    garantovany na vsetkych systemoch a moze crashnut bez lame/ffmpeg
    MP3 dekodera. Demucs defaultne produkuje WAV, co je spolahlive.
    """
    import demucs.separate

    out_dir = os.path.join(workdir, "demucs_out")
    os.makedirs(out_dir, exist_ok=True)

    demucs.separate.main([
        "--model", "htdemucs_ft",
        "--two-stems", "vocals",
        "-o", out_dir,
        audio_path,
    ])

    # Demucs uklada do out_dir/htdemucs_ft/<track_name>/
    track_name = Path(audio_path).stem
    vocals_wav = os.path.join(out_dir, "htdemucs_ft", track_name, "vocals.wav")
    no_vocals_wav = os.path.join(out_dir, "htdemucs_ft", track_name, "no_vocals.wav")

    # Resample na 16kHz mono pre Whisper (Demucs vracia 44.1kHz stereo)
    vocals_16k = os.path.join(out_dir, "vocals_16k.wav")
    _ffmpeg(
        ["ffmpeg", "-y", "-i", vocals_wav, "-ar", "16000", "-ac", "1", vocals_16k],
        timeout=60, step="resample_vocals"
    )

    logger.info(f"Separated: vocals={vocals_16k}, accompaniment={no_vocals_wav}")
    return vocals_16k, no_vocals_wav


def step_transcribe(vocals_path: str, source_lang: str | None = None) -> list[dict]:
    """
    Whisper large-v3: transkripcia s timestampmi.
    source_lang=None -> Whisper automaticky detekuje jazyk (spanielcina, cistina, arabcina...).
    Vracia list segmentov: [{start, end, text}, ...]
    """
    model = get_whisper()
    lang_display = source_lang or "auto-detect"
    logger.info(f"Transcribing... (language={lang_display})")
    result = model.transcribe(
        vocals_path,
        language=source_lang,   # None = auto-detection
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

    Fallback hierarchia:
    1. Priamo parse JSON array
    2. Extrahovanie JSON array regexom (zvlada prose pred/po JSON)
    3. Prazdny dict -> caller pouzije originalny text ako fallback

    Zvlada: markdown fences, "Here is the translation:" prefix,
    trailing whitespace, unicode apostrofy.
    """
    # Stripni markdown fences
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Najdi prvu [ ... ] strukturu v texte.
    # [\s\S]* namiesto .*? -- spolahlive zachyti multiline JSON aj ked Qwen3 prida prose pred arraym.
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


def step_translate(segments: list[dict], target_lang: str = "sk") -> list[dict]:
    """
    Qwen3-32B: prelozi segmenty, zachova timing.

    OPRAVA: pouziva JSON mode s few-shot promptom namiesto krehkeho
    riadkoveho parsera. Qwen3 pridava verbose prefix ("Here is..."),
    co stary parser rozbijalo. JSON format je deterministicky.
    """
    LANG_NAMES = {
        "sk": "Slovak", "cs": "Czech", "de": "German",
        "fr": "French", "es": "Spanish", "it": "Italian",
        "pl": "Polish", "hu": "Hungarian",
    }
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    pipe = get_qwen()
    translated = []

    batch_size = 20
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]

        items_json = json.dumps(
            [{"id": j, "text": seg["text"]} for j, seg in enumerate(batch)],
            ensure_ascii=False,
        )

        # Few-shot prompt ukazuje presny format vstupu aj vystupu
        prompt = (
            f'Translate each "text" value from English to {lang_name}.\n'
            f'Return ONLY a valid JSON array. No explanation, no markdown, no preamble.\n\n'
            f'Example input:  [{{"id": 0, "text": "Hello world"}}, {{"id": 1, "text": "How are you?"}}]\n'
            f'Example output: [{{"id": 0, "text": "Ahoj svet"}}, {{"id": 1, "text": "Ako sa mas?"}}]\n\n'
            f'Input: {items_json}\n'
            f'Output:'
        )

        response = pipe(
            [{"role": "user", "content": prompt}],
            return_full_text=False,
        )
        raw = response[0]["generated_text"].strip()
        lines = _parse_translation_json(raw, len(batch))

        for j, seg in enumerate(batch):
            translated.append({
                **seg,
                "translated": lines.get(j, seg["text"]),  # fallback = original text
            })

    logger.info(f"Translated {len(translated)} segments")
    return translated


def step_tts_clone(
    segments: list[dict],
    reference_audio_path: str,
    workdir: str,
    target_lang: str = "sk",
) -> str:
    """
    CosyVoice2 zero-shot voice cloning: syntetizuje prelozene segmenty
    s klonovanym hlasom. Vracia cestu k vysledku.

    OPRAVA: ref_audio.float() a audio_data.float() — CosyVoice2 interna
    infer moze failnut ak tensor ma nespravny dtype (napr. float16 z torchaudio).
    """
    import torchaudio

    model = get_cosyvoice()
    tts_sample_rate = 22050  # CosyVoice2 vystupny sample rate

    # Nacitaj referencny hlas (prvych 10 sekund)
    ref_audio, ref_sr = torchaudio.load(reference_audio_path)
    if ref_sr != 16000:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 16000)
    ref_audio = ref_audio[:, : 16000 * 10]  # max 10s referencie
    ref_audio = ref_audio.float()            # OPRAVA: explicitny float32

    all_audio_chunks: list[np.ndarray] = []
    prev_end = 0.0

    for seg in segments:
        # Pridaj ticho ak je medzera medzi segmentmi
        gap = seg["start"] - prev_end
        if gap > 0.05:
            silence = np.zeros(int(gap * tts_sample_rate), dtype=np.float32)
            all_audio_chunks.append(silence)

        # TTS synteza
        try:
            output = list(model.inference_zero_shot(
                tts_text=seg["translated"],
                prompt_text="",
                prompt_speech_16k=ref_audio,
                stream=False,
            ))
            if output:
                # OPRAVA: .float() pred .numpy() — garantuje float32 dtype
                audio_data = output[0]["tts_speech"].float().numpy().flatten()
                all_audio_chunks.append(audio_data)
        except Exception as e:
            logger.warning(f"TTS failed for segment '{seg['translated'][:60]}': {e}")
            # Fallback: ticho rovnako dlhe ako originalny segment
            dur = seg["end"] - seg["start"]
            all_audio_chunks.append(np.zeros(int(dur * tts_sample_rate), dtype=np.float32))

        prev_end = seg["end"]

    final_audio = (
        np.concatenate(all_audio_chunks)
        if all_audio_chunks
        else np.zeros(1000, dtype=np.float32)
    )
    out_path = os.path.join(workdir, "dubbed_voice.wav")
    sf.write(out_path, final_audio, tts_sample_rate)
    logger.info(f"TTS complete: {out_path}")
    return out_path


def step_mix_final(
    original_video: str,
    dubbed_voice: str,
    accompaniment: str,
    workdir: str,
    output_path: str,
) -> str:
    """
    FFmpeg: zmiesaj novy hlas + hudba/zvuky + zachovaj video track.
    Hlas: 0 dB, Sprievod: 0.7x (-3 dB) — jemne potlaceny pod hlasom.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,       # [0] video
        "-i", dubbed_voice,          # [1] novy hlas
        "-i", accompaniment,         # [2] hudba + zvuky
        "-filter_complex",
        "[1:a]volume=1.0[voice];[2:a]volume=0.7[music];[voice][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",             # video bez reenkodovania
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
) -> dict:
    """
    Spusti cely pipeline. Ak reference_audio_path nie je zadany,
    pouzije extrahovany hlas z videa ako referenciu.
    """
    with tempfile.TemporaryDirectory(prefix="dubbing_") as workdir:
        # 1. Extrakcia audia
        raw_audio = step_extract_audio(video_path, workdir)

        # 2. Separacia Demucs
        vocals, accompaniment = step_separate_audio(raw_audio, workdir)

        # Referencia pre klonovanie = povodny hlas ak nie je zadany
        ref_audio = reference_audio_path or vocals

        # 3. Transkripcia
        segments = step_transcribe(vocals, source_lang)

        # 4. Preklad
        segments = step_translate(segments, target_lang)

        # 5. TTS s klonovanim hlasu
        dubbed_voice = step_tts_clone(segments, ref_audio, workdir, target_lang)

        # 6. Finalny mix
        step_mix_final(video_path, dubbed_voice, accompaniment, workdir, output_path)

        total_seconds = segments[-1]["end"] if segments else 0
        return {
            "output_path": output_path,
            "duration_seconds": round(total_seconds, 1),
            "segments_count": len(segments),
            "target_lang": target_lang,
        }
