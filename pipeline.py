"""
pipeline.py - AI Dubbing Pipeline
Kroky: FFmpeg -> Whisper -> Helsinki-NLP preklad -> pyannote diarizacia -> XTTS v2 (per-speaker voice cloning) -> FFmpeg mix

Verzia 1.3:
- Pridana pyannote speaker diarization (rozlisenie hlasov)
- Kazdy speaker dostane vlastny ref_audio a vlastny XTTS hlas
- COQUI_TTS_HOME -> /workspace/models (perzistentny)
- torchaudio.load monkey-patch -> soundfile (torchcodec workaround)
- torch.load weights_only=False pre XTTS (PyTorch 2.6 compat)
- Text normalizacia pre TTS (cisla s medzerami atd.)
"""

import os
import json
import re
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path

# Presmeruj Coqui TTS cache na perzistentny Volume
os.environ.setdefault("COQUI_TTS_HOME", "/workspace/models")

# torchaudio 2.11+ odstranilo set_audio_backend(), defaultuje na torchcodec (nie je nainštalovany).
# Monkey-patch torchaudio.load -> soundfile hned pri importe modulu.
def _patch_torchaudio_load():
    try:
        import torchaudio
        import soundfile as _sf
        if getattr(torchaudio, "_cml_patched", False):
            return
        _orig = torchaudio.load
        def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
                     channels_first=True, format=None, backend=None, **kw):
            try:
                data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
                if num_frames > 0:
                    data = data[frame_offset:frame_offset + num_frames]
                elif frame_offset > 0:
                    data = data[frame_offset:]
                import torch as _t
                return _t.from_numpy(data.T if channels_first else data), sr
            except Exception:
                return _orig(path, frame_offset=frame_offset, num_frames=num_frames,
                             normalize=normalize, channels_first=channels_first)
        torchaudio.load = _sf_load
        torchaudio._cml_patched = True
    except ImportError:
        pass

_patch_torchaudio_load()

import torch
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
CACHE_DIR  = Path(os.environ.get("CACHE_DIR",  "/workspace/cache"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

_whisper_model   = None
_qwen_pipe       = None
_xtts_model      = None
_diarize_pipeline = None


def _ffmpeg(cmd: list[str], timeout: int = 120, step: str = "ffmpeg") -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timeout ({timeout}s) in step '{step}'.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise RuntimeError(f"FFmpeg failed in step '{step}': {stderr}")


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


# --- Pipeline kroky ---

def step_extract_audio(video_path: str, workdir: str) -> str:
    out = os.path.join(workdir, "audio_raw.wav")
    _ffmpeg(["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", out],
            timeout=120, step="extract_audio")
    return out


SEPARATOR_VENV = "/venv-separator/bin/python"
SEPARATOR_MODEL = "MDX23C-8KFFT-InstVoc_HQ.ckpt"
SEPARATOR_MODEL_DIR = str(MODEL_DIR / "mdx23c")


def step_separate_audio(audio_path: str, workdir: str) -> tuple[str, str]:
    """
    MDX23C separacia cez /venv-separator (audio-separator).
    Vracia (vocals_path, accompaniment_path).
    Fallback: ak venv neexistuje, vrati povodne audio pre oba vystupy.
    """
    import shutil
    if not os.path.exists(SEPARATOR_VENV):
        logger.warning("audio-separator venv nenajdeny — preskakujem separaciu, pouzivam povodne audio")
        vocals = os.path.join(workdir, "vocals.wav")
        shutil.copy(audio_path, vocals)
        return vocals, audio_path

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


def step_transcribe(vocals_path: str, source_lang: str | None = None) -> list[dict]:
    model = get_whisper()
    if source_lang and source_lang.lower() in ("auto", ""):
        source_lang = None
    logger.info(f"Transcribing... (language={source_lang or 'auto-detect'})")
    result = model.transcribe(vocals_path, language=source_lang, word_timestamps=True, verbose=False)
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result["segments"]]
    logger.info(f"Transcribed {len(segments)} segments")
    return segments


def step_diarize(vocals_path: str) -> dict[str, list[tuple[float, float]]]:
    """
    pyannote diarizacia — vrati slovnik {speaker_id: [(start, end), ...]}
    Ak HF_API_TOKEN nie je nastaveny, vrati jedineho default speakera.
    """
    hf_token = os.environ.get("HF_API_TOKEN") or os.environ.get("RUNPOD_SECRET_HF_API_TOKEN", "")
    if not hf_token:
        logger.warning("HF_API_TOKEN nie je nastaveny — preskakujem diarizaciu, pouzivam jedineho speakera")
        return {"SPEAKER_00": []}

    try:
        pipeline = get_diarize_pipeline()
        data, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
        audio = {"waveform": torch.from_numpy(data.T), "sample_rate": sr}
        result = pipeline(audio)
        sd = result.speaker_diarization

        speakers: dict[str, list[tuple[float, float]]] = {}
        for turn, _, speaker in sd.itertracks(yield_label=True):
            speakers.setdefault(speaker, []).append((turn.start, turn.end))

        logger.info(f"Diarization: {len(speakers)} speaker(s) — {list(speakers.keys())}")
        return speakers
    except Exception as e:
        logger.warning(f"Diarizacia zlyhala: {e} — pouzivam jedineho speakera")
        return {"SPEAKER_00": []}


def step_assign_speakers(segments: list[dict], speaker_turns: dict[str, list[tuple[float, float]]]) -> list[dict]:
    """
    Prirad kazdenmu segmentu speaker_id podla prekryvu s diarizacnymi segmentmi.
    Ak nie su k dispozicii diarizacne data, vsetky segmenty dostanu SPEAKER_00.
    """
    if not any(turns for turns in speaker_turns.values()):
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for speaker, turns in speaker_turns.items():
            for (s, e) in turns:
                overlap = min(seg["end"], e) - max(seg["start"], s)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
        seg["speaker"] = best_speaker

    speaker_counts = {}
    for seg in segments:
        speaker_counts[seg["speaker"]] = speaker_counts.get(seg["speaker"], 0) + 1
    logger.info(f"Speaker assignment: {speaker_counts}")
    return segments


def step_extract_speaker_refs(
    segments: list[dict],
    vocals_path: str,
    workdir: str,
    min_duration: float = 5.0,
    max_duration: float = 15.0,
) -> dict[str, str]:
    """
    Pre kazdeho speakera extrahuj ref audio z jeho najdlhsich segmentov.
    Vracia {speaker_id: path_to_ref_wav}.
    """
    from collections import defaultdict
    speaker_segs = defaultdict(list)
    for seg in segments:
        speaker_segs[seg["speaker"]].append(seg)

    refs = {}
    for speaker, segs in speaker_segs.items():
        # Zorad podla dlzky (najdlhsie prve)
        segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)

        # Zbieraj segmenty kym nemame aspon min_duration sekund
        collected = []
        total = 0.0
        for s in segs_sorted:
            dur = s["end"] - s["start"]
            if dur < 1.0:
                continue
            collected.append(s)
            total += dur
            if total >= max_duration:
                break

        if total < 1.0:
            logger.warning(f"Speaker {speaker}: prilis malo audia ({total:.1f}s) — pouzivam cely vocals")
            refs[speaker] = vocals_path
            continue

        # Extrahuj a spoj audio segmenty pomocou FFmpeg
        ref_path = os.path.join(workdir, f"ref_{speaker}.wav")
        if len(collected) == 1:
            s = collected[0]
            _ffmpeg([
                "ffmpeg", "-y", "-i", vocals_path,
                "-ss", str(s["start"]), "-to", str(s["end"]),
                "-ar", "22050", "-ac", "1", ref_path,
            ], timeout=30, step=f"ref_extract_{speaker}")
        else:
            # Viac segmentov — concat cez FFmpeg filter
            parts = []
            for idx, s in enumerate(collected):
                part = os.path.join(workdir, f"ref_{speaker}_part{idx}.wav")
                _ffmpeg([
                    "ffmpeg", "-y", "-i", vocals_path,
                    "-ss", str(s["start"]), "-to", str(s["end"]),
                    "-ar", "22050", "-ac", "1", part,
                ], timeout=30, step=f"ref_part_{speaker}_{idx}")
                parts.append(part)

            # Concat list file
            list_file = os.path.join(workdir, f"ref_{speaker}_list.txt")
            with open(list_file, "w") as f:
                for p in parts:
                    f.write(f"file '{p}'\n")
            _ffmpeg([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-ar", "22050", "-ac", "1", ref_path,
            ], timeout=60, step=f"ref_concat_{speaker}")

        logger.info(f"Speaker {speaker}: ref audio {total:.1f}s -> {ref_path}")
        refs[speaker] = ref_path

    return refs


def step_translate(segments: list[dict], target_lang: str = "cs") -> list[dict]:
    pipe = get_qwen()
    translated = []
    batch_size = 50
    texts = [seg["text"] for seg in segments]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        results = pipe(batch_texts, max_length=512)
        for j, result in enumerate(results):
            translated.append({**segments[i + j], "translated": result["translation_text"]})
        logger.info(f"Translated batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    logger.info(f"Translated {len(translated)} segments")
    return translated


def _num_to_words(m, lang="cs"):
    try:
        from num2words import num2words
        return num2words(int(m.group(0).replace('\xa0', '').replace(' ', '')), lang=lang)
    except Exception:
        return m.group(0)

def _normalize_text(t: str, lang="cs") -> str:
    # Spoj tisícové medzery: "1 700" -> "1700"
    t = re.sub(r'(\d)\s(\d{3})\b', r'\1\2', t)
    # Preveď čísla na slová
    t = re.sub(r'\b\d+\b', lambda m: _num_to_words(m, lang), t)
    t = re.sub(r'  +', ' ', t)
    return t.strip()


def step_tts_clone(
    segments: list[dict],
    speaker_refs: dict[str, str],
    workdir: str,
    target_lang: str = "cs",
) -> str:
    """
    XTTS v2 voice cloning — kazdy speaker ma vlastny ref audio.
    """
    XTTS_LANG_MAP = {
        "sk": "sk", "cs": "cs", "de": "de", "fr": "fr",
        "es": "es", "it": "it", "pl": "pl", "hu": "hu",
        "uk": "uk", "ru": "ru",
    }
    xtts_lang = XTTS_LANG_MAP.get(target_lang, "en")
    model = get_xtts()
    tts_sample_rate = 24000

    # Validuj ref audio pre kazdeho speakera
    for speaker, ref_path in speaker_refs.items():
        ref_data, ref_sr = sf.read(ref_path, dtype="float32")
        ref_duration = len(ref_data) / ref_sr
        logger.info(f"Speaker {speaker} ref: {ref_duration:.1f}s @ {ref_sr}Hz")
        if ref_duration < 3.0:
            logger.warning(f"Speaker {speaker}: ref audio prilis kratke ({ref_duration:.1f}s)")

    all_audio_chunks: list[np.ndarray] = []
    prev_end = 0.0

    for i, seg in enumerate(segments):
        text = _normalize_text(seg.get("translated", seg.get("text", "")), lang=target_lang)
        if not text:
            continue

        gap = seg["start"] - prev_end
        if gap > 0.05:
            all_audio_chunks.append(np.zeros(int(gap * tts_sample_rate), dtype=np.float32))

        speaker = seg.get("speaker", "SPEAKER_00")
        ref_path = speaker_refs.get(speaker, list(speaker_refs.values())[0])

        try:
            tmp_out = os.path.join(workdir, f"seg_{i:04d}.wav")
            model.tts_to_file(
                text=text,
                speaker_wav=ref_path,
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
            peak = np.abs(audio_data).max()
            if peak > 0:
                audio_data = audio_data / peak * 0.9
            all_audio_chunks.append(audio_data)
            logger.info(f"Seg {i} [{speaker}]: '{text[:40]}' -> {len(audio_data)/tts_sample_rate:.2f}s")
        except Exception as e:
            logger.warning(f"TTS failed seg {i} [{speaker}] '{text[:50]}': {type(e).__name__}: {e}", exc_info=True)
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


def step_generate_srt(segments: list[dict], workdir: str) -> tuple[str, str]:
    """Vygeneruje SRT titulky — originalne aj prelozene."""
    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    orig_srt = os.path.join(workdir, "subtitles_orig.srt")
    cs_srt   = os.path.join(workdir, "subtitles_cs.srt")

    with open(orig_srt, "w", encoding="utf-8") as fo, \
         open(cs_srt,  "w", encoding="utf-8") as fc:
        for i, seg in enumerate(segments, 1):
            start = fmt_time(seg["start"])
            end   = fmt_time(seg["end"])
            orig_text = seg.get("text", "").strip()
            cs_text   = seg.get("translated", "").strip()
            if orig_text:
                fo.write(f"{i}\n{start} --> {end}\n{orig_text}\n\n")
            if cs_text:
                fc.write(f"{i}\n{start} --> {end}\n{cs_text}\n\n")

    logger.info(f"SRT vygenerovane: {orig_srt}, {cs_srt}")
    return orig_srt, cs_srt


def step_mix_final(
    original_video: str,
    dubbed_voice: str,
    accompaniment: str,
    segments: list[dict],
    workdir: str,
    output_path: str,
) -> str:
    orig_srt, cs_srt = step_generate_srt(segments, workdir)

    # Temp video bez titulkov
    tmp_video = os.path.join(workdir, "dubbed_no_subs.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", dubbed_voice,
        "-i", accompaniment,
        "-filter_complex",
        "[1:a]volume=1.0[voice];[2:a]volume=0.5[music];[voice][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", tmp_video,
    ]
    _ffmpeg(cmd, timeout=600, step="final_mix")

    # Napali titulky — originalne hore (biele), preklad dole (zlte)
    cmd2 = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-vf",
        (
            f"subtitles={cs_srt}:force_style='Alignment=2,FontSize=18,PrimaryColour=&H00FFFF00,OutlineColour=&H00000000,Outline=2',"
            f"subtitles={orig_srt}:force_style='Alignment=8,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'"
        ),
        "-c:a", "copy",
        output_path,
    ]
    _ffmpeg(cmd2, timeout=600, step="burn_subtitles")
    logger.info(f"Final video s titulkami: {output_path}")
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
    Spusti cely pipeline s per-speaker voice cloning.
    Cache sa uklada do CACHE_DIR/<job_id>/ na perzistentnom Volume.
    """
    job_cache = CACHE_DIR / job_id
    job_cache.mkdir(parents=True, exist_ok=True)

    transcription_cache = job_cache / "transcription.json"
    translation_cache   = job_cache / "translation.json"
    dubbed_voice_cache  = job_cache / "dubbed_voice.wav"

    with tempfile.TemporaryDirectory(prefix=f"dubbing_{job_id}_") as workdir:
        # 1. Extrakcia audia
        raw_audio = step_extract_audio(video_path, workdir)

        # 2. Separacia audia (MDX23C) — vocals pre Whisper/pyannote/cloning, accompaniment pre mix
        vocals, accompaniment = step_separate_audio(raw_audio, workdir)

        # 3. Transkripcia (cache)
        if transcription_cache.exists():
            logger.info(f"Loading cached transcription: {transcription_cache}")
            segments = json.loads(transcription_cache.read_text())
        else:
            segments = step_transcribe(vocals, source_lang)
            transcription_cache.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
            logger.info(f"Transcription cached: {transcription_cache}")

        # 4. Preklad (cache)
        if translation_cache.exists():
            logger.info(f"Loading cached translation: {translation_cache}")
            segments = json.loads(translation_cache.read_text())
        else:
            segments = step_translate(segments, target_lang)
            translation_cache.write_text(json.dumps(segments, ensure_ascii=False, indent=2))
            logger.info(f"Translation cached: {translation_cache}")

        # 5. Diarizacia + priradenie speakerov
        diarization_cache = job_cache / "diarization.json"
        if diarization_cache.exists():
            logger.info(f"Loading cached diarization: {diarization_cache}")
            speaker_turns_raw = json.loads(diarization_cache.read_text())
            speaker_turns = {k: [tuple(t) for t in v] for k, v in speaker_turns_raw.items()}
        else:
            speaker_turns = step_diarize(raw_audio)
            diarization_cache.write_text(json.dumps(
                {k: list(v) for k, v in speaker_turns.items()}, indent=2
            ))
            logger.info(f"Diarization cached: {diarization_cache}")

        segments = step_assign_speakers(segments, speaker_turns)

        # Uloz translation s speaker info
        translation_cache.write_text(json.dumps(segments, ensure_ascii=False, indent=2))

        # 6. Ref audio pre kazdeho speakera
        # Ak je zadany externy ref_audio, pouzije sa pre vsetkych speakerov (fallback)
        if reference_audio_path:
            speakers = list(set(seg.get("speaker", "SPEAKER_00") for seg in segments))
            speaker_refs = {s: reference_audio_path for s in speakers}
            logger.info(f"Using provided ref_audio for all {len(speakers)} speaker(s)")
        else:
            speaker_refs = step_extract_speaker_refs(segments, vocals, workdir)

        # 7. TTS (cache)
        if dubbed_voice_cache.exists():
            logger.info(f"Loading cached dubbed voice: {dubbed_voice_cache}")
            dubbed_voice = str(dubbed_voice_cache)
        else:
            dubbed_voice_tmp = step_tts_clone(segments, speaker_refs, workdir, target_lang)
            shutil.copy(dubbed_voice_tmp, dubbed_voice_cache)
            dubbed_voice = str(dubbed_voice_cache)
            logger.info(f"Dubbed voice cached: {dubbed_voice_cache}")

        # 8. Finalny mix + titulky
        step_mix_final(video_path, dubbed_voice, accompaniment, segments, workdir, output_path)

        total_seconds = segments[-1]["end"] if segments else 0
        speakers_found = list(set(seg.get("speaker", "SPEAKER_00") for seg in segments))
        return {
            "output_path": output_path,
            "duration_seconds": round(total_seconds, 1),
            "segments_count": len(segments),
            "target_lang": target_lang,
            "speakers": speakers_found,
        }
