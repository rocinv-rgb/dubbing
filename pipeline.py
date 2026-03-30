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

# torchaudio 2.11+ monkey-patch -> soundfile (torchcodec workaround)

import torch
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models"))
CACHE_DIR  = Path(os.environ.get("CACHE_DIR",  "/workspace/cache"))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --- Konfigurácia zlučovania segmentov ---
# Prepisatelne z job inputu (pozri run_dubbing_pipeline parameter pause_marker)
PAUSE_MARKER: str | None = None   # None = ignoruj pauzy, "..." = vloz marker medzi vety
MAX_MERGE_PAUSE_S: float = 1.0    # max pauza medzi segmentmi na zlucenie (s)
MAX_MERGE_BLOCK_S: float = 7.0    # max dlzka zluceneho bloku (s)
SEGMENT_GAP_MS: float = 75.0      # buffer (ms) — TTS skonci aspon X ms pred zaciatkom dalsej vety

_whisper_model   = None
_qwen_pipe       = None
_qwen_model      = None
_qwen_tokenizer  = None
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


LANG_NAMES = {
    "cs": "Czech", "sk": "Slovak", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pl": "Polish", "hu": "Hungarian",
}

def _translate_segment_qwen(text: str, target_lang: str, duration: float, model, tokenizer) -> str:
    """Preloži segment Qwen3 s ohladom na dlzku (target word count)."""
    # Odhadneme max pocet slov pre target jazyk: ~2.5 slova/sekunda
    max_words = max(3, int(duration * 2.5))
    lang_name = LANG_NAMES.get(target_lang, target_lang.upper())

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional dubbing translator. "
                f"Translate the following English text to {lang_name}. "
                f"The translation MUST fit within {max_words} words maximum "
                f"because it needs to match a {duration:.1f} second audio slot. "
                f"Be concise. Preserve meaning. Output ONLY the translation, nothing else. /no_think"
            )
        },
        {"role": "user", "content": text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # Odstran <think> bloky ak existuju
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return response


def step_translate(segments: list[dict], target_lang: str = "cs") -> list[dict]:
    pipe = get_translator()
    translated = []
    batch_size = 50
    texts = [seg["text"] for seg in segments]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        results = pipe(batch_texts, max_length=512)
        for j, result in enumerate(results):
            seg = segments[i + j]
            trans = result["translation_text"]
            # Post-process: ak je preklad viac ako 40% dlhsi ako original (pocet znakov),
            # skusime ho skratit odstranenım poslednej vety
            orig_len = len(seg["text"])
            trans_len = len(trans)
            if trans_len > orig_len * 1.4:
                # Skrat na poslednu bodku/otaznik/vyksricnik
                sentences = re.split(r'(?<=[.!?])\s+', trans)
                if len(sentences) > 1:
                    trans = ' '.join(sentences[:-1])
                    logger.info(f"Seg {i+j}: skrateny preklad {trans_len} -> {len(trans)} znakov")
            translated.append({**seg, "translated": trans})
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
    reference_audio_path,  # str alebo dict {speaker_id: path}
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

    reference_audio_path moze byt:
    - str: jedna referencna stopa pre vsetkych speakerov
    - dict {speaker_id: path}: per-speaker referencie
    """
    XTTS_LANG_MAP = {
        "sk": "sk", "cs": "cs", "de": "de", "fr": "fr",
        "es": "es", "it": "it", "pl": "pl", "hu": "hu",
        "uk": "uk", "ru": "ru",
    }
    xtts_lang = XTTS_LANG_MAP.get(target_lang, "en")
    model = get_xtts()
    tts_sample_rate = 24000

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
            model.tts_to_file(
                text=text,
                speaker_wav=ref_wav,
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

    # Mix audio — ciste video bez titulkov (SRT su ulozene osobitne v cache)
    cmd = [
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", dubbed_voice,
        "-i", accompaniment,
        "-filter_complex",
        "[1:a]volume=1.0[voice];[2:a]volume=0.5[music];[voice][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ]
    _ffmpeg(cmd, timeout=600, step="final_mix")
    logger.info(f"Final video: {output_path}")
    return output_path


# --- Hlavna funkcia ---

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
    pause_marker: str | None = None,
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
    srt_orig_cache      = job_cache / "subtitles_orig.srt"
    srt_cs_cache        = job_cache / "subtitles_cs.srt"

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

        # 8. SRT titulky do cache aj vedla output videa (player ich najde automaticky)
        orig_srt, cs_srt = step_generate_srt(segments, workdir)
        import shutil as _shutil
        _shutil.copy(orig_srt, srt_orig_cache)
        _shutil.copy(cs_srt, srt_cs_cache)
        # Uloz vedla output videa — rovnaky nazov ako video
        output_stem = os.path.splitext(output_path)[0]
        srt_out_cs   = output_stem + ".srt"
        srt_out_orig = output_stem + ".en.srt"
        _shutil.copy(cs_srt, srt_out_cs)
        _shutil.copy(orig_srt, srt_out_orig)
        logger.info(f"SRT ulozene: {srt_out_cs}, {srt_out_orig}")

        # 9. Finalny mix — ciste video
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
