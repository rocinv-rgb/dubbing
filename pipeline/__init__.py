"""
pipeline/__init__.py - AI Dubbing Pipeline
Kroky: FFmpeg -> Whisper -> Helsinki-NLP preklad -> pyannote diarizacia -> XTTS v2 (per-speaker voice cloning) -> FFmpeg mix

Verzia 1.3 (modularna):
- Rozdeleny do modulov: config, models, audio, transcribe, translate, diarize, tts, mix
- Zachovana 100% funkcionalita oproti povodnej pipeline.py
"""

import os
import json
import shutil
import logging
import tempfile
from pathlib import Path

# Presmeruj Coqui TTS cache na perzistentny Volume
os.environ.setdefault("COQUI_TTS_HOME", "/workspace/models")

# torchaudio 2.11+ monkey-patch -> soundfile (torchcodec workaround)
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

from .config import (
    MODEL_DIR, CACHE_DIR, DEVICE,
    PAUSE_MARKER, MAX_MERGE_PAUSE_S, MAX_MERGE_BLOCK_S, SEGMENT_GAP_MS,
)
from .audio import step_extract_audio, step_separate_audio, step_prepare_audio
from .transcribe import step_transcribe
from .translate import step_translate, LANG_NAMES
from .diarize import step_diarize, step_assign_speakers, step_extract_speaker_refs
from .tts import (
    step_tts_clone,
    enrich_segments_with_available_duration,
    merge_speaker_blocks,
)
from .mix import step_mix_final, step_generate_srt

logger = logging.getLogger(__name__)


def run_dubbing_pipeline(
    video_path: str,
    reference_audio_path: str | None,
    target_lang: str,
    source_lang: str | None,
    output_path: str,
    job_id: str = "local",
    pause_marker: str | None = None,
    use_openvoice: bool = False,
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
            # Enrichment: pre kazdy segment spocita available_duration = slot + pauza do dalsiho
            video_end = segments[-1]["end"] if segments else 0
            segments = enrich_segments_with_available_duration(segments, video_end=video_end)
            dubbed_voice_tmp = step_tts_clone(segments, speaker_refs, workdir, target_lang, use_openvoice=use_openvoice)
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
