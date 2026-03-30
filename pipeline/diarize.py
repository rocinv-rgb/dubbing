"""
pipeline/diarize.py - Diarizácia, priradenie a extrakcia ref. audia pre speakerov
"""

import os
import logging
import torch
import soundfile as sf
from collections import defaultdict
from .models import get_diarize_pipeline
from .audio import _ffmpeg

logger = logging.getLogger(__name__)


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
