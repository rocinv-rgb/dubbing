"""
pipeline/transcribe.py - Transkripcia pomocou Whisper
"""

import logging
from .models import get_whisper

logger = logging.getLogger(__name__)


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
