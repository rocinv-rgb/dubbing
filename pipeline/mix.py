"""
pipeline/mix.py - Finálny mix videa a generovanie SRT titulkov
"""

import os
import logging
from .audio import _ffmpeg

logger = logging.getLogger(__name__)


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
