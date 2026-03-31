# AI Dubbing Pipeline

Automated dubbing of foreign-language video into Czech/Slovak using GPU-accelerated AI.
Preserves per-speaker voice identity via zero-shot voice cloning.

## Pipeline

```
FFmpeg          → audio extraction (WAV 16kHz mono)
MDX23C          → voice/music separation (preserves background music and effects)
Whisper v3      → transcription + timestamps (auto language detect)
Helsinki-NLP    → translation (EN→CS/SK and others)
pyannote 3.1    → speaker diarization (per-speaker ref audio extraction)
XTTS v2         → TTS + zero-shot voice cloning per speaker
FFmpeg          → final mix (dubbed voice + background music + original video)
```

## Features

- Per-speaker voice cloning — each speaker gets their own cloned voice
- Canvas-based timing — no cumulative drift, segments placed at exact timestamps
- Time-stretching via FFmpeg atempo — TTS fits into available slot
- Persistent cache — transcription, translation, diarization cached on RunPod volume
- SRT subtitles generated alongside output video (original + translated)

## Infrastructure

| Phase | Hardware | Cost per job |
|-------|----------|-------------|
| Now | RunPod H100/A100 80GB | ~$0.10–0.20/min of video |
| Later | Mac Studio M3 Ultra 256GB | ~$0 (electricity only) |

RunPod Network Volume (~35GB): models pre-downloaded once, mounted on every pod start.
No cold-start model downloads — pipeline is ready in seconds.

## Input

```json
{
  "video_url": "/workspace/video.mp4",
  "target_lang": "cs",
  "source_lang": "en",
  "job_id": "my_test"
}
```

Supported target languages: `cs`, `sk`, `de`, `fr`, `es`, `it`, `pl`, `hu`

## Output

```json
{
  "output_video_url": "/workspace/output_my_test.mp4",
  "duration_seconds": 142.3,
  "segments_count": 47,
  "cost_estimate_usd": 0.65
}
```

## Project Structure

```
h_v1.py              — RunPod serverless handler (entry point)
pipeline/
  __init__.py        — orchestration (run_dubbing_pipeline)
  config.py          — constants, paths, tuneable parameters
  audio.py           — FFmpeg audio extraction + MDX23C voice separation
  transcribe.py      — Whisper v3 transcription
  translate.py       — Helsinki-NLP translation
  diarize.py         — pyannote speaker diarization + per-speaker ref audio extraction
  tts.py             — XTTS v2 voice cloning + canvas placement + time-stretching
  mix.py             — FFmpeg final mix + SRT subtitle generation
  models.py          — model loaders (cached singletons, loaded once per pod lifetime)
```
