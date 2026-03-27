"""
handler.py - RunPod Serverless Entry Point

Vstup (job["input"]):
{
    "video_url":           "https://...",   # povinne
    "target_lang":         "sk",            # povinne  (sk/cs/de/fr/es/it/pl/hu)
    "source_lang":         "en",            # optional, default "en"
    "reference_audio_url": "https://..."    # optional — ak chyba, pouzije sa
}                                           # vlastny hlas extrahovany z videa

Vystup:
{
    "output_video_url":  "https://...",   # S3/R2 URL, alebo data:video/mp4;base64,...
    "duration_seconds":  142.3,
    "segments_count":    47,
    "cost_estimate_usd": 0.65             # pri cene $5/min
}
"""

import os
import uuid
import base64
import logging
import tempfile
import subprocess
import requests

import runpod
from pipeline import run_dubbing_pipeline

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("handler")

# --- Konfig z env vars ---
OUTPUT_BUCKET_URL = os.environ.get("OUTPUT_BUCKET_URL", "")
S3_ENDPOINT       = os.environ.get("S3_ENDPOINT", "")
S3_ACCESS_KEY     = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY     = os.environ.get("S3_SECRET_KEY", "")
S3_BUCKET         = os.environ.get("S3_BUCKET", "dubbing-outputs")

SUPPORTED_LANGS = {"sk", "cs", "de", "fr", "es", "it", "pl", "hu"}
BASE64_SIZE_LIMIT = 50 * 1024 * 1024  # 50 MB


# --- Pomocne funkcie ---

YT_DOMAINS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com")


def _is_yt_url(url: str) -> bool:
    return any(d in url for d in YT_DOMAINS)


def download_video(url: str, dest_path: str, job_id: str) -> str:
    """
    Stiahne video z URL do dest_path.
    - YouTube / yt-dlp kompatibilné URL -> yt-dlp (max 1080p MP4)
    - Priame HTTP URL -> requests stream
    """
    if _is_yt_url(url):
        logger.info(f"[{job_id}] yt-dlp download: {url}")
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", dest_path,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr[-500:]}")
    else:
        logger.info(f"[{job_id}] Direct download: {url}")
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
        logger.info(f"[{job_id}] Downloaded {downloaded / 1024 / 1024:.1f} MB")

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise RuntimeError(f"Downloaded file missing or empty: {dest_path}")

    logger.info(f"[{job_id}] Video ready: {os.path.getsize(dest_path) // 1024 // 1024} MB")
    return dest_path


def download_file(url: str, dest_path: str, job_id: str) -> str:
    """Stiahne obecny subor (audio, ref) cez HTTP. Timeout 5 min."""
    logger.info(f"[{job_id}] Downloading file -> {dest_path}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
    logger.info(f"[{job_id}] Downloaded {downloaded / 1024 / 1024:.1f} MB")
    return dest_path


def upload_file(local_path: str, job_id: str) -> str:
    """
    Nahraje subor na S3/R2 a vrati URL.
    Fallback: base64 data URI pre male subory (testovanie bez bucketu).
    """
    if OUTPUT_BUCKET_URL:
        import boto3
        logger.info(f"[{job_id}] Uploading to S3/R2...")
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT or None,
            aws_access_key_id=S3_ACCESS_KEY or None,
            aws_secret_access_key=S3_SECRET_KEY or None,
        )
        key = f"outputs/{job_id}.mp4"
        s3.upload_file(local_path, S3_BUCKET, key)
        url = f"{OUTPUT_BUCKET_URL}/{key}"
        logger.info(f"[{job_id}] Uploaded: {url}")
        return url

    # Fallback: base64 — len pre dev/testing, nie pre produkcne videa
    size = os.path.getsize(local_path)
    if size > BASE64_SIZE_LIMIT:
        logger.warning(
            f"[{job_id}] OUTPUT_BUCKET_URL not set and file is {size / 1024 / 1024:.0f} MB "
            f"(> {BASE64_SIZE_LIMIT // 1024 // 1024} MB limit). "
            f"Nastav OUTPUT_BUCKET_URL + S3_* env vars pre produkcne nasadenie."
        )
        raise RuntimeError(
            f"File too large ({size // 1024 // 1024} MB) for base64 fallback. "
            "Set OUTPUT_BUCKET_URL environment variable."
        )
    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    logger.info(f"[{job_id}] Returning base64 ({size // 1024} KB) — dev mode")
    return f"data:video/mp4;base64,{encoded}"


# --- RunPod handler ---

def handler(job: dict) -> dict:
    """
    Hlavny RunPod serverless handler.
    Vrati dict s vysledkom alebo {"error": "..."} pri zlyhaní.
    """
    job_id = job.get("id", str(uuid.uuid4())[:8])
    job_input = job.get("input", {})

    logger.info(f"[{job_id}] Job started")

    # --- Validacia vstupu ---
    video_url = job_input.get("video_url", "").strip()
    if not video_url:
        logger.error(f"[{job_id}] Missing video_url")
        return {"error": "Missing required field: video_url"}

    target_lang = job_input.get("target_lang", "sk").strip().lower()
    if target_lang not in SUPPORTED_LANGS:
        logger.error(f"[{job_id}] Unsupported target_lang: {target_lang}")
        return {
            "error": f"Unsupported target_lang '{target_lang}'. "
                     f"Supported: {sorted(SUPPORTED_LANGS)}"
        }

    source_lang    = job_input.get("source_lang", "en").strip().lower()
    ref_audio_url  = job_input.get("reference_audio_url", "").strip() or None

    logger.info(
        f"[{job_id}] {source_lang} -> {target_lang} | "
        f"ref_audio={'yes' if ref_audio_url else 'no (auto)'} | "
        f"video={video_url}"
    )

    # --- Spracovanie ---
    with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as workdir:
        try:
            # Stiahnutie videa (yt-dlp pre YouTube, requests pre priame URL)
            video_path = os.path.join(workdir, "input.mp4")
            download_video(video_url, video_path, job_id)

            # Stiahnutie referencneho audia (ak zadane)
            ref_audio_path = None
            if ref_audio_url:
                ref_audio_path = os.path.join(workdir, "reference.wav")
                download_file(ref_audio_url, ref_audio_path, job_id)

            output_path = os.path.join(workdir, "output_dubbed.mp4")

            # Pipeline
            logger.info(f"[{job_id}] Starting pipeline...")
            result = run_dubbing_pipeline(
                video_path=video_path,
                reference_audio_path=ref_audio_path,
                target_lang=target_lang,
                source_lang=source_lang,
                output_path=output_path,
            )
            logger.info(
                f"[{job_id}] Pipeline done | "
                f"{result['duration_seconds']}s | "
                f"{result['segments_count']} segments"
            )

            # Upload
            output_url = upload_file(output_path, job_id)

            # Cena: $5/min
            duration_min = result["duration_seconds"] / 60
            cost = round(duration_min * 5.0, 2)

            logger.info(f"[{job_id}] Job complete | cost_estimate=${cost}")
            return {
                "output_video_url":  output_url,
                "duration_seconds":  result["duration_seconds"],
                "segments_count":    result["segments_count"],
                "target_lang":       result["target_lang"],
                "cost_estimate_usd": cost,
            }

        except Exception as e:
            logger.exception(f"[{job_id}] Job failed: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Na RunPod Pode nie je test_input.json -> serverless.start() crashuje.
    # Ak najde test_input.json, spusti job priamo (Pod / lokalne testovanie).
    # Ak nie, spusti serverless worker (Serverless endpoint).
    test_input_path = os.path.join(os.path.dirname(__file__), "test_input.json")

    if os.path.exists(test_input_path):
        import json as _json
        logger.info("[local] Found test_input.json, running single job...")
        with open(test_input_path) as f:
            test_job = _json.load(f)
        job = {"id": "local-test", "input": test_job.get("input", test_job)}
        result = handler(job)
        print("\n=== RESULT ===")
        print(_json.dumps(result, indent=2, ensure_ascii=False))
    else:
        logger.info("[serverless] No test_input.json, starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})
