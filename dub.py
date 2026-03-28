"""
dub.py - CLI wrapper pre AI dubbing pipeline

Pouzitie:
    python dub.py --url "https://youtube.com/watch?v=xxx" --lang sk
    python dub.py --url "https://youtube.com/watch?v=xxx" --lang cs
    python dub.py --file video.mp4 --lang sk
    python dub.py --url "..." --lang sk --ref mojhlas.wav
"""

import argparse
import os
import sys
import time
import logging
import subprocess
import tempfile
from pathlib import Path

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dub")

SUPPORTED_LANGS = {
    "sk": "Slovak",
    "cs": "Czech",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pl": "Polish",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "ru": "Russian",
}


def check_yt_dlp() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_video(url: str, output_dir: str) -> str:
    """
    Stiahne video cez yt-dlp.
    Format: najlepsia kvalita videa + audio, max 1080p (viac je zbytocne pre dubbing).
    """
    if not check_yt_dlp():
        logger.error("yt-dlp nie je nainstalovany. Spusti: pip install yt-dlp")
        sys.exit(1)

    out_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--no-playlist",          # len jedno video, nie cely playlist
        "--quiet",
        "--progress",
        url,
    ]

    logger.info(f"Sťahujem: {url}")
    result = subprocess.run(cmd, check=True, timeout=600)

    # Najdi stiahnuty subor
    mp4_files = list(Path(output_dir).glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError("yt-dlp nestiahol žiadny MP4 súbor.")

    path = str(mp4_files[0])
    size_mb = os.path.getsize(path) / 1024 / 1024
    logger.info(f"Stiahnuté: {Path(path).name} ({size_mb:.1f} MB)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="AI Dubbing — prelož video do iného jazyka s klonovaním hlasu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Príklady:
  python dub.py --url "https://youtube.com/watch?v=xxx" --lang sk
  python dub.py --url "https://youtu.be/xxx" --lang cs
  python dub.py --file video.mp4 --lang de
  python dub.py --url "..." --lang sk --ref mojhlas.wav --out output/
        """,
    )

    # Vstup — buď URL alebo lokálny súbor
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url",  help="YouTube (alebo iný) URL")
    src.add_argument("--file", help="Lokálny video súbor")

    parser.add_argument(
        "--lang",
        required=True,
        choices=sorted(SUPPORTED_LANGS.keys()),
        metavar=f"{{{','.join(sorted(SUPPORTED_LANGS.keys()))}}}",
        help="Cieľový jazyk (sk, cs, de, fr, es, it, pl, hu, uk, ru)",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help="Zdrojový jazyk — ak nie je zadaný, Whisper ho detekuje automaticky",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Referenčné audio pre klonovanie hlasu (WAV, 3-10s). "
             "Ak nie je zadané, použije sa hlas extrahovaný z videa.",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Výstupný adresár (default: aktuálny adresár)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Zachovaj dočasné súbory (debug)",
    )

    args = parser.parse_args()

    # Validacia
    if args.lang not in SUPPORTED_LANGS:
        parser.error(f"Nepodporovaný jazyk: {args.lang}")

    os.makedirs(args.out, exist_ok=True)

    # --- Vstupný súbor ---
    work_tmp = tempfile.mkdtemp(prefix="dub_dl_") if not args.keep_temp else args.out

    try:
        if args.url:
            video_path = download_video(args.url, work_tmp)
        else:
            video_path = args.file
            if not os.path.exists(video_path):
                logger.error(f"Súbor neexistuje: {video_path}")
                sys.exit(1)

        # --- Výstupný súbor ---
        stem = Path(video_path).stem
        lang_name = SUPPORTED_LANGS[args.lang]
        output_path = os.path.join(args.out, f"{stem}__{args.lang}.mp4")

        # --- Spustenie pipeline ---
        logger.info(f"Jazyk: {lang_name} ({args.lang})")
        logger.info(f"Výstup: {output_path}")
        if args.ref:
            logger.info(f"Referenčný hlas: {args.ref}")
        else:
            logger.info("Referenčný hlas: auto (extrahovaný z videa)")

        print()
        print("─" * 50)
        print(f"  Video:  {Path(video_path).name}")
        print(f"  Jazyk:  {lang_name}")
        print(f"  Výstup: {output_path}")
        print("─" * 50)
        print()

        t_start = time.time()

        from pipeline import run_dubbing_pipeline
        result = run_dubbing_pipeline(
            video_path=video_path,
            reference_audio_path=args.ref,
            target_lang=args.lang,
            source_lang=args.source_lang or None,  # None = Whisper auto-detect
            output_path=output_path,
        )

        elapsed = time.time() - t_start
        duration_min = result["duration_seconds"] / 60

        print()
        print("─" * 50)
        print(f"  ✓ Hotovo za {elapsed:.0f}s")
        print(f"  Dĺžka videa:  {duration_min:.1f} min")
        print(f"  Segmenty:     {result['segments_count']}")
        print(f"  Výstup:       {output_path}")
        print("─" * 50)
        print()

    except KeyboardInterrupt:
        print("\nPrerušené.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Chyba: {e}")
        sys.exit(1)
    finally:
        if not args.keep_temp and args.url:
            import shutil
            shutil.rmtree(work_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
