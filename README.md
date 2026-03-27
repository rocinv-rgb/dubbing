# AI Dubbing Pipeline — Projektový kontext

## Čo to je a prečo

Osobný nástroj na preklad zahraničných videí (španielčina, čínština, arabčina...)
do slovenčiny/češtiny. Motivácia: sledovať cudzojazyčný obsah bez jazykovej bariéry,
za zlomok ceny ručného prekladu.

Vedľajší efekt: keď príde Mac Studio M3 Ultra (čakacia doba 12-16 týždňov),
nevyťažená kapacita sa predá ako služba. Mac sa zaplatí za 2-3 dni pri $5/min videa.

---

## Infraštruktúra

| Fáza | Hardware | Cena | Náklady na job |
|------|----------|------|----------------|
| Teraz | RunPod A100 80GB | $1.39/hod | ~$0.08-0.15/min videa |
| Neskôr | Mac Studio M3 Ultra 256GB | $8,099 jednorazovo | ~$0 (elektrina) |

**RunPod Network Volume** (~35GB): modely sa pred-stiahnu raz, mountujú sa pri každom pode.
Bez toho by každý cold start sťahoval 80GB = neúnosné.

---

## Pipeline (6 krokov)

```
1. FFmpeg          extrakcia audia z videa → WAV 16kHz mono
2. Demucs          separácia hlasu od hudby/zvukov (htdemucs_ft, --two-stems)
3. Whisper v3      transkripcia → segmenty s timestampmi (auto-detect jazyka)
4. Qwen3-32B       preklad segmentov → JSON array (few-shot prompt)
5. CosyVoice2      TTS + zero-shot klonovanie hlasu tvorcu
6. FFmpeg          mix: nový hlas (0dB) + sprievod (0.7x) + pôvodné video
```

---

## Súbory

### `Dockerfile`
**Kritické rozhodnutia:**
- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` — A100 vyžaduje CUDA 12.1
- PyTorch sa inštaluje **pred** `requirements.txt` s explicitným `--index-url pytorch.whl/cu121`
  → pip bez toho stiahne CPU build a na A100 nič nebude fungovať
- `python3.10-dev` odstránené — nie je potrebné pre runtime
- `apt-get clean` — menší image
- `HEALTHCHECK` overuje CUDA dostupnosť každých 30s

### `requirements.txt`
- `torch/torchvision/torchaudio` **NIE SÚ tu** — sú v Dockerfile s CUDA verziou
  → keby boli tu, pip by prepísal CUDA build CPU buildom
- `boto3` — S3/R2 upload výsledkov
- `pydub` + `ffmpeg-python` — spoľahlivý MP3/audio handling

### `pipeline.py`
**Opravy ktoré sa riešili:**

1. **Demucs `--mp3` odstránené** → torchaudio MP3 backend nie je garantovaný
   na všetkých systémoch. Demucs defaultne produkuje WAV. Po separácii
   sa pridal ffmpeg resample na 16kHz mono (Demucs vracia 44.1kHz stereo).

2. **Qwen3 parser** → pôvodný riadkový parser (`[0] text`) bol krehký,
   Qwen3-32B pridáva verbose prefix "Here is the translation...".
   Riešenie: JSON mode s few-shot promptom + `_parse_translation_json()`
   ktorý zvláda markdown fences, multiline JSON, prose pred arraym.
   Regex: `[\s\S]*` namiesto `.*?` — spoľahlivejší pre multiline.

3. **CosyVoice2 dtype** → `ref_audio.float()` a `tts_speech.float()` pred
   `.numpy()` — CosyVoice2 môže failnúť ak tensor má float16 dtype z torchaudio.

4. **FFmpeg timeouty** → `_ffmpeg()` helper namiesto priameho `subprocess.run`.
   Warm pod môže bežať hodiny, zaseknutý FFmpeg bez timeoutu zablokuje worker navždy.
   - `extract_audio`: 120s
   - `resample_vocals`: 60s
   - `final_mix`: 600s (dlhé video + AAC enkódovanie)
   Pri timeoutu/chybe vypíše posledných 500 znakov stderr pre debug.

5. **Whisper auto-detect** → `source_lang=None` → Whisper sám rozozná
   španielčinu, čínštinu, arabčinu. Netreba špecifikovať pri každom volaní.

**VRAM na A100 80GB:**
```
Whisper large-v3:  ~3 GB
Qwen3-32B bf16:   ~65 GB
CosyVoice2-0.5B:   ~4 GB
Celkom:           ~72 GB  ✓ (8GB buffer)
```
Ak by Qwen3-32B nestačil: Qwen3-14B (~29GB) alebo externé API.

### `handler.py`
RunPod Serverless entry point. Kľúčové vlastnosti:
- `job_id` v každom log riadku — v RunPod logoch okamžite vidíš ktorý job čo robí
- Validácia `target_lang` pred spustením pipeline — zmysluplná chyba klientovi
- `upload_file`: S3/R2 ak je `OUTPUT_BUCKET_URL` nastavený, inak base64 (dev mode)
  pri >50MB súbore bez bucketu hodí `RuntimeError` s návodom čo nastaviť
- `download_file` timeout 300s — 10-min video môže byť 500MB+

### `dub.py`
CLI wrapper pre osobné použitie (nie RunPod Serverless).
```bash
# Základné použitie
python dub.py --url "https://youtube.com/watch?v=xxx" --lang sk
python dub.py --url "https://youtu.be/xxx" --lang cs
python dub.py --file video.mp4 --lang de

# S vlastným referenčným hlasom
python dub.py --url "..." --lang sk --ref mojhlas.wav --out output/

# Debug — zachová temp súbory
python dub.py --url "..." --lang sk --keep-temp
```
- `--url` stiahne video cez `yt-dlp` (max 1080p, MP4)
- `--source-lang` nie je povinný — Whisper detekuje automaticky
- Výstup: `<názov_videa>__sk.mp4` v aktuálnom adresári

---

## Nasadenie (TODO — ďalší krok)

Potrebné účty:
- [docker.com](https://docker.com) — Docker Desktop
- [hub.docker.com](https://hub.docker.com) — registry pre image
- [runpod.io](https://runpod.io) — $20 kredit vystačí na testovanie

Postup:
```bash
# 1. Build a push
docker build --platform linux/amd64 -t <dockerhub_user>/dubbing:latest .
docker push <dockerhub_user>/dubbing:latest

# 2. RunPod Network Volume — pred-stiahnuť modely
huggingface-cli download openai/whisper-large-v3 \
    --local-dir /workspace/models/whisper
huggingface-cli download FunAudioLLM/CosyVoice2-0.5B \
    --local-dir /workspace/models/CosyVoice2-0.5B
huggingface-cli download Qwen/Qwen3-32B \
    --local-dir /workspace/models/qwen3-32b

# 3. RunPod template
#    Image: <dockerhub_user>/dubbing:latest
#    GPU: A100 80GB
#    Volume: /workspace → Network Volume
#    Env: MODEL_DIR=/workspace/models
```

---

## Vízia (osobný agent)

Aktuálny stav: pipeline hotový, CLI hotový, nasadenie TODO.

Budúcnosť (po Mac Studiu):
```
Teraz:      python dub.py --url "..." --lang sk
Neskôr:     agent dostane URL a preloží sám bez parametrov
Finál:      "daj mi toto video po slovensky" → hotovo
```

Agent orchestruje pipeline cez function calling — `dub.py` sa stane
jedným z nástrojov vedľa smart home, kalendára atď.
Lokálny Qwen3-32B na Mac Studiu = $0/dotaz vs $0.15 externé API.
