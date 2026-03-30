"""
pipeline/translate.py - Preklad segmentov (Helsinki-NLP)
"""

import re
import logging
import torch
from .models import get_translator

logger = logging.getLogger(__name__)

LANG_NAMES = {
    "cs": "Czech", "sk": "Slovak", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "pl": "Polish", "hu": "Hungarian",
}


def _num_to_words(m, lang="cs"):
    raw = m.group(0).replace('\xa0', '').replace(' ', '')
    try:
        from num2words import num2words
        return num2words(int(raw), lang=lang)
    except Exception:
        digit_words = {"0":"nula","1":"jedna","2":"dva","3":"tri","4":"styri",
                       "5":"pet","6":"sest","7":"sedem","8":"osem","9":"devet"}
        return " ".join(digit_words.get(c, c) for c in raw)


def _normalize_text(t: str, lang="cs") -> str:
    # Spoj tisícové medzery: "1 700" -> "1700"
    t = re.sub(r'(\d)\s(\d{3})\b', r'\1\2', t)
    # Preveď čísla na slová
    t = re.sub(r'\b\d+\b', lambda m: _num_to_words(m, lang), t)
    t = re.sub(r'  +', ' ', t)
    return t.strip()


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
