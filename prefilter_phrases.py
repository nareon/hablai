#!/usr/bin/env python
import json
import re
from pathlib import Path

from tqdm import tqdm

# ==========================
#   ПУТИ
# ==========================

# исходный индекс (на NVMe)
PHRASE_INDEX_IN = Path("corpus/jsonl/phrase_index.jsonl")

# результат префильтра — на sda2
PHRASE_INDEX_OUT = Path(
    "/media/ol/SSD2T_Photo/hablai/corpus/jsonl/phrase_index_prefiltered.jsonl"
)

# ==========================
#   ПРЕФИЛЬТР
# ==========================

_ES_LETTERS = set("abcdefghijklmnñopqrstuvwxyzáéíóúü")

# набор символов, считающихся «чистой пунктуацией»
_PUNCT_CHARS = set("!¡?¿.,;:()[]{}«»\"'-")


def _es_like(word: str) -> bool:
    """
    Грубая проверка: слово похоже на испанское?
    Разрешаем буквы испанского алфавита + апостроф.
    """
    w = word.lower().strip("¡!¿?.,;:()[]{}\"'«»")
    if not w:
        return False
    letters = [c for c in w if c.isalpha()]
    if not letters:
        return False
    return all(c in _ES_LETTERS for c in letters)


def clean_phrase(phrase: str) -> str:
    """
    Очистка строки:
    - убираем управляющие символы
    - убираем emoji / non-BMP
    - схлопываем повторяющуюся пунктуацию
      ("! !" -> "!", "!!!" -> "!")
    - отрезаем ведущую/замыкающую пунктуацию
    - нормализуем пробелы
    """
    # 1) убрать управляющие символы (NUL и пр.) U+0000–U+001F и U+007F
    phrase = re.sub(r"[\u0000-\u001F\u007F]", " ", phrase)

    # 2) убрать emoji и прочие не-BMP символы
    phrase = re.sub(r"[\U00010000-\U0010FFFF]", "", phrase)

    # 3) схлопнуть конструкции вида "! !" -> "!"
    phrase = re.sub(r"([!¡?¿])\s+\1", r"\1", phrase)

    # 4) "!!!" -> "!", "¿¿" -> "¿" и т.п.
    phrase = re.sub(r"[!¡?¿]{2,}", lambda m: m.group(0)[0], phrase)

    # 5) убрать ведущие последовательности пунктуации
    phrase = re.sub(r'^[!¡?¿\.,;:(){}\[\]«»"\'\-]+', "", phrase)

    # 6) убрать замыкающие последовательности пунктуации
    phrase = re.sub(r'[!¡?¿\.,;:(){}\[\]«»"\'\-]+$', "", phrase)

    # 7) нормализовать пробелы
    phrase = re.sub(r"\s+", " ", phrase).strip()

    return phrase


def simple_prefilter(rec: dict) -> bool:
    """
    Детерминированная фильтрация до LLM.
    True = оставить, False = выкинуть.
    В начале очищаем управляющие символы, emoji и лишнюю пунктуацию.
    """
    raw = rec["phrase"]
    phrase = clean_phrase(raw)

    if not phrase:
        return False

    # записываем очищенную фразу обратно
    rec["phrase"] = phrase

    tokens = phrase.split()
    n_eff = len(tokens)

    # базовая длина по словам — уже по очищенному тексту
    if n_eff < 2 or n_eff > 5:
        return False

    lower_tokens = [t.lower() for t in tokens]

    # если слишком много "чисто пунктуационных" токенов — отбрасываем
    punct_tokens = sum(
        1 for t in tokens if all(ch in _PUNCT_CHARS for ch in t)
    )
    if punct_tokens / n_eff > 0.4:
        return False

    # 1. URL, почта, домены
    if "http://" in phrase or "https://" in phrase or "www." in phrase:
        return False
    if "@" in phrase or ".com" in phrase or ".net" in phrase or ".org" in phrase:
        return False

    # 2. Много цифр или шаблон даты
    digits = sum(ch.isdigit() for ch in phrase)
    if digits >= 3:
        return False
    if re.search(r"\d{1,2}\s+\w+\s+\d{4}", phrase):
        return False  # типичный "19 Mayo 2017"

    # 3. Много шумовой пунктуации (после чистки это почти не нужно, но оставим)
    if phrase.count("!") + phrase.count("?") >= 4:
        return False

    # 4. Заголовки КАПСОМ
    if phrase.isupper():
        return False

    # 5. Слишком коротко по символам
    if len(phrase) < 6:
        return False

    # 6. Полностью из "шумовых" служебных слов
    noise_words = {"por", "se", "de", "y", "yo", "te", "la", "el", "al", "en", "lo", "que"}
    if all(t.lower().strip("¡!¿?.,") in noise_words for t in tokens):
        return False

    # 7. Начало с ¡/¿ и без типичного глагола → скорее мусор
    common_verbs = {
        "es", "está", "estoy", "eres", "soy", "somos", "son",
        "quiero", "quieres", "quiere", "quieren",
        "tengo", "tienes", "tiene", "tenemos",
        "puedo", "puedes", "puede", "podemos",
        "vamos", "voy", "ven", "venga",
        "dime", "di", "haz", "ve",
        "mira", "pienso", "creo", "sabes", "sabe",
        "habla", "hablo", "hable", "hablemos",
        "déjame", "déjate",
    }

    def has_common_verb() -> bool:
        for t in lower_tokens:
            if t.strip("¡!¿?.,;:()[]{}\"'«»") in common_verbs:
                return True
        return False

    if phrase.startswith(("¡", "¿")) and not has_common_verb():
        return False

    # 8. Проверка "похожести на испанский"
    es_like_count = 0
    es_total = 0
    for t in tokens:
        if any(c.isalpha() for c in t):
            es_total += 1
            if _es_like(t):
                es_like_count += 1
    if es_total > 0 and es_like_count / es_total < 0.5:
        return False

    return True


def main():
    PHRASE_INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with PHRASE_INDEX_IN.open("r", encoding="utf-8") as inp, \
            PHRASE_INDEX_OUT.open("w", encoding="utf-8") as out:

        for line in tqdm(inp, desc="prefiltering phrase_index"):
            total += 1
            rec = json.loads(line)
            if simple_prefilter(rec):
                kept += 1
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Prefilter done.")
    print(f"Total records: {total}")
    print(f"Kept after prefilter: {kept}")


if __name__ == "__main__":
    main()
