import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ========= НАСТРОЙКИ =========
MAX_DOCS = 6_000_000              # сколько документов взять
OUTPUT_JSONL = Path("corpus/jsonl/c4_es_sample.jsonl")
SOURCE_NAME = "c4_es"
SPLIT = "train"                   # train/validation
# =============================

# Простейшая очистка
RE_SPACES = re.compile(r"\s+")
RE_URL = re.compile(r"https?://\S+")
RE_TAG = re.compile(r"<[^>]+>")

def clean_text(text: str) -> str:
    text = RE_URL.sub(" ", text)
    text = RE_TAG.sub(" ", text)
    text = RE_SPACES.sub(" ", text)
    return text.strip()


def main():
    print(f"Loading allenai/c4 config='es', split='{SPLIT}' (streaming=True)...")

    ds = load_dataset(
        "allenai/c4",   # это и есть C4/mC4
        "es",           # конкретно испанский
        split=SPLIT,
        streaming=True,
    )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for row in tqdm(ds, desc="reading c4-es"):
            text = row.get("text", "")
            if not text:
                continue

            text = clean_text(text)
            if not text:
                continue

            # фильтр по длине, при желании подстрой
            if len(text) < 20 or len(text) > 2000:
                continue

            obj = {
                "source": SOURCE_NAME,
                "text": text,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

            if count >= MAX_DOCS:
                break

    print("-----")
    print("Сохранено документов:", count)
    print("Файл:", OUTPUT_JSONL.resolve())


if __name__ == "__main__":
    main()
