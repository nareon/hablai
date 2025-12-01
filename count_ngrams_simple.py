import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Входной объединённый корпус
INPUT_JSONL = Path("corpus/jsonl/mix_es_60os_40c4.jsonl")

# Выходные файлы с частотами
UNIGRAMS_OUT = Path("corpus/jsonl/freq_unigrams.json")
NGRAMS_OUT   = Path("corpus/jsonl/freq_ngrams_2_5.json")

# Ограничение по длине строки (для фильтрации мусора)
MIN_CHARS = 5
MAX_CHARS = 5000


def main():
    unigram_counter = Counter()
    ngram_counter = Counter()

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"reading {INPUT_JSONL.name}"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
                continue

            # очень простая токенизация по пробелам
            tokens = text.split()
            if not tokens:
                continue

            # униграммы
            unigram_counter.update(tokens)

            # n-граммы 2–5 слов
            L = len(tokens)
            for n in range(2, 6):
                if L < n:
                    break
                for i in range(L - n + 1):
                    ngram = " ".join(tokens[i:i+n])
                    ngram_counter[ngram] += 1

    # Сохраняем результаты
    print("Сохранение частот...")

    UNIGRAMS_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Преобразуем Counter в обычный словарь для json
    with UNIGRAMS_OUT.open("w", encoding="utf-8") as f_out:
        json.dump(unigram_counter, f_out, ensure_ascii=False)

    with NGRAMS_OUT.open("w", encoding="utf-8") as f_out:
        json.dump(ngram_counter, f_out, ensure_ascii=False)

    print("Готово.")
    print("Unigrams:", UNIGRAMS_OUT.resolve())
    print("N-grams 2–5:", NGRAMS_OUT.resolve())
    print("Всего уникальных слов:", len(unigram_counter))
    print("Всего уникальных n-грамм 2–5:", len(ngram_counter))


if __name__ == "__main__":
    main()
