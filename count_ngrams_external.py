import json
import os
import tempfile
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import heapq

# ==========================
#   КОНФИГУРАЦИЯ
# ==========================

# Входной корпус
INPUT_JSONL = Path("corpus/jsonl/mix_es_60os_40c4.jsonl")

# Выходные файлы
OUTPUT_UNI        = Path("corpus/jsonl/freq_unigrams.jsonl")
OUTPUT_NGRAMS_2_4 = Path("corpus/jsonl/freq_ngrams_2_4.jsonl")
OUTPUT_NGRAMS_5   = Path("corpus/jsonl/freq_ngrams_5.jsonl")

# Размер батча (кол-во строк корпуса)
BATCH_SIZE   = 300_000        # можно менять: 200k–500k
# Пороги для 5-грамм
BATCH_MIN_5  = 5              # минимум повторов 5-граммы в одном батче
GLOBAL_MIN_5 = 30             # минимум повторов 5-граммы в итоговом словаре

# Ограничения на длину строки
MIN_CHARS = 5
MAX_CHARS = 5000

# ==========================
#   УТИЛИТЫ
# ==========================

def spill_counter(counter: Counter, tmp_list: list):
    """Записать Counter -> временный .jsonl файл, сохранить путь в tmp_list."""
    fd, fname = tempfile.mkstemp(prefix="ngr_", suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        for key, val in counter.items():
            out.write(json.dumps({"text": key, "count": val}, ensure_ascii=False) + "\n")
    tmp_list.append(fname)


def sort_jsonl(input_file: str) -> str:
    """
    Отсортировать JSONL-файл по полю 'text', вернуть путь к .sorted-файлу.
    (Файл читается в память; делаем это на уровне уже агрегированных частот, а не сырого корпуса.)
    """
    sorted_file = input_file + ".sorted"
    items = []

    with open(input_file, "r", encoding="utf-8") as inp:
        for line in inp:
            row = json.loads(line)
            items.append(row)

    items.sort(key=lambda x: x["text"])

    with open(sorted_file, "w", encoding="utf-8") as out:
        for obj in items:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return sorted_file


def multiway_merge_sorted(files, output_path: Path):
    """Слить много отсортированных JSONL-файлов в один, суммируя count (без порога)."""
    streams = [open(f, "r", encoding="utf-8") for f in files]

    def row_iter(stream):
        for line in stream:
            obj = json.loads(line)
            yield obj["text"], obj["count"]

    iterators = [row_iter(s) for s in streams]
    merged = heapq.merge(*iterators, key=lambda x: x[0])

    with output_path.open("w", encoding="utf-8") as out:
        last_key = None
        acc = 0

        for key, val in merged:
            if key != last_key and last_key is not None:
                out.write(json.dumps({"text": last_key, "count": acc}, ensure_ascii=False) + "\n")
                acc = 0
            last_key = key
            acc += val

        if last_key is not None:
            out.write(json.dumps({"text": last_key, "count": acc}, ensure_ascii=False) + "\n")

    for s in streams:
        s.close()


def multiway_merge_sorted_with_min(files, output_path: Path, min_count: int):
    """Слить много отсортированных JSONL-файлов в один, суммируя count и применяя порог по частоте."""
    streams = [open(f, "r", encoding="utf-8") for f in files]

    def row_iter(stream):
        for line in stream:
            obj = json.loads(line)
            yield obj["text"], obj["count"]

    iterators = [row_iter(s) for s in streams]
    merged = heapq.merge(*iterators, key=lambda x: x[0])

    with output_path.open("w", encoding="utf-8") as out:
        last_key = None
        acc = 0

        for key, val in merged:
            if key != last_key and last_key is not None:
                if acc >= min_count:
                    out.write(json.dumps({"text": last_key, "count": acc}, ensure_ascii=False) + "\n")
                acc = 0
            last_key = key
            acc += val

        if last_key is not None and acc >= min_count:
            out.write(json.dumps({"text": last_key, "count": acc}, ensure_ascii=False) + "\n")

    for s in streams:
        s.close()

# ==========================
#   ОСНОВНОЙ ПРОЦЕСС
# ==========================

def process():
    # списки временных файлов
    tmp_uni = []
    tmp_2_4 = []
    tmp_5 = []

    counter_uni = Counter()
    counter_2_4 = Counter()
    counter_5 = Counter()

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Reading corpus")):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text", "")

            if not text:
                continue
            if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
                continue

            tokens = text.split()
            L = len(tokens)
            if not L:
                continue

            # униграммы
            counter_uni.update(tokens)

            # 2–5-граммы
            for n in range(2, 6):
                if L < n:
                    break
                for j in range(L - n + 1):
                    ngram = " ".join(tokens[j:j+n])
                    if n < 5:
                        counter_2_4[ngram] += 1
                    else:
                        counter_5[ngram] += 1

            # сброс батча
            if (i + 1) % BATCH_SIZE == 0:
                print(f"--- Flushing batch at {i+1} lines")

                spill_counter(counter_uni, tmp_uni)
                spill_counter(counter_2_4, tmp_2_4)

                # 5-граммы: оставляем только те, что достаточно частые в батче
                if counter_5:
                    filtered_5 = Counter({ng: c for ng, c in counter_5.items()
                                          if c >= BATCH_MIN_5})
                    if filtered_5:
                        spill_counter(filtered_5, tmp_5)

                counter_uni.clear()
                counter_2_4.clear()
                counter_5.clear()

    # хвостовой батч
    if counter_uni:
        spill_counter(counter_uni, tmp_uni)
    if counter_2_4:
        spill_counter(counter_2_4, tmp_2_4)
    if counter_5:
        filtered_5 = Counter({ng: c for ng, c in counter_5.items()
                              if c >= BATCH_MIN_5})
        if filtered_5:
            spill_counter(filtered_5, tmp_5)

    print("Sorting temporary files...")

    sorted_uni  = [sort_jsonl(f) for f in tmp_uni]
    sorted_2_4  = [sort_jsonl(f) for f in tmp_2_4]
    sorted_5    = [sort_jsonl(f) for f in tmp_5]

    print("Merging unigrams...")
    multiway_merge_sorted(sorted_uni, OUTPUT_UNI)

    print("Merging 2–4-grams...")
    multiway_merge_sorted(sorted_2_4, OUTPUT_NGRAMS_2_4)

    print(f"Merging 5-grams with GLOBAL_MIN_5 = {GLOBAL_MIN_5} ...")
    multiway_merge_sorted_with_min(sorted_5, OUTPUT_NGRAMS_5, GLOBAL_MIN_5)

    print("Done.")
    print("Unigrams:", OUTPUT_UNI.resolve())
    print("Ngrams 2–4:", OUTPUT_NGRAMS_2_4.resolve())
    print("Ngrams 5:", OUTPUT_NGRAMS_5.resolve())


if __name__ == "__main__":
    process()
