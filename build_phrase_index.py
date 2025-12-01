import json
from pathlib import Path
from math import log
from tqdm import tqdm

UNIGRAMS = Path("corpus/jsonl/freq_unigrams.jsonl")
NGRAMS_2_4 = Path("corpus/jsonl/freq_ngrams_2_4.jsonl")
NGRAMS_5 = Path("corpus/jsonl/freq_ngrams_5.jsonl")

PHRASE_INDEX = Path("corpus/jsonl/phrase_index.jsonl")

F_MIN = 5        # минимальная частота фразы, чтобы вообще учитывать
MAX_PHRASES = None  # можно ограничить top-N, если захочешь


def load_unigrams():
    freq = {}
    with UNIGRAMS.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="loading unigrams"):
            obj = json.loads(line)
            w = obj["text"]
            c = obj["count"]
            freq[w] = c
    return freq


def process_ngrams(ngram_path, freq_word, out):
    with ngram_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"processing {ngram_path.name}"):
            obj = json.loads(line)
            phrase = obj["text"]
            freq_phrase = obj["count"]
            if freq_phrase < F_MIN:
                continue

            tokens = phrase.split()
            n = len(tokens)
            if n < 2 or n > 5:
                continue

            wf = []
            w_imp = 0.0
            for w in tokens:
                fw = freq_word.get(w, 1)  # редким даём 1
                wf.append(fw)
                w_imp += log(fw + 1.0)

            rec = {
                "phrase": phrase,
                "freq_phrase": freq_phrase,
                "n": n,
                "tokens": tokens,
                "word_freqs": wf,
                "word_importance": w_imp,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    freq_word = load_unigrams()
    PHRASE_INDEX.parent.mkdir(parents=True, exist_ok=True)

    with PHRASE_INDEX.open("w", encoding="utf-8") as out:
        process_ngrams(NGRAMS_2_4, freq_word, out)
        process_ngrams(NGRAMS_5, freq_word, out)

    print("Index written to:", PHRASE_INDEX.resolve())


if __name__ == "__main__":
    main()
