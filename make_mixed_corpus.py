import json
import random
from pathlib import Path
from tqdm import tqdm

OS_JSONL = Path("corpus/jsonl/opensubs_es.jsonl")
C4_JSONL = Path("corpus/jsonl/c4_es_sample.jsonl")
OUT_JSONL = Path("corpus/jsonl/mix_es_60os_40c4.jsonl")

SHARE_OS = 0.60
SHARE_C4 = 0.40

def estimate_keep_prob():
    s_os = OS_JSONL.stat().st_size
    s_c4 = C4_JSONL.stat().st_size

    print(f"OpenSubs size: {s_os / (1024**3):.2f} GB")
    print(f"C4 size:       {s_c4 / (1024**3):.2f} GB")

    p_keep_os = (SHARE_OS / SHARE_C4) * (s_c4 / s_os)
    p_keep_os = min(1.0, p_keep_os)

    print(f"Keep probability for OS lines: {p_keep_os:.4f}")
    return p_keep_os


def main():
    random.seed(42)

    p_keep_os = estimate_keep_prob()

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    os_tokens = 0
    c4_tokens = 0

    with OUT_JSONL.open("w", encoding="utf-8") as out:

        # Sample OpenSubtitles
        with OS_JSONL.open("r", encoding="utf-8") as f_os:
            for line in tqdm(f_os, desc="sampling OpenSubtitles"):
                if random.random() > p_keep_os:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                if not text:
                    continue
                os_tokens += len(text.split())
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Add whole C4
        with C4_JSONL.open("r", encoding="utf-8") as f_c4:
            for line in tqdm(f_c4, desc="adding C4"):
                obj = json.loads(line)
                text = obj["text"]
                if not text:
                    continue
                c4_tokens += len(text.split())
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    total = os_tokens + c4_tokens
    print("-----")
    print("OS tokens:", os_tokens)
    print("C4 tokens:", c4_tokens)
    print("Final OS share:", os_tokens / total)
    print("Final C4 share:", c4_tokens / total)
    print("Output:", OUT_JSONL.resolve())


if __name__ == "__main__":
    main()
