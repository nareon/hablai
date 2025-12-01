import json
from pathlib import Path

INPUT_TXT = Path("corpus/opensubs2024_es/es.txt")
OUTPUT_JSONL = Path("corpus/jsonl/opensubs_es.jsonl")


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_TXT.open("r", encoding="utf-8", errors="ignore") as inp, \
         OUTPUT_JSONL.open("w", encoding="utf-8") as out:

        for i, line in enumerate(inp):
            line = line.strip()
            if not line:
                continue

            obj = {
                "id": f"os_{i}",
                "source": "opensubs",
                "text": line,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Wrote:", OUTPUT_JSONL.resolve())


if __name__ == "__main__":
    main()
