import argparse
import json

STRATEGIES = ["1 абзац", "3 абзаца", "только разделы", "N1..N2 токенов без разрыва абзаца", "overlap 1 абзац"]


def recall_at_k(retrieved: list[str], gold_doc: str, k: int = 5) -> float:
    return 1.0 if gold_doc in retrieved[:k] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="jsonl: {question, gold_doc, gold_answer}")
    ap.add_argument("--docs", default="./data")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.qa, encoding="utf-8")]
    print(f"questions={len(rows)} strategies={len(STRATEGIES)}")
    print("TODO: подключить retrieve() для каждой стратегии и посчитать recall@k.")
    for s in STRATEGIES:
        print(f" - {s}: not evaluated yet")


if __name__ == "__main__":
    main()
