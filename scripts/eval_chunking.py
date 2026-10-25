import argparse
import json
from pathlib import Path
from typing import Any
from core.config import settings
from helpers.chunking_eval import evaluate_strategy
from helpers.text_clean import clean_text, drop_service_parts, split_by_strategy
from service.pdf_parser import parse_pdf

STRATEGIES = [
    "paragraphs_1",
    "paragraphs_3",
    "sections",
    "token_window_no_split",
    "paragraph_overlap",
]


def load_documents(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(
        file for file in path.rglob("*") if file.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    documents: list[dict[str, Any]] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".pdf":
            pages = parse_pdf(file.read_bytes())
        else:
            text = file.read_text(encoding="utf-8")
            pages = [(i + 1, page) for i, page in enumerate(text.split("\f"))]
        cleaned = [(page_no, clean_text(text)) for page_no, text in pages if text.strip()]
        documents.append({"id": file.name, "pages": drop_service_parts(cleaned)})
    if not documents:
        raise ValueError(f"No PDF/TXT/MD documents found in {path}")
    return documents


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise ValueError(f"No QA rows found in {path}")
    for row_no, row in enumerate(rows, start=1):
        if not row.get("question"):
            raise ValueError(f"QA row {row_no} is missing 'question'")
        if "gold_pages" not in row and "gold_page" not in row:
            raise ValueError(f"QA row {row_no} needs 'gold_pages' or 'gold_page'")
    return rows


def build_chunks(documents: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for page_no, text in document["pages"]:
            page_chunks = split_by_strategy(
                text,
                strategy=strategy,
                chunk_size=settings.CHUNK_SIZE,
            )
            chunks.extend(
                {"doc_id": document["id"], "page": page_no, "text": chunk}
                for chunk in page_chunks
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG chunking strategies")
    parser.add_argument("--qa", required=True, type=Path, help="JSONL with question, gold_doc, gold_pages")
    parser.add_argument("--docs", required=True, type=Path, help="Directory containing source PDFs/TXT/MD")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    documents = load_documents(args.docs)
    questions = load_questions(args.qa)
    if not 10 <= len(questions) <= 15:
        parser.error(f"expected 10–15 labeled questions, got {len(questions)}")

    from service.embeddings import get_embedding_model

    model = get_embedding_model()
    question_embeddings = model.encode(
        [row["question"] for row in questions], normalize_embeddings=True
    )
    reports: dict[str, Any] = {}

    for strategy in STRATEGIES:
        chunks = build_chunks(documents, strategy)
        if not chunks:
            raise ValueError(f"Chunking strategy {strategy!r} produced no chunks")
        chunk_embeddings = model.encode(
            [chunk["text"] for chunk in chunks], normalize_embeddings=True, show_progress_bar=True
        )
        result = evaluate_strategy(
            chunks,
            chunk_embeddings,
            questions,
            question_embeddings,
            args.top_k,
        )
        reports[strategy] = {"chunks": len(chunks), **result}
        print(
            f"{strategy:24} chunks={len(chunks):5} "
            f"Recall@{args.top_k}={result['recall_at_k']:.3f}"
        )

    if args.output:
        args.output.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
