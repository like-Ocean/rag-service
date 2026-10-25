from pathlib import Path
from typing import Any
import numpy as np


def _gold_pages(row: dict[str, Any]) -> set[int]:
    values = row.get("gold_pages", [row.get("gold_page")])
    return {int(value) for value in values if value is not None}


def _matches_gold_doc(chunk_doc_id: str, gold_doc: str | None) -> bool:
    if not gold_doc:
        return True
    return Path(chunk_doc_id).name == Path(gold_doc).name


def evaluate_strategy(
    chunks: list[dict[str, Any]],
    chunk_embeddings: np.ndarray,
    question_rows: list[dict[str, Any]],
    question_embeddings: np.ndarray,
    top_k: int
) -> dict[str, Any]:
    if not chunks:
        return {"recall_at_k": 0.0, "questions": []}

    scores = question_embeddings @ chunk_embeddings.T
    recalls: list[float] = []
    details: list[dict[str, Any]] = []
    for row, row_scores in zip(question_rows, scores):
        gold_pages = _gold_pages(row)
        gold_doc = row.get("gold_doc")
        relevant_pages = {
            (chunk["doc_id"], chunk["page"])
            for chunk in chunks
            if _matches_gold_doc(chunk["doc_id"], gold_doc) and chunk["page"] in gold_pages
        }
        if not relevant_pages:
            raise ValueError(
                f"No indexed chunk matches gold_doc={gold_doc!r}, gold_pages={sorted(gold_pages)}"
            )

        ranking = np.argsort(row_scores)[::-1][:top_k]
        found_pages = {
            (chunks[int(index)]["doc_id"], chunks[int(index)]["page"])
            for index in ranking
            if _matches_gold_doc(chunks[int(index)]["doc_id"], gold_doc)
            and chunks[int(index)]["page"] in gold_pages
        }
        recall = len(found_pages) / len(relevant_pages)
        recalls.append(recall)
        details.append({"question": row["question"], "recall_at_k": recall})

    return {"recall_at_k": sum(recalls) / len(recalls), "questions": details}
