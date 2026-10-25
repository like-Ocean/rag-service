import numpy as np
import pytest
from pydantic import ValidationError

from core.config import settings
from helpers.chunking_eval import evaluate_strategy
from helpers.text_clean import clean_text, split_by_strategy, split_text
from schemas.rag import AskRequest
from service.generation_service import build_prompt


def test_top_k_must_be_at_least_one():
    with pytest.raises(ValidationError):
        AskRequest(query="test query", course_id="course1", top_k=0)


def test_prompt_chunk_limit_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_PER_CHUNK", 5)

    prompt = build_prompt("question", [{"doc_id": "doc", "page": 1, "text": "abcdefgh"}])

    assert "abcde" in prompt
    assert "abcdefgh" not in prompt


def test_semantic_chunking_preserves_numbered_list_items():
    text = clean_text(
        "Вопросы для самоконтроля\n"
        "1. Что такое сущность?\n"
        "2. Чем сущность отличается от намерения?\n"
        "3. Приведите примеры сущностей."
    )

    chunks = split_text(text, chunk_size=100, overlap=0, strategy="semantic")
    joined = "\n".join(chunks)

    assert "1. Что такое сущность?" in joined
    assert "2. Чем сущность отличается от намерения?" in joined
    assert "3. Приведите примеры сущностей." in joined
    assert "Вопросы для самоконтроля" in joined


def test_paragraph_strategy_keeps_list_items_as_individual_blocks():
    text = "1. Первый пункт списка.\n2. Второй пункт списка.\n3. Третий пункт списка."

    chunks = split_by_strategy(text, "paragraphs_1", chunk_size=100)

    assert chunks == [
        "1. Первый пункт списка.",
        "2. Второй пункт списка.",
        "3. Третий пункт списка.",
    ]


def test_token_window_strategy_does_not_split_an_oversized_paragraph():
    paragraph = "Это один цельный абзац, который превышает очень маленький лимит чанка."

    chunks = split_by_strategy(paragraph, "token_window_no_split", chunk_size=2)

    assert chunks == [paragraph]


def test_paragraph_overlap_repeats_the_previous_logical_block():
    text = "Первый абзац здесь.\n\nВторой абзац здесь.\n\nТретий абзац здесь."

    chunks = split_by_strategy(text, "paragraph_overlap", chunk_size=5)

    assert len(chunks) == 3
    assert "Первый абзац здесь." in chunks[0]
    assert "Первый абзац здесь." in chunks[1]
    assert "Второй абзац здесь." in chunks[1]
    assert "Второй абзац здесь." in chunks[2]
    assert "Третий абзац здесь." in chunks[2]


def test_evaluator_calculates_page_recall_at_k():
    chunks = [
        {"doc_id": "lecture.pdf", "page": 1, "text": "irrelevant"},
        {"doc_id": "lecture.pdf", "page": 2, "text": "relevant"},
        {"doc_id": "lecture.pdf", "page": 3, "text": "also relevant"},
    ]
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.8, 0.2]])
    rows = [{"question": "question", "gold_doc": "lecture.pdf", "gold_pages": [2, 3]}]
    question_embeddings = np.array([[1.0, 0.0]])

    result = evaluate_strategy(chunks, embeddings, rows, question_embeddings, top_k=1)

    assert result["recall_at_k"] == 0.5
