import importlib

import pytest
from fastapi.testclient import TestClient

from core.config import settings
from main import app

ask_router = importlib.import_module("routers.ask")


def chunks(count):
    return [
        {"doc_id": f"doc-{i}", "doc_title": f"Document {i}", "page": i, "text": f"Text {i}"}
        for i in range(1, count + 1)
    ]


@pytest.fixture
def ask_with_answer(monkeypatch):
    monkeypatch.setattr(settings, "SERVICE_TOKEN", "test-token")

    def request(answer, candidates=None, top_k=None):
        monkeypatch.setattr(ask_router, "retrieve", lambda *a, **kw: candidates or [])

        async def generate(query, context):
            return answer

        monkeypatch.setattr(ask_router, "generate_answer", generate)
        payload = {"query": "question", "course_id": "course"}
        if top_k is not None:
            payload["top_k"] = top_k
        response = TestClient(app).post(
            "/api/ask", json=payload, headers={"X-Service-Token": "test-token"}
        )
        assert response.status_code == 200
        return response.json()

    return request


def test_only_cited_sources_are_returned_and_links_are_renumbered(ask_with_answer):
    result = ask_with_answer("Third [S3], first [S1], third again [S3].", chunks(5))
    assert result["answer"] == "Third [S1], first [S2], third again [S1]."
    assert result["sources"] == [
        {"doc": "Document 3", "page": 3, "fragment": "Text 3"},
        {"doc": "Document 1", "page": 1, "fragment": "Text 1"},
    ]


@pytest.mark.parametrize("answer", ["У меня недостаточно информации", "Уточните вопрос?", "Answer"])
def test_answer_without_citations_has_no_sources(ask_with_answer, answer):
    result = ask_with_answer(answer, chunks(5))
    assert result == {"answer": answer, "sources": []}


def test_unknown_citations_are_removed_without_exposing_sources(ask_with_answer):
    result = ask_with_answer("Unknown [S0] [S99]. Valid [S2].", chunks(3))
    assert result["answer"] == "Unknown  . Valid [S1]."
    assert [source["doc"] for source in result["sources"]] == ["Document 2"]


def test_empty_context_cannot_produce_sources(ask_with_answer):
    result = ask_with_answer("Unknown [S1].")
    assert result == {"answer": "Unknown .", "sources": []}


@pytest.mark.parametrize("requested,configured,expected", [(12000, 5, 16), (None, 12000, 16), (3, 5, 3)])
def test_retrieval_and_generation_context_are_bounded(monkeypatch, requested, configured, expected):
    monkeypatch.setattr(settings, "SERVICE_TOKEN", "test-token")
    monkeypatch.setattr(settings, "TOP_K", configured)
    calls = []

    def retrieve(query, course_id, top_k, mode, **filters):
        calls.append(top_k)
        return chunks(12000)  # Even a faulty retriever must not flood the LLM or response.

    async def generate(query, context):
        assert len(context) == expected
        return " ".join(f"[S{i}]" for i in range(1, len(context) + 1))

    monkeypatch.setattr(ask_router, "retrieve", retrieve)
    monkeypatch.setattr(ask_router, "generate_answer", generate)
    payload = {"query": "question", "course_id": "course"}
    if requested is not None:
        payload["top_k"] = requested
    response = TestClient(app).post(
        "/api/ask", json=payload, headers={"X-Service-Token": "test-token"}
    )
    assert response.status_code == 200
    assert calls == [expected]
    assert len(response.json()["sources"]) == expected
