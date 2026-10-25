import asyncio
import json
import os
from datetime import date

from fastapi.testclient import TestClient
from pydantic import SecretStr

from core.config import settings
from core.runtime_config import RuntimeConfigStore
from helpers.metadata import (
    build_metadata_where,
    matches_keywords,
    matches_publish_date,
    parse_keywords,
    serialize_keywords,
)
from main import app
from schemas.admin import RuntimeConfigPatch
from service import admin_service, ingest_service


def test_keyword_metadata_round_trip_and_all_keyword_filter():
    encoded = serialize_keywords(["  NLP ", "rag", "nlp", ""])

    assert parse_keywords(encoded) == ["NLP", "rag"]
    assert matches_keywords({"keywords": encoded}, ["RAG", "nlp"])
    assert not matches_keywords({"keywords": encoded}, ["RAG", "missing"])


def test_metadata_where_and_publish_date_filters():
    where = build_metadata_where(course_id="course1", author="Author")
    assert where == {"$and": [{"course_id": "course1"}, {"author": "Author"}]}
    assert matches_publish_date(
        {"publish_date": "2025-06-01"}, date(2025, 1, 1), date(2025, 12, 31)
    )
    assert not matches_publish_date({"publish_date": "2024-12-31"}, date(2025, 1, 1), None)


def test_runtime_config_persists_secrets_but_never_returns_them(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "TOP_K", 5)
    path = tmp_path / "runtime-config.json"
    store = RuntimeConfigStore(path)

    public, updated = store.update(
        RuntimeConfigPatch(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY=SecretStr("test-key-not-real"),
            TOP_K=7,
        )
    )

    assert updated == ["LLM_PROVIDER", "OPENROUTER_API_KEY", "TOP_K"]
    assert public["LLM_PROVIDER"] == "openrouter"
    assert public["OPENROUTER_API_KEY_SET"] is True
    assert "OPENROUTER_API_KEY" not in public
    assert json.loads(path.read_text(encoding="utf-8"))["OPENROUTER_API_KEY"] == "test-key-not-real"
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_documents_api_requires_separate_admin_token(monkeypatch):
    import routers.admin as admin_router

    monkeypatch.setattr(settings, "ADMIN_SERVICE_TOKEN", "")
    client = TestClient(app)
    unconfigured = client.get("/api/documents", headers={"X-Admin-Token": "anything"})

    monkeypatch.setattr(settings, "ADMIN_SERVICE_TOKEN", "test-admin-token")
    monkeypatch.setattr(admin_router, "list_documents", lambda **kwargs: [])

    denied = client.get("/api/documents")
    wrong_token = client.get(
        "/api/documents", headers={"X-Admin-Token": settings.SERVICE_TOKEN}
    )
    allowed = client.get("/api/documents", headers={"X-Admin-Token": "test-admin-token"})

    assert unconfigured.status_code == 503
    assert denied.status_code == 401
    assert wrong_token.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json() == []


def test_config_api_masks_api_key_and_accepts_local_ollama_settings(monkeypatch):
    import routers.admin as admin_router

    class FakeRuntimeConfigStore:
        def public_config(self):
            return {
                "LLM_PROVIDER": "ollama",
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_MODEL": "qwen2.5:7b",
                "OPENROUTER_API_KEY_SET": True,
            }

        def update(self, patch):
            assert patch.LLM_PROVIDER == "ollama"
            assert patch.OLLAMA_MODEL == "qwen2.5:7b"
            return self.public_config(), list(patch.model_fields_set)

    monkeypatch.setattr(admin_router, "runtime_config_store", FakeRuntimeConfigStore())
    monkeypatch.setattr(settings, "ADMIN_SERVICE_TOKEN", "test-admin-token")
    client = TestClient(app)
    headers = {"X-Admin-Token": "test-admin-token"}

    current = client.get("/api/config", headers=headers)
    updated = client.patch(
        "/api/config",
        headers=headers,
        json={
            "LLM_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "OLLAMA_MODEL": "qwen2.5:7b",
        },
    )

    assert current.status_code == 200
    assert "OPENROUTER_API_KEY" not in current.json()["config"]
    assert current.json()["config"]["OPENROUTER_API_KEY_SET"] is True
    assert updated.status_code == 200
    assert updated.json()["config"]["LLM_PROVIDER"] == "ollama"


def test_reindex_reports_manual_uploads_as_not_reindexable(monkeypatch):
    class FakeCollection:
        def get(self, where=None, include=None):
            return {
                "metadatas": [
                    {
                        "doc_id": "manual-doc",
                        "course_id": "course1",
                        "doc_title": "Manual PDF",
                        "author": "Teacher",
                        "keywords": serialize_keywords(["test"]),
                    }
                ]
            }

    monkeypatch.setattr(admin_service, "get_collection", lambda: FakeCollection())
    calls = []

    async def fake_ingest_document(**kwargs):
        calls.append(kwargs)
        return 3

    monkeypatch.setattr(admin_service, "ingest_document", fake_ingest_document)
    results = asyncio.run(admin_service.reindex_documents())

    assert len(results) == 1
    assert results[0].status == "not_reindexable"
    assert calls == []


def test_ingest_stores_publication_metadata_and_moodle_source_pointer(monkeypatch):
    class FakeCollection:
        def __init__(self):
            self.metadatas = None

        def delete(self, where):
            pass

        def add(self, ids, documents, embeddings, metadatas):
            self.metadatas = metadatas

    collection = FakeCollection()

    async def fake_load_document_bytes(file_url, raw, mime_type):
        return b"pdf bytes", mime_type

    monkeypatch.setattr(ingest_service, "_load_document_bytes", fake_load_document_bytes)
    monkeypatch.setattr(ingest_service, "parse_document", lambda pdf, filename, mime: [(1, "Body text")])
    monkeypatch.setattr(ingest_service, "embed_texts", lambda texts: [[0.1] * 3 for _ in texts])
    monkeypatch.setattr(ingest_service, "get_collection", lambda: collection)

    count = asyncio.run(
        ingest_service.ingest_document(
            doc_id="doc1",
            course_id="course1",
            title="Guide",
            author="Teacher",
            file_url="https://moodle.example/pluginfile.php/1/file.pdf",
            publish_date=date(2025, 4, 12),
            keywords=["RAG", "Moodle"],
        )
    )

    assert count == 1
    metadata = collection.metadatas[0]
    assert metadata["publish_date"] == "2025-04-12"
    assert parse_keywords(metadata["keywords"]) == ["RAG", "Moodle"]
    assert metadata["source_url"].startswith("https://moodle.example/")
