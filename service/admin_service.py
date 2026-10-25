import logging
from datetime import date
from typing import Any
from core.chroma import get_collection
from helpers.metadata import (
    build_metadata_where, matches_keywords,
    matches_publish_date, parse_keywords,
)
from schemas.admin import DocumentRecord, ReindexItemResult
from service.ingest_service import ingest_document

logger = logging.getLogger(__name__)


def _document_metadata(
    *,
    course_id: str | None = None,
    author: str | None = None
) -> dict[str, dict[str, Any]]:
    where = build_metadata_where(course_id=course_id, author=author)
    collection = get_collection()
    result = (
        collection.get(where=where, include=["metadatas"])
        if where
        else collection.get(include=["metadatas"])
    )

    grouped: dict[str, dict[str, Any]] = {}
    for metadata in result.get("metadatas", []):
        metadata = metadata or {}
        doc_id = metadata.get("doc_id")
        if not doc_id:
            continue
        record = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "course_id": str(metadata.get("course_id", "")),
                "title": str(metadata.get("doc_title", doc_id)),
                "filename": metadata.get("source_filename"),
                "author": str(metadata.get("author", "")),
                "publish_date": metadata.get("publish_date"),
                "keywords": parse_keywords(metadata.get("keywords")),
                "source_url": metadata.get("source_url"),
                "chunks": 0,
            },
        )
        record["chunks"] += 1
        if not record["source_url"] and metadata.get("source_url"):
            record["source_url"] = metadata["source_url"]
    return grouped


def list_documents(
    *,
    course_id: str | None = None,
    author: str | None = None,
    publish_date_from: date | None = None,
    publish_date_to: date | None = None,
    keywords: list[str] | None = None
) -> list[DocumentRecord]:
    records = _document_metadata(
        course_id=course_id,
        author=author,
    )
    filtered = [
        record
        for record in records.values()
        if matches_keywords(record, keywords)
        and matches_publish_date(record, publish_date_from, publish_date_to)
    ]
    filtered.sort(key=lambda item: (item["course_id"], item["title"].casefold(), item["doc_id"]))
    return [
        DocumentRecord(
            doc_id=record["doc_id"],
            course_id=record["course_id"],
            title=record["title"],
            filename=record["filename"],
            author=record["author"],
            publish_date=record["publish_date"],
            keywords=record["keywords"],
            chunks=record["chunks"],
            reindexable=bool(record["source_url"]),
        )
        for record in filtered
    ]


async def reindex_documents(doc_ids: list[str] | None = None) -> list[ReindexItemResult]:
    records = _document_metadata()
    requested = list(dict.fromkeys(doc_ids)) if doc_ids is not None else list(records)
    results: list[ReindexItemResult] = []

    for doc_id in requested:
        record = records.get(doc_id)
        if record is None:
            results.append(ReindexItemResult(doc_id=doc_id, status="not_found"))
            continue
        if not record["source_url"]:
            results.append(
                ReindexItemResult(
                    doc_id=doc_id,
                    status="not_reindexable",
                    detail="No Moodle file URL is stored; upload the document again to reindex it.",
                )
            )
            continue

        try:
            chunk_count = await ingest_document(
                doc_id=doc_id,
                course_id=record["course_id"],
                title=record["title"],
                author=record["author"],
                file_url=record["source_url"],
                publish_date=record["publish_date"],
                keywords=record["keywords"],
            )
            results.append(
                ReindexItemResult(doc_id=doc_id, status="indexed", chunks=chunk_count)
            )
        except Exception as exc:
            logger.error("Reindex failed for doc_id=%s (%s)", doc_id, type(exc).__name__)
            results.append(
                ReindexItemResult(
                    doc_id=doc_id,
                    status="failed",
                    detail="Reindex failed; inspect backend logs.",
                )
            )
    return results
