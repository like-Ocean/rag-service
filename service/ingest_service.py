from datetime import date
from urllib.parse import unquote, urlsplit
import httpx
from core.chroma import get_collection
from core.config import settings
from helpers.metadata import serialize_keywords
from helpers.text_clean import clean_text, drop_service_parts, split_text
from service.document_parser import parse_document
from service.embeddings import embed_texts


async def _load_document_bytes(file_url: str | None, raw: bytes | None, mime_type: str | None) -> tuple[bytes, str | None]:
    if raw is not None:
        return raw, mime_type
    if not file_url:
        raise ValueError("нужен file_url или raw bytes")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(file_url, headers={"X-Service-Token": settings.SERVICE_TOKEN})
        resp.raise_for_status()
        return resp.content, mime_type or resp.headers.get("content-type")


def _filename_from_url(file_url: str | None) -> str | None:
    if not file_url:
        return None
    return unquote(urlsplit(file_url).path.rsplit("/", 1)[-1]) or None


async def ingest_document(
    doc_id: str, course_id: str,
    title: str, author: str,
    file_url: str | None = None,
    raw: bytes | None = None,
    publish_date: date | str | None = None,
    keywords: list[str] | None = None,
    filename: str | None = None,
    mime_type: str | None = None
):
    document_bytes, detected_mime_type = await _load_document_bytes(file_url, raw, mime_type)
    source_filename = filename or _filename_from_url(file_url)
    pages = parse_document(document_bytes, source_filename, detected_mime_type)
    pages = [(n, clean_text(t)) for n, t in pages if t.strip()]
    pages = drop_service_parts(pages)

    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for page_no, text in pages:
        for ch in split_text(
            text,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP,
            strategy=settings.CHUNK_STRATEGY,
        ):
            idx = len(chunks)
            chunks.append(ch)
            metadata = {
                "doc_id": doc_id,
                "doc_title": title or doc_id,
                "page": page_no,
                "chunk_no": idx,
                "course_id": course_id,
                "author": author,
                "keywords": serialize_keywords(keywords),
            }
            if source_filename:
                metadata["source_filename"] = source_filename
            if publish_date is not None:
                metadata["publish_date"] = (
                    publish_date.isoformat() if isinstance(publish_date, date) else publish_date
                )
            if file_url:
                metadata["source_url"] = file_url
            metadatas.append(metadata)
            ids.append(f"{doc_id}:::{idx}")

    if not chunks:
        return 0

    collection = get_collection()
    collection.delete(where={"doc_id": doc_id})
    collection.add(ids=ids, documents=chunks, embeddings=embed_texts(chunks), metadatas=metadatas)
    return len(chunks)


def delete_document(doc_id: str):
    get_collection().delete(where={"doc_id": doc_id})
