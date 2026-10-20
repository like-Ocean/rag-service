import httpx
from core.config import settings
from core.chroma import get_collection
from helpers.text_clean import clean_text, drop_service_parts, split_recursive
from service.embeddings import embed_texts
from service.pdf_parser import parse_pdf


async def _load_pdf_bytes(file_url: str | None, raw: bytes | None):
    if raw is not None:
        return raw
    assert file_url, "нужен file_url или raw bytes"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(file_url, headers={"X-Service-Token": settings.SERVICE_TOKEN})
        resp.raise_for_status()
        return resp.content


async def ingest_document(
    doc_id: str, course_id: str, title: str, 
    author: str, file_url: str | None = None, raw: bytes | None = None
):
    pdf_bytes = await _load_pdf_bytes(file_url, raw)
    pages = parse_pdf(pdf_bytes)
    pages = [(n, clean_text(t)) for n, t in pages if t.strip()]
    pages = drop_service_parts(pages)

    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for page_no, text in pages:
        for j, ch in enumerate(split_recursive(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
            idx = len(chunks)
            chunks.append(ch)
            metadatas.append({
                "doc_id": doc_id, "doc_title": title or doc_id,
                "page": page_no, "chunk_no": idx,
                "course_id": course_id, "author": author,
            })
            ids.append(f"{doc_id}:::{idx}")

    if not chunks:
        return 0

    collection = get_collection()
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    collection.add(ids=ids, documents=chunks, embeddings=embed_texts(chunks), metadatas=metadatas)
    return len(chunks)


def delete_document(doc_id: str):
    get_collection().delete(where={"doc_id": doc_id})
