import re
from datetime import date
from rank_bm25 import BM25Okapi
from core.chroma import get_collection
from core.config import settings
from helpers.metadata import build_metadata_where, matches_keywords, matches_publish_date


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked_ids in rank_lists:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranked_ids):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def _chunks_from_query(result: dict) -> list[dict]:
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    return [
        {"id": chunk_id, "text": text, **(metadata or {})}
        for chunk_id, text, metadata in zip(ids, documents, metadatas)
    ]


def _maybe_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not settings.RERANKER_ENABLED or not chunks:
        return chunks[:top_k]

    from service.reranking_service import rerank_chunks

    return rerank_chunks(query, chunks, top_k)


def retrieve(
    query: str, course_id: str,
    top_k: int | None = None,
    mode: str | None = None,
    *,
    author: str | None = None,
    publish_date_from: date | None = None,
    publish_date_to: date | None = None,
    keywords: list[str] | None = None,
):
    top_k = settings.TOP_K if top_k is None else top_k
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    mode = mode or settings.SEARCH_MODE
    if mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"Unknown search mode: {mode}")

    candidate_k = max(top_k, settings.RERANKER_CANDIDATES) if settings.RERANKER_ENABLED else top_k
    collection = get_collection()
    where = build_metadata_where(
        course_id=course_id,
        author=author,
    )
    post_filter_required = bool(keywords or publish_date_from or publish_date_to)

    if mode == "dense":
        result_limit = collection.count() if post_filter_required else candidate_k
        if result_limit < 1:
            return []
        result = collection.query(
            query_texts=[query],
            n_results=result_limit,
            where=where,
            include=["documents", "metadatas"],
        )
        chunks = [
            chunk for chunk in _chunks_from_query(result)
            if matches_keywords(chunk, keywords)
            and matches_publish_date(chunk, publish_date_from, publish_date_to)
        ]
        return _maybe_rerank(query, chunks, top_k)

    corpus = collection.get(where=where, include=["documents", "metadatas"])
    ids: list[str] = corpus.get("ids", [])
    documents: list[str] = corpus.get("documents", [])
    metadatas: list[dict] = corpus.get("metadatas", [])
    items = [
        {"id": chunk_id, "text": text, **(metadata or {})}
        for chunk_id, text, metadata in zip(ids, documents, metadatas)
    ]
    items = [
        item for item in items
        if matches_keywords(item, keywords)
        and matches_publish_date(item, publish_date_from, publish_date_to)
    ]
    if not items:
        return []

    corpus_by_id = {item["id"]: item for item in items}
    query_tokens = _tokenize(query)
    tokenized_docs = [_tokenize(item["text"]) for item in items]
    bm25_rank: list[str] = []
    if query_tokens and any(tokenized_docs):
        bm25 = BM25Okapi(tokenized_docs)
        sparse_scores = bm25.get_scores(query_tokens)
        matching_indexes = [
            index
            for index, tokens in enumerate(tokenized_docs)
            if set(query_tokens).intersection(tokens)
        ]
        bm25_rank = [
            ids[index]
            for index in sorted(
                matching_indexes, key=lambda i: sparse_scores[i], reverse=True
            )
        ]

    if mode == "bm25":
        return _maybe_rerank(query, [corpus_by_id[chunk_id] for chunk_id in bm25_rank[:candidate_k]], top_k)

    dense_limit = collection.count() if post_filter_required else min(candidate_k, len(items))
    if dense_limit < 1:
        return []
    dense_result = collection.query(
        query_texts=[query],
        n_results=dense_limit,
        where=where,
        include=["documents", "metadatas"],
    )
    allowed_ids = set(corpus_by_id)
    dense_rank = [
        chunk_id for chunk_id in dense_result.get("ids", [[]])[0] if chunk_id in allowed_ids
    ]
    fused_rank = _rrf([dense_rank, bm25_rank])
    fused_chunks = [corpus_by_id[chunk_id] for chunk_id in fused_rank[:candidate_k]]
    return _maybe_rerank(query, fused_chunks, top_k)
