from rank_bm25 import BM25Okapi
from core.config import settings
from core.chroma import get_collection


def _rrf(rank_lists: list[list[int]], k: int = 60):
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def retrieve(query: str, course_id: str, top_k: int | None = None, mode: str | None = None):
    top_k = top_k or settings.TOP_K
    mode = mode or settings.SEARCH_MODE
    collection = get_collection()

    where = {"course_id": course_id}
    dense = collection.query(query_texts=[query], n_results=top_k, where=where)
    dense_docs: list[str] = dense.get("documents", [[]])[0]
    dense_meta: list[dict] = dense.get("metadatas", [[]])[0]

    if mode == "dense" or not dense_docs:
        return [{"text": d, **m} for d, m in zip(dense_docs, dense_meta)]

    pool = collection.query(query_texts=[query], n_results=max(top_k * 4, 20), where=where)
    pool_docs: list[str] = pool.get("documents", [[]])[0]
    pool_meta: list[dict] = pool.get("metadatas", [[]])[0]
    tokenized = [d.lower().split() for d in pool_docs]
    bm25 = BM25Okapi(tokenized)
    bm25_rank = sorted(
        range(len(pool_docs)),
        key=lambda i: bm25.get_scores(query.lower().split())[i],
        reverse=True
    )
    dense_rank = list(range(len(dense_docs)))

    if mode == "bm25":
        order = bm25_rank[:top_k]
        return [{"text": pool_docs[i], **pool_meta[i]} for i in order]

    fused = _rrf([dense_rank, bm25_rank[: len(dense_rank) or 1]])[:top_k]

    out = []
    for i in fused:
        if i < len(pool_docs):
            out.append({"text": pool_docs[i], **pool_meta[i]})
    return out
