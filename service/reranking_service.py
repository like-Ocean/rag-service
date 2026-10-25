from core.config import settings

_model = None


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(settings.RERANKER_MODEL)
    return _model


def reset_reranker() -> None:
    global _model
    _model = None


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []
    model = get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
    return [chunk for chunk, _score in ranked[:top_k]]
