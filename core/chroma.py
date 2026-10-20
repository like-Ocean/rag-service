import chromadb

from core.config import settings

_client = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    return _client


def get_collection():
    from service.embeddings import BGEM3EmbeddingFunction

    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=BGEM3EmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
