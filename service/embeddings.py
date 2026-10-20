from typing import Any

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction


_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from core.config import settings

        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]):
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


class BGEM3EmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        model = get_embedding_model()
        arr = model.encode(list(input), normalize_embeddings=True)
        return list(arr)

    @staticmethod
    def name() -> str:
        return "bge-m3"

    def get_config(self) -> dict[str, Any]:
        from core.config import settings

        return {"model": settings.EMBEDDING_MODEL}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "BGEM3EmbeddingFunction":
        return BGEM3EmbeddingFunction()
