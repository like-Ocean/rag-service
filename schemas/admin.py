from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr, field_validator


class DocumentRecord(BaseModel):
    doc_id: str
    course_id: str
    title: str
    filename: str | None = None
    author: str
    publish_date: str | None = None
    keywords: list[str] = Field(default_factory=list)
    chunks: int
    reindexable: bool


class ReindexRequest(BaseModel):
    doc_ids: list[str] | None = Field(default=None, min_length=1)


class ReindexItemResult(BaseModel):
    doc_id: str
    status: Literal["indexed", "not_reindexable", "not_found", "failed"]
    chunks: int = 0
    detail: str | None = None


class ReindexResponse(BaseModel):
    results: list[ReindexItemResult]


class RuntimeConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    LLM_PROVIDER: Literal["ollama", "openrouter"] | None = None
    OLLAMA_BASE_URL: AnyHttpUrl | None = None
    OLLAMA_MODEL: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    OLLAMA_TIMEOUT: int | None = Field(default=None, ge=1, le=900)
    OPENROUTER_API_KEY: SecretStr | None = None
    OPENROUTER_BASE_URL: AnyHttpUrl | None = None
    OPENROUTER_MODEL: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    TOP_K: int | None = Field(default=None, ge=1, le=1000)
    SEARCH_MODE: Literal["dense", "bm25", "hybrid"] | None = None
    RERANKER_ENABLED: bool | None = None
    RERANKER_MODEL: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    RERANKER_CANDIDATES: int | None = Field(default=None, ge=1, le=1000)
    PDF_PARSER: Literal["pymupdf", "pix2text"] | None = None
    CHUNK_STRATEGY: Literal["semantic", "recursive"] | None = None
    CHUNK_SIZE: int | None = Field(default=None, ge=1, le=10000)
    CHUNK_OVERLAP: int | None = Field(default=None, ge=0, le=5000)
    CONTEXT_PER_CHUNK: int | None = Field(default=None, ge=1, le=100000)
    FAITHFULNESS_CHECK_ENABLED: bool | None = None
    RAG_PROMPT: str | None = Field(
        default=None,
        min_length=1,
        max_length=20000,
    )

    @field_validator("RAG_PROMPT")
    @classmethod
    def prompt_has_required_placeholders(cls, value: str | None) -> str | None:
        if value is not None and ("{context}" not in value or "{query}" not in value):
            raise ValueError("RAG_PROMPT must contain {context} and {query}")
        return value


class RuntimeConfigResponse(BaseModel):
    config: dict[str, Any]
    updated_fields: list[str] = Field(default_factory=list)
    reindex_required: bool = False
