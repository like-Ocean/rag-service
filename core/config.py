from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment
    ENV: str = "development"
    DEBUG: bool = True

    # Application
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_RELOAD: bool = True
    APP_LOG_LEVEL: str = "debug"

    SERVICE_TOKEN: str = "change-me"
    ADMIN_SERVICE_TOKEN: str = ""

    CHROMA_HOST: str = "127.0.0.1"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "moodle_docs"

    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Парсер PDF: pymupdf (быстро, только текст) | pix2text (бесплатный
    # аналог Mathpix: формулы -> LaTeX, таблицы -> markdown)
    PDF_PARSER: str = "pymupdf"

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_TIMEOUT: int = 120

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "qwen/qwen-2.5-7b-instruct"

    TOP_K: int = Field(default=5, ge=1)
    SEARCH_MODE: str = "hybrid"  # dense | bm25 | hybrid
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_CANDIDATES: int = Field(default=20, ge=1)
    CHUNK_STRATEGY: str = "semantic"  # recursive | semantic
    CHUNK_SIZE: int = Field(default=800, ge=1)  # приблизительные токены
    CHUNK_OVERLAP: int = Field(default=100, ge=0)  # приблизительные токены
    CONTEXT_PER_CHUNK: int = Field(default=1200, ge=1)  # символов на чанк в промпте
    FAITHFULNESS_CHECK_ENABLED: bool = False
    RUNTIME_CONFIG_PATH: str = "data/runtime_config.json"
    RAG_PROMPT: str = (
        "Ты — ассистент преподавателя. Используй только следующий контекст "
        "для ответа на вопрос. Если ответа нет, напиши: 'У меня недостаточно "
        "информации'. Контекст:\n{context}\nВопрос: {query}"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
