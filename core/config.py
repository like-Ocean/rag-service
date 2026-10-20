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

    TOP_K: int = 5
    SEARCH_MODE: str = "hybrid"  # dense | bm25 | hybrid
    CHUNK_SIZE: int = 800  # токенов, ТЗ рекомендует 500-1000
    CHUNK_OVERLAP: int = 100
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
