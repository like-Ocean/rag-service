# RAG-микросервис умного поиска Moodle

Бэкенд по ТЗ: FastAPI принимает запросы от PHP-плагинов Moodle, делает ETL → поиск (dense + BM25 + RRF) → генерацию (Ollama локально / OpenRouter для отладки).

## Структура

- `main.py` — точка входа, lifespan, CORS, префикс `/api`
- `core/` — `config.py` (настройки из `.env`), `security.py` (токен Moodle↔API), `chroma.py` (клиент + коллекция `moodle_docs`)
- `routers/` — `health`, `ingest`, `ask` (тонкие, вся логика в `service/`)
- `schemas/rag.py` — Pydantic-контракт с фронтом/PHP
- `service/` — `ingest_service`, `retrieval_service`, `generation_service`, `embeddings` (BGE-M3 + `BGEM3EmbeddingFunction`), `pdf_parser`
- `helpers/text_clean.py` — чистка и выкидывание оглавления/лит-ры, базовый чанкинг
- `scripts/eval_chunking.py` — заготовка эксперимента по чанкингу из ТЗ (пока stub)
- `tests/test_health.py` — smoke: health доступен, ask без токена не пускает

## Контракт с Moodle/PHP

Заголовок: `X-Service-Token: <SERVICE_TOKEN>` (хранится в админке Moodle, в браузер не отдаётся).

- `GET /api/health` — без токена, `{"status":"ok","chroma":"ok (N chunks)"}` или `unavailable`
- `POST /api/ingest` `{doc_id, course_id, title, author, file_url}` — push из event observer; повторный ingest того же `doc_id` сначала удаляет старые чанки (переиндексация)
- `POST /api/ingest/file?doc_id=&course_id=&title=&author=` + multipart `file` — ручная загрузка без Moodle (для отладки)
- `DELETE /api/ingest/{doc_id}` — удалить документ
- `POST /api/ask` `{query, course_id, user_id?, top_k?, mode?}` → `{answer, sources: [{doc, page, fragment}]}`. `course_id` проверяется в PHP через `is_enrolled()`, сервис дополнительно фильтрует Chroma по `course_id`. Нет контекста → `"У меня недостаточно информации"`.
- `top_k` (дефолт `TOP_K=5`): сколько чанков уходят в промпт. `mode`: `hybrid` (дефолт) | `dense` | `bm25`.

Метаданные чанка в Chroma: `doc_id, doc_title, page, chunk_no, course_id, author` — по ним фильтрация прав и блок «Источники» (собирается из метаданных, не из текста модели).

## Парсинг PDF (`PDF_PARSER` в `.env`)

- `pymupdf` (дефолт) — быстро, только текст
- `pix2text` — бесплатный локальный аналог Mathpix: формулы → LaTeX, таблицы → markdown, медленнее, при первом запуске докачивает ~1 ГБ в `~/.pix2text`. Не установлен — молча откат на `pymupdf` с warning.
- Пин `rapidocr>=3.9.1,<3.10`: `cnstd 1.2.x` несовместим с `3.10+`

## Эмбеддинги и Chroma

- Модель `BAAI/bge-m3` (1024 dim), грузится лениво при первом ingest (`service/embeddings.py`)
- Коллекция создаётся с `BGEM3EmbeddingFunction` — один энкодер для записи и запроса (фикс рассогласования с дефолтной `DefaultEmbeddingFunction` на 384 dim)
- Версии связаны: клиент `chromadb>=1.5` ↔ образ `chromadb/chroma:1.5.9` в `docker-compose.yml`. При смене мажорной версии коллекцию пересоздать (старая создавалась с другой EF)

## LLM: локальная + OpenRouter

- `LLM_PROVIDER=ollama` — прод по ТЗ: `OLLAMA_BASE_URL` + `OLLAMA_MODEL=qwen2.5:7b`
- `LLM_PROVIDER=openrouter` — отладка без GPU: `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`. Переключение требует рестарта uvicorn (`--reload` смену `.env` не подхватывает, в логе должно быть `llm=openrouter`)

## Запуск (OrbStack = обычный docker)

```bash
cd rag-service
cp .env.example .env   # затем заполнить SERVICE_TOKEN / OPENROUTER_API_KEY
uv sync
docker compose up chroma -d
curl http://127.0.0.1:8001/api/v2/heartbeat
uv run uvicorn main:app --reload
curl http://127.0.0.1:8000/api/health
brew install ollama && ollama serve & ollama pull qwen2.5:7b
```

Swagger: `http://127.0.0.1:8000/docs`

E2E-проверка: `POST /api/ingest/file` (`doc_id=test1`, `course_id=course1`, файл PDF) → `{"chunks": N}` (первый раз долго — качается BGE-M3) → `POST /api/ask` с тем же `course_id` → `{answer, sources}`.

## Известные ограничения (следующие шаги по ТЗ)

- Точечный RAG: модель видит только `top_k` чанков (каждый обрезан до ~1200 символов), весь документ не видит — глобальные вопросы («сколько всего…», «перечисли все…») страдают
- Чанкинг только базовый, рвёт списки; семантический чанкинг + `eval_chunking.py` (5 стратегий × 10-15 вопросов) — не сделаны
- Нет реранкера, RRF базовый; нет `GET /documents`, `POST /reindex`, `GET/PATCH /config` под PHP-админку (п.1.3.4 ТЗ); картинки из PDF не возвращаются; прод-compose под Linux+NVIDIA — отдельно
