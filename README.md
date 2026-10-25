# RAG-микросервис умного поиска Moodle

Бэкенд по ТЗ: FastAPI принимает запросы от PHP-плагинов Moodle, делает ETL → поиск (dense + BM25 + RRF) → генерацию (Ollama локально / OpenRouter для отладки).

## Структура

- `main.py` — точка входа, lifespan, CORS, префикс `/api`
- `core/` — `config.py` (настройки из `.env`), `security.py` (токен Moodle↔API), `chroma.py` (клиент + коллекция `moodle_docs`)
- `routers/` — `health`, `ingest`, `ask` (тонкие, вся логика в `service/`)
- `schemas/rag.py` — Pydantic-контракт с фронтом/PHP
- `service/` — `ingest_service`, `retrieval_service`, `reranking_service`, `generation_service`, `embeddings` (BGE-M3 + `BGEM3EmbeddingFunction`), `pdf_parser`
- `helpers/text_clean.py` — чистка и чанкинг с сохранением границ абзацев/пунктов списка
- `helpers/chunking_eval.py` — вычисление dense Recall@k для эксперимента
- `scripts/eval_chunking.py` — сравнение 5 стратегий чанкинга на размеченных вопросах (dense Recall@k)
- `tests/test_health.py` — smoke: health доступен, ask без токена не пускает

## Контракт с Moodle/PHP

Обычные API Moodle→FastAPI защищены `X-Service-Token: <SERVICE_TOKEN>`. Административные endpoints используют отдельный `X-Admin-Token: <ADMIN_SERVICE_TOKEN>`. Оба секрета хранятся только на сервере и не отдаются браузеру.

- `GET /api/health` — без токена, `{"status":"ok","chroma":"ok (N chunks)"}` или `unavailable`
- `POST /api/ingest` `{doc_id, course_id, title, author, file_url, filename?, mime_type?, publish_date?, keywords?}` — push из event observer; повторный ingest того же `doc_id` сначала удаляет старые чанки (переиндексация)
- `POST /api/ingest/file?doc_id=&course_id=&title=&author=&publish_date=&keywords=` + multipart `file` — ручная загрузка без Moodle (для отладки)
- `DELETE /api/ingest/{doc_id}` — удалить документ
- `POST /api/ask` `{query, course_id, user_id?, top_k?, mode?}` → `{answer, sources: [{doc, page, fragment}]}`. `course_id` проверяется в PHP через `is_enrolled()`, сервис дополнительно фильтрует Chroma по `course_id`. Нет контекста → `"У меня недостаточно информации"`.
- `top_k` (дефолт `TOP_K=5`, минимум 1): сколько чанков уходят в промпт. `mode`: `hybrid` (дефолт) | `dense` | `bm25`.
- `CONTEXT_PER_CHUNK=1200`: максимальное число символов каждого найденного чанка, включаемых в промпт.
- В режиме генерации ответ может содержать метки `[S1]`, `[S2]` и т.д.; они соответствуют элементам `sources` в том же порядке.

Метаданные чанка в Chroma: `doc_id, doc_title, page, chunk_no, course_id, author` — по ним фильтрация прав и блок «Источники» (собирается из метаданных, не из текста модели).
Дополнительно хранятся `publish_date` и `keywords` (JSON-массив в скалярном metadata-поле Chroma). `POST /api/ask` принимает необязательные фильтры `author`, `publish_date_from`, `publish_date_to`, `keywords`; автора/курс фильтрует Chroma, даты и ключевые слова — backend (документ должен содержать все переданные keywords).

## Административный API Moodle (п.1.3.4 ТЗ)

Все административные endpoints требуют отдельный серверный `X-Admin-Token`; его нужно задать в `.env` как `ADMIN_SERVICE_TOKEN` и настроить для серверной части Moodle/PHP. При пустом токене admin API закрыт (503). PHP всё равно обязан проверять права текущего пользователя до проксирования: FastAPI видит только доверенный серверный токен, а не роль/сессию Moodle. Токен нельзя передавать в браузер.

- `GET /api/documents` — список проиндексированных документов: ID, курс, заголовок, автор, дата, ключевые слова, число чанков, можно ли переиндексировать. Поддерживает фильтры `course_id`, `author`, `publish_date_from`, `publish_date_to`, повторяемый `keywords`.
- `POST /api/reindex` — без тела переиндексирует все документы с сохранённой Moodle-ссылкой; `{"doc_ids":["..."]}` переиндексирует выбранные. Документы, загруженные вручную через `/ingest/file`, не имеют Moodle-ссылки и требуют повторной загрузки файла. Сервис хранит указатель на Moodle-файл, не его копию.
- `GET /api/config` — текущие настройки для UI; секреты не выдаются, возвращается только `OPENROUTER_API_KEY_SET`.
- `PATCH /api/config` — меняет и сохраняет разрешённые настройки моделей/поиска, в том числе `LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OPENROUTER_MODEL`, `TOP_K`, `SEARCH_MODE`, `CHUNK_*`, `PDF_PARSER`, `RERANKER_*`, `RAG_PROMPT`. API key можно передать при PATCH; в ответе он никогда не возвращается. `SERVICE_TOKEN` и параметры Chroma через этот endpoint не меняются.

Runtime overrides сохраняются в `RUNTIME_CONFIG_PATH` (`data/runtime_config.json`, права файла ограничены владельцем) и перекрывают `.env` после перезапуска. Настройки LLM применяются сразу; изменение парсера/стратегии/параметров чанкинга возвращает `reindex_required=true` — документы нужно переиндексировать.

Пример выбора локальной Ollama из админки:

```json
{"LLM_PROVIDER":"ollama","OLLAMA_BASE_URL":"http://127.0.0.1:11434","OLLAMA_MODEL":"qwen2.5:7b"}
```

## Парсинг PDF (`PDF_PARSER` в `.env`)

- `pymupdf` (дефолт) — быстро, только текст
- `pix2text` — бесплатный локальный аналог Mathpix: формулы → LaTeX, таблицы → markdown, медленнее, при первом запуске докачивает ~1 ГБ в `~/.pix2text`. Не установлен — молча откат на `pymupdf` с warning.
- Пин `rapidocr>=3.9.1,<3.10`: `cnstd 1.2.x` несовместим с `3.10+`

## Поддерживаемые документы

- `.pdf` — PyMuPDF или Pix2Text, см. настройки выше.
- `.docx` — извлекаются абзацы и таблицы через `python-docx`.
- `.pptx` — извлекается текст и содержимое таблиц; источники указывают номер слайда.
- Для `.docx` указывается номер явно размеченной страницы, если документ содержит разрыв страницы; если разметки нет, текст относится к логической странице 1. Точные физические страницы DOCX требуют конвертации/рендеринга офисного документа.
- Legacy `.doc` пока не поддерживается; см. TODO в конце README.

## Чанкинг и его оценка

- `CHUNK_STRATEGY=semantic` (дефолт) сохраняет абзацы и отдельные пункты списков как логические блоки, объединяя их до лимита `CHUNK_SIZE`; слишком длинный блок при необходимости делится по предложениям/словам.
- `CHUNK_STRATEGY=recursive` — совместимая рекурсивная нарезка, сначала по структурным блокам, затем по более мелким границам.
- `CHUNK_SIZE` и `CHUNK_OVERLAP` задаются в приблизительных токенах (используется эвристика 4 символа на токен).
- После смены стратегии или параметров уже загруженные документы нужно переиндексировать — существующие чанки автоматически не меняются.
- Для сравнения стратегий подготовь JSONL с 10–15 вопросами, именем исходного документа и известными релевантными страницами:

```jsonl
{"question":"Что должен содержать отчет?","gold_doc":"guide.pdf","gold_pages":[19]}
```

Положи исходные PDF в отдельную папку, затем выполни:

```bash
uv run python scripts/eval_chunking.py --qa data/qa.jsonl --docs data/pdfs --top-k 5 --output data/chunking-report.json
```

Скрипт сравнивает `paragraphs_1`, `paragraphs_3`, `sections`, `token_window_no_split`, `paragraph_overlap`; считает dense Recall@k по размеченным страницам и не меняет рабочую коллекцию Chroma. Для запуска модели BGE-M3 нужны интернет при первом скачивании и ресурсы компьютера.

## Эмбеддинги и Chroma

- Модель `BAAI/bge-m3` (1024 dim), грузится лениво при первом ingest (`service/embeddings.py`)
- Коллекция создаётся с `BGEM3EmbeddingFunction` — один энкодер для записи и запроса (фикс рассогласования с дефолтной `DefaultEmbeddingFunction` на 384 dim)
- Версии связаны: клиент `chromadb>=1.5` ↔ образ `chromadb/chroma:1.5.9` в `docker-compose.yml`. При смене мажорной версии коллекцию пересоздать (старая создавалась с другой EF)

## Retrieval и защита ответа

- Dense использует BGE-M3; BM25 ранжирует все чанки выбранного курса; hybrid объединяет их общие ID через RRF, а не позиции из разных списков.
- BM25 сейчас загружает текст всех чанков курса при каждом запросе. Для большого корпуса понадобится отдельный кэш/индекс sparse-поиска.
- `RERANKER_ENABLED=False` по умолчанию. При `True` CrossEncoder `RERANKER_MODEL` пересортирует до `RERANKER_CANDIDATES` кандидатов; модель скачивается при первом запуске reranker и работает локально.
- Системная инструкция помечает документы как недоверенные данные, требует не выполнять найденные в них команды, отвечать только по контексту, добавлять ссылки `[S#]`, а при неоднозначности задавать уточняющий вопрос.
- `FAITHFULNESS_CHECK_ENABLED=False` по умолчанию. При включении ответ проверяется вторым вызовом той же LLM; неподтверждённый черновик заменяется на «У меня недостаточно информации». Это увеличивает задержку и, при OpenRouter, примерно удваивает вызовы/расходы; оценка самой LLM не является строгим доказательством истинности.
- Логи `/api/ask` содержат режим поиска, число найденных чанков, провайдера и время; текст вопроса и фрагменты документов не логируются.

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

E2E-проверка: `POST /api/ingest/file` (`doc_id=test1`, `course_id=course1`, файл PDF/DOCX/PPTX) → `{"chunks": N}` (первый раз долго — качается BGE-M3) → `POST /api/ask` с тем же `course_id` → `{answer, sources}`.

## Известные ограничения (следующие шаги по ТЗ)

- Точечный RAG: модель видит только `top_k` чанков (каждый ограничен `CONTEXT_PER_CHUNK` символами), весь документ не видит — глобальные вопросы («сколько всего…», «перечисли все…») могут страдать
- Чанкинг теперь сохраняет логические границы, но варианты ещё надо оценить на реальных 10–15 вопросах и согласовать с заказчиком
- BM25 пока перестраивается из полного текста курса на каждом запросе; reranker и faithfulness judge выключены по умолчанию
- Moodle-интеграция должна проверять capabilities администратора/учителя перед проксированием admin API; сервис проверяет отдельный admin token, но сам не знает роли и сессии Moodle
- Legacy `.doc` пока не поддерживается; рассмотреть LibreOffice для конвертации, если он нужен заказчику
- Точные физические номера страниц DOCX требуют офисного рендерера; без него используются явные разрывы страниц, иначе логическая страница 1
- Картинки/формулы из DOCX/PPTX и вывод изображений в цитатах не реализованы; Pix2Text распознаёт формулы только в режиме PDF
- Прод-compose под Linux+NVIDIA — отдельно
