import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, SecretStr

from core.config import settings
from schemas.admin import RuntimeConfigPatch

logger = logging.getLogger(__name__)

CONFIG_FIELDS = (
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "TOP_K",
    "SEARCH_MODE",
    "RERANKER_ENABLED",
    "RERANKER_MODEL",
    "RERANKER_CANDIDATES",
    "PDF_PARSER",
    "CHUNK_STRATEGY",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CONTEXT_PER_CHUNK",
    "FAITHFULNESS_CHECK_ENABLED",
    "RAG_PROMPT",
)
REINDEX_FIELDS = {"PDF_PARSER", "CHUNK_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP"}


def _plain_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        elif isinstance(value, AnyHttpUrl):
            value = str(value).rstrip("/")
        if value is None:
            if key == "OPENROUTER_API_KEY":
                value = ""
            else:
                raise ValueError(f"{key} cannot be null; omit it to keep the current value")
        normalized[key] = value
    return normalized


class RuntimeConfigStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.overrides: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            validated = RuntimeConfigPatch.model_validate(raw)
            self.overrides = _plain_values(validated.model_dump(exclude_unset=True))
            self._validate_effective(self.overrides)
            self._apply(self.overrides)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid runtime config file (%s)", type(exc).__name__)
            self.overrides = {}

    @staticmethod
    def _apply(values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(settings, key, value)

    @staticmethod
    def _validate_effective(values: dict[str, Any]) -> None:
        effective = {key: getattr(settings, key) for key in CONFIG_FIELDS}
        effective.update(values)
        if effective["LLM_PROVIDER"] == "openrouter" and not effective["OPENROUTER_API_KEY"]:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        if effective["CHUNK_OVERLAP"] > effective["CHUNK_SIZE"]:
            raise ValueError("CHUNK_OVERLAP cannot be greater than CHUNK_SIZE")

    def _save(self, values: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="runtime-config-", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(values, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def public_config(self) -> dict[str, Any]:
        values = {key: getattr(settings, key) for key in CONFIG_FIELDS if key != "OPENROUTER_API_KEY"}
        values["OPENROUTER_API_KEY_SET"] = bool(settings.OPENROUTER_API_KEY)
        return values

    def update(self, patch: RuntimeConfigPatch) -> tuple[dict[str, Any], list[str]]:
        values = _plain_values(patch.model_dump(exclude_unset=True))
        if not values:
            raise ValueError("At least one configuration field is required")

        merged = {**self.overrides, **values}
        self._validate_effective(merged)
        self._save(merged)
        self.overrides = merged
        self._apply(values)
        if "RERANKER_MODEL" in values:
            from service.reranking_service import reset_reranker

            reset_reranker()
        return self.public_config(), list(values)


runtime_config_store = RuntimeConfigStore(settings.RUNTIME_CONFIG_PATH)
