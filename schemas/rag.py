from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Устойчивый ID документа в Moodle")
    course_id: str = Field(..., description="Курс-владелец, для фильтра прав")
    title: str = ""
    author: str = ""
    file_url: str | None = Field(None, description="Moodle отдаёт файл ссылкой (pluginfile.php)")
    filename: str | None = None
    mime_type: str | None = None
    publish_date: date | None = None
    keywords: list[str] = Field(default_factory=list)


class ChunkInfo(BaseModel):
    chunk_no: int
    page: int
    text_preview: str


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    detail: str = "indexed"


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3)
    course_id: str = Field(..., description="Проверен в PHP через is_enrolled(), не из браузера")
    user_id: str = ""
    top_k: int | None = Field(default=None, ge=1)
    mode: Literal["dense", "bm25", "hybrid"] | None = None
    author: str | None = None
    publish_date_from: date | None = None
    publish_date_to: date | None = None
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_publish_date_range(self):
        if (
            self.publish_date_from is not None
            and self.publish_date_to is not None
            and self.publish_date_from > self.publish_date_to
        ):
            raise ValueError("publish_date_from must be on or before publish_date_to")
        return self


class Source(BaseModel):
    doc: str
    page: int
    fragment: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class ErrorResponse(BaseModel):
    detail: str


class MessageResponse(BaseModel):
    message: str
