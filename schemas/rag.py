from typing import Literal
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Устойчивый ID документа в Moodle")
    course_id: str = Field(..., description="Курс-владелец, для фильтра прав")
    title: str = ""
    author: str = ""
    file_url: str | None = Field(None, description="Moodle отдаёт файл ссылкой (pluginfile.php)")


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
    top_k: int | None = None
    mode: Literal["dense", "bm25", "hybrid"] | None = None


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
