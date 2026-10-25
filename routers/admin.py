from datetime import date
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query
from core.runtime_config import runtime_config_store
from core.security import verify_admin_token
from schemas.admin import (
    DocumentRecord,
    ReindexRequest,
    ReindexResponse,
    RuntimeConfigPatch,
    RuntimeConfigResponse,
)
from service.admin_service import list_documents, reindex_documents

router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(verify_admin_token)],
)


@router.get("/documents", response_model=list[DocumentRecord])
async def documents(
    course_id: str | None = None,
    author: str | None = None,
    publish_date_from: date | None = None,
    publish_date_to: date | None = None,
    keywords: Annotated[list[str] | None, Query()] = None,
):
    if publish_date_from and publish_date_to and publish_date_from > publish_date_to:
        raise HTTPException(422, "publish_date_from must be on or before publish_date_to")
    return list_documents(
        course_id=course_id,
        author=author,
        publish_date_from=publish_date_from,
        publish_date_to=publish_date_to,
        keywords=keywords,
    )


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(data: ReindexRequest | None = None):
    return ReindexResponse(results=await reindex_documents(data.doc_ids if data else None))


@router.get("/config", response_model=RuntimeConfigResponse)
async def get_config():
    return RuntimeConfigResponse(config=runtime_config_store.public_config())


@router.patch("/config", response_model=RuntimeConfigResponse)
async def patch_config(data: RuntimeConfigPatch):
    try:
        config, updated_fields = runtime_config_store.update(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RuntimeConfigResponse(
        config=config,
        updated_fields=updated_fields,
        reindex_required=bool(
            set(updated_fields) & {"PDF_PARSER", "CHUNK_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP"}
        ),
    )
