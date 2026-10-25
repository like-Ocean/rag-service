import logging
from datetime import date
from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from core.security import verify_service_token
from schemas.rag import IngestRequest, IngestResponse
from service import ingest_service
from service.document_parser import UnsupportedDocumentFormatError

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)


@router.post("", response_model=IngestResponse, dependencies=[Depends(verify_service_token)])
async def ingest(data: IngestRequest):
    try:
        n = await ingest_service.ingest_document(
            doc_id=data.doc_id, course_id=data.course_id,
            title=data.title,
            author=data.author,
            file_url=data.file_url,
            filename=data.filename,
            mime_type=data.mime_type,
            publish_date=data.publish_date,
            keywords=data.keywords,
        )
    except UnsupportedDocumentFormatError as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e)) from e
    except Exception as e:
        logger.error("Document ingest failed (%s)", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document ingest failed; inspect backend logs",
        ) from e
    return IngestResponse(doc_id=data.doc_id, chunks=n)


@router.post("/file", response_model=IngestResponse, dependencies=[Depends(verify_service_token)])
async def ingest_file(
    doc_id: str, course_id: str, file: Annotated[UploadFile, File()],
    title: str = "", author: str = "", publish_date: date | None = None,
    keywords: Annotated[list[str] | None, Query()] = None
):
    raw = await file.read()
    try:
        n = await ingest_service.ingest_document(
            doc_id, course_id, title, author,
            raw=raw, filename=file.filename,
            mime_type=file.content_type,
            publish_date=publish_date,
            keywords=keywords or [],
        )
    except UnsupportedDocumentFormatError as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e)) from e
    except Exception as e:
        logger.error("Document ingest failed (%s)", type(e).__name__)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Document ingest failed") from e
    return IngestResponse(doc_id=doc_id, chunks=n)


@router.delete("/{doc_id}", dependencies=[Depends(verify_service_token)])
async def delete_doc(doc_id: str):
    ingest_service.delete_document(doc_id)
    return {"message": f"{doc_id} deleted"}
