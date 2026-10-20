from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from core.security import verify_service_token
from schemas.rag import IngestRequest, IngestResponse
from service import ingest_service

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, dependencies=[Depends(verify_service_token)])
async def ingest(data: IngestRequest):
    try:
        n = await ingest_service.ingest_document(
            doc_id=data.doc_id, course_id=data.course_id,
            title=data.title, author=data.author, file_url=data.file_url,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"ingest failed: {e}")
    return IngestResponse(doc_id=data.doc_id, chunks=n)


@router.post("/file", response_model=IngestResponse, dependencies=[Depends(verify_service_token)])
async def ingest_file(
    doc_id: str, course_id: str, file: UploadFile = File(...),
    title: str = "", author: str = ""
):
    raw = await file.read()
    n = await ingest_service.ingest_document(doc_id, course_id, title, author, raw=raw)
    return IngestResponse(doc_id=doc_id, chunks=n)


@router.delete("/{doc_id}", dependencies=[Depends(verify_service_token)])
async def delete_doc(doc_id: str):
    ingest_service.delete_document(doc_id)
    return {"message": f"{doc_id} deleted"}
