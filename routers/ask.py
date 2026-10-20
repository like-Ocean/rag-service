from fastapi import APIRouter, Depends

from core.security import verify_service_token
from schemas.rag import AskRequest, AskResponse, Source
from service.retrieval_service import retrieve
from service.generation_service import generate_answer

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse, dependencies=[Depends(verify_service_token)])
async def ask(data: AskRequest):
    chunks = retrieve(data.query, data.course_id, data.top_k, data.mode)
    answer = await generate_answer(data.query, chunks)
    sources = [Source(
            doc=c.get("doc_title", c.get("doc_id", "")),
            page=int(c.get("page", 0)), fragment=c["text"][:500]
        ) for c in chunks]
    return AskResponse(answer=answer, sources=sources)
