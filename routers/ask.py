import logging
from time import perf_counter
from fastapi import APIRouter, Depends
from core.config import settings
from core.security import verify_service_token
from schemas.rag import AskRequest, AskResponse, Source
from service.generation_service import generate_answer
from service.retrieval_service import retrieve

router = APIRouter(prefix="/ask", tags=["ask"])
logger = logging.getLogger(__name__)


@router.post("", response_model=AskResponse, dependencies=[Depends(verify_service_token)])
async def ask(data: AskRequest):
    started = perf_counter()
    try:
        chunks = retrieve(
            data.query, data.course_id, data.top_k, data.mode,
            author=data.author,
            publish_date_from=data.publish_date_from,
            publish_date_to=data.publish_date_to,
            keywords=data.keywords,
        )
        answer = await generate_answer(data.query, chunks)
    except Exception:
        logger.exception("RAG request failed")
        raise

    logger.info(
        "RAG request completed mode=%s retrieved=%d provider=%s duration_ms=%d",
        data.mode or settings.SEARCH_MODE,
        len(chunks),
        settings.LLM_PROVIDER,
        round((perf_counter() - started) * 1000),
    )
    sources = [
        Source(
            doc=c.get("doc_title", c.get("doc_id", "")),
            page=int(c.get("page", 0)),
            fragment=c["text"][:500],
        )
        for c in chunks
    ]
    return AskResponse(answer=answer, sources=sources)
