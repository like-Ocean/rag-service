from fastapi import APIRouter
from core.chroma import get_collection

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    try:
        count = get_collection().count()
        chroma = f"ok ({count} chunks)"
    except Exception as e:
        chroma = f"unavailable: {e}"
    return {"status": "ok", "chroma": chroma}
