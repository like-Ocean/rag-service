from fastapi import Header, HTTPException, status

from core.config import settings


async def verify_service_token(x_service_token: str | None = Header(default=None)):
    if not x_service_token or x_service_token != settings.SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )
    return True
