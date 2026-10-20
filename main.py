import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from routers import routes

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"RAG service started (env={settings.ENV}, llm={settings.LLM_PROVIDER})")
    yield
    print("RAG service stopped")


app = FastAPI(
    title="Moodle RAG Service",
    version="0.1.0",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in routes:
    app.include_router(router, prefix="/api")

os.makedirs("/tmp/rag_uploads", exist_ok=True)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_RELOAD,
        log_level=settings.APP_LOG_LEVEL.lower(),
    )
