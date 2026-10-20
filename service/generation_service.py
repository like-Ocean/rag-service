import httpx
from core.config import settings

SYSTEM_PROMPT = settings.RAG_PROMPT


def build_prompt(query: str, chunks: list[dict]):
    context = "\n\n".join(
        f"[{c.get('doc_title', c.get('doc_id'))}, стр. {c.get('page')}]: {c['text'][:1200]}"
        for c in chunks
    )
    return SYSTEM_PROMPT.format(context=context or "(контекст пуст)", query=query)


async def _ask_ollama(prompt: str):
    async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


async def _ask_openrouter(prompt: str):
    async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
            json={"model": settings.OPENROUTER_MODEL,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def generate_answer(query: str, chunks: list[dict]):
    if not chunks:
        return "У меня недостаточно информации"
    prompt = build_prompt(query, chunks)
    if settings.LLM_PROVIDER == "openrouter":
        return await _ask_openrouter(prompt)
    return await _ask_ollama(prompt)
