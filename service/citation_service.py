import re

from core.limits import MAX_ANSWER_SOURCES

CITATION = re.compile(r"\[S[0-9]+\]")


def has_valid_citations(answer: str, chunks: list[dict]) -> bool:
    citations = set(CITATION.findall(answer))
    available = {f"[S{i}]" for i in range(1, min(len(chunks), MAX_ANSWER_SOURCES) + 1)}
    return bool(citations) and citations <= available


def select_cited_chunks(answer: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Keep cited evidence and preserve positional [S#] links after filtering."""
    available = {f"[S{i}]": chunk for i, chunk in enumerate(chunks[:MAX_ANSWER_SOURCES], 1)}
    selected: list[dict] = []
    replacements: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        citation = match.group()
        if citation not in available:
            return ""
        if citation not in replacements:
            selected.append(available[citation])
            replacements[citation] = f"[S{len(selected)}]"
        return replacements[citation]

    return CITATION.sub(replace, answer), selected
