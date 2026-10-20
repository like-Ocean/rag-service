import re

JUNK_HEADERS = [
    "оглавление",
    "содержание",
    "список литературы",
    "список использованных источников",
    "титульный лист",
]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*[-•·]\s+", "", text)
    return text.strip()


def drop_service_parts(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    kept: list[tuple[int, str]] = []
    for page_no, text in pages:
        head = text.strip()[:200].lower()
        if any(h in head for h in JUNK_HEADERS) and len(text.strip()) < 3000:
            continue
        kept.append((page_no, text))
    return kept


def split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[str] = []
    current = ""

    def _push():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    words = text.split(" ")
    for w in words:
        if len(current) + len(w) + 1 > chunk_size * 4 and current:
            _push()
            tail = chunks[-1][-overlap * 4 :] if chunks else ""
            current = tail + " " + w if tail else w
        else:
            current = (current + " " + w).strip() if current else w
    _push()
    _ = separators
    return [c for c in chunks if c]
