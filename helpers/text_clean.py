import re

JUNK_HEADERS = [
    "оглавление",
    "содержание",
    "список литературы",
    "список использованных источников",
    "титульный лист",
]

LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•·▪◦]\s+|\d+[.)]\s+)")
NUMBERED_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+){1,}[.)]?\s+\S")
KNOWN_SECTION_RE = re.compile(r"^\s*(?:вопросы для самоконтроля|заключение|введение)\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def drop_service_parts(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    kept: list[tuple[int, str]] = []
    for page_no, text in pages:
        head = text.strip()[:200].lower()
        if any(h in head for h in JUNK_HEADERS) and len(text.strip()) < 3000:
            continue
        kept.append((page_no, text))
    return kept


def _is_heading(line: str) -> bool:
    return bool(NUMBERED_SECTION_RE.match(line) or KNOWN_SECTION_RE.match(line))


def _logical_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    paragraph_lines: list[str] = []
    list_item_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            blocks.append(("paragraph", " ".join(paragraph_lines).strip()))
            paragraph_lines.clear()

    def flush_list_item() -> None:
        if list_item_lines:
            blocks.append(("list", " ".join(list_item_lines).strip()))
            list_item_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_list_item()
            flush_paragraph()
            continue

        if _is_heading(line):
            flush_list_item()
            flush_paragraph()
            blocks.append(("heading", line))
        elif LIST_ITEM_RE.match(line):
            flush_paragraph()
            flush_list_item()
            list_item_lines.append(line)
        elif list_item_lines:
            list_item_lines.append(line)
        else:
            paragraph_lines.append(line)

    flush_list_item()
    flush_paragraph()
    return [(kind, value) for kind, value in blocks if value]


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if current and len(current) + len(word) + 1 > max_chars:
                    pieces.append(current)
                    current = ""
                current = f"{current} {word}".strip()
            continue

        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        pieces.append(current)
    return pieces


def _pack_blocks(
    blocks: list[str], max_chars: int,
    overlap_chars: int = 0,
    max_blocks: int | None = None,
    overlap_blocks: int = 0,
    split_oversized: bool = True
) -> list[str]:
    """Pack complete logical blocks; only split a block if it exceeds the cap."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    overlap_chars = max(0, min(overlap_chars, max_chars - 1))

    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(current))
            current.clear()

    for block in blocks:
        if len(block) > max_chars:
            flush()
            if split_oversized:
                chunks.extend(_split_oversized(block, max_chars))
            else:
                chunks.append(block)
            continue

        candidate_size = len("\n\n".join([*current, block]))
        exceeds_count = max_blocks is not None and len(current) >= max_blocks
        if current and (candidate_size > max_chars or exceeds_count):
            previous = current[:]
            flush()
            if overlap_blocks:
                current.extend(previous[-overlap_blocks:])
            elif overlap_chars:
                carry: list[str] = []
                carry_limit = min(overlap_chars, max_chars - len(block) - 2)
                for previous_block in reversed(previous):
                    proposed = [previous_block, *carry]
                    if carry_limit < 1 or len("\n\n".join(proposed)) > carry_limit:
                        break
                    carry = proposed
                current.extend(carry)
        current.append(block)

    flush()
    return chunks


def _sections(blocks: list[tuple[str, str]]) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] = []
    for kind, text in blocks:
        if kind == "heading" and current:
            sections.append(current)
            current = []
        current.append(text)
    if current:
        sections.append(current)
    return sections


def split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    blocks = [value for _, value in _logical_blocks(text)]
    return _pack_blocks(blocks, max_chars=chunk_size * 4, overlap_chars=overlap * 4)


def split_by_strategy(text: str, strategy: str, chunk_size: int) -> list[str]:
    blocks_with_kind = _logical_blocks(text)
    blocks = [value for _, value in blocks_with_kind]
    max_chars = chunk_size * 4

    if strategy == "paragraphs_1":
        return blocks
    if strategy == "paragraphs_3":
        return _pack_blocks(blocks, max_chars=max_chars, max_blocks=3, split_oversized=False)
    if strategy == "sections":
        chunks: list[str] = []
        for section in _sections(blocks_with_kind):
            chunks.extend(_pack_blocks(section, max_chars=max_chars, split_oversized=False))
        return chunks
    if strategy == "token_window_no_split":
        return _pack_blocks(blocks, max_chars=max_chars, split_oversized=False)
    if strategy == "paragraph_overlap":
        return _pack_blocks(blocks, max_chars=max_chars, overlap_blocks=1, split_oversized=False)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def split_text(text: str, chunk_size: int, overlap: int, strategy: str = "semantic") -> list[str]:
    if strategy == "semantic":
        blocks = [value for _, value in _logical_blocks(text)]
        return _pack_blocks(blocks, max_chars=chunk_size * 4, overlap_chars=overlap * 4)
    if strategy == "recursive":
        return split_recursive(text, chunk_size, overlap)
    raise ValueError(f"Unknown CHUNK_STRATEGY: {strategy}")
