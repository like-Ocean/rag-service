import logging
import fitz
from core.config import settings

logger = logging.getLogger(__name__)

_p2t = None


def _parse_pymupdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text() or ""))
    return pages


def _get_p2t():
    global _p2t
    if _p2t is None:
        from pix2text import Pix2Text

        _p2t = Pix2Text()
    return _p2t


def _element_to_md(el: dict) -> str:
    kind = str(el.get("type", "text")).lower()
    if "formula" in kind:
        latex = el.get("latex") or el.get("text") or ""
        return f"$${latex}$$" if latex.strip() else ""
    for key in ("markdown", "text", "latex"):
        val = el.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _parse_pix2text(pdf_bytes: bytes) -> list[tuple[int, str]]:
    import numpy as np

    p2t = _get_p2t()
    pages: list[tuple[int, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA -> RGB
                img = img[:, :, :3]
            elements = p2t.recognize_page(img) or []
            md = "\n\n".join(filter(None, (_element_to_md(el) for el in elements)))
            pages.append((i, md))
    return pages


def parse_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    if settings.PDF_PARSER == "pix2text":
        try:
            return _parse_pix2text(pdf_bytes)
        except ImportError:
            logger.warning("pix2text не установлен, откат на pymupdf (pip install pix2text)")
            return _parse_pymupdf(pdf_bytes)
    return _parse_pymupdf(pdf_bytes)
