"""Document-to-text readers: PDF, DOCX, CSV/TXT -> plain text."""

import io

from docx import Document
from loguru import logger
from pypdf import PdfReader


class DocReadError(Exception):
    """Raised when a document has no extractable text layer."""


def read_document(filename: str, data: bytes) -> str:
    """Decode an uploaded document to text by extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _pdf(data)
    if name.endswith(".docx"):
        return _docx(data)
    return data.decode("utf-8", errors="replace")


def _pdf(data: bytes) -> str:
    """Extract the text layer of a PDF (scanned images need OCR: not in MVP)."""
    try:
        pages = [
            page.extract_text() or ""
            for page in PdfReader(io.BytesIO(data)).pages
        ]
    except Exception as exc:  # malformed PDF
        raise DocReadError(f"unreadable PDF: {exc}") from exc
    text = "\n".join(pages).strip()
    if not text:
        raise DocReadError(
            "PDF has no text layer (scanned image?); OCR is not in the MVP"
        )
    return text


def _docx(data: bytes) -> str:
    """Extract paragraph and table-cell text from a DOCX."""
    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocReadError(f"unreadable DOCX: {exc}") from exc
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(",".join(cells))
    text = "\n".join(parts).strip()
    if not text:
        raise DocReadError("DOCX contains no text")
    logger.debug("docx read: paragraphs={}", len(parts))
    return text
