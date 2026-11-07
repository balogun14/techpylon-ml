"""Document-to-text readers: PDF, DOCX, images, CSV/TXT -> plain text.

PDFs are read from their text layer; a PDF without one (scanned document)
is rasterised and OCR'd. Image uploads are OCR'd directly.
"""

import io

from docx import Document
from loguru import logger
from pypdf import PdfReader

from extractor.ocr import IMAGE_EXTS, ocr_image, ocr_pdf


class DocReadError(Exception):
    """Raised when a document has no extractable text."""


def read_document(filename: str, data: bytes) -> str:
    """Decode an uploaded document to text by extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _pdf_text_layer(data)
        if text:
            logger.debug("pdf text layer: chars={}", len(text))
            return text
        return ocr_pdf(data)
    if name.endswith(".docx"):
        return _docx(data)
    if name.endswith(IMAGE_EXTS):
        return ocr_image(data)
    return data.decode("utf-8", errors="replace")


def _pdf_text_layer(data: bytes) -> str:
    """Extract the native text layer of a PDF, empty string if none."""
    try:
        pages = [
            page.extract_text() or ""
            for page in PdfReader(io.BytesIO(data)).pages
        ]
    except Exception:
        return ""  # malformed or scanned; the OCR path may still read it
    return "\n".join(pages).strip()


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
