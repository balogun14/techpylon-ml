"""OCR for scanned PDFs and images via RapidOCR (onnxruntime, bundled models).

Pipeline: scanned PDF -> PyMuPDF rasterises pages to PNG -> RapidOCR reads
each page. Image uploads are OCR'd directly. Text fidelity is lower than a
native text layer, so downstream confidence floors matter more here.
"""

from functools import lru_cache
from typing import Any

from loguru import logger

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
_RASTER_DPI = 200


@lru_cache(maxsize=1)
def _engine() -> Any:
    """Load the OCR engine once; model init costs a few seconds."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _run_ocr(png: bytes) -> list[str]:
    """OCR one PNG, returning its text lines in reading order."""
    result, _ = _engine()(png)
    if not result:
        return []
    return [line[1] for line in result]


def ocr_image(data: bytes) -> str:
    """OCR a standalone image (png/jpg/tiff/...)."""
    lines = _run_ocr(data)
    if not lines:
        raise _empty()
    return "\n".join(lines)


def ocr_pdf(data: bytes) -> str:
    """Rasterise every PDF page and OCR it (scanned documents)."""
    import fitz  # pymupdf

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise _empty(f"unreadable PDF: {exc}") from exc
    pages: list[str] = []
    with doc:
        for page in doc:
            pix = page.get_pixmap(dpi=_RASTER_DPI)
            lines = _run_ocr(pix.tobytes("png"))
            if lines:
                pages.append("\n".join(lines))
    text = "\n".join(pages).strip()
    if not text:
        raise _empty()
    logger.info("ocr: pages={} chars={}", len(pages), len(text))
    return text


def _empty(reason: str = "no readable text found by OCR") -> Exception:
    from extractor.readers import DocReadError

    return DocReadError(reason)
