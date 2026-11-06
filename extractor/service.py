"""FastAPI service for the TechPylon ML extractor (port 8100).

``POST /extract`` accepts a multipart ``file`` field (pdf, docx, csv, txt)
and returns, per ``docs/api-contract.md``:

- Free text (and pdf/docx): a single object
  ``{"doc_type": "free_text", "extractor_version": "0.2.0",
  "fields": {...}, "narrative": {...}}`` carrying all six field keys.
- CSV (filename ends with ``.csv``, case-insensitive): a wrapper object
  ``{"doc_type": "csv_line_listing", "extractor_version": "0.2.0",
  "cases": [<response>, ...]}`` with one entry per CSV data row and no
  top-level ``fields``/``narrative``.

Documents with no extractable text layer (scanned PDFs) return 422.
"""

from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from loguru import logger

from extractor import __version__
from extractor.config import settings
from extractor.deterministic import extract_csv
from extractor.heuristic import extract_text
from extractor.readers import DocReadError, read_document
from extractor.schemas import ExtractResponse

app = FastAPI(title="TechPylon Extractor", version=__version__)

MAX_UPLOAD = 10 << 20  # 10 MiB, mirrors the Go API's read limit


def _single(case: ExtractResponse) -> dict[str, Any]:
    """Serialize one case, dropping the CSV-only ``cases`` field."""
    data = case.model_dump(mode="json")
    data.pop("cases", None)
    return data


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile) -> dict[str, Any]:
    """Extract field drafts from an uploaded document (CSV: one per row)."""
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="file exceeds 10 MiB")
    filename = file.filename or "upload.txt"
    try:
        text = read_document(filename, data)
    except DocReadError as exc:
        logger.warning("doc read failed: file={} err={}", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if filename.lower().endswith(".csv"):
        cases = extract_csv(text)
        logger.info("csv extract: file={} rows={}", filename, len(cases))
        return {
            "doc_type": "csv_line_listing",
            "extractor_version": __version__,
            "cases": [_single(case) for case in cases],
        }
    case = extract_text(text)
    logger.info("free-text extract: file={} chars={}", filename, len(text))
    return _single(case)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
