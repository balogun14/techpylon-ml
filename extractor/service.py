"""FastAPI service for the TechPylon ML extractor (port 8100).

``POST /extract`` response shapes (see ``docs/api-contract.md``):

- Free text: a single object ``{"doc_type": "free_text",
  "extractor_version": "0.1.0", "fields": {...}, "narrative": {...}}``
  carrying all six field keys.
- CSV (filename ends with ``.csv``, case-insensitive, or content type
  mentions csv): a wrapper object ``{"doc_type": "csv_line_listing",
  "extractor_version": "0.1.0", "cases": [<response>, ...]}`` with one
  entry per CSV data row and no top-level ``fields``/``narrative``.
"""

from typing import Any

from fastapi import FastAPI
from loguru import logger

from extractor import __version__
from extractor.config import settings
from extractor.deterministic import extract_csv
from extractor.heuristic import extract_text
from extractor.schemas import ExtractRequest, ExtractResponse

app = FastAPI(title="TechPylon Extractor", version=__version__)


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
def extract(req: ExtractRequest) -> dict[str, Any]:
    """Extract field drafts (CSV: one case per data row)."""
    is_csv = (
        req.filename.lower().endswith(".csv")
        or "csv" in req.content_type.lower()
    )
    if is_csv:
        cases = extract_csv(req.text)
        logger.info("csv extract: file={} rows={}", req.filename, len(cases))
        return {
            "doc_type": "csv_line_listing",
            "extractor_version": __version__,
            "cases": [_single(case) for case in cases],
        }
    case = extract_text(req.text)
    logger.info("free-text extract: file={}", req.filename)
    return _single(case)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
