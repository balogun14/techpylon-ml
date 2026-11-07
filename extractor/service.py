"""FastAPI service for the TechPylon ML extractor (port 8100).

``POST /extract`` accepts a multipart ``file`` field (pdf, docx, csv, txt,
images) and returns, per ``docs/api-contract.md``:

- Free text (and pdf/docx/images): a single case object.
- CSV: ``{"doc_type": "csv_line_listing", ..., "cases": [<case>, ...]}``.

After extraction, an enrichment pipeline runs on every case:

1. trained seriousness/expectedness classifiers, when artefacts exist
   (``make train-models`` -> ``models/classifier.joblib``);
2. LLM fallback for low-confidence labelled fields, when
   ``TECHPYLON_LLM_API_KEY`` is set;
3. MedDRA-style coding of product/event values;
4. PII scan of the source text.

Every enrichment stage degrades gracefully: a failure logs a warning and
leaves the draft untouched, so extraction never fails because enrichment
did.
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
    """Serialize one case, dropping nested CSV wrappers and nulls."""
    data = case.model_dump(mode="json", exclude_none=True)
    data.pop("cases", None)
    return data


def _enrich(case: ExtractResponse, text: str) -> ExtractResponse:
    """Attach coding, PII, model and LLM sections; never raise."""
    from extractor import coding, llm_fallback, models, pii, reference

    try:
        used = models.apply(case, text)
        if used:
            case.model_fields_used = used
    except Exception as exc:
        logger.warning("model stage failed: {}", exc)
    try:
        used = llm_fallback.apply(case, text)
        if used:
            case.llm_fields_used = used
    except Exception as exc:
        logger.warning("llm stage failed: {}", exc)
    try:
        case.coded_terms = coding.code_case(case)
    except Exception as exc:
        logger.warning("coding stage failed: {}", exc)
    try:
        if reference.refine_expectedness(case):
            logger.debug("expectedness refined from reference label")
    except Exception as exc:
        logger.warning("reference stage failed: {}", exc)
    try:
        case.pii = pii.scan(text)
    except Exception as exc:
        logger.warning("pii stage failed: {}", exc)
    return case


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
            "cases": [_single(_enrich(case, text)) for case in cases],
        }
    case = _enrich(extract_text(text), text)
    logger.info("free-text extract: file={} chars={}", filename, len(text))
    return _single(case)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
