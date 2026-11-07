# techpylon-extractor

ML extraction service for TechPylon (FastAPI, port 8100). Turns uploaded
documents into `FieldDraft` extractions per `docs/api-contract.md`:
deterministic one-case-per-row extraction for CSV line listings, regex
heuristics for free text, PDF text-layer and DOCX paragraph/table reading
(`extractor/readers.py`). Scanned PDFs (no text layer) and image uploads
(png/jpg/tiff) are rasterised and OCR'd (`extractor/ocr.py`, RapidOCR +
PyMuPDF — pip-only, no system binaries); OCR fidelity is lower than a native
text layer, so the reviewer gate matters more on those.

Run with `make install` then `make run`; extract via
`POST /extract` with a multipart `file` field.
