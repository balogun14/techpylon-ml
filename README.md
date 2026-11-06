# techpylon-extractor

ML extraction service for TechPylon (FastAPI, port 8100). Turns uploaded
documents into `FieldDraft` extractions per `docs/api-contract.md`:
deterministic one-case-per-row extraction for CSV line listings, regex
heuristics for free text, PDF text-layer and DOCX paragraph/table reading
(`extractor/readers.py`). Scanned-image PDFs have no text layer and return
422 — OCR is a post-MVP item.

Run with `make install` then `make run`; extract via
`POST /extract` with a multipart `file` field (pdf, docx, csv, txt).
