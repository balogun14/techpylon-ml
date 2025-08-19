# techpylon-extractor

ML extraction service for TechPylon (FastAPI, port 8100). Turns raw document
text into `FieldDraft` extractions per `docs/api-contract.md`: deterministic
one-case-per-row extraction for CSV line listings, regex heuristics for free
text.

Run with `make install` then `make run`; extract via `POST /extract`.
