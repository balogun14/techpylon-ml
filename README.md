# techpylon-extractor

ML extraction service for TechPylon (FastAPI, port 8100). Turns raw adverse-event
documents — PDF (text layer or scanned/OCR), DOCX, CSV line listings, plain text,
images — into structured, coded, review-ready case drafts. Every extracted field
carries a confidence score, the source span it came from, and its provenance.
Nothing files itself: drafts go to the reviewer gate (`techpylon-web` via
`techpylon-api`).

Version: 0.4.0

## Pipeline

```
upload (pdf/docx/csv/txt/png/...)
  │
  ├─ readers.py      document → text: PDF text layer (pypdf), DOCX paragraphs +
  │                  tables (python-docx), UTF-8 for csv/txt
  ├─ ocr.py          scanned PDFs → PyMuPDF raster (200 DPI) → RapidOCR;
  │                  image uploads OCR'd directly. No system binaries needed.
  │
  ├─ extraction
  │   ├─ deterministic.py   CSV line listings: synonym column map, one case per
  │   │                     row, confidence 1.0, cell-level source spans
  │   └─ heuristic.py       free text: labelled-field regexes, seriousness and
  │                         expectedness as classification decisions, negation-
  │                         aware, label-boundary and sentence-boundary aware;
  │                         French/Portuguese labels via multilang.py
  │
  └─ enrichment (each stage fails open — enrichment never breaks extraction)
      ├─ models.py         trained TF-IDF + logistic-regression classifiers for
      │                    seriousness/expectedness, when artefacts exist
      │                    (origin "model", confidence capped at 0.9)
      ├─ llm_fallback.py   OpenAI-compatible LLM fills low-confidence labelled
      │                    fields; values must be verbatim source substrings
      │                    (origin "llm"); off unless TECHPYLON_LLM_API_KEY set
      ├─ reference.py      expectedness refined against reference labels
      ├─ coding.py         product/event values coded against the curated
      │                    dictionary (exact → alias → fuzzy ladder)
      └─ pii.py            email/phone/NIN/date-of-birth scan with offsets
```

## Quick start

```powershell
make install          # uv sync
make run              # uvicorn on :8100
make check-all        # format + lint + typecheck + test
```

Smoke:

```powershell
curl -F "file=@report.csv" http://localhost:8100/extract
curl -F "file=@scan.pdf"   http://localhost:8100/extract
```

Response shape (free text; CSV wraps one case per row in `cases`):

```json
{
  "doc_type": "free_text",
  "extractor_version": "0.4.0",
  "fields": {
    "patient":   { "value": "...", "confidence": 0.85, "source_span": {"start":0,"end":10,"quote":"..."},
                   "origin": "heuristic", "needs_review": false, "edited": false }
  },
  "narrative": { "...": "same FieldDraft shape" },
  "coded_terms":      [ { "field": "product", "text": "aspirin", "code": "PT-DRG-004",
                          "term": "aspirin", "method": "exact", "confidence": 1.0 } ],
  "pii":              [ { "kind": "phone", "value": "0803...", "start": 210, "end": 221 } ],
  "model_fields_used": [ "seriousness" ],
  "llm_fields_used":   null
}
```

`origin`: `deterministic` | `heuristic` | `model` | `llm` (→ `reviewer` after the
gate). `needs_review` auto-flags below the confidence floor (0.60). Fields whose
confidence cannot be trusted are flagged, never silently passed. Scanned
documents with no readable text return **422**.

## Quality system

The bench is the contract between extraction changes and reality.

| Command | What it does |
|---|---|
| `make bench` | Runs the 12-case annotated gold set (`bench/gold_cases.json`); per-field acceptance; fails below 0.9; writes `bench/report.json` |
| `make calibrate` | Buckets bench predictions by confidence and computes empirical accuracy per bin → `bench/calibration.json`; `calibrated()` maps raw confidence through it |
| `make train-models` | Trains the seriousness/expectedness classifiers on the bench → `extractor/artefacts/decision_models.joblib` |
| `make validate` | bench + calibration + `artefacts/MODEL_CARD.md` + `artefacts/validation-report.json` — the per-release validation file |

**Growing the bench:** add a case to `gold_cases.json` — `id`, `doc_type`,
`text`, and `expected` (per-field gold values). Gold annotations must come from a
credentialed safety reviewer; the file header records that policy. Seed set is
12; the product target is 1,400 annotated cases across 6 therapeutic areas.
Retrain (`make train-models`) after every bench growth — the model card tracks it.

## Data sources

The coding dictionary and reference labels ship as **curated internal files**
(`extractor/dictionary.py`, `extractor/reference_labels.py`) so the service runs
with zero licensed dependencies. They are placeholders until the licensed/free
sources below are wired — do not present them as MedDRA.

| Source | Status | Role | Licensing |
|---|---|---|---|
| **RxNorm** (NLM RxNav API) | wired — `extractor/sources.py` | normalises drug names, cross-checks the drug dictionary | free, no key |
| **openFDA FAERS** | wired — `extractor/sources.py` | real-world drug→event report counts; data-driven expectedness priors and future classifier training data | free, no key, rate-limited |
| **DailyMed SPL** | wired — `extractor/sources.py` | structured product labels → labelled (expected) event lists | free, no key |
| **NAFDAC registered products** | ingestion ready — `extractor/nafdac.py` (CSV) | Nigerian product dictionary extension; safety alerts feed | no public bulk API — CSV export from the NAFDAC product registry |
| **MedDRA** | licensing decision open (PRD) | the regulator-expected AE coding dictionary | commercial license (MSSO/UMC) |
| **DrugBank** | not used | drug indications/DDI enrichment | commercial (free academic tier exists) |
| **WHO UMC / VigiBase** | not used | global ICSR signal statistics | via national centre (NAFDAC) |

Cached responses live under `data/source_cache/` (git-ignored). Source calls are
fail-open: a network error degrades to the internal dictionary, never to a
failed extraction.

### Training data position (honest version)

The trained classifiers and the bench are only as good as their annotations.
Real model training for this product draws on three layers, in order of trust:

1. **Reviewer corrections** captured by the API (`/training`) — the customer's
   own edge cases; adapted into aliases by `extractor/adapt.py`.
2. **Credentialed gold annotations** — the bench; grows toward 1,400.
3. **Public regulatory data** (FAERS event distributions, DailyMed labels,
   NAFDAC alerts) — weak supervision: priors and expectedness signals, never
   gold truth.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `TECHPYLON_LLM_API_KEY` | unset | enables the LLM fallback stage |
| `TECHPYLON_LLM_BASE_URL` | `https://api.openai.com/v1` | any OpenAI-compatible endpoint |
| `TECHPYLON_LLM_MODEL` | `gpt-4o-mini` | fallback model |
| (port) | 8100 | `make run` / uvicorn |

## Layout

```
extractor/
  service.py        FastAPI app: /extract, /healthz, enrichment orchestration
  readers.py        document → text (pdf/docx/csv/txt)
  ocr.py            scanned PDF + image OCR (RapidOCR + PyMuPDF)
  deterministic.py  CSV line-listing extraction + narrative builder
  heuristic.py      free-text extraction (EN core; FR/PT via multilang)
  multilang.py      additional label patterns and seriousness terms
  models.py         trained decision-field classifiers (apply/available)
  train_models.py   classifier training CLI
  llm_fallback.py   LLM stage for low-confidence fields
  coding.py         coding ladder against the dictionary
  dictionary.py     curated internal drug/event dictionary (PT-style codes)
  reference.py      expectedness refinement from reference labels
  reference_labels.py  curated product → expected-event lists
  sources.py        RxNorm, openFDA FAERS + DailyMed SPL clients (cached, fail-open)
  nafdac.py         NAFDAC registered-products CSV ingestion
  adapt.py          reviewer-correction → customer alias learning
  calibrate.py      confidence calibration
  artefacts.py      model card + validation report generation
  schemas.py        pydantic contract models (FieldDraft, ExtractResponse, ...)
  config.py         settings
bench/
  gold_cases.json   annotated bench (gold = credentialed reviewer)
  harness.py        run_bench() + CLI
tests/              pytest suite (bench, enrich, models/llm, extraction)
```

## Development

Python 3.12, uv-managed. `make check-all` before every push: ruff format +
ruff (line 80, numpy docstrings), mypy `--strict`, pytest. Trusted boundaries
(API payloads) are pydantic-validated. Generated files (`bench/report.json`,
`bench/calibration.json`, `extractor/artefacts/`, `customer_aliases.json`,
`data/source_cache/`) are git-ignored — regenerate with `make validate`.

## Known limitations

- Heuristics are regex-based: they are the measurable baseline, not the ceiling.
  The correction loop (adapt.py) and bench growth are what move acceptance.
- OCR fidelity on real-world phone photos is unproven — pilot intake should
  still prefer structured channels; the reviewer gate is the backstop.
- The internal dictionary is not MedDRA; regulator submissions must code
  against the licensed dictionary once the MedDRA decision lands.
- Calibration is indicative at seed-set size; do not quote it as a validated
  sensitivity/specificity figure.
