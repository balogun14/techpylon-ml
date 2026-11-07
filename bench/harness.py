"""Regression bench: score the extractor against the gold seed set.

Loads ``bench/gold_cases.json``, extracts each case with the current
heuristics/deterministic paths, and scores field-level agreement. The
aggregate report is written to ``bench/report.json``. Run as a module
(``python -m bench.harness``) it exits 1 when overall acceptance is
below the threshold or any structural check fails.
"""

import json
import re
import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from extractor.config import settings
from extractor.deterministic import extract_csv
from extractor.heuristic import extract_text
from extractor.schemas import FIELD_KEYS

BENCH_DIR = Path(__file__).resolve().parent
GOLD_PATH = BENCH_DIR / "gold_cases.json"
REPORT_PATH = BENCH_DIR / "report.json"

_TRAILING_PUNCT = re.compile(r"[\s.,;:]+$")


class BenchCase(BaseModel):
    """One annotated gold case from the seed set."""

    id: str
    doc_type: str
    text: str
    expected: dict[str, str] | None = None
    note: str | None = None


class BenchSuite(BaseModel):
    """The gold file: a description plus its annotated cases."""

    description: str
    cases: list[BenchCase]


class FieldResult(BaseModel):
    """Comparison of one predicted field against its gold value."""

    expected: str
    actual: str
    match: bool
    confidence: float
    needs_review: bool


class CaseResult(BaseModel):
    """Per-case outcome for one gold case.

    Free-text cases carry per-``FieldResult`` comparisons; CSV line
    listings carry named structural checks instead of gold values.
    """

    case_id: str
    doc_type: str
    passed: bool
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    structural: dict[str, bool] = Field(default_factory=dict)


class BenchReport(BaseModel):
    """Aggregate bench results across the seed set."""

    total_cases: int
    field_totals: dict[str, int]
    field_matches: dict[str, int]
    field_acceptance: dict[str, float]
    overall_acceptance: float
    needs_review_correct: int
    needs_review_total: int
    structural_ok: bool
    threshold: float
    passed: bool
    cases: list[CaseResult]


def normalize_value(value: str) -> str:
    """Casefold, collapse whitespace and strip trailing punctuation."""
    collapsed = re.sub(r"\s+", " ", value).strip()
    return _TRAILING_PUNCT.sub("", collapsed).casefold()


def _score_free_text(case: BenchCase) -> CaseResult:
    """Extract one free-text case and compare each field to its gold."""
    resp = extract_text(case.text)
    expected = case.expected or {}
    fields = {
        key: FieldResult(
            expected=expected.get(key, ""),
            actual=draft.value,
            match=normalize_value(draft.value)
            == normalize_value(expected.get(key, "")),
            confidence=draft.confidence,
            needs_review=draft.needs_review,
        )
        for key, draft in resp.fields.items()
        if key in FIELD_KEYS
    }
    return CaseResult(
        case_id=case.id,
        doc_type=case.doc_type,
        passed=all(f.match for f in fields.values()),
        fields=fields,
    )


def _check_csv(case: BenchCase) -> CaseResult:
    """Structurally validate the CSV line-listing case (no gold values)."""
    data_rows = max(
        len([ln for ln in case.text.splitlines() if ln.strip()]) - 1, 0
    )
    drafts = extract_csv(case.text)
    checks = {
        "one_draft_per_data_row": len(drafts) == data_rows,
        "deterministic_confidence": all(
            field.origin == "deterministic" and field.confidence == 1.0
            for draft in drafts
            for field in draft.fields.values()
        ),
    }
    return CaseResult(
        case_id=case.id,
        doc_type=case.doc_type,
        passed=all(checks.values()),
        structural=checks,
    )


def run_bench(threshold: float = 0.9) -> BenchReport:
    """Run every gold case and write ``bench/report.json``.

    Parameters
    ----------
    threshold : float
        Minimum overall field acceptance for the report to pass.

    Returns
    -------
    BenchReport
        Per-field acceptance, per-case results and needs-review stats.
    """
    suite = BenchSuite.model_validate(
        json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    )
    cases = [
        _check_csv(c)
        if c.doc_type == "csv_line_listing"
        else _score_free_text(c)
        for c in suite.cases
    ]
    field_matches = {key: 0 for key in FIELD_KEYS}
    field_totals = {key: 0 for key in FIELD_KEYS}
    needs_review_correct = 0
    needs_review_total = 0
    for result in cases:
        for key, field in result.fields.items():
            field_totals[key] += 1
            needs_review_total += 1
            if field.match:
                field_matches[key] += 1
            # the flag is correct when it marks exactly the fields a
            # reviewer should see: wrong values or low confidence
            should_review = not field.match or (
                field.confidence < settings.needs_review_floor
            )
            if field.needs_review == should_review:
                needs_review_correct += 1
    field_acceptance = {
        key: field_matches[key] / field_totals[key]
        for key in FIELD_KEYS
        if field_totals[key]
    }
    predictions = sum(field_totals.values())
    overall = sum(field_matches.values()) / predictions if predictions else 0.0
    structural_ok = all(
        all(result.structural.values()) for result in cases if result.structural
    )
    report = BenchReport(
        total_cases=len(cases),
        field_totals=field_totals,
        field_matches=field_matches,
        field_acceptance=field_acceptance,
        overall_acceptance=overall,
        needs_review_correct=needs_review_correct,
        needs_review_total=needs_review_total,
        structural_ok=structural_ok,
        threshold=threshold,
        passed=overall >= threshold and structural_ok,
        cases=cases,
    )
    REPORT_PATH.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "bench: cases={} overall={:.3f} structural_ok={} passed={}",
        len(cases),
        overall,
        structural_ok,
        report.passed,
    )
    return report


if __name__ == "__main__":
    sys.exit(0 if run_bench().passed else 1)
