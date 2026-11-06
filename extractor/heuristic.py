"""Heuristic regex extraction from free-text adverse event reports."""

import re

from extractor import __version__
from extractor.deterministic import build_narrative
from extractor.schemas import (
    FIELD_KEYS,
    ExtractResponse,
    FieldDraft,
    SourceSpan,
)

_LABELLED: dict[str, tuple[re.Pattern[str], ...]] = {
    "patient": (
        re.compile(
            r"patient(?:\s+name)?\s*[:\-]\s*(?P<v>[^\n]+)", re.IGNORECASE
        ),
    ),
    "product": (
        re.compile(
            r"(?:drug|product|medication)\s*[:\-]\s*(?P<v>[^\n]+)",
            re.IGNORECASE,
        ),
    ),
    "event": (
        re.compile(
            r"(?:adverse\s+event|reaction|event)\s*[:\-]\s*(?P<v>[^\n]+)",
            re.IGNORECASE,
        ),
        re.compile(r"experienced\s+(?P<v>[^\n]+)", re.IGNORECASE),
    ),
    "reporter": (
        re.compile(
            r"(?:reported\s+by|reporter)\s*[:\-]\s*(?P<v>[^\n]+)",
            re.IGNORECASE,
        ),
    ),
}

_SERIOUS = re.compile(
    r"hospitalis\w*|hospitaliz\w*|life[\s-]threatening|death|fatal"
    r"|disability|incapacitat\w*|congenital|birth\s+defect"
    r"|cancer\w*|malignan\w*|tumou?r|medically\s+important",
    re.IGNORECASE,
)
_EXPECTED = re.compile(
    r"\b(?:known|labelled|labeled|expected)\b", re.IGNORECASE
)
_UNEXPECTED = re.compile(r"\b(?:unexpected|new)\b", re.IGNORECASE)

_MATCH_CONFIDENCE = 0.85
_CLASSIFY_CONFIDENCE = 0.75
_UNKNOWN_CONFIDENCE = 0.5


def _clean(raw: str) -> str:
    """Trim whitespace and trailing punctuation from a captured value."""
    return raw.strip().rstrip(".,;").strip()


_NEXT_LABEL = re.compile(
    r"\s+(?=(?:patient|medication|drug|product|adverse\s+event|reaction"
    r"|event|reported\s+by|reporter|seriousness|expectedness)\s*[:\-])",
    re.IGNORECASE,
)


def _cut_at_next_label(value: str) -> str:
    """Stop a captured value where the next ``label:`` starts.

    Documents often pack several ``label: value`` pairs on one line
    ("Patient: B. Ade. Reaction: nausea."); the capture for one label
    must not swallow the next.
    """
    return _NEXT_LABEL.split(value.strip(), maxsplit=1)[0]


def _labelled_field(text: str, key: str) -> FieldDraft:
    """Match a labelled ``key: value`` pattern for one field."""
    for pattern in _LABELLED[key]:
        match = pattern.search(text)
        if match is None:
            continue
        raw = match.group("v")
        value = _clean(_cut_at_next_label(raw))
        if not value:
            continue
        start = match.start("v") + (len(raw) - len(raw.lstrip()))
        return FieldDraft(
            value=value,
            confidence=_MATCH_CONFIDENCE,
            source_span=SourceSpan(
                start=start, end=start + len(value), quote=value
            ),
        )
    return FieldDraft(value="", confidence=0.0)


def _seriousness(text: str) -> FieldDraft:
    """Classify seriousness from criterion keywords."""
    match = _SERIOUS.search(text)
    if match is None:
        return FieldDraft(value="non-serious", confidence=_CLASSIFY_CONFIDENCE)
    return FieldDraft(
        value="serious",
        confidence=_CLASSIFY_CONFIDENCE,
        source_span=SourceSpan(
            start=match.start(), end=match.end(), quote=match.group()
        ),
    )


def _expectedness(text: str) -> FieldDraft:
    """Classify expectedness from labelled hints, else unknown."""
    for pattern, value in (
        (_EXPECTED, "expected"),
        (_UNEXPECTED, "unexpected"),
    ):
        match = pattern.search(text)
        if match is not None:
            return FieldDraft(
                value=value,
                confidence=_CLASSIFY_CONFIDENCE,
                source_span=SourceSpan(
                    start=match.start(), end=match.end(), quote=match.group()
                ),
            )
    return FieldDraft(value="unknown", confidence=_UNKNOWN_CONFIDENCE)


def extract_text(text: str) -> ExtractResponse:
    """Extract all field drafts from free text with regex heuristics."""
    fields: dict[str, FieldDraft] = {
        key: _labelled_field(text, key)
        for key in ("patient", "product", "event", "reporter")
    }
    fields["seriousness"] = _seriousness(text)
    fields["expectedness"] = _expectedness(text)
    values = {key: fields[key].value for key in FIELD_KEYS}
    found = any(
        values[key] for key in ("patient", "product", "event", "reporter")
    )
    narrative = build_narrative(
        values,
        origin="heuristic",
        confidence=_MATCH_CONFIDENCE if found else 0.0,
    )
    return ExtractResponse(
        doc_type="free_text",
        extractor_version=__version__,
        fields=fields,
        narrative=narrative,
    )
