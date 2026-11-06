"""Deterministic extraction from CSV line listings.

One ``ExtractResponse`` per CSV data row. Every present cell becomes a
field with ``origin="deterministic"`` and ``confidence=1.0``; its
``source_span`` quotes the cell text at its approximate offset in the
raw input.
"""

import csv
import re

from extractor import __version__
from extractor.schemas import (
    FIELD_KEYS,
    ExtractResponse,
    FieldDraft,
    FieldOrigin,
    SourceSpan,
)

COLUMN_ALIASES: dict[str, str] = {
    "patient": "patient",
    "patient initials": "patient",
    "drug": "product",
    "product": "product",
    "suspect product": "product",
    "adverse event": "event",
    "reaction": "event",
    "event": "event",
    "reporter": "reporter",
    "reported by": "reporter",
    "serious": "seriousness",
    "seriousness": "seriousness",
    "expected": "expectedness",
    "expectedness": "expectedness",
}


def _rows(text: str) -> list[tuple[int, str, list[str]]]:
    """Split raw CSV text into (char offset, raw line, cells) triples."""
    # shortcut: one CSV row per line; upgrade if quoted multi-line
    # cells ever appear in the input
    rows: list[tuple[int, str, list[str]]] = []
    pos = 0
    for line in text.splitlines():
        cells = next(csv.reader([line]), [])
        if any(cell.strip() for cell in cells):
            rows.append((pos, line, cells))
        pos += len(line) + 1
    return rows


def _cell_span(line: str, line_start: int, cell: str) -> SourceSpan | None:
    """Locate ``cell`` inside its raw line, falling back to line start."""
    if not cell:
        return None
    idx = line.find(cell)
    start = line_start + idx if idx >= 0 else line_start
    return SourceSpan(start=start, end=start + len(cell), quote=cell)


def build_narrative(
    values: dict[str, str], *, origin: FieldOrigin, confidence: float
) -> FieldDraft:
    """Assemble a narrative draft from extracted field values."""
    v = {key: values.get(key, "").strip() for key in FIELD_KEYS}
    # narrative reads naturally when the event phrase doesn't repeat
    # its own label prefix, e.g. "Event: Patient experienced ..."
    event = re.sub(
        r"^patient\s+(?:experienced\s+)?", "", v["event"], flags=re.IGNORECASE
    )
    clauses: list[str] = []
    if v["patient"]:
        clauses.append(f"Patient {v['patient']}")
    if event:
        clauses.append(f"experienced {event}")
    if v["product"]:
        clauses.append(f"after taking {v['product']}")
    if v["reporter"]:
        clauses.append(f"reported by {v['reporter']}")
    body = " ".join(clauses) or "No case details were extracted"
    seriousness = v["seriousness"] or "unknown"
    expectedness = v["expectedness"] or "unknown"
    return FieldDraft(
        value=(
            f"{body}. Seriousness: {seriousness}. Expectedness: {expectedness}."
        ),
        confidence=confidence,
        origin=origin,
    )


def extract_csv(text: str) -> list[ExtractResponse]:
    """Parse CSV text and return one deterministic draft per data row."""
    rows = _rows(text)
    if not rows:
        return []
    columns: dict[int, str] = {}
    for idx, name in enumerate(rows[0][2]):
        key = COLUMN_ALIASES.get(name.strip().lower())
        if key is not None:
            columns[idx] = key
    drafts: list[ExtractResponse] = []
    for line_start, line, cells in rows[1:]:
        fields: dict[str, FieldDraft] = {}
        values: dict[str, str] = {}
        for key in FIELD_KEYS:
            col = next((i for i, k in columns.items() if k == key), None)
            if col is None or col >= len(cells):
                fields[key] = FieldDraft(
                    value="", confidence=0.0, origin="deterministic"
                )
                continue
            cell = cells[col].strip()
            values[key] = cell
            fields[key] = FieldDraft(
                value=cell,
                confidence=1.0,
                source_span=_cell_span(line, line_start, cell),
                origin="deterministic",
            )
        drafts.append(
            ExtractResponse(
                doc_type="csv_line_listing",
                extractor_version=__version__,
                fields=fields,
                narrative=build_narrative(
                    values, origin="deterministic", confidence=1.0
                ),
            )
        )
    return drafts
