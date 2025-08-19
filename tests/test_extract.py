"""Tests for the TechPylon extractor (deterministic, heuristic, API)."""

from fastapi.testclient import TestClient

from extractor.deterministic import extract_csv
from extractor.heuristic import extract_text
from extractor.schemas import (
    FIELD_KEYS,
    ExtractResponse,
    FieldDraft,
    SourceSpan,
)
from extractor.service import app

CSV_TEXT = (
    "Patient Initials,Suspect Product,Adverse Event,Reported By,"
    "Serious,Expected\n"
    "JD,Aspirin,Headache,Alice,yes,expected\n"
    "MK,Ibuprofen,Nausea,Bob,no,unexpected\n"
)

FREE_TEXT = (
    "Patient name: Jane Doe\n"
    "Medication: Aspirin\n"
    "Reaction: Headache\n"
    "Reported by: Dr. Smith\n"
    "The event was fatal.\n"
)

client = TestClient(app)


def test_csv_messy_headers() -> None:
    """Messy CSV headers map to field keys, one draft per row."""
    drafts = extract_csv(CSV_TEXT)
    assert len(drafts) == 2
    first = drafts[0]
    assert first.doc_type == "csv_line_listing"
    assert first.extractor_version == "0.1.0"
    assert set(first.fields) == set(FIELD_KEYS)
    patient = first.fields["patient"]
    assert patient.value == "JD"
    assert patient.confidence == 1.0
    assert patient.origin == "deterministic"
    assert patient.source_span is not None
    span = patient.source_span
    assert CSV_TEXT[span.start : span.end] == "JD"
    assert all(f.confidence == 1.0 for f in first.fields.values())
    assert "Headache" in first.narrative.value
    assert first.narrative.origin == "deterministic"


def test_free_text_source_spans() -> None:
    """Labelled fields carry real offsets matching their quote."""
    resp = extract_text(FREE_TEXT)
    assert resp.doc_type == "free_text"
    patient = resp.fields["patient"]
    assert patient.value == "Jane Doe"
    assert patient.source_span is not None
    span = patient.source_span
    assert FREE_TEXT[span.start : span.end] == "Jane Doe"
    assert resp.fields["product"].value == "Aspirin"
    assert resp.fields["event"].value == "Headache"
    assert resp.fields["reporter"].value == "Dr. Smith"


def test_free_text_unfound_field() -> None:
    """Missing fields are empty, confidence 0, flagged for review."""
    resp = extract_text("Reaction: Rash\n")
    reporter = resp.fields["reporter"]
    assert reporter.value == ""
    assert reporter.confidence == 0.0
    assert reporter.source_span is None
    assert reporter.needs_review is True


def test_seriousness_classification() -> None:
    """Criterion keywords classify seriousness, with the keyword quoted."""
    resp = extract_text("Patient: X\nReaction: Rash\nOutcome: fatal\n")
    serious = resp.fields["seriousness"]
    assert serious.value == "serious"
    assert serious.source_span is not None
    assert serious.source_span.quote == "fatal"
    mild = extract_text("Patient: X\nReaction: mild rash\n")
    assert mild.fields["seriousness"].value == "non-serious"


def test_needs_review_floor() -> None:
    """needs_review flips on automatically below the 0.60 floor."""
    low = FieldDraft(value="x", confidence=0.5)
    assert low.needs_review is True
    high = FieldDraft(value="x", confidence=0.9)
    assert high.needs_review is False
    forced = FieldDraft(value="x", confidence=0.5, needs_review=False)
    assert forced.needs_review is False


def test_field_draft_shape() -> None:
    """FieldDraft serializes exactly the contract keys."""
    draft = FieldDraft(
        value="Headache",
        confidence=0.87,
        source_span=SourceSpan(start=120, end=128, quote="Headache"),
        origin="deterministic",
    )
    assert set(draft.model_dump()) == {
        "value",
        "confidence",
        "source_span",
        "origin",
        "needs_review",
        "edited",
    }


def test_healthz() -> None:
    """GET /healthz reports ok."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_csv_end_to_end() -> None:
    """CSV uploads return the cases wrapper, one case per row."""
    resp = client.post(
        "/extract", json={"filename": "listing.csv", "text": CSV_TEXT}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_type"] == "csv_line_listing"
    assert body["extractor_version"] == "0.1.0"
    assert set(body) == {"doc_type", "extractor_version", "cases"}
    assert len(body["cases"]) == 2
    case = body["cases"][0]
    assert set(case) == {"doc_type", "extractor_version", "fields", "narrative"}
    draft = case["fields"]["patient"]
    assert draft["value"] == "JD"
    assert set(draft) == {
        "value",
        "confidence",
        "source_span",
        "origin",
        "needs_review",
        "edited",
    }


def test_extract_free_text_end_to_end() -> None:
    """Free-text uploads return a single response in contract shape."""
    resp = client.post(
        "/extract",
        json={
            "filename": "report_17.txt",
            "content_type": "text/plain",
            "text": FREE_TEXT,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"doc_type", "extractor_version", "fields", "narrative"}
    assert body["doc_type"] == "free_text"
    assert set(body["fields"]) == set(FIELD_KEYS)
    assert body["narrative"]["origin"] == "heuristic"


def test_csv_content_type_detection() -> None:
    """CSV is also detected from the content type."""
    resp = client.post(
        "/extract",
        json={
            "filename": "listing.dat",
            "content_type": "text/csv",
            "text": CSV_TEXT,
        },
    )
    assert resp.json()["doc_type"] == "csv_line_listing"


def test_extract_response_rejects_bad_confidence() -> None:
    """Confidence outside [0, 1] is rejected at the trust boundary."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractResponse(
            doc_type="free_text",
            extractor_version="0.1.0",
            fields={
                "patient": FieldDraft(value="x", confidence=1.5),
            },
            narrative=FieldDraft(value="n"),
        )
