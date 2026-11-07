"""Tests for the TechPylon extractor: deterministic, heuristic, readers, API."""

import io

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from extractor.deterministic import extract_csv
from extractor.heuristic import extract_text
from extractor.readers import DocReadError, read_document
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
    assert first.extractor_version == "0.3.0"
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
        "/extract",
        files={"file": ("listing.csv", CSV_TEXT.encode(), "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_type"] == "csv_line_listing"
    assert body["extractor_version"] == "0.3.0"
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
        files={"file": ("report_17.txt", FREE_TEXT.encode(), "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"doc_type", "extractor_version", "fields", "narrative"}
    assert body["doc_type"] == "free_text"
    assert set(body["fields"]) == set(FIELD_KEYS)
    assert body["narrative"]["origin"] == "heuristic"


# --- document readers: PDF and DOCX ---


def _minimal_pdf(text: str) -> bytes:
    """Build a small but valid one-page PDF with a text layer."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF"
    ).encode()
    return bytes(out)


def test_read_pdf() -> None:
    """PDF text layer is extracted and flows through extraction."""
    data = _minimal_pdf("Patient: Jane Doe. Reaction: fatal rash.")
    text = read_document("scan.pdf", data)
    assert "Jane Doe" in text
    resp = extract_text(text)
    assert resp.fields["patient"].value == "Jane Doe"
    assert resp.fields["seriousness"].value == "serious"


def test_read_scanned_pdf_ocr() -> None:
    """A scanned PDF (image-only) falls back to the OCR path."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    draw.text((50, 50), "Patient: Jane Doe", fill="black", font=font)
    draw.text((50, 200), "Reaction: fatal rash", fill="black", font=font)
    png = io.BytesIO()
    img.save(png, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=1200, height=400)
    page.insert_image(fitz.Rect(0, 0, 1200, 400), stream=png.getvalue())
    scanned = doc.tobytes()

    text = read_document("scan.pdf", scanned)
    low = text.lower()
    assert "jane doe" in low
    assert "rash" in low
    resp = extract_text(text)
    assert resp.fields["patient"].value != ""
    assert resp.fields["seriousness"].value == "serious"


def test_read_image_ocr() -> None:
    """Image uploads are OCR'd directly."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1200, 200), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    draw.text((50, 60), "Patient: John Bello", fill="black", font=font)
    png = io.BytesIO()
    img.save(png, format="PNG")

    text = read_document("photo.png", png.getvalue())
    assert "bello" in text.lower()


def test_read_garbage_doc_fails_cleanly() -> None:
    """Bytes that are neither PDF nor image end as 422, not a crash."""
    with pytest.raises(DocReadError):
        read_document("scan.pdf", b"%PDF-1.4 not a real pdf")


def test_read_docx() -> None:
    """DOCX paragraphs and table cells become extractable text."""
    from docx import Document

    doc = Document()
    doc.add_paragraph("Patient: Jane Doe")
    doc.add_paragraph("Reaction: hospitalisation for rash")
    doc.add_paragraph("Reported by: Dr. Smith")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Case id"
    table.rows[0].cells[1].text = "PV-2024-0117"
    buf = io.BytesIO()
    doc.save(buf)
    text = read_document("report.docx", buf.getvalue())
    assert "Jane Doe" in text
    assert "PV-2024-0117" in text  # table cells are not lost
    resp = extract_text(text)
    assert resp.fields["patient"].value == "Jane Doe"
    assert resp.fields["seriousness"].value == "serious"
    assert resp.fields["reporter"].value == "Dr. Smith"


def test_extract_pdf_and_docx_end_to_end() -> None:
    """The service accepts pdf/docx multipart uploads directly."""
    resp = client.post(
        "/extract",
        files={
            "file": (
                "scan.pdf",
                _minimal_pdf("Patient: R. Okoye. Medication: Aspirin."),
                "application/pdf",
            )
        },
    )
    assert resp.status_code == 200
    assert resp.json()["fields"]["patient"]["value"] == "R. Okoye"

    from docx import Document

    doc = Document()
    doc.add_paragraph("Patient: B. Ade. Reaction: nausea.")
    buf = io.BytesIO()
    doc.save(buf)
    resp = client.post(
        "/extract",
        files={
            "file": ("report.docx", buf.getvalue(), "application/octet-stream")
        },
    )
    assert resp.status_code == 200
    assert resp.json()["fields"]["patient"]["value"] == "B. Ade"


def test_narrative_strips_event_label_prefix() -> None:
    """Narrative does not duplicate the event value's label prefix."""
    resp = extract_text(
        "Patient: Jane Doe\n"
        "Event: Patient experienced a severe rash five days into treatment.\n"
        "Medication: Aspirin\n"
    )
    assert "experienced Patient experienced" not in resp.narrative.value
    assert "a severe rash" in resp.narrative.value


def test_extract_response_rejects_bad_confidence() -> None:
    """Confidence outside [0, 1] is rejected at the trust boundary."""
    with pytest.raises(ValidationError):
        ExtractResponse(
            doc_type="free_text",
            extractor_version="0.3.0",
            fields={
                "patient": FieldDraft(value="x", confidence=1.5),
            },
            narrative=FieldDraft(value="n"),
        )
