"""Tests for enrichment stages: coding, reference, PII, multilang, API."""

import sys
import types

import pytest
from fastapi.testclient import TestClient

from extractor.coding import code_case
from extractor.heuristic import extract_text
from extractor.pii import scan
from extractor.reference import refine_expectedness
from extractor.schemas import ExtractResponse
from extractor.service import app

client = TestClient(app)


def _case(text: str) -> ExtractResponse:
    """Extract one free-text case through the heuristic stage."""
    return extract_text(text)


# --- coding ---


def test_coding_exact() -> None:
    """Exact dictionary hits code with confidence 1.0."""
    case = _case("Medication: aspirin\nReaction: headache\n")
    terms = {t.field: t for t in code_case(case)}
    assert terms["product"].code == "PT-DRG-004"
    assert terms["product"].term == "aspirin"
    assert terms["product"].method == "exact"
    assert terms["product"].confidence == 1.0
    assert terms["event"].code == "PT-EVT-001"
    assert terms["event"].method == "exact"


def test_coding_fuzzy() -> None:
    """Fuzzy matches carry the ratio as confidence and the PT code."""
    case = _case("Drug: Amoxicillin 500mg\nReaction: severe rash\n")
    terms = {t.field: t for t in code_case(case)}
    product = terms["product"]
    assert product.method == "fuzzy"
    assert product.term == "amoxicillin"
    assert product.confidence == 1.0  # token subset -> ratio 100
    event = terms["event"]
    assert event.method == "fuzzy"
    assert event.term == "rash"
    assert 0.85 <= event.confidence <= 1.0


def test_coding_internal_alias() -> None:
    """Curated aliases (e.g. acetaminophen) code exactly."""
    case = _case("Medication: Acetaminophen\nReaction: nausea\n")
    terms = {t.field: t for t in code_case(case)}
    product = terms["product"]
    assert product.term == "paracetamol"
    assert product.code == "PT-DRG-001"
    assert product.method == "exact"


def test_coding_adapt_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aliases from adapt.load_aliases code via the alias method."""
    fake = types.ModuleType("extractor.adapt")
    fake.load_aliases = lambda: {"paralemol": "paracetamol"}
    monkeypatch.setitem(sys.modules, "extractor.adapt", fake)
    case = _case("Medication: Paralemol\nReaction: nausea\n")
    terms = {t.field: t for t in code_case(case)}
    product = terms["product"]
    assert product.term == "paracetamol"
    assert product.method == "alias"
    assert product.code == "PT-DRG-001"


def test_coding_without_adapt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or empty adapt module degrades to exact/fuzzy coding."""
    monkeypatch.setitem(sys.modules, "extractor.adapt", None)
    case = _case("Medication: aspirin\nReaction: headache\n")
    terms = {t.field: t for t in code_case(case)}
    assert terms["product"].method == "exact"

    bare = types.ModuleType("extractor.adapt")  # no load_aliases
    monkeypatch.setitem(sys.modules, "extractor.adapt", bare)
    assert code_case(case) == code_case(case)


def test_coding_skips_empty_fields() -> None:
    """Empty field values produce no coded terms."""
    case = _case("Reaction: mild rash\n")
    case.fields["event"].value = ""
    terms = code_case(case)
    assert terms == []


# --- reference expectedness refinement ---


def test_reference_expected_up() -> None:
    """Unknown expectedness is raised to expected for labelled pairs."""
    case = _case(
        "Patient: Ngozi Eze\n"
        "Product: Artesunate 100mg\n"
        "Reaction: severe vomiting\n"
        "Reported by: Pharm. T. Bello\n"
    )
    before = case.fields["expectedness"]
    assert before.value == "unknown"
    assert before.confidence < 0.75
    assert refine_expectedness(case) is True
    field = case.fields["expectedness"]
    assert field.value == "expected"
    assert field.confidence == 0.85
    assert field.origin == before.origin


def test_reference_unexpected_down() -> None:
    """Unknown expectedness drops to unexpected for unlabelled pairs."""
    case = _case(
        "Patient: Emeka Nwosu, 56M\n"
        "Drug: Artemether/Lumefantrine\n"
        "Event: Jaundice and dark urine noticed on day 3\n"
        "Reported by: Dr. A. Hassan\n"
    )
    before = case.fields["expectedness"]
    assert before.value == "unknown"
    assert refine_expectedness(case) is True
    field = case.fields["expectedness"]
    assert field.value == "unexpected"
    assert field.confidence == 0.7
    assert field.origin == before.origin


def test_reference_never_overrides_high_confidence() -> None:
    """A 0.75-confidence draft (heuristic keyword) is left alone."""
    case = _case(
        "Patient: Jane Doe\n"
        "Drug: Amoxicillin 500mg\n"
        "Event: severe rash\n"
        "Reported by: Dr. K. Adeyemi\n"
        "Rash is a known labelled reaction, considered expected.\n"
    )
    before = case.fields["expectedness"]
    assert before.value == "expected"
    assert before.confidence == 0.75
    assert refine_expectedness(case) is False
    assert case.fields["expectedness"].value == "expected"
    assert case.fields["expectedness"].confidence == 0.75


def test_reference_untouched_without_coding() -> None:
    """Uncoded product or event means no refinement."""
    case = _case("Reaction: mild rash\n")
    case.fields["product"].value = ""
    assert refine_expectedness(case) is False


# --- PII ---


def test_pii_offsets() -> None:
    """Every finding's offsets slice its value out of the text."""
    text = (
        "Contact ada.obi@example.com or +234 803 123 4567. "
        "NIN: 12345678901. DOB: 14/03/1988. Alt: 08031234567."
    )
    findings = scan(text)
    assert findings == sorted(findings, key=lambda f: f.start)
    by_kind: dict[str, object] = {}
    for finding in findings:
        assert text[finding.start : finding.end] == finding.value
        by_kind.setdefault(finding.kind, finding)
    assert by_kind["email"].value == "ada.obi@example.com"  # type: ignore
    assert by_kind["phone"].value == "+234 803 123 4567"  # type: ignore
    assert by_kind["national_id"].value == "12345678901"  # type: ignore
    assert by_kind["date_of_birth"].value == "14/03/1988"  # type: ignore


def test_pii_dedup_overlap() -> None:
    """Bare digit runs do not duplicate labelled NIN/phone matches."""
    text = "NIN 12345678901 for verification"
    matches = [f for f in scan(text) if f.value == "12345678901"]
    assert len(matches) == 1
    assert matches[0].kind == "national_id"

    findings = scan("Call 08031234567 now")
    assert [f.kind for f in findings] == ["phone"]
    assert findings[0].value == "08031234567"


def test_pii_no_hits() -> None:
    """Clean text yields no findings."""
    assert scan("Patient recovered well after treatment.") == []


# --- multilang ---


def test_multilang_french() -> None:
    """French labelled reports extract all fields and seriousness."""
    case = _case(
        "Patient : Marie Dupont\n"
        "Médicament : paracétamol\n"
        "Effet indésirable : nausée\n"
        "Rapporté par : Dr. Martin\n"
        "hospitalisation\n"
    )
    assert case.fields["patient"].value == "Marie Dupont"
    assert case.fields["product"].value == "paracétamol"
    assert case.fields["event"].value == "nausée"
    assert case.fields["reporter"].value == "Dr. Martin"
    assert case.fields["seriousness"].value == "serious"


def test_multilang_serious_terms() -> None:
    """French and Portuguese seriousness keywords classify serious."""
    assert (
        _case("Patient: X\nReaction: rash\ndécès\n").fields["seriousness"].value
        == "serious"
    )
    assert (
        _case("Patient: X\nReaction: rash\nóbito\n").fields["seriousness"].value
        == "serious"
    )


# --- service wiring ---


def test_service_codes_and_scans_pii() -> None:
    """Free-text uploads carry non-empty coded_terms and pii."""
    text = (
        "Patient: Test User\n"
        "Medication: paracetamol 500mg\n"
        "Reaction: headache\n"
        "Reported by: Dr. Who\n"
        "Contact test@example.com, DOB 01/02/1990.\n"
    )
    resp = client.post(
        "/extract",
        files={"file": ("report.txt", text.encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    coded = data["coded_terms"]
    assert coded
    assert {t["field"] for t in coded} == {"product", "event"}
    pii = data["pii"]
    assert pii
    kinds = {p["kind"] for p in pii}
    assert "email" in kinds
    assert "date_of_birth" in kinds
