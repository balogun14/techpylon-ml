"""Tests for the trained-model and LLM-fallback enrichment stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from extractor import llm_fallback, models, train_models
from extractor.heuristic import extract_text
from extractor.schemas import ExtractResponse, FieldDraft
from extractor.service import app

SERIOUS_TEXT = (
    "Patient: Yemi Ogun, 70F\n"
    "Product: Lisinopril 10mg\n"
    "Event: Angio-oedema with airway compromise, life-threatening episode.\n"
    "Reported by: Dr. C. Eze, LASUTH\n"
    "Admitted to ICU; known class effect, labelled as expected.\n"
)

LLM_TEXT = (
    "Patient Jane Doe developed a rash after Aspirin. Reported by Dr. Smith."
)


@pytest.fixture()
def trained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Train on the seed set into tmp and point models at the artefact."""
    artefact = tmp_path / "decision_models.joblib"
    train_models.main(out=artefact)
    monkeypatch.setattr(models, "_ARTEFACT", artefact)
    return artefact


def _llm_case() -> ExtractResponse:
    """Case with empty/low-confidence fields for the fallback to fill."""
    return ExtractResponse(
        doc_type="free_text",
        extractor_version="0.4.0",
        fields={
            "patient": FieldDraft(value="", confidence=0.0),
            "product": FieldDraft(value="Aspirin", confidence=0.9),
            "event": FieldDraft(value="rash", confidence=0.5),
            "reporter": FieldDraft(value="", confidence=0.0),
            "seriousness": FieldDraft(value="serious", confidence=0.9),
            "expectedness": FieldDraft(value="expected", confidence=0.9),
        },
        narrative=FieldDraft(value="n"),
    )


# --- trained decision classifiers ---


def test_train_models_bundle(trained: Path) -> None:
    """Training writes a bundle with one pipeline per decision field."""
    assert trained.exists()
    bundle = train_models.main(out=trained)
    assert bundle["n_cases"] >= 10
    assert "seriousness" in bundle
    assert "expectedness" in bundle
    assert bundle["trained_at"]


def test_models_available_and_apply(trained: Path) -> None:
    """With artefacts, apply overrides seriousness confidently."""
    assert models.available() is True
    case = extract_text(SERIOUS_TEXT)
    used = models.apply(case, SERIOUS_TEXT)
    assert "seriousness" in used
    draft = case.fields["seriousness"]
    assert draft.value == "serious"
    assert draft.origin == "model"
    assert 0.6 <= draft.confidence <= 0.9


def test_models_noop_without_artefact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing artefacts: available() False and apply never raises."""
    monkeypatch.setattr(models, "_ARTEFACT", tmp_path / "missing.joblib")
    assert models.available() is False
    case = extract_text(SERIOUS_TEXT)
    assert models.apply(case, SERIOUS_TEXT) == []


def test_service_uses_model_fields(trained: Path) -> None:
    """Service-level: /extract reports model_fields_used when trained."""
    client = TestClient(app)
    resp = client.post(
        "/extract",
        files={"file": ("report.txt", SERIOUS_TEXT.encode(), "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_fields_used"]
    assert set(body["model_fields_used"]) <= {"seriousness", "expectedness"}


# --- LLM fallback ---


def test_llm_enabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """enabled() tracks TECHPYLON_LLM_API_KEY."""
    monkeypatch.delenv("TECHPYLON_LLM_API_KEY", raising=False)
    assert llm_fallback.enabled() is False
    monkeypatch.setenv("TECHPYLON_LLM_API_KEY", "sk-test")
    assert llm_fallback.enabled() is True


def test_llm_fills_empty_and_low_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canned LLM JSON fills empty/low fields with verbatim spans."""
    monkeypatch.setenv("TECHPYLON_LLM_API_KEY", "sk-test")
    answer = json.dumps(
        {
            "patient": "Jane Doe",
            "event": "a rash",
            "reporter": "Dr. Smith",
            "product": "Acetylsalicylic acid",  # not a substring -> ignored
        }
    )
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": f"```json\n{answer}\n```"}}]
        }

    monkeypatch.setattr(llm_fallback, "_post", fake_post)
    case = _llm_case()
    used = llm_fallback.apply(case, LLM_TEXT)

    assert "product" not in used  # confidence 0.9, not a target
    assert set(used) == {"patient", "event", "reporter"}
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["temperature"] == 0
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    for key in used:
        draft = case.fields[key]
        assert draft.origin == "llm"
        assert draft.confidence == 0.7
        span = draft.source_span
        assert span is not None
        assert LLM_TEXT[span.start : span.end] == span.quote


def test_llm_rejects_non_substring_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM values that are not verbatim substrings never enter the draft."""
    monkeypatch.setenv("TECHPYLON_LLM_API_KEY", "sk-test")
    answer = json.dumps({"patient": "Janet Dough", "reporter": "Dr. Smith"})
    monkeypatch.setattr(
        llm_fallback,
        "_post",
        lambda url, headers, payload: {
            "choices": [{"message": {"content": answer}}]
        },
    )
    case = _llm_case()
    used = llm_fallback.apply(case, LLM_TEXT)
    assert used == ["reporter"]
    assert case.fields["patient"].value == ""


def test_llm_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network/parse failures degrade to no fills."""
    monkeypatch.setenv("TECHPYLON_LLM_API_KEY", "sk-test")

    def boom(
        url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise RuntimeError("down")

    monkeypatch.setattr(llm_fallback, "_post", boom)
    case = _llm_case()
    assert llm_fallback.apply(case, LLM_TEXT) == []
