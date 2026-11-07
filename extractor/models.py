"""Trained-classifier stage for decision fields (seriousness, expectedness).

Stub shipped with the enrichment pipeline; replace with the real
implementation (TF-IDF + logistic-regression artefacts produced by
``make train-models``).
"""

from extractor.schemas import ExtractResponse, FieldDraft


def available() -> bool:
    """Report whether trained classifier artefacts exist."""
    return False


def apply(case: ExtractResponse, text: str) -> list[str]:
    """Override decision fields with model predictions; return keys used."""
    return []


def _unused_draft() -> FieldDraft:  # pragma: no cover - shape reference
    return FieldDraft(value="", confidence=0.0, origin="model")
