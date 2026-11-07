"""Trained-classifier stage for decision fields (seriousness, expectedness).

Loads the joblib bundle produced by ``python -m extractor.train_models``
(TF-IDF + logistic regression per decision field) and overrides a case's
decision fields when the model is confident enough. Model confidence is
capped at 0.9 so it never outranks deterministic extraction.

Every failure degrades to "no overrides" - enrichment must never break
extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from extractor.schemas import ExtractResponse, FieldDraft

# Module-level, resolved lazily on first use so tests can monkeypatch it.
_ARTEFACT: Path | None = None

_MIN_PROBABILITY = 0.6
_MAX_CONFIDENCE = 0.9


def _artefact_path() -> Path:
    """Resolve the artefact path once, on first use."""
    global _ARTEFACT
    if _ARTEFACT is None:
        _ARTEFACT = (
            Path(__file__).resolve().parent
            / "artefacts"
            / "decision_models.joblib"
        )
    return _ARTEFACT


def _load_bundle() -> dict[str, Any] | None:
    """Load the trained bundle, or None when absent/corrupt."""
    path = _artefact_path()
    if not path.exists():
        return None
    try:
        import joblib

        bundle = joblib.load(path)
    except Exception as exc:
        logger.warning("could not load model artefact: {}", exc)
        return None
    return bundle if isinstance(bundle, dict) else None


def available() -> bool:
    """Report whether trained classifier artefacts exist."""
    return _artefact_path().exists()


def apply(case: ExtractResponse, text: str) -> list[str]:
    """Override decision fields with model predictions; return keys used."""
    try:
        bundle = _load_bundle()
        if bundle is None or not text.strip():
            return []
        used: list[str] = []
        for key in ("seriousness", "expectedness"):
            pipe: Any = bundle.get(key)
            if not hasattr(pipe, "predict_proba"):
                continue
            proba = pipe.predict_proba([text])[0]
            best = int(proba.argmax())
            pred = str(pipe.classes_[best])
            prob = float(proba[best])
            if prob < _MIN_PROBABILITY:
                continue
            case.fields[key] = FieldDraft(
                value=pred,
                # Cap at 0.9: model confidence must not outrank the
                # deterministic extractor's exact-match confidence.
                confidence=min(round(prob, 2), _MAX_CONFIDENCE),
                origin="model",
                source_span=None,
            )
            used.append(key)
        return used
    except Exception as exc:
        logger.warning("model stage failed: {}", exc)
        return []
