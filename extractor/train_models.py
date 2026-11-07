"""Train the seriousness/expectedness decision classifiers (#5).

Usage::

    python -m extractor.train_models

Loads ``bench/gold_cases.json`` and trains one TF-IDF + logistic-regression
pipeline per decision field ("seriousness", "expectedness"), then saves a
joblib bundle to ``extractor/artefacts/decision_models.joblib`` for the
``extractor.models`` enrichment stage.

Rerun after every bench growth: the seed set is tiny, so the artefacts are
indicative only until the bench reaches its 1,400-case target.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BENCH_PATH = Path(__file__).resolve().parents[1] / "bench" / "gold_cases.json"
ARTEFACT_PATH = (
    Path(__file__).resolve().parent / "artefacts" / "decision_models.joblib"
)
DECISION_FIELDS: tuple[str, ...] = ("seriousness", "expectedness")


def _pipeline() -> Pipeline:
    """Build the TF-IDF + balanced logistic-regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), lowercase=True, sublinear_tf=True
                ),
            ),
            (
                "clf",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def main(out: Path | None = None) -> dict[str, Any]:
    """Train one pipeline per decision field and save the joblib bundle.

    Returns the bundle that was written (useful for tests and the bench
    harness). ``out`` overrides the artefact path.
    """
    cases: list[dict[str, Any]] = json.loads(
        BENCH_PATH.read_text(encoding="utf-8")
    )["cases"]
    labelled = [c for c in cases if isinstance(c.get("expected"), dict)]
    target = out or ARTEFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_cases": len(labelled),
    }
    for key in DECISION_FIELDS:
        xs = [c["text"] for c in labelled if c["expected"].get(key)]
        ys = [c["expected"][key] for c in labelled if c["expected"].get(key)]
        pipe = _pipeline()
        pipe.fit(xs, ys)
        accuracy = float(pipe.score(xs, ys))
        logger.info("{}: train accuracy={:.2f} n={}", key, accuracy, len(ys))
        print(f"{key}: train accuracy {accuracy:.2f} on {len(ys)} cases")
        bundle[key] = pipe

    import joblib

    joblib.dump(bundle, target)
    print(
        "Caveat: trained on the tiny seed set "
        f"({len(labelled)} cases) - indicative only; the real bench "
        "target is 1,400 cases. Retrain after every bench growth."
    )
    return bundle


if __name__ == "__main__":
    main()
