"""Governance artefacts: model card and validation report (#12).

``write_artefacts`` renders the bench results into a human-readable
``MODEL_CARD.md`` and a machine-readable ``validation-report.json``
under the requested output directory (created if missing).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from extractor import __version__


def _model_card(bench: dict[str, Any]) -> str:
    """Render the markdown model card from a bench report dump."""
    acceptance: dict[str, float] = bench.get("field_acceptance", {})
    overall = float(bench.get("overall_acceptance", 0.0))
    lines = [
        "# TechPylon extractor — model card",
        "",
        f"- Version: {__version__}",
        f"- Date: {datetime.now(UTC).date().isoformat()}",
        "",
        "## Intended use",
        "",
        "Draft adverse-event fields (patient, product, event, reporter,",
        "seriousness, expectedness) from free-text reports and CSV line",
        "listings, for pharmacist reviewer confirmation. Not intended",
        "for unsupervised regulatory submission.",
        "",
        "## Bench acceptance (seed set)",
        "",
        "| field | acceptance |",
        "| --- | --- |",
    ]
    lines += [f"| {key} | {value:.3f} |" for key, value in acceptance.items()]
    lines += [
        f"| **overall** | **{overall:.3f}** |",
        "",
        "## Known limitations",
        "",
        "- Extraction is regex-heuristic only; novel layouts, label",
        'wording or negations ("no hospitalisation") can be missed or',
        "mis-captured.",
        "- OCR fidelity bounds quality: scanning artefacts propagate",
        "into field values and downstream classification.",
        "- Confidence calibration is fit on a small seed set; bin",
        "accuracies are indicative, not statistically significant.",
    ]
    return "\n".join(lines) + "\n"


def _calibration_summary(
    calibration: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Reduce the full calibration to edges plus per-bin accuracy."""
    if calibration is None:
        return None
    return {
        "bin_edges": calibration.get("bin_edges"),
        "bins": [
            {
                "low": bucket.get("low"),
                "high": bucket.get("high"),
                "n": bucket.get("n"),
                "accuracy": bucket.get("accuracy"),
            }
            for bucket in calibration.get("bins", [])
        ],
    }


def write_artefacts(
    bench: dict[str, Any],
    calibration: dict[str, Any] | None,
    out_dir: str = "artefacts",
) -> list[str]:
    """Write ``MODEL_CARD.md`` and ``validation-report.json``.

    Parameters
    ----------
    bench : dict
        Dump of a ``bench.harness.BenchReport`` (``model_dump()``).
    calibration : dict | None
        Output of ``extractor.calibrate.fit_calibration``, if fitted.
    out_dir : str
        Output directory, created with mkdir -p semantics.

    Returns
    -------
    list[str]
        Paths of the written ``MODEL_CARD.md`` and
        ``validation-report.json``.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    card_path = directory / "MODEL_CARD.md"
    report_path = directory / "validation-report.json"
    card_path.write_text(_model_card(bench), encoding="utf-8")
    validation = {
        "extractor_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "bench": {
            "overall_acceptance": bench.get("overall_acceptance"),
            "field_acceptance": bench.get("field_acceptance", {}),
            "needs_review_correct": bench.get("needs_review_correct"),
            "needs_review_total": bench.get("needs_review_total"),
            "structural_ok": bench.get("structural_ok"),
        },
        "calibration": _calibration_summary(calibration),
    }
    report_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    logger.info("artefacts written: {}, {}", card_path, report_path)
    return [str(card_path), str(report_path)]
