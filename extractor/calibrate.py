"""Confidence calibration fitted from the bench seed set.

``fit_calibration`` buckets every field prediction from a bench run
into confidence bins and records the empirical accuracy per bin.
Honest caveat: the seed set is tiny (11 free-text cases today), so the
resulting numbers are indicative of confidence ranking, not a
statistically significant calibration. Re-fit as the seed set grows;
do not treat a bin accuracy as a true probability estimate.
"""

import bisect
import json
from pathlib import Path
from typing import Any

from loguru import logger

from bench.harness import BenchReport, run_bench

CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "bench" / "calibration.json"
)
BIN_EDGES: tuple[float, ...] = (0.0, 0.6, 0.75, 0.85, 1.0)


def _bin_index(
    confidence: float, edges: list[float] | tuple[float, ...]
) -> int:
    """Return the index of the half-open bin holding ``confidence``."""
    return min(bisect.bisect_right(edges, confidence) - 1, len(edges) - 2)


def fit_calibration() -> dict[str, Any]:
    """Fit empirical accuracy per confidence bin from a bench run.

    Runs the bench, buckets each free-text field prediction by its raw
    confidence (edges 0, 0.6, 0.75, 0.85, 1.0) and records the
    empirical match rate per bin. Writes ``bench/calibration.json``.

    Caveat: seed-set-sized data gives indicative, not statistical,
    calibration — treat bin accuracies as sanity signals only.

    Returns
    -------
    dict
        ``{"bin_edges": [...], "bins": [{low, high, n, matches,
        accuracy}, ...]}`` with ``accuracy`` null for empty bins.
    """
    report: BenchReport = run_bench()
    bins: list[dict[str, Any]] = [
        {"low": low, "high": high, "n": 0, "matches": 0, "accuracy": None}
        for low, high in zip(BIN_EDGES, BIN_EDGES[1:], strict=False)
    ]
    for case in report.cases:
        for field in case.fields.values():
            bucket = bins[_bin_index(field.confidence, BIN_EDGES)]
            bucket["n"] += 1
            if field.match:
                bucket["matches"] += 1
    for bucket in bins:
        if bucket["n"]:
            bucket["accuracy"] = bucket["matches"] / bucket["n"]
    calibration: dict[str, Any] = {
        "bin_edges": list(BIN_EDGES),
        "bins": bins,
    }
    CALIBRATION_PATH.write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    logger.info("calibration written: {}", CALIBRATION_PATH)
    return calibration


def load_calibration() -> dict[str, Any] | None:
    """Load the fitted calibration, or None when it has not been fit."""
    if not CALIBRATION_PATH.exists():
        return None
    data: dict[str, Any] = json.loads(
        CALIBRATION_PATH.read_text(encoding="utf-8")
    )
    return data


def calibrated(confidence: float, calibration: dict[str, Any]) -> float:
    """Map a raw confidence through its bin's empirical accuracy.

    Falls back to the raw confidence when the bin is empty or missing
    so that uncalibrated confidences degrade gracefully.
    """
    edges = calibration.get("bin_edges", list(BIN_EDGES))
    bins = calibration.get("bins", [])
    idx = _bin_index(confidence, edges)
    if 0 <= idx < len(bins):
        accuracy = bins[idx].get("accuracy")
        if accuracy is not None:
            return float(accuracy)
    return confidence


if __name__ == "__main__":
    fit_calibration()
