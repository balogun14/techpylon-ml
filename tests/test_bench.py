"""Tests for the bench harness, calibration, aliases and artefacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.harness import REPORT_PATH, run_bench  # noqa: E402
from extractor.adapt import build_aliases  # noqa: E402
from extractor.artefacts import write_artefacts  # noqa: E402
from extractor.calibrate import calibrated, fit_calibration  # noqa: E402


def test_bench_seed_set_acceptance() -> None:
    """The seed set runs and overall acceptance is at least 0.85."""
    report = run_bench()
    assert report.total_cases == 12
    assert report.overall_acceptance >= 0.85
    assert report.structural_ok
    assert REPORT_PATH.exists()
    dumped = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert dumped["overall_acceptance"] == report.overall_acceptance


def test_bench_csv_structural_checks() -> None:
    """The CSV case yields one deterministic draft per data row."""
    report = run_bench()
    csv_case = next(c for c in report.cases if c.doc_type == "csv_line_listing")
    assert csv_case.passed
    assert csv_case.structural == {
        "one_draft_per_data_row": True,
        "deterministic_confidence": True,
    }


def test_alias_builder_maps_and_skips(tmp_path: Path) -> None:
    """Corrections become aliases; identical/empty/non-target skip."""
    records = [
        {
            "case_id": "c1",
            "field": "product",
            "draft_value": "Amoxil caps",
            "final_value": "Amoxicillin 500mg",
            "ts": "2026-01-01T00:00:00Z",
        },
        {
            # identical after normalization -> skipped
            "case_id": "c2",
            "field": "product",
            "draft_value": "Paracetamol  syrup",
            "final_value": "Paracetamol syrup",
            "ts": "2026-01-01T00:00:00Z",
        },
        {
            # empty draft -> skipped
            "case_id": "c3",
            "field": "event",
            "draft_value": "",
            "final_value": "nausea",
            "ts": "2026-01-01T00:00:00Z",
        },
        {
            # not a learned field -> skipped
            "case_id": "c4",
            "field": "patient",
            "draft_value": "J. Doe",
            "final_value": "Jane Doe",
            "ts": "2026-01-01T00:00:00Z",
        },
    ]
    out = tmp_path / "aliases.json"
    aliases = build_aliases(records, out_path=str(out))
    assert aliases == {"amoxil caps": "Amoxicillin 500mg"}
    assert json.loads(out.read_text(encoding="utf-8")) == aliases


def test_calibration_bins_written() -> None:
    """fit_calibration bins every prediction and writes the file."""
    calibration = fit_calibration()
    assert calibration["bin_edges"] == [0.0, 0.6, 0.75, 0.85, 1.0]
    # 11 free-text gold cases x 6 fields = 66 predictions
    assert sum(bucket["n"] for bucket in calibration["bins"]) == 66
    for bucket in calibration["bins"]:
        if bucket["n"]:
            assert bucket["accuracy"] == bucket["matches"] / bucket["n"]
    written = json.loads(
        Path("bench/calibration.json").read_text(encoding="utf-8")
    )
    assert written == calibration


def test_calibrated_maps_through_bin() -> None:
    """calibrated() returns the bin accuracy, else the raw value."""
    calibration = {
        "bin_edges": [0.0, 0.6, 0.75, 0.85, 1.0],
        "bins": [
            {"low": 0.0, "high": 0.6, "n": 0, "matches": 0, "accuracy": None},
            {"low": 0.6, "high": 0.75, "n": 2, "matches": 1, "accuracy": 0.5},
        ],
    }
    assert calibrated(0.7, calibration) == 0.5
    # empty bin falls back to the raw confidence
    assert calibrated(0.3, calibration) == 0.3
    # unknown bin (no bins at all) falls back too
    assert calibrated(0.9, {}) == 0.9


def test_artefacts_written_and_parseable(tmp_path: Path) -> None:
    """MODEL_CARD.md is non-empty and validation-report.json parses."""
    report = run_bench()
    calibration = {
        "bin_edges": [0.0, 0.6, 0.75, 0.85, 1.0],
        "bins": [
            {"low": 0.0, "high": 0.6, "n": 1, "matches": 1, "accuracy": 1.0}
        ],
    }
    out = tmp_path / "artefacts"
    paths = write_artefacts(report.model_dump(), calibration, out_dir=str(out))
    card = Path(paths[0])
    validation = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
    assert card.name == "MODEL_CARD.md"
    assert len(card.read_text(encoding="utf-8")) > 0
    assert validation["bench"]["overall_acceptance"] == (
        report.overall_acceptance
    )
    assert validation["calibration"] is not None
    assert validation["calibration"]["bins"][0]["accuracy"] == 1.0

    # no calibration fitted -> null in the validation report
    paths = write_artefacts(report.model_dump(), None, out_dir=str(out))
    validation = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
    assert validation["calibration"] is None
