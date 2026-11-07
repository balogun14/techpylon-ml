"""Tests for free regulatory-data sources: no network in tests."""

from pathlib import Path

import httpx
import pytest

from extractor import sources
from extractor.nafdac import extend_aliases, load_products
from extractor.reference import faers_support
from extractor.schemas import ExtractResponse, FieldDraft

_RXNAV_PAYLOAD = {
    "rxcuiStatus": "active",
    "idGroup": {"rxcui": ["161"], "name": "acetaminophen"},
}
_RXNAV_PAYLOAD_RXNORMID = {
    "idGroup": {"rxnormId": ["161"]},
}
_FAERS_PAYLOAD = {
    "meta": {"disclaimer": "openFDA"},
    "results": [
        {"term": "HEADACHE", "count": 70},
        {"term": "RASH", "count": 30},
    ],
}


def _patch_get(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict | None = None,
    error: Exception | None = None,
) -> list[int]:
    """Patch sources._get (counting calls) and the cache directory."""
    calls = [0]

    def fake_get(url: str) -> httpx.Response:
        calls[0] += 1
        if error is not None:
            raise error
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(sources, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sources, "_get", fake_get)
    return calls


# --- sources.normalize_drug ---


def test_normalize_drug_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A match returns the first candidate's rxcui and name."""
    calls = _patch_get(monkeypatch, tmp_path, _RXNAV_PAYLOAD)
    result = sources.normalize_drug("  Paracetamol ")
    assert result == {"rxcui": "161", "name": "acetaminophen"}
    assert calls[0] == 1
    assert (tmp_path / "rxnorm_paracetamol.json").exists()


def test_normalize_drug_rxnormid_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RxNav's newer 'rxnormId' key is accepted too; name falls back."""
    _patch_get(monkeypatch, tmp_path, _RXNAV_PAYLOAD_RXNORMID)
    assert sources.normalize_drug("paracetamol") == {
        "rxcui": "161",
        "name": "paracetamol",
    }


def test_normalize_drug_no_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty idGroup is no match, not an error."""
    _patch_get(monkeypatch, tmp_path, {"idGroup": {}})
    assert sources.normalize_drug("zzzzdrug") is None


def test_normalize_drug_error_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A network error degrades to None (fail-open)."""
    _patch_get(monkeypatch, tmp_path, error=httpx.ConnectError("down"))
    assert sources.normalize_drug("paracetamol") is None


def test_normalize_drug_cache_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second call is served from cache, not the network."""
    calls = _patch_get(monkeypatch, tmp_path, _RXNAV_PAYLOAD)
    first = sources.normalize_drug("paracetamol")
    second = sources.normalize_drug("paracetamol")
    assert first == second == {"rxcui": "161", "name": "acetaminophen"}
    assert calls[0] == 1


def test_corrupted_cache_refetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupt cache file is ignored and the fetch re-runs."""
    calls = _patch_get(monkeypatch, tmp_path, _RXNAV_PAYLOAD)
    (tmp_path / "rxnorm_paracetamol.json").write_text("{not json")
    assert sources.normalize_drug("paracetamol") == {
        "rxcui": "161",
        "name": "acetaminophen",
    }
    assert calls[0] == 1


# --- sources.faers_top_events / drug_event_priors ---


def test_faers_404_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 404 (no matching data) yields [], not an exception."""
    request = httpx.Request("GET", "https://api.fda.gov/x")
    error = httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404)
    )
    _patch_get(monkeypatch, tmp_path, error=error)
    assert sources.faers_top_events("unknowndrug") == []


def test_faers_parses_counts_and_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Results are parsed to term/count dicts and truncated to limit."""
    calls = _patch_get(monkeypatch, tmp_path, _FAERS_PAYLOAD)
    events = sources.faers_top_events("aspirin", limit=1)
    assert events == [{"term": "HEADACHE", "count": 70}]
    assert calls[0] == 1


def test_priors_share_math(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shares are count/total rounded to 3 decimals."""
    _patch_get(monkeypatch, tmp_path, _FAERS_PAYLOAD)
    priors = sources.drug_event_priors("aspirin")
    assert priors == [
        {"term": "HEADACHE", "count": 70, "share": 0.7},
        {"term": "RASH", "count": 30, "share": 0.3},
    ]


def test_priors_fail_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Any underlying error yields [] (fail-open)."""
    _patch_get(monkeypatch, tmp_path, error=httpx.ConnectError("down"))
    assert sources.drug_event_priors("aspirin") == []


# --- nafdac ---


_CSV_TEXT = (
    "Product Name, Active Ingredient ,Manufacturer,"
    "Registration Number,Status\n"
    "  Paracetamol Tablets 500mg , paracetamol ,"
    "Emzor Pharma,A4-000001,registered\n"
    ",aspirin,Other Pharma,C4-000003,registered\n"
    "Artesunate Tablets 100mg,artesunate,Fidson,"
    "B4-000002,registered\n"
)


def test_load_products(tmp_path: Path) -> None:
    """Header casing is liberal; values stripped; name lowercased."""
    path = tmp_path / "registry.csv"
    path.write_text(_CSV_TEXT, encoding="utf-8")
    products = load_products(path)
    assert len(products) == 2  # row without product_name skipped
    assert products[0]["name"] == "paracetamol tablets 500mg"
    assert products[0]["active_ingredient"] == "paracetamol"
    assert products[0]["manufacturer"] == "Emzor Pharma"
    assert products[0]["registration_number"] == "A4-000001"
    assert products[0]["status"] == "registered"


def test_load_products_bad_columns(tmp_path: Path) -> None:
    """Missing required columns raise ValueError."""
    path = tmp_path / "bad.csv"
    path.write_text("name,ingredient\nparacetamol,paracetamol\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_products(path)


def test_extend_aliases_shape() -> None:
    """Aliases map normalized variants to product names."""
    products = [
        {
            "name": "paracetamol tablets 500mg",
            "active_ingredient": "paracetamol",
        },
        {
            "name": "ors-zinc sachets",
            "active_ingredient": "oral rehydration salts + zinc",
        },
    ]
    aliases = extend_aliases(products)
    assert aliases["paracetamol"] == "paracetamol tablets 500mg"
    assert aliases["oral rehydration salts + zinc"] == "ors-zinc sachets"
    # strength-stripped product-name variant
    assert aliases["paracetamol tablets"] == "paracetamol tablets 500mg"


# --- reference.faers_support ---


def _case(product: str, event: str) -> ExtractResponse:
    """Minimal one-case response with product/event fields set."""
    return ExtractResponse(
        doc_type="free_text",
        extractor_version="test",
        fields={
            "product": FieldDraft(value=product),
            "event": FieldDraft(value=event),
        },
        narrative=FieldDraft(value=""),
    )


def test_faers_support_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """FAERS terms sharing a 3-char prefix with the event are kept."""
    seen: list[str] = []

    def fake_priors(drug: str) -> list[dict[str, object]]:
        seen.append(drug)
        return [
            {"term": "headache", "count": 7, "share": 0.7},
            {"term": "nausea", "count": 3, "share": 0.3},
        ]

    monkeypatch.setattr(sources, "drug_event_priors", fake_priors)
    support = faers_support(_case("aspirin", "headache"))
    assert support == [{"term": "headache", "count": 7, "share": 0.7}]
    assert seen == ["aspirin"]


def test_faers_support_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No shared prefix means no corroboration."""
    monkeypatch.setattr(
        sources,
        "drug_event_priors",
        lambda drug: [{"term": "headache", "count": 7, "share": 0.7}],
    )
    assert faers_support(_case("aspirin", "rash")) == []


def test_faers_support_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A priors failure degrades to [] (fail-open)."""

    def boom(drug: str) -> list[dict[str, object]]:
        raise RuntimeError("offline")

    monkeypatch.setattr(sources, "drug_event_priors", boom)
    assert faers_support(_case("aspirin", "headache")) == []
