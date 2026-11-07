"""Tests for the NAFDAC feed, product matching, and verify flow."""

import json
from typing import Any

import httpx
import pytest

from extractor import nafdac


def _row(
    idx: int, date: str, title: str, url: str, kind: str, maker: str
) -> str:
    return (
        f'<tr data-row_id="{idx}"><td>{date}</td>'
        f'<td><a title="t" href="{url}">{title}</a></td>'
        f"<td>{kind}</td><td>Drugs</td><td>{maker}</td></tr>"
    )


TWO_ROWS = (
    "<table>"
    + _row(
        0,
        "19-Aug-26",
        "Public Alert No. 043/2026 - Paracetamol syrup watchlist",
        "https://nafdac.gov.ng/alert-43/",
        "Safety Alert",
        "Mofus Ltd",
    )
    + _row(
        1,
        "01-Jul-25",
        "Public Alert No. 020/2025 - Falsified Amoxicillin",
        "https://nafdac.gov.ng/alert-20/",
        "Recall",
        "Acme Pharma",
    )
    + "</table>"
)

TOKEN_PAGE = (
    '<input name="__RequestVerificationToken" type="hidden" '
    'value="tok123" /></form>'
)


class FakeResponse:
    """Minimal stand-in for an httpx response."""

    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Mirror httpx: raise on error status."""
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=None,
                response=None,  # type: ignore[arg-type]
            )


def test_fetch_alerts_parses_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Two alert rows parse into entries and persist."""
    monkeypatch.setattr(nafdac, "ALERTS_PATH", tmp_path / "alerts.json")
    monkeypatch.setattr(
        nafdac.httpx, "get", lambda *a, **k: FakeResponse(TWO_ROWS)
    )
    alerts = nafdac.fetch_alerts()
    assert len(alerts) == 2
    first = alerts[0]
    assert first["date"] == "19-Aug-26"
    assert "043/2026" in first["title"]
    assert first["url"] == "https://nafdac.gov.ng/alert-43/"
    assert first["manufacturer"] == "Mofus Ltd"
    # saved and reloadable
    saved = json.loads((tmp_path / "alerts.json").read_text(encoding="utf-8"))
    assert len(saved) == 2


def test_alerts_for_product(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Product matching hits the saved feed case-insensitively."""
    monkeypatch.setattr(nafdac, "ALERTS_PATH", tmp_path / "alerts.json")
    monkeypatch.setattr(
        nafdac.httpx, "get", lambda *a, **k: FakeResponse(TWO_ROWS)
    )
    nafdac.fetch_alerts()
    hits = nafdac.alerts_for_product("AMOXICILLIN")
    assert len(hits) == 1
    assert "Falsified" in hits[0]["title"]
    assert nafdac.alerts_for_product("ibuprofen") == []


def test_verify_product_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful verification posts token, captcha and cert number."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/":
            return httpx.Response(200, text=TOKEN_PAGE)
        return httpx.Response(
            200,
            text='<div class="alert alert-success">A4-10023L: Registered</div>',
        )

    result = nafdac.verify_product(
        "A4-10023L", "captcha-tok", transport=httpx.MockTransport(handler)
    )
    assert result["status"] == "ok"
    assert "Registered" in result["detail"]
    posted = [c for c in calls if c.method == "POST"][0]
    body = posted.read().decode()
    assert "captcha-tok" in body and "tok123" in body


def test_verify_product_captcha_failed() -> None:
    """A stale captcha token maps to captcha_failed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=TOKEN_PAGE)
        return httpx.Response(
            200, text='<div class="alert alert-danger">Captcha failed</div>'
        )

    result = nafdac.verify_product(
        "A4-1", "stale-token", transport=httpx.MockTransport(handler)
    )
    assert result["status"] == "captcha_failed"


def test_verify_product_not_found() -> None:
    """An unknown certificate maps to not_found."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text=TOKEN_PAGE)
        return httpx.Response(
            200,
            text='<div class="alert alert-danger">Certificate not found</div>',
        )

    result = nafdac.verify_product(
        "X0-0", "tok", transport=httpx.MockTransport(handler)
    )
    assert result["status"] == "not_found"
