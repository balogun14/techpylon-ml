"""NAFDAC integration: registered-products CSV ingestion, safety-alert
feed, and single-product verification via the NAPAMS portal.

Sample data, NOT AUTHORITATIVE: ``data/nafdac/`` ships a handful of
illustrative product rows only. Real bulk registries must come from an
official NAFDAC product-registry CSV export or an FOI request (there is
no public bulk API). Rows are validated at the trust boundary: missing
required columns raise ``ValueError``; rows without a product name are
skipped.

The safety-alert feed (``fetch_alerts``) reads the public NAFDAC
"Recalls and Alerts" page — genuine regulatory data, no auth.

``verify_product`` queries the public NAPAMS product-verification form,
which is protected by reCAPTCHA v3: pass a ``captcha_token`` obtained in
a real browser session; without one the portal rejects the request.
"""  # noqa: D205

import csv
import html as html_mod
import json
import re
import sys
from pathlib import Path

import httpx
from loguru import logger

_DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "nafdac"
    / "sample_registered_products.csv"
)
ALERTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "nafdac" / "alerts.json"
)
NAPAMS_BASE = "https://registration.nafdac.gov.ng"
_ALERTS_URL = "https://nafdac.gov.ng/category/recalls-and-alerts/"
_COLUMNS = (
    "product_name",
    "active_ingredient",
    "manufacturer",
    "registration_number",
    "status",
)
_STRENGTH = re.compile(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|iu)\b", re.IGNORECASE)
_ROW = re.compile(r'<tr data-row_id="\d+"[^>]*>(.*?)</tr>', re.DOTALL)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_LINK = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _norm(value: str) -> str:
    """Normalize a term the way the coding stage compares aliases."""
    return " ".join(value.split()).casefold()


def load_products(path: str | Path) -> list[dict[str, str]]:
    """Load a NAFDAC registered-products CSV export.

    Header names are matched liberally (case- and space-insensitive)
    against ``product_name``, ``active_ingredient``,
    ``manufacturer``, ``registration_number``, ``status``. The
    product name is re-keyed to ``name`` and lowercased.

    Parameters
    ----------
    path : str or Path
        CSV file path.

    Returns
    -------
    list of dict
        One normalized dict per usable row; rows without a
        ``product_name`` are skipped.

    Raises
    ------
    ValueError
        When any required column is absent from the header.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = {
            (name or "").strip().casefold().replace(" ", "_"): name
            for name in (reader.fieldnames or [])
        }
        missing = [column for column in _COLUMNS if column not in header]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        products: list[dict[str, str]] = []
        for raw in reader:
            row = {
                column: (raw.get(header[column]) or "").strip()
                for column in _COLUMNS
            }
            if not row["product_name"]:
                continue
            row["name"] = row.pop("product_name").casefold()
            products.append(row)
    return products


def extend_aliases(products: list[dict[str, str]]) -> dict[str, str]:
    """Build coding-stage alias pairs from loaded NAFDAC products.

    Produces ``{"<normalized variant>": "<product name>"}`` entries in
    the same shape as ``extractor.adapt`` aliases: keys are normalized
    (whitespace-collapsed, casefolded) ingredient names and
    strength-stripped product-name variants; values are the product
    names. Returned only — nothing is written to
    ``customer_aliases.json``; the caller decides what to learn from.

    Parameters
    ----------
    products : list of dict
        Output of :func:`load_products`.

    Returns
    -------
    dict
        Alias pairs suitable for the coding stage's alias convention.
    """
    aliases: dict[str, str] = {}
    for product in products:
        name = product.get("name", "").strip()
        if not name:
            continue
        ingredient = product.get("active_ingredient", "").strip()
        if ingredient:
            aliases.setdefault(_norm(ingredient), name)
        bare = _norm(_STRENGTH.sub("", name))
        if bare and bare != _norm(name):
            aliases.setdefault(bare, name)
    return aliases


def fetch_alerts(
    url: str = _ALERTS_URL, save: bool = True
) -> list[dict[str, str]]:
    """Scrape the public NAFDAC Recalls and Alerts table.

    The category page server-renders a Ninja Table with every published
    alert (date, title with permalink, type, product type,
    manufacturer).

    Parameters
    ----------
    url : str
        Category page to scrape.
    save : bool
        Write the parsed feed to ``data/nafdac/alerts.json``.

    Returns
    -------
    list of dict
        One entry per alert row, newest first as published.
    """
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    alerts: list[dict[str, str]] = []
    for row in _ROW.findall(response.text):
        cells = [_TAG.sub("", cell).strip() for cell in _CELL.findall(row)]
        if len(cells) < 3:
            continue
        link = _LINK.search(row)
        entry = {
            "date": cells[0],
            "title": html_mod.unescape(cells[1]),
            "url": link.group(1) if link else "",
            "type": cells[2] if len(cells) > 2 else "",
            "product_type": cells[3] if len(cells) > 3 else "",
            "manufacturer": cells[4] if len(cells) > 4 else "",
        }
        if entry["title"]:
            alerts.append(entry)
    logger.info("nafdac alerts: {} entries from {}", len(alerts), url)
    if save and alerts:
        ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERTS_PATH.write_text(
            json.dumps(alerts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return alerts


def load_alerts() -> list[dict[str, str]]:
    """Load the last-saved alert feed (empty if never fetched)."""
    if not ALERTS_PATH.exists():
        return []
    data = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def alerts_for_product(name: str) -> list[dict[str, str]]:
    """Return saved alerts whose title mentions the product name."""
    needle = _norm(name)
    return [
        alert
        for alert in load_alerts()
        if needle and needle in _norm(alert.get("title", ""))
    ]


class NafdacVerifyError(Exception):
    """Raised when the NAPAMS verification flow cannot complete."""


def verify_product(
    certificate_number: str,
    captcha_token: str,
    base_url: str = NAPAMS_BASE,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, str]:
    """Verify one product registration number against NAPAMS.

    The portal's verify form is reCAPTCHA-v3-protected: ``captcha_token``
    must be produced by a real browser session against the NAPAMS site
    key. The anti-forgery token and cookies are fetched automatically.

    Returns
    -------
    dict
        ``{"status": "ok" | "not_found" | "captcha_failed",
        "detail": <page message>}``.
    """
    with httpx.Client(
        base_url=base_url,
        timeout=30,
        follow_redirects=True,
        transport=transport,
    ) as client:
        page = client.get("/")
        page.raise_for_status()
        match = re.search(
            r'name="__RequestVerificationToken"[^>]*value="([^"]+)"',
            page.text,
        )
        if match is None:
            raise NafdacVerifyError("no anti-forgery token on NAPAMS page")
        result = client.post(
            "/Home/VerifyProduct",
            data={
                "CertificateNumber": certificate_number,
                "captchaToken": captcha_token,
                "__RequestVerificationToken": match.group(1),
            },
        )
        result.raise_for_status()
        message = ""
        alert = re.search(r"alert[^>]*>\s*([^<]{3,200})", result.text)
        if alert is not None:
            message = html_mod.unescape(alert.group(1)).strip()
    lowered = message.casefold()
    if "captcha" in lowered:
        status = "captcha_failed"
    elif not message or "not" in lowered and "found" in lowered:
        status = "not_found"
    else:
        status = "ok"
    logger.info(
        "nafdac verify {}: {} ({})", certificate_number, status, message
    )
    return {"status": status, "detail": message}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--alerts":
        feed = fetch_alerts()
        print(f"nafdac: {len(feed)} alerts saved to {ALERTS_PATH}")
        for alert in feed[:5]:
            print(f"  [{alert['date']}] {alert['title'][:90]}")
            print(f"      {alert['type']} | {alert['manufacturer']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # needs a browser-obtained reCAPTCHA token as the third arg
        if len(sys.argv) < 4:
            print(
                "usage: python -m extractor.nafdac",
                "--verify <cert-number> <captcha-token>",
            )
            raise SystemExit(1)
        print(verify_product(sys.argv[2], sys.argv[3]))
    else:
        csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_CSV
        try:
            loaded = load_products(csv_path)
        except (OSError, ValueError) as exc:
            print(f"nafdac: could not load {csv_path}: {exc}")
            raise SystemExit(1) from exc
        print(f"nafdac: {len(loaded)} products from {csv_path}")
        for product in loaded[:3]:
            print(f"  {product}")
