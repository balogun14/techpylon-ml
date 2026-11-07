"""Free regulatory-data clients: RxNorm (RxNav) and openFDA FAERS.

All calls are cached under ``data/source_cache/`` (one JSON file per
query key, 7-day TTL, corrupted entries refetched) and fail-open: any
network, parsing, or cache error degrades to a cached value or
``None``/``[]`` with a warning — enrichment never raises because a
free source is down.
"""

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "source_cache"
_TTL_SECONDS = 7 * 24 * 60 * 60
_TIMEOUT = 10.0

_RXNAV_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}"
_FAERS_URL = (
    "https://api.fda.gov/drug/event.json"
    '?search=patient.drug.medicinalproduct:"{drug}"'
    "&count=patient.reaction.reactionmeddrapt.exact"
)
_DAILYMED_SEARCH_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    "?drug_name={drug}"
)
_DAILYMED_SPL_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
)
_SPL_MAX_BYTES = 5 * 1024 * 1024
_SPL_MAX_TERMS = 30
_SPL_BULLET_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-•*·]\s*)")


def _get(url: str) -> httpx.Response:
    """GET ``url`` with the module timeout; raise on HTTP errors."""
    response = httpx.get(url, timeout=_TIMEOUT)
    response.raise_for_status()
    return response


def _cache_path(key: str) -> Path:
    """Filesystem path for one query key inside the cache directory."""
    slug = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return _CACHE_DIR / f"{slug}.json"


def _cache_read(key: str, max_age: float = _TTL_SECONDS) -> Any | None:
    """Return cached JSON for ``key``; None when missing or stale."""
    path = _cache_path(key)
    try:
        if time.time() - path.stat().st_mtime > max_age:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_write(key: str, data: Any) -> None:
    """Persist ``data`` for ``key`` (mkdir on first use)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(data), encoding="utf-8")


def _cached_call(key: str, fetch: Callable[[], Any]) -> Any:
    """Return cached data for ``key``, fetching and storing when absent.

    Fail-open: a fetch error falls back to any cached value (even
    stale) or None, with a warning; a cache-write error still returns
    the freshly fetched data.
    """
    cached = _cache_read(key)
    if cached is not None:
        return cached
    try:
        data = fetch()
    except Exception as exc:
        logger.warning("sources: '{}' failed: {}", key, exc)
        return _cache_read(key, max_age=float("inf"))
    try:
        _cache_write(key, data)
    except Exception as exc:
        logger.warning("sources: cache write '{}' failed: {}", key, exc)
    return data


def _fetch_rxcui(name: str) -> dict[str, str] | None:
    """Query RxNav for ``name``; None when there is no candidate."""
    response = _get(_RXNAV_URL.format(name=quote(name.strip())))
    group = response.json().get("idGroup") or {}
    # RxNav has used both "rxcui" and (since 2025) "rxnormId"
    rxcuis = group.get("rxcui") or group.get("rxnormId") or []
    if not rxcuis:
        return None
    return {
        "rxcui": str(rxcuis[0]),
        "name": str(group.get("name") or name.strip()),
    }


def _fetch_faers(drug: str, limit: int) -> list[dict[str, Any]]:
    """Query openFDA FAERS reaction counts for ``drug``.

    404 (no matching data) and 403 (rate limit) return ``[]`` with a
    warning; other HTTP errors raise for the caller to fail open.
    """
    url = _FAERS_URL.format(drug=quote(drug.strip()))
    try:
        response = _get(url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (403, 404):
            logger.warning(
                "faers: no data or rate-limited for '{}' ({})",
                drug,
                exc.response.status_code,
            )
            return []
        raise
    results = response.json().get("results") or []
    return [
        {"term": str(item["term"]), "count": int(item["count"])}
        for item in results[:limit]
    ]


def normalize_drug(name: str) -> dict[str, str] | None:
    """Normalize a drug name against the RxNorm RxNav API.

    Parameters
    ----------
    name : str
        Free-text drug name as extracted from a document.

    Returns
    -------
    dict or None
        ``{"rxcui": str, "name": str}`` for the first RxNorm
        candidate, or None on no match or any error (fail-open).
    """
    key = f"rxnorm:{name.strip().casefold()}"
    result: dict[str, str] | None = _cached_call(
        key, lambda: _fetch_rxcui(name)
    )
    return result


def faers_top_events(drug: str, limit: int = 10) -> list[dict[str, Any]]:
    """Top reported reaction terms for a drug from openFDA FAERS.

    Parameters
    ----------
    drug : str
        Medicinal product name to search for.
    limit : int, default 10
        Maximum number of reaction terms to return.

    Returns
    -------
    list of dict
        ``[{"term": str, "count": int}, ...]`` sorted by report count;
        empty when there is no data, a rate limit applies, or any
        error occurs (fail-open).
    """
    key = f"faers:{drug.strip().casefold()}:{limit}"
    result: list[dict[str, Any]] | None = _cached_call(
        key, lambda: _fetch_faers(drug, limit)
    )
    return result if result is not None else []


def drug_event_priors(drug: str) -> list[dict[str, Any]]:
    """FAERS top events with each term's share of reported reactions.

    Cached transitively via :func:`faers_top_events`; the share math
    re-runs per call and is pure.

    Parameters
    ----------
    drug : str
        Medicinal product name to search for.

    Returns
    -------
    list of dict
        ``[{"term", "count", "share"}, ...]`` with ``share`` rounded
        to 3 decimals; empty on no data or any error (fail-open).
    """
    try:
        events = faers_top_events(drug)
        total = sum(int(event["count"]) for event in events)
        if total <= 0:
            return []
        return [
            {
                "term": event["term"],
                "count": event["count"],
                "share": round(int(event["count"]) / total, 3),
            }
            for event in events
        ]
    except Exception as exc:
        logger.warning("sources: priors for '{}' failed: {}", drug, exc)
        return []


def _fetch_spl_search(drug: str) -> list[dict[str, str]]:
    """Query DailyMed for SPL documents matching ``drug``."""
    url = _DAILYMED_SEARCH_URL.format(drug=quote(drug.strip()))
    results = _get(url).json().get("data") or []
    return [
        {
            "setid": str(item.get("setid", "")),
            "title": str(item.get("title", "")),
        }
        for item in results[:3]
    ]


def search_spl(drug: str) -> list[dict[str, str]] | None:
    """Search DailyMed SPL documents by drug name.

    Parameters
    ----------
    drug : str
        Free-text drug name as extracted from a document.

    Returns
    -------
    list of dict or None
        First few ``{"setid": str, "title": str}`` results, or None
        on no match or any error (fail-open).
    """
    key = f"dailymed_search:{drug.strip().casefold()}"
    result: list[dict[str, str]] | None = _cached_call(
        key, lambda: _fetch_spl_search(drug)
    )
    return result


def _local_tag(element: ET.Element) -> str:
    """Tag name of ``element`` without any ``{namespace}`` prefix."""
    return str(element.tag).rsplit("}", 1)[-1]


def _spl_adverse_chunks(root: ET.Element) -> list[str]:
    """Text chunks under the section headed 'adverse reactions'.

    Namespace-agnostic: headings match on local tag name, and the
    section body is the matched heading's parent element. Returns
    [] when no such heading exists.
    """
    for element in root.iter():
        for child in element:
            if not isinstance(child.tag, str):
                continue
            if _local_tag(child) not in ("heading", "title"):
                continue
            if "adverse reaction" not in (child.text or "").casefold():
                continue
            chunks: list[str] = []
            for node in element.iter():
                heading = (
                    isinstance(node.tag, str)
                    and _local_tag(node) in ("heading", "title")
                )
                if heading:
                    continue
                for chunk in (node.text, node.tail):
                    if chunk:
                        chunks.append(chunk)
            return chunks
    return []


def _spl_terms(chunks: list[str]) -> list[str]:
    """Candidate event terms: bullets stripped, 1-6 words, deduped."""
    terms: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for line in re.split(r"[\n;]+", chunk):
            words = _SPL_BULLET_RE.sub("", line).strip().split()
            if not 1 <= len(words) <= 6:
                continue
            token = " ".join(words).casefold()
            if not any(ch.isalnum() for ch in token):
                continue
            if token in seen:
                continue
            seen.add(token)
            terms.append(" ".join(words))
            if len(terms) >= _SPL_MAX_TERMS:
                return terms
    return terms


def _fetch_spl_label_events(drug: str) -> list[dict[str, str]]:
    """Fetch and parse the first SPL's adverse-reactions section."""
    results = search_spl(drug)
    if not results or not results[0].get("setid"):
        return []
    setid = str(results[0]["setid"])
    content = _get(_DAILYMED_SPL_URL.format(setid=quote(setid))).content
    if len(content) > _SPL_MAX_BYTES:
        raise ValueError(f"SPL '{setid}' exceeds {_SPL_MAX_BYTES} bytes")
    chunks = _spl_adverse_chunks(ET.fromstring(content))
    if not chunks:
        logger.debug(
            "dailymed: no adverse-reactions section for '{}'", drug
        )
        return []
    return [{"term": term, "setid": setid} for term in _spl_terms(chunks)]


def spl_label_events(drug: str) -> list[dict[str, str]]:
    """Labelled (expected) adverse-event terms from the drug's SPL.

    Resolves the SPL setid via :func:`search_spl`, fetches the SPL
    XML (parse capped at 5 MB), and extracts candidate event terms
    from the adverse-reactions section.

    Parameters
    ----------
    drug : str
        Medicinal product name to search for.

    Returns
    -------
    list of dict
        ``[{"term": str, "setid": str}, ...]`` — labelled (expected)
        events for the drug, capped at 30 terms; empty when the drug
        has no SPL, the label has no usable adverse-reactions
        section, or any error occurs (fail-open).
    """
    key = f"dailymed_spl:{drug.strip().casefold()}"
    result: list[dict[str, str]] | None = _cached_call(
        key, lambda: _fetch_spl_label_events(drug)
    )
    return result if result is not None else []


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "paracetamol"
    normalized = normalize_drug(query)
    if normalized is None:
        print(f"rxnorm: offline/unavailable or no match for '{query}'")
    else:
        print(f"rxnorm: {normalized}")
    events = faers_top_events(query)
    if not events:
        print(f"faers: offline/unavailable or no data for '{query}'")
    else:
        for event in events:
            print(f"faers: {event['count']:>8}  {event['term']}")
