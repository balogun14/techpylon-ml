"""LLM fallback stage for low-confidence labelled fields (#6, cost-aware).

When ``TECHPYLON_LLM_API_KEY`` is set, fields that are empty or below the
0.6 confidence floor get one OpenAI-compatible chat-completions call that
must return every value as a verbatim substring of the source text.
Non-substring answers are rejected, so hallucinations never enter the
draft. Every failure degrades to "no fills" - enrichment must never break
extraction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from loguru import logger

from extractor.schemas import ExtractResponse, FieldDraft, SourceSpan

_TARGET_FIELDS: tuple[str, ...] = ("patient", "product", "event", "reporter")
_MIN_CONFIDENCE = 0.6
_FILL_CONFIDENCE = 0.7
_TIMEOUT_SECONDS = 15.0


def enabled() -> bool:
    """Report whether an LLM endpoint is configured."""
    return bool(os.getenv("TECHPYLON_LLM_API_KEY"))


def _targets(case: ExtractResponse) -> list[str]:
    """Fields that are empty or below the confidence floor."""
    out: list[str] = []
    for key in _TARGET_FIELDS:
        draft = case.fields.get(key)
        if (
            draft is None
            or not draft.value
            or draft.confidence < _MIN_CONFIDENCE
        ):
            out.append(key)
    return out


def _parse(content: str) -> dict[str, Any]:
    """Parse the model's JSON answer, tolerating code fences/prose."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match is None:
        return {}
    obj = json.loads(match.group(0))
    return obj if isinstance(obj, dict) else {}


def _post(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    """One chat-completions POST; split out so tests can stub it."""
    resp = httpx.post(
        url, headers=headers, json=payload, timeout=_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def apply(case: ExtractResponse, text: str) -> list[str]:
    """Fill low-confidence fields from the LLM; return keys filled."""
    if not enabled():
        return []
    targets = _targets(case)
    if not targets:
        return []
    base_url = os.getenv(
        "TECHPYLON_LLM_BASE_URL", "https://api.openai.com/v1"
    ).rstrip("/")
    payload = {
        "model": os.getenv("TECHPYLON_LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract ONLY the missing fields "
                    f"({', '.join(targets)}) from the adverse-event text "
                    'as strict JSON {"patient": "...", ...}; every value '
                    'MUST be a verbatim substring of the text; use "" '
                    "if absent."
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    try:
        data = _post(
            base_url + "/chat/completions",
            {"Authorization": f"Bearer {os.environ['TECHPYLON_LLM_API_KEY']}"},
            payload,
        )
        content = data["choices"][0]["message"]["content"]
        parsed = _parse(content)
    except Exception as exc:
        logger.warning("llm fallback failed: {}", exc)
        return []

    used: list[str] = []
    for key in targets:
        value = parsed.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value not in text:
            logger.warning(
                "llm value for {} is not a verbatim substring; rejected",
                key,
            )
            continue
        start = text.index(value)
        case.fields[key] = FieldDraft(
            value=value,
            confidence=_FILL_CONFIDENCE,
            origin="llm",
            source_span=SourceSpan(
                start=start, end=start + len(value), quote=value
            ),
        )
        used.append(key)
    return used
