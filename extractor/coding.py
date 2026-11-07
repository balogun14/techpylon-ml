"""Coding stage: map product/event values to dictionary terms.

Method ladder per field (one best ``CodedTerm`` max):

1. exact — casefold match against the dictionary or the curated alias
   map (confidence 1.0, method ``exact``);
2. alias — aliases learned by ``extractor.adapt.load_aliases`` (another
   stage's alias learner, loaded defensively: it may not exist yet);
3. fuzzy — rapidfuzz ``token_set_ratio`` >= 85, confidence = ratio/100.
"""

import importlib

from rapidfuzz import fuzz

from extractor.dictionary import (
    DRUG_ALIASES,
    DRUGS,
    EVENT_ALIASES,
    EVENTS,
)
from extractor.schemas import CodedTerm, ExtractResponse

_FUZZY_THRESHOLD = 85
_ALIAS_CONFIDENCE = 0.9


def _norm(value: str) -> str:
    """Normalise a term for case-insensitive comparison."""
    return " ".join(value.split()).casefold()


def _alias_index(
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """Map each normalized alias to its normalized canonical term."""
    index: dict[str, str] = {}
    for term, variants in aliases.items():
        for variant in variants:
            index.setdefault(_norm(variant), _norm(term))
    return index


_DRUG_ALIAS_INDEX = _alias_index(DRUG_ALIASES)
_EVENT_ALIAS_INDEX = _alias_index(EVENT_ALIASES)


def _load_adapt_aliases() -> dict[str, str]:
    """Best-effort alias map from adapt.py; {} when unavailable.

    Not cached: adapt.py may appear (or change) between calls, and the
    alias stage costs a single import + dict read.
    """
    try:
        adapt = importlib.import_module("extractor.adapt")
    except ImportError:
        return {}
    loader = getattr(adapt, "load_aliases", None)
    if not callable(loader):
        return {}
    try:
        loaded = loader()
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    mapping: dict[str, str] = {}
    for key, value in loaded.items():
        if isinstance(value, str):
            mapping[_norm(str(key))] = _norm(value)
        elif isinstance(value, (list, tuple)):
            for variant in value:
                mapping[_norm(str(variant))] = _norm(str(key))
    return mapping


def _best_fuzzy(
    value: str,
    codes: dict[str, str],
    aliases: dict[str, tuple[str, ...]],
) -> tuple[str, float] | None:
    """Return (canonical term, confidence) for the best fuzzy match."""
    best_term = ""
    best_ratio = 0.0
    for term in codes:
        for candidate in (term, *aliases.get(term, ())):
            ratio = fuzz.token_set_ratio(value, candidate)
            if ratio > best_ratio:
                best_term = term
                best_ratio = ratio
    if best_ratio < _FUZZY_THRESHOLD:
        return None
    return best_term, round(best_ratio / 100, 2)


def _code_field(
    field: str,
    value: str,
    codes: dict[str, str],
    alias_variants: dict[str, tuple[str, ...]],
    alias_index: dict[str, str],
    adapt_map: dict[str, str],
) -> CodedTerm | None:
    """Code one field value via the exact -> alias -> fuzzy ladder."""
    norm = _norm(value)
    if not norm:
        return None
    if norm in codes:
        return CodedTerm(
            field=field,
            text=value,
            code=codes[norm],
            term=norm,
            method="exact",
            confidence=1.0,
        )
    canon = alias_index.get(norm)
    if canon is not None and canon in codes:
        return CodedTerm(
            field=field,
            text=value,
            code=codes[canon],
            term=canon,
            method="exact",
            confidence=1.0,
        )
    canon = adapt_map.get(norm)
    if canon is not None and canon in codes:
        return CodedTerm(
            field=field,
            text=value,
            code=codes[canon],
            term=canon,
            method="alias",
            confidence=_ALIAS_CONFIDENCE,
        )
    fuzzy = _best_fuzzy(norm, codes, alias_variants)
    if fuzzy is None:
        return None
    term, confidence = fuzzy
    return CodedTerm(
        field=field,
        text=value,
        code=codes[term],
        term=term,
        method="fuzzy",
        confidence=confidence,
    )


def code_case(case: ExtractResponse) -> list[CodedTerm]:
    """Code the product and event fields of one case draft."""
    adapt_map = _load_adapt_aliases()
    terms: list[CodedTerm] = []
    for key, codes, variants, alias_index in (
        ("product", DRUGS, DRUG_ALIASES, _DRUG_ALIAS_INDEX),
        ("event", EVENTS, EVENT_ALIASES, _EVENT_ALIAS_INDEX),
    ):
        draft = case.fields.get(key)
        if draft is None or not draft.value.strip():
            continue
        coded = _code_field(
            key, draft.value, codes, variants, alias_index, adapt_map
        )
        if coded is not None:
            terms.append(coded)
    return terms
