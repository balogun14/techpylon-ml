"""Multilingual label patterns merged into the heuristic extractor.

``EXTRA_LABELLED`` maps a FIELD_KEYS entry to compiled regexes with a
named group ``v`` — they are tried after the English label patterns.
Because ``heuristic._LABELLED`` merges these entries over its English
defaults (``**EXTRA_LABELLED`` wins in the dict literal), each pattern
below must also cover the English labels it replaces.
``EXTRA_SERIOUS_TERMS`` is a regex alternation body (no group) of
seriousness keywords in other languages.
"""

import re

_LABEL_TAIL = r")\s*[:\-]\s*(?P<v>[^\n]+)"

_PATIENT = r"(?:patient(?:e)?(?:\s+name)?"
_PRODUCT = (
    r"(?:drug(?:\s+name)?|product|medication"
    r"|m[eé]dicament|traitement|produit"
    r"|medicamento|f[aá]rmaco"
)
_EVENT = (
    r"(?:adverse\s+event|event"
    r"|r[eé]action"
    r"|effet\s+ind[eé]sirable|[eé]v[eé]nement"
    r"|rea[cç][aã]o|evento"
)
_REPORTER = (
    r"(?:reported\s+by|reporter"
    r"|rapport[eé]\s+par|signal[eé]\s+par"
    r"|reportado\s+por|notificado\s+por"
)

EXTRA_LABELLED: dict[str, tuple[re.Pattern[str], ...]] = {
    "patient": (re.compile(_PATIENT + _LABEL_TAIL, re.IGNORECASE),),
    "product": (re.compile(_PRODUCT + _LABEL_TAIL, re.IGNORECASE),),
    "event": (
        re.compile(_EVENT + _LABEL_TAIL, re.IGNORECASE),
        re.compile(r"experienced\s+(?P<v>[^\n]+)", re.IGNORECASE),
    ),
    "reporter": (re.compile(_REPORTER + _LABEL_TAIL, re.IGNORECASE),),
}

EXTRA_SERIOUS_TERMS = (
    r"hospitalisation|d[eé]c[eè]s|mise\s+en\s+jeu\s+du\s+pronostic"
    r"|handicap|malformation\s+cong[eé]nitale|gravement"
    r"|internação|óbito|risco\s+de\s+vida|deficiência|malformação"
)
