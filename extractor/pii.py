"""PII stage: find personal-data fragments in the source text.

Detects emails, phone numbers (Nigerian +234 / 0803... formats plus
generic 10-13 digit runs), 11-digit National Identification Numbers
labelled "NIN", and dates of birth labelled "dob"/"date of birth".
Overlapping matches are deduplicated (earlier, longer, higher-priority
pattern wins) and findings are sorted by start offset.
"""

import re
from typing import Literal

from extractor.schemas import PiiFinding

PiiKind = Literal["email", "phone", "national_id", "date_of_birth"]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DOB = re.compile(
    r"(?:dob|date\s+of\s+birth)\s*[:\-]?\s*"
    r"(?P<v>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)
_NIN = re.compile(r"\bNIN\s*[:\-]?\s*(?P<v>\d{11})\b", re.IGNORECASE)
_PHONE_INTL = re.compile(r"\+234[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4}")
_PHONE_LOCAL = re.compile(r"\b0[789][01][\s-]?\d{3}[\s-]?\d{4}\b")
_PHONE_GENERIC = re.compile(r"\b\d{10,13}\b")

# (kind, pattern, capture-group name or None for the whole match);
# order is the dedup priority: labelled values beat bare digit runs.
_PATTERNS: tuple[tuple[PiiKind, re.Pattern[str], str | None], ...] = (
    ("date_of_birth", _DOB, "v"),
    ("national_id", _NIN, "v"),
    ("email", _EMAIL, None),
    ("phone", _PHONE_INTL, None),
    ("phone", _PHONE_LOCAL, None),
    ("phone", _PHONE_GENERIC, None),
)


def scan(text: str) -> list[PiiFinding]:
    """Return PII findings with character offsets into ``text``."""
    candidates: list[tuple[int, int, int, PiiKind, str]] = []
    for priority, (kind, pattern, group) in enumerate(_PATTERNS):
        for match in pattern.finditer(text):
            if group is not None:
                start, end = match.span(group)
                value = match.group(group)
            else:
                start, end = match.span()
                value = match.group()
            candidates.append((start, -(end - start), priority, kind, value))
    candidates.sort()
    findings: list[PiiFinding] = []
    last_end = -1
    for start, neg_len, _priority, kind, value in candidates:
        if start >= last_end:
            findings.append(
                PiiFinding(
                    kind=kind,
                    value=value,
                    start=start,
                    end=start - neg_len,
                )
            )
            last_end = start - neg_len
    return findings
