"""PII stage: find personal-data fragments in the source text.

Stub shipped with the enrichment pipeline; replace ``scan`` with the real
implementation (email, phone, national ID, date-of-birth regexes).
"""

from extractor.schemas import PiiFinding


def scan(text: str) -> list[PiiFinding]:
    """Return PII findings with character offsets into ``text``."""
    return []
