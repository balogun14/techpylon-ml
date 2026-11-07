"""Coding stage: map product/event values to dictionary terms.

Stub shipped with the enrichment pipeline; replace ``code_case`` with the
real implementation (curated MedDRA-style term list + fuzzy matching).
"""

from extractor.schemas import CodedTerm, ExtractResponse


def code_case(case: ExtractResponse) -> list[CodedTerm]:
    """Code the product and event fields of one case draft."""
    return []
