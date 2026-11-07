"""LLM fallback stage for low-confidence labelled fields.

Stub shipped with the enrichment pipeline; replace with the real
implementation (OpenAI-compatible chat endpoint, structured JSON output,
enabled when TECHPYLON_LLM_API_KEY is set).
"""

from extractor.schemas import ExtractResponse


def enabled() -> bool:
    """Report whether an LLM endpoint is configured."""
    return False


def apply(case: ExtractResponse, text: str) -> list[str]:
    """Fill low-confidence fields from the LLM; return keys filled."""
    return []
