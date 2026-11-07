"""Reference-based expectedness refinement.

Uses the curated ``reference_labels.PRODUCT_KEY`` table to nudge the
expectedness draft of a case when both product and event could be coded.
It only ever fills in low-confidence or unknown values; it must never
override high-confidence or reviewer-edited signals.
"""

from typing import Any

from loguru import logger

from extractor import coding, sources
from extractor.dictionary import EVENTS
from extractor.reference_labels import PRODUCT_KEY
from extractor.schemas import ExtractResponse

_UP_CONFIDENCE = 0.85
_UP_FLOOR = 0.75
_DOWN_CONFIDENCE = 0.7
_TOLERABILITY_CODE = EVENTS["none observed"]


def refine_expectedness(case: ExtractResponse) -> bool:
    """Refine expectedness from coded product/event; True if changed.

    - event in the product's expected list and confidence < 0.75 (and
      not already "expected"): set "expected" at 0.85, origin unchanged;
    - product and event coded but the event is not expected and the
      draft is "unknown": set "unexpected" at 0.7. Tolerability events
      ("none observed") are not adverse events and never become
      "unexpected".
    """
    field = case.fields.get("expectedness")
    if field is None or field.edited:
        return False
    terms = coding.code_case(case)
    product = next((t for t in terms if t.field == "product"), None)
    event = next((t for t in terms if t.field == "event"), None)
    if product is None or event is None:
        return False
    expected = PRODUCT_KEY.get(product.term.casefold().strip(), [])
    if event.code in expected:
        if field.confidence < _UP_FLOOR and field.value != "expected":
            field.value = "expected"
            field.confidence = _UP_CONFIDENCE
            return True
        return False
    if field.value == "unknown" and event.code != _TOLERABILITY_CODE:
        field.value = "unexpected"
        field.confidence = _DOWN_CONFIDENCE
        return True
    return False


def faers_support(case: ExtractResponse) -> list[dict[str, Any]]:
    """FAERS report counts corroborating the coded event.

    Reviewer-visible corroboration only: this never changes field
    values and does not participate in ``refine_expectedness`` — the
    API surfaces the returned priors beside the draft so a reviewer
    can see how often the coded event is reported for the coded
    product in FAERS. A FAERS term corroborates the event when it
    shares at least the first three characters with the coded event
    term (case-insensitive).

    Parameters
    ----------
    case : ExtractResponse
        One case draft with (ideally) coded product and event fields.

    Returns
    -------
    list of dict
        ``[{"term", "count", "share"}, ...]`` for matching FAERS
        terms; empty when product/event cannot be coded, FAERS is
        unavailable, or anything else goes wrong (fail-open).
    """
    try:
        terms = coding.code_case(case)
        product = next((t for t in terms if t.field == "product"), None)
        event = next((t for t in terms if t.field == "event"), None)
        if product is None or event is None:
            return []
        priors = sources.drug_event_priors(product.term)
        prefix = event.term[:3].casefold()
        return [
            prior
            for prior in priors
            if str(prior["term"]).casefold().startswith(prefix)
        ]
    except Exception as exc:
        logger.warning("reference: faers_support failed: {}", exc)
        return []
