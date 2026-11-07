"""Reference-based expectedness refinement.

Uses the curated ``reference_labels.PRODUCT_KEY`` table to nudge the
expectedness draft of a case when both product and event could be coded.
It only ever fills in low-confidence or unknown values; it must never
override high-confidence or reviewer-edited signals.
"""

from extractor import coding
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
