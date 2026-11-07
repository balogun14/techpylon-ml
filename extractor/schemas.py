"""Pydantic models shared by the TechPylon extraction pipeline.

Shapes mirror ``docs/api-contract.md``: every extracted field is a
``FieldDraft`` and one extraction produces an ``ExtractResponse``. The
enrichment stage attaches optional, additive sections (coding, PII,
model provenance) that reviewers and the API can consume.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from extractor.config import settings

FIELD_KEYS: tuple[str, ...] = (
    "patient",
    "product",
    "event",
    "reporter",
    "seriousness",
    "expectedness",
)

FieldOrigin = Literal["deterministic", "heuristic", "model", "llm"]


class SourceSpan(BaseModel):
    """Character offset range of a quoted fragment of the source text."""

    start: int
    end: int
    quote: str


class FieldDraft(BaseModel):
    """One extracted (or later reviewed) field value plus provenance."""

    value: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_span: SourceSpan | None = None
    origin: FieldOrigin = "heuristic"
    needs_review: bool = False
    edited: bool = False

    @model_validator(mode="after")
    def _flag_low_confidence(self) -> "FieldDraft":
        """Flag the draft for review when confidence is low."""
        if (
            "needs_review" not in self.model_fields_set
            and self.confidence < settings.needs_review_floor
        ):
            self.needs_review = True
        return self


class CodedTerm(BaseModel):
    """A coding-dictionary term matched to an extracted field value."""

    field: str
    text: str
    code: str
    term: str
    method: Literal["exact", "alias", "fuzzy"]
    confidence: float = Field(ge=0, le=1)


class PiiFinding(BaseModel):
    """A personal-data fragment found in the source text."""

    kind: Literal["email", "phone", "national_id", "date_of_birth"]
    value: str
    start: int
    end: int


class ExtractResponse(BaseModel):
    """Extraction result for one document (or one CSV row via ``cases``)."""

    doc_type: str
    extractor_version: str
    fields: dict[str, FieldDraft]
    narrative: FieldDraft
    cases: list["ExtractResponse"] | None = None
    # enrichment (additive, optional)
    coded_terms: list[CodedTerm] | None = None
    pii: list[PiiFinding] | None = None
    model_fields_used: list[str] | None = None
    llm_fields_used: list[str] | None = None
