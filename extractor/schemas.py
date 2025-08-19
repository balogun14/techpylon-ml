"""Pydantic models shared by the TechPylon extraction pipeline.

Shapes mirror ``docs/api-contract.md``: every extracted field is a
``FieldDraft`` and one extraction produces an ``ExtractResponse``.
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

FieldOrigin = Literal["deterministic", "heuristic"]


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


class ExtractRequest(BaseModel):
    """Body of ``POST /extract``."""

    filename: str
    content_type: str = "text/plain"
    text: str


class ExtractResponse(BaseModel):
    """Extraction result for one document (or one CSV row via ``cases``)."""

    doc_type: str
    extractor_version: str
    fields: dict[str, FieldDraft]
    narrative: FieldDraft
    cases: list["ExtractResponse"] | None = None
