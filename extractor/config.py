"""Static settings for the extractor service."""

from pydantic import BaseModel


class Settings(BaseModel):
    """Runtime knobs for schemas and the HTTP service."""

    port: int = 8100
    needs_review_floor: float = 0.60


settings = Settings()
