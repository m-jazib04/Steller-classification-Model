"""Pydantic schemas for API validation."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    alpha: float = Field(..., ge=0, le=360, description="Right ascension (degrees)")
    delta: float = Field(..., ge=-90, le=90, description="Declination (degrees)")
    u: float = Field(..., ge=0, le=30, description="u-band magnitude")
    g: float = Field(..., ge=0, le=30, description="g-band magnitude")
    r: float = Field(..., ge=0, le=30, description="r-band magnitude")
    i: float = Field(..., ge=0, le=30, description="i-band magnitude")
    z: float = Field(..., ge=0, le=30, description="z-band magnitude")
    redshift: float = Field(..., ge=-1, le=10, description="Redshift value")

    @field_validator("alpha", "delta", "u", "g", "r", "i", "z", "redshift", mode="before")
    @classmethod
    def sanitize_numeric(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("Value cannot be empty")
        return value


class ProbabilityItem(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    prediction: Literal["GALAXY", "QSO", "STAR"]
    confidence: float
    probabilities: list[ProbabilityItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    accuracy: float | None = None


class ModelInfoResponse(BaseModel):
    features: list[str]
    classes: list[str]
    accuracy: float
    model_type: str
