"""Learning API response schemas (CIH Phase 7).

Responses reuse the engine's learning value objects verbatim so the
on-the-wire shapes are exactly what the deterministic build layer produces.
Every value is derived by :mod:`predictron_engine.learning` — this module
only wraps them in response envelopes.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from predictron_engine.learning.models import (
    LearningCalibration,
    LearningConfidence,
    LearningDigest,
    LearningObservation,
    LearningPattern,
    LearningPeriod,
    LearningPeriodKind,
    LearningRecommendation,
)


class LearningSummaryResponse(BaseModel):
    """Live learning summary over the caller's evaluated population."""

    scope: str = Field(...)
    period_kind: LearningPeriodKind = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    engine_version: str = Field(...)
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    digest: LearningDigest = Field(default_factory=LearningDigest)
    calibration: LearningCalibration = Field(default_factory=LearningCalibration)
    confidence: LearningConfidence = Field(default_factory=LearningConfidence)
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)
    knowledge: dict[str, list[LearningPeriod]] = Field(default_factory=dict)
    patterns: list[LearningPattern] = Field(default_factory=list)
    observations: list[LearningObservation] = Field(default_factory=list)
    recommendations: list[LearningRecommendation] = Field(default_factory=list)


class LearningKnowledgeResponse(BaseModel):
    """Per-dimension knowledge view (sector/stage/country/...)."""

    dimension: str = Field(...)
    scope: str = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    entries: list[LearningPeriod] = Field(default_factory=list)


class LearningPatternResponse(BaseModel):
    """Per-dimension cohort patterns for the whole evaluated population."""

    scope: str = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    patterns: list[LearningPattern] = Field(default_factory=list)


class LearningObservationResponse(BaseModel):
    """Canonical observations about the evaluated population."""

    scope: str = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    observations: list[LearningObservation] = Field(default_factory=list)


class LearningConfidenceResponse(BaseModel):
    """Population confidence statistics for the learning snapshot."""

    scope: str = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    confidence: LearningConfidence = Field(default_factory=LearningConfidence)
    calibration: LearningCalibration = Field(default_factory=LearningCalibration)


class LearningBiasResponse(BaseModel):
    """Confidence-bias view per attribute dimension."""

    scope: str = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    knowledge: dict[str, list[LearningPeriod]] = Field(default_factory=dict)


class LearningRecommendationsResponse(BaseModel):
    """Deterministic, rule-based recommendations for the population."""

    scope: str = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    recommendations: list[LearningRecommendation] = Field(default_factory=list)


class LearningReportsResponse(BaseModel):
    """Recorded learning snapshots (headers only)."""

    period_kind: LearningPeriodKind = Field(...)
    reports: list[dict[str, object]] = Field(default_factory=list)


class LearningReportDetailResponse(BaseModel):
    """Full recorded learning snapshot payload plus its integrity hash."""

    snapshot_id: str = Field(...)
    report_id: str = Field(...)
    content_hash: str = Field(...)
    recorded_at: datetime = Field(...)
    payload: dict[str, object] = Field(default_factory=dict)

    def verify(self) -> bool:
        """``True`` when the stored hash matches the recomputed payload."""
        import json

        from predictron_engine.learning.models import LearningSnapshot

        try:
            snapshot = LearningSnapshot.model_validate(self.payload)
        except Exception:
            return False
        material = json.dumps(
            snapshot.analytic_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        import hashlib

        return hashlib.sha256(material.encode("utf-8")).hexdigest() == self.content_hash


__all__ = [
    "LearningBiasResponse",
    "LearningConfidenceResponse",
    "LearningKnowledgeResponse",
    "LearningObservationResponse",
    "LearningPatternResponse",
    "LearningRecommendationsResponse",
    "LearningReportDetailResponse",
    "LearningReportsResponse",
    "LearningSummaryResponse",
]
