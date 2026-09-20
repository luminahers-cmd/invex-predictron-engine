"""Feature definitions for the GROWTH category.

Deterministic growth metrics derived from signal timelines and records.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureComputer

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.25


def _make_feature(
    feature_id: str,
    feature_name: str,
    description: str,
    value_type: ValueType = ValueType.FLOAT,
    dependencies: list[str] | None = None,
    source_fields: list[str] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.GROWTH,
        description=description,
        value_type=value_type,
        dependencies=dependencies or [],
        source_fields=source_fields or [],
        min_value=min_value,
        max_value=max_value,
        tags=tags or ["growth"],
    )


def _age_days(timestamp: datetime, as_of: datetime) -> float:
    return (as_of - timestamp).total_seconds() / _SECONDS_PER_DAY


def _get_timeline(ctx: dict[str, Any]) -> Any:
    return ctx.get("timeline")


def _tl_ev(company_id: str, field: str) -> list[dict[str, str]]:
    return [{"source_type": "timeline", "source_id": company_id,
             "source_field": field}]


FUNDING_VELOCITY_DEFINITION = _make_feature(
    feature_id="funding_velocity",
    feature_name="Funding Velocity",
    description="Funding rounds per year over the company's history",
    value_type=ValueType.FLOAT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["growth", "funding", "velocity"],
)


def compute_funding_velocity(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    from predictron_engine.dataset.signals.model import SignalType
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    funding = [
        s for s in timeline.signals
        if s.signal_type == SignalType.FUNDING_ROUND
    ]
    if not funding:
        return 0.0, []
    if len(funding) < 2:
        val = round(1.0 / max(_DAYS_PER_YEAR, 1.0), 6)
        return val, _tl_ev(timeline.company_id, "funding_rounds")
    span = funding[-1].timestamp - funding[0].timestamp
    span_days = span.total_seconds() / _SECONDS_PER_DAY
    span_years = max(span_days / _DAYS_PER_YEAR, 1e-9)
    velocity = round(len(funding) / span_years, 6)
    return velocity, _tl_ev(timeline.company_id, "funding_rounds")


HIRING_VELOCITY_DEFINITION = _make_feature(
    feature_id="hiring_velocity",
    feature_name="Hiring Velocity",
    description="Employee milestone signals per year",
    value_type=ValueType.FLOAT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["growth", "hiring", "velocity"],
)


def compute_hiring_velocity(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    from predictron_engine.dataset.signals.model import SignalType
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    milestones = [
        s for s in timeline.signals
        if s.signal_type == SignalType.EMPLOYEE_MILESTONE
    ]
    if not milestones:
        return 0.0, []
    if len(milestones) < 2:
        val = round(1.0 / max(_DAYS_PER_YEAR, 1.0), 6)
        return val, _tl_ev(timeline.company_id, "employee_milestones")
    span = milestones[-1].timestamp - milestones[0].timestamp
    span_days = span.total_seconds() / _SECONDS_PER_DAY
    span_years = max(span_days / _DAYS_PER_YEAR, 1e-9)
    velocity = round(len(milestones) / span_years, 6)
    return velocity, _tl_ev(timeline.company_id, "employee_milestones")


MILESTONE_FREQUENCY_DEFINITION = _make_feature(
    feature_id="milestone_frequency",
    feature_name="Milestone Frequency",
    description="Total milestone signals per year",
    value_type=ValueType.FLOAT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["growth", "milestones", "frequency"],
)


def compute_milestone_frequency(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    from predictron_engine.dataset.signals.model import SignalType
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    types = {
        SignalType.ARR_MILESTONE,
        SignalType.REVENUE_MILESTONE,
        SignalType.EMPLOYEE_MILESTONE,
    }
    milestones = [s for s in timeline.signals if s.signal_type in types]
    if not milestones:
        return 0.0, []
    if len(milestones) < 2:
        val = round(1.0 / max(_DAYS_PER_YEAR, 1.0), 6)
        return val, _tl_ev(timeline.company_id, "milestones")
    span = milestones[-1].timestamp - milestones[0].timestamp
    span_days = span.total_seconds() / _SECONDS_PER_DAY
    span_years = max(span_days / _DAYS_PER_YEAR, 1e-9)
    return round(len(milestones) / span_years, 6), _tl_ev(timeline.company_id, "milestones")


GROWTH_CONSISTENCY_DEFINITION = _make_feature(
    feature_id="growth_consistency",
    feature_name="Growth Consistency",
    description="CV of inter-milestone intervals (lower = more consistent)",
    value_type=ValueType.FLOAT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["growth", "consistency"],
)


def compute_growth_consistency(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    from predictron_engine.dataset.signals.model import SignalType
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    types = {
        SignalType.ARR_MILESTONE,
        SignalType.REVENUE_MILESTONE,
        SignalType.EMPLOYEE_MILESTONE,
    }
    milestones = [s for s in timeline.signals if s.signal_type in types]
    if len(milestones) < 3:
        return None, []
    intervals = [
        (milestones[i + 1].timestamp - milestones[i].timestamp).total_seconds()
        / _SECONDS_PER_DAY
        for i in range(len(milestones) - 1)
    ]
    mean_interval = sum(intervals) / len(intervals)
    if mean_interval == 0:
        return 0.0, []
    variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
    std_dev = math.sqrt(variance)
    cv = round(std_dev / mean_interval, 6) if mean_interval else 0.0
    return cv, _tl_ev(timeline.company_id, "milestone_intervals")


MOMENTUM_SCORE_DEFINITION = _make_feature(
    feature_id="momentum_score",
    feature_name="Momentum Score",
    description="Recency-weighted signal confidence (half-life 180d)",
    value_type=ValueType.FLOAT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["growth", "momentum"],
)


def compute_momentum_score(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    as_of = ctx.get("as_of")
    if not isinstance(as_of, datetime):
        as_of = datetime.now(UTC)
    half_life = 180.0
    score = 0.0
    for signal in timeline.signals:
        age = max(_age_days(signal.timestamp, as_of), 0.0)
        score += signal.confidence * math.exp(-age / half_life)
    score = round(score, 4)
    return score, _tl_ev(timeline.company_id, "momentum")


GROWTH_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (FUNDING_VELOCITY_DEFINITION, compute_funding_velocity),
    (HIRING_VELOCITY_DEFINITION, compute_hiring_velocity),
    (MILESTONE_FREQUENCY_DEFINITION, compute_milestone_frequency),
    (GROWTH_CONSISTENCY_DEFINITION, compute_growth_consistency),
    (MOMENTUM_SCORE_DEFINITION, compute_momentum_score),
]
