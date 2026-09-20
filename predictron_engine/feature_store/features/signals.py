"""Feature definitions for the SIGNALS category.

Deterministic signal-derived features from company timelines.
"""

from __future__ import annotations

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
    source_fields: list[str] | None = None,
    min_value: float | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.SIGNALS,
        description=description,
        value_type=value_type,
        source_fields=source_fields or [],
        min_value=min_value,
        tags=tags or ["signals"],
    )


def _get_timeline(ctx: dict[str, Any]) -> Any:
    return ctx.get("timeline")


def _tl_ev(company_id: str, field: str) -> list[dict[str, str]]:
    return [{"source_type": "timeline", "source_id": company_id,
             "source_field": field}]


def _ref_as_of(ctx: dict[str, Any]) -> datetime:
    as_of = ctx.get("as_of")
    return as_of if isinstance(as_of, datetime) else datetime.now(UTC)


def _span_years(timeline: Any) -> float | None:
    first = timeline.first
    last = timeline.last
    if first is None or last is None or first is last:
        return None
    days = (last.timestamp - first.timestamp).total_seconds() / _SECONDS_PER_DAY
    return float(max(days / _DAYS_PER_YEAR, 1e-9))


SIGNAL_FREQUENCY_DEFINITION = _make_feature(
    feature_id="signal_frequency",
    feature_name="Signal Frequency",
    description="Total signals per year across all signal types",
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["signals", "frequency"],
)


def compute_signal_frequency(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    if timeline.signal_count == 0:
        return 0.0, []
    span = _span_years(timeline)
    if span is None:
        return round(float(timeline.signal_count), 4), []
    freq = round(timeline.signal_count / span, 4)
    return freq, _tl_ev(timeline.company_id, "signal_count")


SIGNAL_DIVERSITY_DEFINITION = _make_feature(
    feature_id="signal_diversity",
    feature_name="Signal Diversity",
    description="Number of distinct signal types present",
    value_type=ValueType.INT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["signals", "diversity"],
)


def compute_signal_diversity(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    types = {s.signal_type.value for s in timeline.signals}
    return len(types), _tl_ev(timeline.company_id, "signal_types")


SIGNAL_RECENCY_DEFINITION = _make_feature(
    feature_id="signal_recency",
    feature_name="Signal Recency",
    description="Days since the most recent signal of any type",
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["signals", "recency", "freshness"],
)


def compute_signal_recency(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None or timeline.last is None:
        return None, []
    as_of = _ref_as_of(ctx)
    delta = (as_of - timeline.last.timestamp).total_seconds()
    days = round(delta / _SECONDS_PER_DAY, 4)
    return max(days, 0.0), _tl_ev(timeline.company_id, "last_signal")


ACTIVITY_SCORE_DEFINITION = _make_feature(
    feature_id="activity_score",
    feature_name="Activity Score",
    description="Signal frequency * diversity * recency bonus",
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["signals", "activity", "composite"],
)


def compute_activity_score(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None or timeline.signal_count == 0:
        return None, []
    as_of = _ref_as_of(ctx)
    span = _span_years(timeline)
    if span is None:
        freq = float(timeline.signal_count)
    else:
        freq = timeline.signal_count / span
    types = {s.signal_type.value for s in timeline.signals}
    diversity = len(types)
    delta = (as_of - timeline.last.timestamp).total_seconds()
    days_since_last = delta / _SECONDS_PER_DAY
    recency_bonus = max(1.0 / (1.0 + days_since_last / 365.25), 0.01)
    score = round(freq * diversity * recency_bonus, 4)
    return score, _tl_ev(timeline.company_id, "activity")


SIGNALS_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (SIGNAL_FREQUENCY_DEFINITION, compute_signal_frequency),
    (SIGNAL_DIVERSITY_DEFINITION, compute_signal_diversity),
    (SIGNAL_RECENCY_DEFINITION, compute_signal_recency),
    (ACTIVITY_SCORE_DEFINITION, compute_activity_score),
]
