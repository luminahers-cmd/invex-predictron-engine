"""Feature definitions for the FOUNDER category.

Deterministic founder-related features from records and outcomes.
"""

from __future__ import annotations

from typing import Any

from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureComputer


def _make_feature(
    feature_id: str,
    feature_name: str,
    description: str,
    value_type: ValueType = ValueType.INT,
    source_fields: list[str] | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.FOUNDER,
        description=description,
        value_type=value_type,
        source_fields=source_fields or [],
        tags=tags or ["founder"],
    )


def _ev(source_id: str, source_field: str) -> list[dict[str, str]]:
    return [{"source_type": "record", "source_id": source_id,
             "source_field": source_field}]


FOUNDER_COUNT_DEFINITION = _make_feature(
    feature_id="founder_count",
    feature_name="Founder Count",
    description="Number of founders / co-founders identified",
    source_fields=["raw_data"],
    tags=["founder", "count"],
)


def compute_founder_count(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    raw = getattr(record, "raw_data", None) or {}
    founders = raw.get("founders") if isinstance(raw, dict) else None
    if isinstance(founders, list):
        return len(founders), _ev(record.record_id, "raw_data.founders")
    linkedin = None
    if isinstance(raw, dict):
        linkedin = raw.get("founder_linkedin_urls")
    if linkedin is None:
        linkedin = getattr(record, "founder_linkedin_urls", None)
    if isinstance(linkedin, list) and linkedin:
        return len(linkedin), _ev(record.record_id, "founder_linkedin_urls")
    return None, []


REPEAT_FOUNDER_DEFINITION = _make_feature(
    feature_id="repeat_founder_indicator",
    feature_name="Repeat Founder Indicator",
    description="1 if any founder has prior founding history, else 0",
    value_type=ValueType.INT,
    source_fields=["raw_data"],
    tags=["founder", "repeat"],
)


def compute_repeat_founder_indicator(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    raw = getattr(record, "raw_data", None) or {}
    founders = raw.get("founders") if isinstance(raw, dict) else None
    if isinstance(founders, list):
        for founder in founders:
            if isinstance(founder, dict) and founder.get("prior_companies"):
                return 1, _ev(record.record_id, "raw_data.founders")
    return 0, []


FOUNDER_CHANGE_COUNT_DEFINITION = _make_feature(
    feature_id="founder_change_count",
    feature_name="Founder Change Count",
    description="Number of founder changes detected in outcome events",
    value_type=ValueType.INT,
    source_fields=["outcome.founder_changes"],
    tags=["founder", "changes"],
)


def compute_founder_change_count(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    outcome = ctx.get("outcome")
    if outcome is None:
        return 0, []
    events = getattr(outcome, "outcome_events", None) or []
    change_count = 0
    for event in events:
        etype = None
        if isinstance(event, dict):
            etype = event.get("event_type")
        elif hasattr(event, "event_type"):
            etype = event.event_type
        if etype == "founder_change":
            change_count += 1
    if change_count > 0:
        ev = [{
            "source_type": "outcome",
            "source_id": outcome.outcome_id,
            "source_field": "founder_changes",
        }]
        return change_count, ev
    return change_count, []


FOUNDER_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (FOUNDER_COUNT_DEFINITION, compute_founder_count),
    (REPEAT_FOUNDER_DEFINITION, compute_repeat_founder_indicator),
    (FOUNDER_CHANGE_COUNT_DEFINITION, compute_founder_change_count),
]
