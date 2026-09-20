"""Feature definitions for the FUNDING category.

Deterministic funding features from signal timelines and records.
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


def _make_feature(
    feature_id: str,
    feature_name: str,
    description: str,
    value_type: ValueType = ValueType.FLOAT,
    dependencies: list[str] | None = None,
    source_fields: list[str] | None = None,
    min_value: float | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.FUNDING,
        description=description,
        value_type=value_type,
        dependencies=dependencies or [],
        source_fields=source_fields or [],
        min_value=min_value,
        tags=tags or ["funding"],
    )


def _get_timeline(ctx: dict[str, Any]) -> Any:
    return ctx.get("timeline")


def _funding_signals(timeline: Any) -> list[Any]:
    from predictron_engine.dataset.signals.model import SignalType
    return [
        s for s in timeline.signals
        if s.signal_type == SignalType.FUNDING_ROUND
    ]


def _tl_ev(company_id: str, field: str) -> list[dict[str, str]]:
    return [{"source_type": "timeline", "source_id": company_id,
             "source_field": field}]


def _is_numeric_amount(val: Any) -> bool:
    return isinstance(val, int | float) and not isinstance(val, bool)


TOTAL_FUNDING_DEFINITION = _make_feature(
    feature_id="total_funding",
    feature_name="Total Funding",
    description="Total known funding raised in USD",
    source_fields=["timeline.signals.metadata.amount_usd"],
    min_value=0.0,
    tags=["funding", "total"],
)


def compute_total_funding(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    funding = _funding_signals(timeline)
    total = 0.0
    for signal in funding:
        amount = signal.metadata.get("amount_usd")
        if _is_numeric_amount(amount):
            total += float(amount)
    return round(total, 4), _tl_ev(timeline.company_id, "funding_amounts")


FUNDING_ROUND_COUNT_DEFINITION = _make_feature(
    feature_id="funding_round_count",
    feature_name="Funding Round Count",
    description="Number of funding rounds recorded",
    value_type=ValueType.INT,
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["funding", "count", "rounds"],
)


def compute_funding_round_count(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    count = len(_funding_signals(timeline))
    return count, _tl_ev(timeline.company_id, "funding_rounds")


AVERAGE_ROUND_SIZE_DEFINITION = _make_feature(
    feature_id="average_round_size",
    feature_name="Average Round Size",
    description="Average funding round size in USD",
    source_fields=["timeline.signals.metadata.amount_usd"],
    min_value=0.0,
    tags=["funding", "average", "round_size"],
)


def compute_average_round_size(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    funding = _funding_signals(timeline)
    amounts = []
    for signal in funding:
        amount = signal.metadata.get("amount_usd")
        if _is_numeric_amount(amount) and amount > 0:
            amounts.append(float(amount))
    if not amounts:
        return None, []
    avg = round(sum(amounts) / len(amounts), 4)
    return avg, _tl_ev(timeline.company_id, "funding_amounts")


INVESTOR_COUNT_DEFINITION = _make_feature(
    feature_id="investor_count",
    feature_name="Investor Count",
    description="Number of unique investors across all rounds",
    value_type=ValueType.INT,
    source_fields=["timeline.signals.metadata.investors"],
    min_value=0.0,
    tags=["funding", "investors"],
)


def compute_investor_count(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    funding = _funding_signals(timeline)
    investors: set[str] = set()
    for signal in funding:
        inv_list = signal.metadata.get("investors", [])
        if not isinstance(inv_list, list):
            continue
        for inv in inv_list:
            if isinstance(inv, str):
                investors.add(inv)
            elif isinstance(inv, dict):
                name = inv.get("name", "")
                if isinstance(name, str) and name:
                    investors.add(name)
    return len(investors), _tl_ev(timeline.company_id, "investors")


FUNDING_RECENCY_DEFINITION = _make_feature(
    feature_id="funding_recency",
    feature_name="Funding Recency",
    description="Days since the most recent funding round",
    source_fields=["timeline.signals"],
    min_value=0.0,
    tags=["funding", "recency"],
)


def compute_funding_recency(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    timeline = _get_timeline(ctx)
    if timeline is None:
        return None, []
    funding = _funding_signals(timeline)
    if not funding:
        return None, []
    as_of = ctx.get("as_of", datetime.now(UTC))
    if not isinstance(as_of, datetime):
        as_of = datetime.now(UTC)
    delta = (as_of - funding[-1].timestamp).total_seconds()
    days = round(delta / _SECONDS_PER_DAY, 4)
    return max(days, 0.0), _tl_ev(timeline.company_id, "last_funding")


FUNDING_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (TOTAL_FUNDING_DEFINITION, compute_total_funding),
    (FUNDING_ROUND_COUNT_DEFINITION, compute_funding_round_count),
    (AVERAGE_ROUND_SIZE_DEFINITION, compute_average_round_size),
    (INVESTOR_COUNT_DEFINITION, compute_investor_count),
    (FUNDING_RECENCY_DEFINITION, compute_funding_recency),
]
