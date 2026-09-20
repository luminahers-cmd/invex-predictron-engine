"""Feature definitions for the BENCHMARK category.

Deterministic benchmark-derived reference features.
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
    value_type: ValueType = ValueType.FLOAT,
    dependencies: list[str] | None = None,
    source_fields: list[str] | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.BENCHMARK,
        description=description,
        value_type=value_type,
        dependencies=dependencies or [],
        source_fields=source_fields or [],
        tags=tags or ["benchmark"],
    )


def _bench_ev(field: str) -> list[dict[str, str]]:
    return [{"source_type": "benchmark", "source_id": "benchmark_context",
             "source_field": field}]


BENCHMARK_SIMILARITY_DEFINITION = _make_feature(
    feature_id="benchmark_similarity",
    feature_name="Benchmark Similarity",
    description="Similarity score to benchmark reference cases (0-1)",
    dependencies=["company_age", "funding_stage", "industry"],
    source_fields=["benchmark_context"],
    tags=["benchmark", "similarity"],
)


def compute_benchmark_similarity(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    bench_ctx = ctx.get("benchmark_context")
    if bench_ctx is None:
        return None, []
    val = bench_ctx.get("similarity_score")
    if isinstance(val, int | float):
        return round(float(val), 4), _bench_ev("similarity_score")
    return None, []


HISTORICAL_SUCCESS_RATE_DEFINITION = _make_feature(
    feature_id="historical_success_rate",
    feature_name="Historical Success Rate",
    description="Historical success rate for similar companies",
    source_fields=["benchmark_context"],
    tags=["benchmark", "success_rate", "historical"],
)


def compute_historical_success_rate(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    bench_ctx = ctx.get("benchmark_context")
    if bench_ctx is None:
        return None, []
    val = bench_ctx.get("historical_success_rate")
    if isinstance(val, int | float):
        return round(float(val), 4), _bench_ev("historical_success_rate")
    return None, []


SECTOR_ACCURACY_REFERENCE_DEFINITION = _make_feature(
    feature_id="sector_accuracy_reference",
    feature_name="Sector Accuracy Reference",
    description="Engine accuracy for this sector from benchmark history",
    source_fields=["benchmark_context"],
    tags=["benchmark", "accuracy", "sector"],
)


def compute_sector_accuracy_reference(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    bench_ctx = ctx.get("benchmark_context")
    if bench_ctx is None:
        return None, []
    val = bench_ctx.get("sector_accuracy")
    if isinstance(val, int | float):
        return round(float(val), 4), _bench_ev("sector_accuracy")
    return None, []


BENCHMARK_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (BENCHMARK_SIMILARITY_DEFINITION, compute_benchmark_similarity),
    (HISTORICAL_SUCCESS_RATE_DEFINITION, compute_historical_success_rate),
    (SECTOR_ACCURACY_REFERENCE_DEFINITION, compute_sector_accuracy_reference),
]
