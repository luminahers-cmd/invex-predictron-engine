"""Decision Intelligence test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from predictron_engine.decision.calibration_layer import CalibrationPoint
from predictron_engine.decision.service import DecisionIntelligenceService
from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    EvidenceReference,
    FeatureCategory,
    FeatureSnapshot,
    FeatureStatus,
    ValueType,
)


def make_snapshot(
    feature_id: str,
    feature_name: str,
    category: FeatureCategory,
    value: Any,
    *,
    company_id: str = "rec-1",
    value_type: ValueType = ValueType.NONE,
    evidence: list[EvidenceReference] | None = None,
    status: FeatureStatus = FeatureStatus.COMPUTED,
    computation_version: str = "1.0.0",
    provenance: dict[str, Any] | None = None,
) -> FeatureSnapshot:
    """Build a FeatureSnapshot with sensible deterministic defaults."""
    resolved_type = value_type
    if resolved_type == ValueType.NONE and value is not None:
        if isinstance(value, bool):
            resolved_type = ValueType.BOOL
        elif isinstance(value, int):
            resolved_type = ValueType.INT
        elif isinstance(value, float):
            resolved_type = ValueType.FLOAT
        elif isinstance(value, str):
            resolved_type = ValueType.STRING
        elif isinstance(value, list):
            resolved_type = ValueType.LIST
        elif isinstance(value, dict):
            resolved_type = ValueType.DICT

    return FeatureSnapshot(
        company_id=company_id,
        feature_id=feature_id,
        feature_name=feature_name,
        category=category,
        value=value,
        value_type=resolved_type,
        status=status,
        computation_version=computation_version,
        computed_at=datetime(2025, 1, 1, tzinfo=UTC),
        provenance=provenance or {},
        evidence_references=evidence or [
            EvidenceReference(
                source_type="timeline",
                source_id=company_id,
                source_field=f"{feature_id}_source",
            ),
        ],
    )


def build_feature_set(
    company_id: str = "rec-1",
    *,
    include_funding: bool = True,
    include_growth: bool = True,
    include_founder: bool = True,
    include_company: bool = True,
    include_graph: bool = True,
    include_signals: bool = True,
    include_benchmark: bool = True,
) -> CompanyFeatureSet:
    """Build a rich feature set exercising every category."""
    features: dict[str, FeatureSnapshot] = {}

    if include_company:
        features["company_age"] = make_snapshot(
            "company_age", "Company Age", FeatureCategory.COMPANY,
            3.0, company_id=company_id,
        )
        features["employee_band"] = make_snapshot(
            "employee_band", "Employee Band", FeatureCategory.COMPANY,
            "11-50", company_id=company_id,
        )
        features["industry"] = make_snapshot(
            "industry", "Industry", FeatureCategory.COMPANY,
            "SaaS", company_id=company_id,
        )

    if include_growth:
        features["funding_velocity"] = make_snapshot(
            "funding_velocity", "Funding Velocity", FeatureCategory.GROWTH,
            0.8, company_id=company_id,
        )
        features["hiring_velocity"] = make_snapshot(
            "hiring_velocity", "Hiring Velocity", FeatureCategory.GROWTH,
            1.5, company_id=company_id,
        )
        features["momentum_score"] = make_snapshot(
            "momentum_score", "Momentum Score", FeatureCategory.GROWTH,
            5.0, company_id=company_id,
        )

    if include_founder:
        features["founder_count"] = make_snapshot(
            "founder_count", "Founder Count", FeatureCategory.FOUNDER,
            2, company_id=company_id,
        )
        features["repeat_founder_indicator"] = make_snapshot(
            "repeat_founder_indicator", "Repeat Founder", FeatureCategory.FOUNDER,
            1.0, company_id=company_id,
        )

    if include_funding:
        features["total_funding"] = make_snapshot(
            "total_funding", "Total Funding", FeatureCategory.FUNDING,
            50_000_000.0, company_id=company_id,
        )
        features["funding_round_count"] = make_snapshot(
            "funding_round_count", "Funding Round Count", FeatureCategory.FUNDING,
            4, company_id=company_id,
        )
        features["investor_count"] = make_snapshot(
            "investor_count", "Investor Count", FeatureCategory.FUNDING,
            15, company_id=company_id,
        )

    if include_graph:
        features["graph_degree"] = make_snapshot(
            "graph_degree", "Graph Degree", FeatureCategory.KNOWLEDGE_GRAPH,
            12.0, company_id=company_id,
        )
        features["graph_density"] = make_snapshot(
            "graph_density", "Graph Density", FeatureCategory.KNOWLEDGE_GRAPH,
            0.4, company_id=company_id,
        )

    if include_signals:
        features["signal_frequency"] = make_snapshot(
            "signal_frequency", "Signal Frequency", FeatureCategory.SIGNALS,
            8.0, company_id=company_id,
        )
        features["activity_score"] = make_snapshot(
            "activity_score", "Activity Score", FeatureCategory.SIGNALS,
            9.0, company_id=company_id,
        )

    if include_benchmark:
        features["benchmark_similarity"] = make_snapshot(
            "benchmark_similarity", "Benchmark Similarity", FeatureCategory.BENCHMARK,
            0.7, company_id=company_id,
        )
        features["historical_success_rate"] = make_snapshot(
            "historical_success_rate", "Historical Success Rate",
            FeatureCategory.BENCHMARK,
            0.6, company_id=company_id,
        )

    return CompanyFeatureSet(
        company_id=company_id,
        features=features,
        build_version="1.0.0",
        built_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def rich_feature_set() -> CompanyFeatureSet:
    """A feature set spanning all seven categories."""
    return build_feature_set()


@pytest.fixture
def sparse_feature_set() -> CompanyFeatureSet:
    """A minimal feature set with only one category."""
    return build_feature_set(
        include_funding=False,
        include_growth=False,
        include_founder=False,
        include_graph=False,
        include_signals=False,
        include_benchmark=False,
    )


@pytest.fixture
def empty_feature_set() -> CompanyFeatureSet:
    """A feature set with no features at all."""
    return CompanyFeatureSet(
        company_id="rec-empty",
        features={},
        build_version="1.0.0",
        built_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def failed_feature_set() -> CompanyFeatureSet:
    """A feature set with failed features."""
    features: dict[str, FeatureSnapshot] = {}
    features["company_age"] = make_snapshot(
        "company_age", "Company Age", FeatureCategory.COMPANY, 3.0,
    )
    features["broken_feature"] = make_snapshot(
        "broken_feature", "Broken Feature", FeatureCategory.GROWTH,
        None,
        status=FeatureStatus.COMPUTATION_ERROR,
        evidence=[],
    )
    features["missing_dep"] = make_snapshot(
        "missing_dep", "Missing Dependency", FeatureCategory.FUNDING,
        None,
        status=FeatureStatus.MISSING_DEPENDENCY,
        evidence=[],
    )
    return CompanyFeatureSet(
        company_id="rec-failed",
        features=features,
        build_version="1.0.0",
        built_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def service() -> DecisionIntelligenceService:
    return DecisionIntelligenceService()


@pytest.fixture
def calibration_points() -> list[CalibrationPoint]:
    return [
        CalibrationPoint(
            benchmark_case_id="case-1",
            expected_confidence=0.80,
            actual_confidence=0.75,
            timestamp="2025-01-01T00:00:00Z",
        ),
        CalibrationPoint(
            benchmark_case_id="case-2",
            expected_confidence=0.65,
            actual_confidence=0.70,
            timestamp="2025-02-01T00:00:00Z",
        ),
        CalibrationPoint(
            benchmark_case_id="case-3",
            expected_confidence=0.50,
            actual_confidence=0.45,
            timestamp="2025-03-01T00:00:00Z",
        ),
    ]


def _strip_non_deterministic(d: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that change between runs to allow determinism comparisons."""
    import copy
    d = copy.deepcopy(d)
    d.pop("report_id", None)
    d.pop("report_timestamp", None)
    d.pop("trace_id", None)
    d.pop("computation_timestamp", None)
    cal = d.get("calibration")
    if isinstance(cal, dict):
        cal.pop("timestamp", None)
    nodes = d.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict):
                node.pop("timestamp", None)
    trace = d.get("trace")
    if isinstance(trace, dict):
        trace.pop("trace_id", None)
        trace.pop("computation_timestamp", None)
        trace_nodes = trace.get("nodes")
        if isinstance(trace_nodes, list):
            for node in trace_nodes:
                if isinstance(node, dict):
                    node.pop("timestamp", None)
    summary = d.get("decision_summary")
    if isinstance(summary, dict):
        summary.pop("trace_id", None)
    return d
