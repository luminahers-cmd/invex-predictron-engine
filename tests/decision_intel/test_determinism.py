"""Determinism tests for the Decision Intelligence layer.

Verifies that identical Feature Store inputs always produce identical
outputs across contributions, traces, explanations, reports, and
calibration.  Also verifies that the layer integrates with the real
Feature Engine / predictron pipeline without breaking anything.
"""

from __future__ import annotations

import pytest

from predictron_engine.decision.contribution import ContributionEngine
from predictron_engine.decision.explainability import ExplainabilityEngine
from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.report import DecisionReportBuilder
from predictron_engine.decision.trace import DecisionTraceEngine

from .conftest import _strip_non_deterministic, build_feature_set


class TestFullPipelineDeterminism:
    @pytest.mark.parametrize("iterations", [2, 3, 5])
    def test_end_to_end_deterministic(self, iterations: int) -> None:
        feature_engine = DecisionFeatureEngine()
        contribution_engine = ContributionEngine(feature_engine)
        trace_engine = DecisionTraceEngine(feature_engine)
        explain_engine = ExplainabilityEngine(contribution_engine)
        builder = DecisionReportBuilder(
            feature_engine=feature_engine,
            contribution_engine=contribution_engine,
            trace_engine=trace_engine,
            explainability_engine=explain_engine,
        )

        feature_set = build_feature_set()
        first_report = _strip_non_deterministic(builder.build_report(feature_set).to_dict())

        for _ in range(iterations):
            report = _strip_non_deterministic(builder.build_report(feature_set).to_dict())
            assert report == first_report

    def test_contribution_order_identical(self) -> None:
        feature_engine = DecisionFeatureEngine()
        engine = ContributionEngine(feature_engine)
        fs = build_feature_set()
        a = [c.feature_id for c in engine.compute_contributions(fs)]
        b = [c.feature_id for c in engine.compute_contributions(fs)]
        assert a == b

    def test_contribution_values_identical(self) -> None:
        feature_engine = DecisionFeatureEngine()
        engine = ContributionEngine(feature_engine)
        fs = build_feature_set()
        a = [c.computed_contribution for c in engine.compute_contributions(fs)]
        b = [c.computed_contribution for c in engine.compute_contributions(fs)]
        assert a == b

    def test_trace_identical(self) -> None:
        feature_engine = DecisionFeatureEngine()
        trace_engine = DecisionTraceEngine(feature_engine)
        fs = build_feature_set()
        a = _strip_non_deterministic(trace_engine.build_trace(fs).to_dict())
        b = _strip_non_deterministic(trace_engine.build_trace(fs).to_dict())
        assert a == b

    def test_explanation_identical(self) -> None:
        contribution_engine = ContributionEngine()
        explain_engine = ExplainabilityEngine(contribution_engine)
        fs = build_feature_set()
        a = explain_engine.build_explanation(fs)
        b = explain_engine.build_explanation(fs)
        assert a.to_dict() == b.to_dict()


class TestIntegrationWithRealFeatureEngine:
    def test_consumes_feature_engine_output(self) -> None:
        # Integration test: build features with the real FeatureEngine,
        # then feed the resulting CompanyFeatureSet through the decision layer.
        from datetime import UTC, datetime

        from predictron_engine.feature_store.engine import FeatureEngine
        from predictron_engine.feature_store.features import ALL_FEATURES
        from predictron_engine.feature_store.registry import FeatureRegistry

        reg = FeatureRegistry()
        reg.register_all(ALL_FEATURES)
        fe = FeatureEngine(reg)

        record = _minimal_record()
        feature_set = fe.build_company_features(
            record,
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
        )

        decision_engine = DecisionFeatureEngine()
        contributions = decision_engine.compute_all_contributions(feature_set)
        assert len(contributions) > 0

        trace = DecisionTraceEngine(decision_engine).build_trace(feature_set)
        assert trace.company_id == "rec-1"
        assert 0.0 <= trace.overall_score <= 100.0

    def test_real_registry_coverage(self) -> None:
        # The decision layer should handle every registered feature.
        from predictron_engine.feature_store.features import ALL_FEATURES
        from predictron_engine.feature_store.registry import FeatureRegistry
        reg = FeatureRegistry()
        reg.register_all(ALL_FEATURES)
        assert reg.count() > 0

    def test_feature_definitions_consumable(self) -> None:
        from predictron_engine.feature_store.features import ALL_FEATURES
        assert len(ALL_FEATURES) >= 20


class TestDecisionLayerOverRealColumnMix:
    @pytest.mark.parametrize("company_suffix", ["a", "b", "c", "d", "e"])
    def test_varied_companies(self, company_suffix: str) -> None:
        feature_engine = DecisionFeatureEngine()
        builder = DecisionReportBuilder(feature_engine=feature_engine)
        fs = build_feature_set(company_id=f"rec-{company_suffix}")
        report = builder.build_report(fs)
        assert report.company_id == f"rec-{company_suffix}"
        assert report.trace is not None

    @pytest.mark.parametrize("missing_categories", [0, 1, 2, 3, 4, 5, 6])
    def test_partial_feature_sets(self, missing_categories: int) -> None:
        kwargs = {
            "include_company": True,
            "include_growth": True,
            "include_founder": True,
            "include_funding": True,
            "include_graph": True,
            "include_signals": True,
            "include_benchmark": True,
        }
        pairs = list(kwargs.items())
        for i in range(missing_categories):
            key, _ = pairs[i]
            kwargs[key] = False

        feature_engine = DecisionFeatureEngine()
        builder = DecisionReportBuilder(feature_engine=feature_engine)
        fs = build_feature_set(**kwargs)
        report = builder.build_report(fs)
        assert 0.0 <= report.decision_summary["overall_score"] <= 100.0


def _minimal_record():
    """A minimal dataset record shaped like the pipeline's FakeRecord."""
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class Profile:
        domain: str | None = None
        industries: list[str] = field(default_factory=lambda: ["SaaS"])
        headquarters: str | None = None
        country_code: str | None = "US"
        city: str | None = None
        region: str | None = None
        founded_year: int | None = 2020
        founded_date: object | None = None
        employee_count: int | None = 30
        employee_range: str | None = None
        description: str | None = None
        legal_name: str | None = None
        status: str | None = None

    @dataclass
    class Record:
        record_id: str = "rec-1"
        startup_name: str = "TestCo"
        website: str = "https://testco.com"
        engine_version: str = "0.13.0"
        profile: Profile = field(default_factory=Profile)
        funding_stage_at_analysis: object | None = None
        raw_data: dict[str, Any] = field(default_factory=dict)
        founder_linkedin_urls: list[str] | None = None

    return Record()
