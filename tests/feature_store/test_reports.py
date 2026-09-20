"""Tests for feature store reports."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureCategory,
    FeatureSnapshot,
    FeatureStatus,
    FeatureStoreSnapshot,
)
from predictron_engine.feature_store.reports import (
    CategoryDistribution,
    ComputationStatistics,
    CoverageReport,
    DependencyGraphReport,
    FeatureReportBuilder,
    FeatureStoreReport,
    FreshnessReport,
)


def make_snap(
    fid: str,
    value=None,
    status: FeatureStatus = FeatureStatus.COMPUTED,
    category: FeatureCategory = FeatureCategory.COMPANY,
    computed_at: datetime | None = None,
    evidence: list | None = None,
) -> FeatureSnapshot:
    from predictron_engine.feature_store.models import ValueType
    return FeatureSnapshot(
        company_id="c1", feature_id=fid, feature_name=fid,
        category=category, value=value,
        value_type=ValueType.NONE,
        status=status,
        computed_at=computed_at or datetime(2024, 1, 1, tzinfo=UTC),
        evidence_references=evidence or [],
    )


class TestReportDataClasses:
    def test_coverage_report_to_dict(self):
        cov = CoverageReport(
            total_registered=10, total_computed=8, coverage_ratio=0.8,
        )
        data = cov.to_dict()
        assert data["total_registered"] == 10
        assert data["coverage_ratio"] == 0.8

    def test_coverage_defaults(self):
        cov = CoverageReport()
        assert cov.total_registered == 0
        assert cov.coverage_ratio == 0.0

    def test_category_distribution_to_dict(self):
        dist = CategoryDistribution(
            distribution={"company": 5}, computed_by_category={"company": 3},
        )
        data = dist.to_dict()
        assert data["distribution"] == {"company": 5}
        assert data["computed_by_category"] == {"company": 3}

    def test_freshness_to_dict(self):
        fresh = FreshnessReport(average_age_days=10.0, stale_count=2)
        data = fresh.to_dict()
        assert data["average_age_days"] == 10.0
        assert data["stale_threshold_days"] == 365

    def test_dependency_graph_to_dict(self):
        graph = DependencyGraphReport(
            edges={"a": ["b"]}, orphans=["b"], max_depth=2, has_cycles=False,
        )
        data = graph.to_dict()
        assert data["edges"] == {"a": ["b"]}
        assert data["max_depth"] == 2

    def test_computation_stats_to_dict(self):
        stats = ComputationStatistics(total_features=10, computed=8)
        data = stats.to_dict()
        assert data["total_features"] == 10
        assert data["computed"] == 8

    def test_store_report_to_dict(self):
        report = FeatureStoreReport(company_count=3)
        data = report.to_dict()
        assert data["company_count"] == 3
        assert "coverage" in data
        assert "missing_features" in data


class TestFeatureReportBuilder:
    @pytest.fixture
    def two_company_sets(self, registry, engine, fake_record):
        from tests.feature_store.conftest import FakeRecord
        rec1 = fake_record
        rec2 = FakeRecord(record_id="c2")
        rec2.profile.founded_year = 2018
        return [
            engine.build_company_features(rec1),
            engine.build_company_features(rec2),
        ]

    def test_build_empty_report(self, report_builder):
        report = report_builder.build_report(feature_sets=[])
        assert report.company_count == 0
        assert report.coverage.total_registered == report_builder._registry.count()

    def test_build_from_snapshot(self, report_builder, two_company_sets):
        snapshot = FeatureStoreSnapshot(
            companies={s.company_id: s for s in two_company_sets},
        )
        report = report_builder.build_report(
            feature_store_snapshot=snapshot,
        )
        assert report.company_count == 2

    def test_coverage_computed(self, report_builder, two_company_sets):
        report = report_builder.build_report(feature_sets=two_company_sets)
        assert report.coverage.total_registered == \
            report_builder._registry.count()
        assert report.coverage.total_computed > 0
        assert report.coverage.coverage_ratio > 0

    def test_category_distribution(self, report_builder, two_company_sets):
        report = report_builder.build_report(feature_sets=two_company_sets)
        assert "company" in report.category_distribution.distribution
        assert report.category_distribution.computed_by_category["company"] > 0

    def test_freshness(self, report_builder, two_company_sets):
        report = report_builder.build_report(feature_sets=two_company_sets)
        assert report.freshness.fresh_count > 0

    def test_dependency_graph_edges(self, report_builder, registry):
        report = report_builder.build_report(feature_sets=[])
        graph = report.dependency_graph
        assert "benchmark_similarity" in graph.edges
        assert "company_age" in graph.edges
        assert "company_age" in graph.edges["benchmark_similarity"]

    def test_orphans_exist(self, report_builder):
        report = report_builder.build_report(feature_sets=[])
        assert len(report.dependency_graph.orphans) > 0
        assert "company_age" in report.dependency_graph.orphans

    def test_max_depth(self, report_builder):
        report = report_builder.build_report(feature_sets=[])
        assert report.dependency_graph.max_depth > 0

    def test_missing_features(self, report_builder, two_company_sets):
        report = report_builder.build_report(feature_sets=two_company_sets)
        assert isinstance(report.missing_features, list)

    def test_validation_summaries(self, report_builder, two_company_sets,
                                  registry):
        from predictron_engine.feature_store.validation import FeatureValidator
        validator = FeatureValidator(registry)
        reports = validator.validate_all(two_company_sets)
        report = report_builder.build_report(
            feature_sets=two_company_sets,
            validation_reports=reports,
        )
        assert len(report.validation_summaries) == 2
        assert "company_id" in report.validation_summaries[0]

    def test_report_deterministic(self, report_builder, two_company_sets):
        r1 = report_builder.build_report(feature_sets=two_company_sets).to_dict()
        r2 = report_builder.build_report(feature_sets=two_company_sets).to_dict()
        r1.pop("generated_at")
        r2.pop("generated_at")
        assert r1 == r2


class TestReportEdgeCases:
    def test_empty_computation_stats(self, report_builder):
        report = report_builder.build_report(feature_sets=[])
        assert report.computation_stats.total_features == 0
        assert report.computation_stats.avg_evidence_per_feature == 0.0

    def test_failed_features_counted(self, report_builder):
        snap = make_snap(
            "bad", status=FeatureStatus.COMPUTATION_ERROR,
        )
        fs = CompanyFeatureSet(company_id="c1", features={"bad": snap})
        report = report_builder.build_report(feature_sets=[fs])
        assert report.computation_stats.computation_error >= 1
        assert report.computation_stats.failed >= 1

    def test_missing_dependency_counted(self, report_builder):
        snap = make_snap(
            "bad", status=FeatureStatus.MISSING_DEPENDENCY,
        )
        fs = CompanyFeatureSet(company_id="c1", features={"bad": snap})
        report = report_builder.build_report(feature_sets=[fs])
        assert report.computation_stats.missing_dependency >= 1

    def test_evidence_stats(self, report_builder):
        with_ev = make_snap(
            "w", value=1.0,
            evidence=[{"source_type": "record", "source_id": "rid",
                       "source_field": "rf"}],
        )
        without_ev = make_snap("n", value=2.0)
        fs = CompanyFeatureSet(
            company_id="c1",
            features={"w": with_ev, "n": without_ev},
        )
        report = report_builder.build_report(feature_sets=[fs])
        assert report.computation_stats.features_with_evidence >= 1
        assert report.computation_stats.features_without_evidence >= 1

    def test_stale_features_counted(self, report_builder):
        old = datetime(2020, 1, 1, tzinfo=UTC)
        snap = make_snap("old", value=1.0, computed_at=old)
        fs = CompanyFeatureSet(company_id="c1", features={"old": snap})
        report = report_builder.build_report(feature_sets=[fs])
        assert report.freshness.stale_count >= 1

    def test_fresh_threshold_configurable(self, registry, report_builder):
        builder = FeatureReportBuilder(registry, stale_threshold_days=1)
        snap = make_snap(
            "fresh", value=1.0,
            computed_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        fs = CompanyFeatureSet(company_id="c1", features={"fresh": snap})
        report = builder.build_report(feature_sets=[fs])
        assert report.freshness.stale_count == 1
        assert report.freshness.stale_threshold_days == 1
