"""Feature Store reports.

Deterministic reports covering feature coverage, completeness, freshness,
category distribution, dependency graph, missing features, and computation statistics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureStatus,
    FeatureStoreSnapshot,
)
from predictron_engine.feature_store.registry import FeatureRegistry
from predictron_engine.feature_store.validation import (
    FeatureValidationReport,
)


@dataclass
class CoverageReport:
    """Feature coverage statistics."""

    total_registered: int = 0
    total_computed: int = 0
    total_failed: int = 0
    total_missing: int = 0
    coverage_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_registered": self.total_registered,
            "total_computed": self.total_computed,
            "total_failed": self.total_failed,
            "total_missing": self.total_missing,
            "coverage_ratio": self.coverage_ratio,
        }


@dataclass
class CategoryDistribution:
    """Feature count by category."""

    distribution: dict[str, int] = field(default_factory=dict)
    computed_by_category: dict[str, int] = field(default_factory=dict)
    failed_by_category: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": dict(sorted(self.distribution.items())),
            "computed_by_category": dict(sorted(self.computed_by_category.items())),
            "failed_by_category": dict(sorted(self.failed_by_category.items())),
        }


@dataclass
class FreshnessReport:
    """Feature freshness statistics."""

    average_age_days: float = 0.0
    oldest_feature_days: float = 0.0
    newest_feature_days: float = 0.0
    stale_count: int = 0
    fresh_count: int = 0
    stale_threshold_days: int = 365

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_age_days": self.average_age_days,
            "oldest_feature_days": self.oldest_feature_days,
            "newest_feature_days": self.newest_feature_days,
            "stale_count": self.stale_count,
            "fresh_count": self.fresh_count,
            "stale_threshold_days": self.stale_threshold_days,
        }


@dataclass
class DependencyGraphReport:
    """Dependency graph analysis."""

    edges: dict[str, list[str]] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)
    max_depth: int = 0
    has_cycles: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": dict(sorted(self.edges.items())),
            "orphans": self.orphans,
            "max_depth": self.max_depth,
            "has_cycles": self.has_cycles,
        }


@dataclass
class ComputationStatistics:
    """Computation performance statistics."""

    total_features: int = 0
    computed: int = 0
    failed: int = 0
    missing_dependency: int = 0
    computation_error: int = 0
    avg_evidence_per_feature: float = 0.0
    features_with_evidence: int = 0
    features_without_evidence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_features": self.total_features,
            "computed": self.computed,
            "failed": self.failed,
            "missing_dependency": self.missing_dependency,
            "computation_error": self.computation_error,
            "avg_evidence_per_feature": self.avg_evidence_per_feature,
            "features_with_evidence": self.features_with_evidence,
            "features_without_evidence": self.features_without_evidence,
        }


@dataclass
class FeatureStoreReport:
    """Complete feature store report."""

    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    company_count: int = 0
    coverage: CoverageReport = field(default_factory=CoverageReport)
    category_distribution: CategoryDistribution = field(
        default_factory=CategoryDistribution
    )
    freshness: FreshnessReport = field(default_factory=FreshnessReport)
    dependency_graph: DependencyGraphReport = field(
        default_factory=DependencyGraphReport
    )
    computation_stats: ComputationStatistics = field(
        default_factory=ComputationStatistics
    )
    missing_features: list[str] = field(default_factory=list)
    validation_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "company_count": self.company_count,
            "coverage": self.coverage.to_dict(),
            "category_distribution": self.category_distribution.to_dict(),
            "freshness": self.freshness.to_dict(),
            "dependency_graph": self.dependency_graph.to_dict(),
            "computation_stats": self.computation_stats.to_dict(),
            "missing_features": self.missing_features,
            "validation_summaries": self.validation_summaries,
        }


class FeatureReportBuilder:
    """Builds deterministic feature store reports."""

    def __init__(
        self,
        registry: FeatureRegistry,
        *,
        stale_threshold_days: int = 365,
    ) -> None:
        self._registry = registry
        self._stale_threshold_days = stale_threshold_days

    def build_report(
        self,
        feature_store_snapshot: FeatureStoreSnapshot | None = None,
        feature_sets: list[CompanyFeatureSet] | None = None,
        validation_reports: list[FeatureValidationReport] | None = None,
    ) -> FeatureStoreReport:
        """Build a complete report from available data."""
        all_sets: list[CompanyFeatureSet] = []
        if feature_store_snapshot is not None:
            all_sets = list(feature_store_snapshot.companies.values())
        elif feature_sets is not None:
            all_sets = feature_sets

        report = FeatureStoreReport(
            company_count=len(all_sets),
        )

        self._build_coverage(all_sets, report)
        self._build_category_distribution(all_sets, report)
        self._build_freshness(all_sets, report)
        self._build_dependency_graph(report)
        self._build_computation_stats(all_sets, report)
        self._build_missing_features(all_sets, report)

        if validation_reports:
            for vr in validation_reports:
                report.validation_summaries.append(vr.to_dict())

        return report

    def _build_coverage(
        self, feature_sets: list[CompanyFeatureSet], report: FeatureStoreReport
    ) -> None:
        total_registered = self._registry.count()
        if not feature_sets:
            report.coverage = CoverageReport(
                total_registered=total_registered,
            )
            return

        all_feature_ids: set[str] = set()
        computed_ids: set[str] = set()
        failed_ids: set[str] = set()

        for fs in feature_sets:
            for fid, snap in fs.features.items():
                all_feature_ids.add(fid)
                if snap.status == FeatureStatus.COMPUTED:
                    computed_ids.add(fid)
                else:
                    failed_ids.add(fid)

        total_computed = len(computed_ids)
        total_failed = len(failed_ids)
        total_missing = total_registered - len(all_feature_ids & set(self._registry.list_ids()))

        coverage_ratio = (
            round(total_computed / total_registered, 4)
            if total_registered > 0
            else 0.0
        )

        report.coverage = CoverageReport(
            total_registered=total_registered,
            total_computed=total_computed,
            total_failed=total_failed,
            total_missing=max(total_missing, 0),
            coverage_ratio=coverage_ratio,
        )

    def _build_category_distribution(
        self, feature_sets: list[CompanyFeatureSet], report: FeatureStoreReport
    ) -> None:
        dist: Counter[str] = Counter()
        computed_dist: Counter[str] = Counter()
        failed_dist: Counter[str] = Counter()

        for defn in self._registry.list_all():
            dist[defn.category.value] += 1

        for fs in feature_sets:
            for snap in fs.features.values():
                if snap.status == FeatureStatus.COMPUTED:
                    computed_dist[snap.category.value] += 1
                else:
                    failed_dist[snap.category.value] += 1

        report.category_distribution = CategoryDistribution(
            distribution=dict(dist),
            computed_by_category=dict(computed_dist),
            failed_by_category=dict(failed_dist),
        )

    def _build_freshness(
        self, feature_sets: list[CompanyFeatureSet], report: FeatureStoreReport
    ) -> None:
        now = datetime.now(UTC)
        ages: list[float] = []
        stale = 0
        fresh = 0

        for fs in feature_sets:
            for snap in fs.features.values():
                age_days = (now - snap.computed_at).total_seconds() / 86400.0
                ages.append(age_days)
                if age_days > self._stale_threshold_days:
                    stale += 1
                else:
                    fresh += 1

        if ages:
            avg_age = round(sum(ages) / len(ages), 4)
            report.freshness = FreshnessReport(
                average_age_days=avg_age,
                oldest_feature_days=round(max(ages), 4),
                newest_feature_days=round(min(ages), 4),
                stale_count=stale,
                fresh_count=fresh,
                stale_threshold_days=self._stale_threshold_days,
            )

    def _build_dependency_graph(
        self, report: FeatureStoreReport
    ) -> None:
        graph = self._registry.dependency_graph()
        orphans = [
            fid for fid, deps in graph.items() if not deps
        ]
        max_depth = 0
        for fid in self._registry.list_ids():
            deps = self._registry.all_dependencies(fid)
            depth = len(deps) - 1
            max_depth = max(max_depth, depth)

        report.dependency_graph = DependencyGraphReport(
            edges=graph,
            orphans=orphans,
            max_depth=max_depth,
            has_cycles=False,
        )

    def _build_computation_stats(
        self, feature_sets: list[CompanyFeatureSet], report: FeatureStoreReport
    ) -> None:
        total = 0
        computed = 0
        failed = 0
        missing_dep = 0
        comp_error = 0
        evidence_total = 0
        with_evidence = 0
        without_evidence = 0

        for fs in feature_sets:
            for snap in fs.features.values():
                total += 1
                if snap.status == FeatureStatus.COMPUTED:
                    computed += 1
                    if snap.evidence_references:
                        with_evidence += 1
                        evidence_total += len(snap.evidence_references)
                    else:
                        without_evidence += 1
                elif snap.status == FeatureStatus.MISSING_DEPENDENCY:
                    missing_dep += 1
                else:
                    comp_error += 1
                    failed += 1

        avg_evidence = (
            round(evidence_total / with_evidence, 4)
            if with_evidence > 0
            else 0.0
        )

        report.computation_stats = ComputationStatistics(
            total_features=total,
            computed=computed,
            failed=failed,
            missing_dependency=missing_dep,
            computation_error=comp_error,
            avg_evidence_per_feature=avg_evidence,
            features_with_evidence=with_evidence,
            features_without_evidence=without_evidence,
        )

    def _build_missing_features(
        self, feature_sets: list[CompanyFeatureSet], report: FeatureStoreReport
    ) -> None:
        registered = set(self._registry.list_ids())
        computed: set[str] = set()
        for fs in feature_sets:
            for fid, snap in fs.features.items():
                if snap.status == FeatureStatus.COMPUTED:
                    computed.add(fid)

        report.missing_features = sorted(registered - computed)
