"""Benchmark metrics — deterministic evaluation metrics for engine output quality.

Computes quantitative metrics from benchmark results to measure:
  - Reasoning consistency
  - Confidence calibration
  - Recommendation quality
  - Explainability coverage
  - Feature extraction completeness
  - Score distribution characteristics

All metrics are deterministic and reproducible. No randomness involved.

Usage:
    from benchmarks.benchmark_metrics import BenchmarkMetrics
    metrics = BenchmarkMetrics()
    report = metrics.compute(results)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from benchmarks.benchmark_runner import CaseResult


@dataclass
class MetricResult:
    """A single computed metric."""

    name: str
    value: float
    description: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class MetricsReport:
    """Complete metrics report from benchmark evaluation."""

    total_cases: int
    successful_cases: int
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def by_name(self) -> dict[str, MetricResult]:
        return {m.name: m for m in self.metrics}

    def get(self, name: str) -> MetricResult | None:
        return self.by_name.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "successful_cases": self.successful_cases,
            "metrics": [m.to_dict() for m in self.metrics],
        }


class BenchmarkMetrics:
    """Computes deterministic evaluation metrics from benchmark results.

    Each metric method takes the full results list and returns a MetricResult.
    The compute() method aggregates all metrics into a MetricsReport.
    """

    def compute(self, results: list[CaseResult]) -> MetricsReport:
        """Compute all metrics from benchmark results.

        Args:
            results: List of CaseResult objects from the benchmark runner.

        Returns:
            MetricsReport with all computed metrics.
        """
        successful = [r for r in results if r.success and r.report is not None]
        metrics: list[MetricResult] = []

        metrics.append(self._score_mean(successful))
        metrics.append(self._score_std_dev(successful))
        metrics.append(self._score_range(successful))
        metrics.append(self._confidence_mean(successful))
        metrics.append(self._confidence_std_dev(successful))
        metrics.append(self._confidence_calibration(successful))
        metrics.append(self._reasoning_consistency(successful))
        metrics.append(self._explainability_coverage(successful))
        metrics.append(self._recommendation_quality(successful))
        metrics.append(self._extraction_completeness(successful))
        metrics.append(self._processing_time_stats(results))
        metrics.append(self._coverage_distribution(results))

        return MetricsReport(
            total_cases=len(results),
            successful_cases=len(successful),
            metrics=metrics,
        )

    def _score_mean(self, results: list[CaseResult]) -> MetricResult:
        """Mean overall score across all successful cases."""
        scores = [r.report.overall_score for r in results if r.report]
        mean = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name="score_mean",
            value=round(mean, 2),
            description="Mean overall score across successful benchmark cases",
            details={"count": len(scores)},
        )

    def _score_std_dev(self, results: list[CaseResult]) -> MetricResult:
        """Standard deviation of overall scores."""
        scores = [r.report.overall_score for r in results if r.report]
        if len(scores) < 2:
            return MetricResult(
                name="score_std_dev", value=0.0,
                description="Standard deviation of overall scores (insufficient data)",
            )
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        std_dev = math.sqrt(variance)
        return MetricResult(
            name="score_std_dev",
            value=round(std_dev, 2),
            description="Standard deviation of overall scores across cases",
            details={"count": len(scores), "variance": round(variance, 2)},
        )

    def _score_range(self, results: list[CaseResult]) -> MetricResult:
        """Score range (min, max, spread)."""
        scores = [r.report.overall_score for r in results if r.report]
        if not scores:
            return MetricResult(
                name="score_range", value=0.0,
                description="Score range (no data)",
            )
        min_s, max_s = min(scores), max(scores)
        return MetricResult(
            name="score_range",
            value=round(max_s - min_s, 2),
            description="Spread between highest and lowest overall scores",
            details={"min": round(min_s, 2), "max": round(max_s, 2)},
        )

    def _confidence_mean(self, results: list[CaseResult]) -> MetricResult:
        """Mean overall confidence across successful cases."""
        confs = [r.report.overall_confidence for r in results if r.report]
        mean = sum(confs) / len(confs) if confs else 0.0
        return MetricResult(
            name="confidence_mean",
            value=round(mean, 4),
            description="Mean overall confidence across successful cases",
            details={"count": len(confs)},
        )

    def _confidence_std_dev(self, results: list[CaseResult]) -> MetricResult:
        """Standard deviation of overall confidence."""
        confs = [r.report.overall_confidence for r in results if r.report]
        if len(confs) < 2:
            return MetricResult(
                name="confidence_std_dev", value=0.0,
                description="Standard deviation of overall confidence (insufficient data)",
            )
        mean = sum(confs) / len(confs)
        variance = sum((c - mean) ** 2 for c in confs) / (len(confs) - 1)
        std_dev = math.sqrt(variance)
        return MetricResult(
            name="confidence_std_dev",
            value=round(std_dev, 4),
            description="Standard deviation of overall confidence across cases",
            details={"count": len(confs), "variance": round(variance, 4)},
        )

    def _confidence_calibration(self, results: list[CaseResult]) -> MetricResult:
        """Check if confidence tracks with data completeness.

        Measures the correlation between confidence and data completeness
        across cases. Good calibration means higher confidence when more
        data is available.
        """
        pairs: list[tuple[float, float]] = []
        for r in results:
            if r.report is None:
                continue
            conf = r.report.overall_confidence
            completeness = r.report.features.data_completeness
            pairs.append((conf, completeness))

        if len(pairs) < 2:
            return MetricResult(
                name="confidence_calibration", value=0.0,
                description="Confidence-calibration correlation (insufficient data)",
            )

        n = len(pairs)
        mean_x = sum(x for _, x in pairs) / n
        mean_y = sum(y for y, _ in pairs) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / (n - 1)
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x, _ in pairs) / (n - 1))
        std_y = math.sqrt(sum((y - mean_y) ** 2 for _, y in pairs) / (n - 1))

        if std_x == 0 or std_y == 0:
            correlation = 0.0
        else:
            correlation = cov / (std_x * std_y)

        return MetricResult(
            name="confidence_calibration",
            value=round(max(-1.0, min(1.0, correlation)), 4),
            description=(
                "Pearson correlation between confidence and data completeness. "
                "Positive values indicate confidence increases with more data."
            ),
            details={
                "n": n,
                "mean_confidence": round(mean_y, 4),
                "mean_completeness": round(mean_x, 4),
            },
        )

    def _reasoning_consistency(self, results: list[CaseResult]) -> MetricResult:
        """Measure reasoning consistency across cases.

        Checks that cases with similar characteristics produce
        similar score patterns by measuring dimension-level score variance
        within the same industry category.
        """
        dimension_scores: dict[str, list[float]] = {}
        for r in results:
            if r.report is None:
                continue
            for s in r.report.scores:
                dimension_scores.setdefault(s.dimension, []).append(s.score)

        if not dimension_scores:
            return MetricResult(
                name="reasoning_consistency", value=0.0,
                description="Reasoning consistency (no score data)",
            )

        dimension_variance: dict[str, float] = {}
        for dim, scores in dimension_scores.items():
            if len(scores) < 2:
                dimension_variance[dim] = 0.0
                continue
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
            dimension_variance[dim] = variance

        overall_variance = (
            sum(dimension_variance.values()) / len(dimension_variance)
            if dimension_variance else 0.0
        )

        consistency_score = max(0.0, 1.0 - (overall_variance / 1000.0))

        return MetricResult(
            name="reasoning_consistency",
            value=round(consistency_score, 4),
            description=(
                "Reasoning consistency score (0-1). Measures how consistent "
                "scoring patterns are across cases. Higher = more consistent."
            ),
            details={
                "dimension_variances": {
                    k: round(v, 2) for k, v in dimension_variance.items()
                },
                "overall_variance": round(overall_variance, 2),
                "dimensions_measured": len(dimension_variance),
            },
        )

    def _explainability_coverage(self, results: list[CaseResult]) -> MetricResult:
        """Measure explainability coverage across cases.

        Checks that outputs have sufficient explanatory content:
        - Scores with rationale text
        - Assessments with summary and rationale
        - Observations with source rules
        - Recommendations with titles
        """
        if not results:
            return MetricResult(
                name="explainability_coverage", value=0.0,
                description="Explainability coverage (no data)",
            )

        total_checks = 0
        passed_checks = 0
        details: dict[str, Any] = {}

        scores_with_rationale = 0
        total_scores = 0
        assessments_with_summary = 0
        total_assessments = 0
        observations_with_rule = 0
        total_observations = 0
        recs_with_title = 0
        total_recs = 0

        for r in results:
            if r.report is None:
                continue

            for s in r.report.scores:
                total_scores += 1
                total_checks += 1
                if s.rationale:
                    scores_with_rationale += 1
                    passed_checks += 1

            for a in r.report.dimension_assessments:
                total_assessments += 1
                total_checks += 1
                if a.summary:
                    assessments_with_summary += 1
                    passed_checks += 1

            for o in r.report.observations:
                total_observations += 1
                total_checks += 1
                if o.source_rule:
                    observations_with_rule += 1
                    passed_checks += 1

            for rec in r.report.recommendations:
                total_recs += 1
                total_checks += 1
                if rec.title:
                    recs_with_title += 1
                    passed_checks += 1

        coverage = passed_checks / total_checks if total_checks > 0 else 0.0

        if total_scores > 0:
            details["scores_rationale_pct"] = round(
                scores_with_rationale / total_scores * 100, 1
            )
        if total_assessments > 0:
            details["assessments_summary_pct"] = round(
                assessments_with_summary / total_assessments * 100, 1
            )
        if total_observations > 0:
            details["observations_rule_pct"] = round(
                observations_with_rule / total_observations * 100, 1
            )
        if total_recs > 0:
            details["recommendations_title_pct"] = round(
                recs_with_title / total_recs * 100, 1
            )

        details["total_checks"] = total_checks
        details["passed_checks"] = passed_checks

        return MetricResult(
            name="explainability_coverage",
            value=round(coverage, 4),
            description=(
                "Fraction of output artifacts with explanatory content (0-1). "
                "Includes score rationales, assessment summaries, observation "
                "source rules, and recommendation titles."
            ),
            details=details,
        )

    def _recommendation_quality(self, results: list[CaseResult]) -> MetricResult:
        """Measure recommendation quality signals.

        Checks that recommendations have:
        - Non-empty action text
        - Confidence scores > 0
        - Priority levels assigned
        - Unique categories
        """
        if not results:
            return MetricResult(
                name="recommendation_quality", value=0.0,
                description="Recommendation quality (no data)",
            )

        total_recs = 0
        with_action = 0
        with_confidence = 0
        with_priority = 0
        unique_categories: set[str] = set()

        for r in results:
            if r.report is None:
                continue
            for rec in r.report.recommendations:
                total_recs += 1
                if rec.action:
                    with_action += 1
                if rec.confidence > 0:
                    with_confidence += 1
                if rec.priority:
                    with_priority += 1
                if rec.category:
                    unique_categories.add(rec.category)

        if total_recs == 0:
            return MetricResult(
                name="recommendation_quality", value=0.0,
                description="Recommendation quality (no recommendations generated)",
            )

        quality_score = (
            (with_action / total_recs) * 0.3
            + (with_confidence / total_recs) * 0.3
            + (with_priority / total_recs) * 0.2
            + min(len(unique_categories) / 5, 1.0) * 0.2
        )

        return MetricResult(
            name="recommendation_quality",
            value=round(quality_score, 4),
            description=(
                "Recommendation quality score (0-1). Based on action completeness, "
                "confidence assignment, priority coverage, and category diversity."
            ),
            details={
                "total_recommendations": total_recs,
                "with_action": with_action,
                "with_confidence": with_confidence,
                "with_priority": with_priority,
                "unique_categories": sorted(unique_categories),
            },
        )

    def _extraction_completeness(self, results: list[CaseResult]) -> MetricResult:
        """Measure feature extraction completeness.

        Computes average data_completeness across all successful cases
        and checks key feature population rates.
        """
        if not results:
            return MetricResult(
                name="extraction_completeness", value=0.0,
                description="Extraction completeness (no data)",
            )

        completeness_scores: list[float] = []
        key_feature_rates: dict[str, int] = {}
        key_feature_counts: dict[str, int] = {}

        key_features = [
            "industry", "business_model", "customer_type",
            "funding_stage", "geography", "team_size_indicator",
        ]

        for r in results:
            if r.report is None:
                continue
            completeness_scores.append(r.report.features.data_completeness)
            for feat in key_features:
                key_feature_counts.setdefault(feat, 0)
                key_feature_rates.setdefault(feat, 0)
                val = getattr(r.report.features, feat, None)
                if val is not None and val != [] and val != "":
                    key_feature_rates[feat] += 1
                key_feature_counts[feat] += 1

        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores else 0.0
        )

        feature_rates = {
            k: round(key_feature_rates[k] / key_feature_counts[k], 2)
            for k in key_feature_counts
            if key_feature_counts[k] > 0
        }

        return MetricResult(
            name="extraction_completeness",
            value=round(avg_completeness, 4),
            description=(
                "Mean data completeness score across all successful cases (0-1). "
                "Higher means more features were successfully extracted."
            ),
            details={
                "key_feature_rates": feature_rates,
                "cases_measured": len(completeness_scores),
            },
        )

    def _processing_time_stats(self, results: list[CaseResult]) -> MetricResult:
        """Compute processing time statistics."""
        times = [r.processing_time_ms for r in results if r.success]
        if not times:
            return MetricResult(
                name="processing_time", value=0.0,
                description="Processing time statistics (no data)",
            )

        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        p50 = sorted(times)[len(times) // 2]

        return MetricResult(
            name="processing_time",
            value=round(mean_time, 2),
            description="Processing time statistics (milliseconds)",
            details={
                "mean_ms": round(mean_time, 2),
                "min_ms": round(min_time, 2),
                "max_ms": round(max_time, 2),
                "p50_ms": round(p50, 2),
                "cases": len(times),
            },
        )

    def _coverage_distribution(self, results: list[CaseResult]) -> MetricResult:
        """Report distribution of industry categories and stages."""
        industries: dict[str, int] = {}

        for r in results:
            if r.report is None:
                continue
            ind = r.report.features.industry or "unknown"
            industries[ind] = industries.get(ind, 0) + 1

        return MetricResult(
            name="coverage_distribution",
            value=float(len(industries)),
            description="Distribution of results across industry categories",
            details={
                "industries": industries,
                "unique_industries": len(industries),
            },
        )
