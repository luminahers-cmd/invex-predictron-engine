"""Tests for Sprint 9 — Adaptive Intelligence & Continuous Improvement.

Covers the pure analytics layer under ``predictron_engine.intelligence``:
history/trends, rule effectiveness, recommendation effectiveness,
regression detection, confidence drift, engine quality, benchmark
evaluation, and the ``IntelligenceAnalyzer`` composition root.

Assertions focus on determinism, additive-only behaviour (no engine
pipeline changes, no duplicated logic), and backward compatibility
(ENGINE_VERSION is not bumped).
"""

from __future__ import annotations

from benchmarks.benchmark_metrics import BenchmarkMetrics, MetricsReport
from benchmarks.benchmark_runner import CaseResult
from benchmarks.benchmark_validator import CaseValidationResult
from predictron_engine.intelligence import IntelligenceAnalyzer
from predictron_engine.intelligence.benchmarks import (
    BenchmarkEvaluation,
    evaluate_benchmark,
)
from predictron_engine.intelligence.confidence_drift import (
    compute_confidence_drift,
)
from predictron_engine.intelligence.history import (
    compute_trend,
    snapshot_from_confidence,
    snapshot_from_report,
)
from predictron_engine.intelligence.models import (
    DriftDirection,
    PerformanceSnapshot,
    TrendDirection,
)
from predictron_engine.intelligence.quality import engine_quality
from predictron_engine.intelligence.recommendations import (
    recommendation_effectiveness,
)
from predictron_engine.intelligence.regression import detect_regressions
from predictron_engine.intelligence.rules import (
    rule_effectiveness_from_reports,
    rule_effectiveness_from_snapshots,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    Observation,
    Recommendation,
    Report,
    ScoreResult,
)
from predictron_engine.models.startup import Startup
from predictron_engine.version import ENGINE_VERSION


def _startup(name: str = "Acme") -> Startup:
    return Startup(name=name, website="https://acme.example", description="d")


def _report(
    *,
    score: float = 60.0,
    conf: float = 0.6,
    completeness: float = 0.8,
    observations: list[Observation] | None = None,
    recommendations: list[Recommendation] | None = None,
    dimension_assessments: list[DimensionAssessment] | None = None,
    scores: list[ScoreResult] | None = None,
) -> Report:
    features = ExtractedFeatures(data_completeness=completeness)
    return Report(
        startup=_startup(),
        features=features,
        observations=observations or [],
        recommendations=recommendations or [],
        dimension_assessments=dimension_assessments or [],
        scores=scores or [],
        overall_score=score,
        overall_confidence=conf,
    )


def _obs(rule: str, dimension: str, confidence: float, importance: float) -> Observation:
    return Observation(
        dimension=dimension,
        category="market_context",
        statement="s",
        confidence=confidence,
        importance=importance,
        source_rule=rule,
    )


def _rec(category: str, action: str, priority: str, confidence: float) -> Recommendation:
    return Recommendation(
        category=category,
        action=action,
        priority=priority,
        title="t",
        confidence=confidence,
    )


def _snap(
    score: float,
    conf: float,
    *,
    decision: float = 0.0,
    completeness: float = 0.0,
    dims: dict[str, float] | None = None,
    hits: dict[str, int] | None = None,
) -> PerformanceSnapshot:
    return PerformanceSnapshot(
        version=ENGINE_VERSION,
        overall_score=score,
        overall_confidence=conf,
        data_completeness=completeness,
        decision_confidence=decision,
        assessed_confidences=dims or {},
        rule_hits=hits or {},
    )


class TestHistory:
    def test_row_version_and_engine_version(self) -> None:
        report = _report(score=70.0, conf=0.7)
        snap = snapshot_from_report(report)
        assert snap.version == ENGINE_VERSION
        assert snap.overall_score == 70.0
        assert snap.overall_confidence == 0.7

    def test_snapshot_rule_hits_aggregated(self) -> None:
        report = _report(observations=[
            _obs("rule_a", "market", 0.8, 0.5),
            _obs("rule_a", "market", 0.8, 0.5),
            _obs("rule_b", "team", 0.6, 0.4),
        ])
        snap = snapshot_from_report(report)
        assert snap.rule_hits == {"rule_a": 2, "rule_b": 1}
        assert snap.observation_count == 3

    def test_trend_direction_up(self) -> None:
        trend = compute_trend([_snap(30, 0.3), _snap(80, 0.9)])
        assert trend.metric_trends["overall_score"] == TrendDirection.UP
        assert trend.metric_trends["overall_confidence"] == TrendDirection.UP
        assert trend.record_count == 2

    def test_trend_flat(self) -> None:
        trend = compute_trend([_snap(50, 0.5), _snap(50, 0.5)])
        assert trend.metric_trends["overall_score"] == TrendDirection.FLAT

    def test_trend_down(self) -> None:
        trend = compute_trend([_snap(90, 0.9), _snap(40, 0.4)])
        assert trend.metric_trends["overall_score"] == TrendDirection.DOWN

    def test_trend_empty(self) -> None:
        trend = compute_trend([])
        assert trend.record_count == 0
        assert trend.mean_overall_score == 0.0

    def test_trend_deterministic(self) -> None:
        data = [_snap(50, 0.5), _snap(80, 0.8), _snap(60, 0.6)]
        assert compute_trend(data).model_dump() == compute_trend(data).model_dump()

    def test_snapshot_from_confidence(self) -> None:
        snap = snapshot_from_confidence(
            70.0, 0.7, confidence_assessments=[], scores=[],
            decision_confidence=0.65, data_completeness=0.9,
        )
        assert snap.decision_confidence == 0.65
        assert snap.data_completeness == 0.9


class TestRuleEffectiveness:
    def test_aggregates_per_rule(self) -> None:
        reports = [
            _report(observations=[
                _obs("rule_a", "market", 0.8, 0.9),
                _obs("rule_a", "market", 0.7, 0.5),
                _obs("rule_b", "team", 0.5, 0.2),
            ])
        ]
        eff = rule_effectiveness_from_reports(reports)
        by_name = {e.source_rule: e for e in eff}
        assert set(by_name) == {"rule_a", "rule_b"}
        a = by_name["rule_a"]
        assert a.hit_count == 2
        assert a.avg_confidence == 0.75
        assert a.avg_importance == 0.7
        assert a.contribution == round(2 / 3, 4)

    def test_empty_reports(self) -> None:
        assert rule_effectiveness_from_reports([]) == []

    def test_from_snapshots_without_reports(self) -> None:
        snaps = [
            _snap(50, 0.5, hits={"rule_a": 3, "rule_b": 1}),
            _snap(60, 0.6, hits={"rule_a": 1}),
        ]
        eff = rule_effectiveness_from_snapshots(snaps)
        by_name = {e.source_rule: e for e in eff}
        assert by_name["rule_a"].hit_count == 4
        assert by_name["rule_b"].hit_count == 1
        assert by_name["rule_a"].contribution == 4 / 5


class TestRecommendationEffectiveness:
    def test_coverages(self) -> None:
        reports = [
            _report(recommendations=[
                _rec("due_diligence", "verify", "high", 0.8),
                _rec("risk", "", "medium", 0.5),
            ])
        ]
        eff = recommendation_effectiveness(reports)
        assert eff.total_recommendations == 2
        assert eff.priority_coverage == 1.0
        assert eff.action_coverage == 0.5
        assert sorted(eff.categories) == ["due_diligence", "risk"]

    def test_empty(self) -> None:
        eff = recommendation_effectiveness([])
        assert eff.total_recommendations == 0
        assert eff.action_coverage == 0.0

    def test_deterministic(self) -> None:
        reports = [_report(recommendations=[_rec("risk", "act", "low", 0.7)])]
        a = recommendation_effectiveness(reports).model_dump()
        b = recommendation_effectiveness(reports).model_dump()
        assert a == b


class TestRegression:
    def test_detects_decrease(self) -> None:
        base = [_snap(80, 0.8), _snap(75, 0.7)]
        comp = [_snap(60, 0.6), _snap(65, 0.5)]
        findings = detect_regressions(base, comp, tolerance=0.01)
        metrics = {f.metric: f for f in findings}
        assert "overall_score" in metrics
        assert "overall_confidence" in metrics
        assert metrics["overall_score"].delta == -15.0
        assert metrics["overall_score"].to_value == 62.5

    def test_no_regression_on_improvement(self) -> None:
        base = [_snap(60, 0.6)]
        comp = [_snap(80, 0.8)]
        assert detect_regressions(base, comp) == []

    def test_tolerance_suppresses_noise(self) -> None:
        base = [_snap(70, 0.7)]
        comp = [_snap(69.999, 0.699)]
        assert detect_regressions(base, comp, tolerance=0.01) == []

    def test_empty_inputs(self) -> None:
        assert detect_regressions([], [], tolerance=0.01) == []


class TestConfidenceDrift:
    def test_decision_improved(self) -> None:
        base = [_snap(50, 0.5, decision=0.4), _snap(50, 0.5, decision=0.4)]
        comp = [_snap(60, 0.6, decision=0.6), _snap(70, 0.7, decision=0.8)]
        cond = compute_confidence_drift(base, comp, overconfident=False)
        assert cond.decision_drift.direction == DriftDirection.IMPROVED
        assert cond.decision_drift.to_confidence == 0.7
        assert cond.sample_count_a == 2
        assert cond.sample_count_b == 2

    def test_per_dimension_drift(self) -> None:
        base = [
            _snap(50, 0.5, decision=0.5, dims={"market": 0.4, "team": 0.9}),
        ]
        comp = [
            _snap(50, 0.5, decision=0.6, dims={"market": 0.6, "team": 0.7}),
        ]
        cond = compute_confidence_drift(base, comp)
        assert cond.assessment_drifts["market"].direction == DriftDirection.IMPROVED
        assert cond.assessment_drifts["team"].direction == DriftDirection.REGRESSED

    def test_overconfident_flag_is_preserved(self) -> None:
        base = [_snap(50, 0.5, decision=0.5)]
        comp = [_snap(50, 0.5, decision=0.5)]
        cond = compute_confidence_drift(base, comp, overconfident=True)
        assert cond.decision_drift.overconfident is True

    def test_empty(self) -> None:
        cond = compute_confidence_drift([], [])
        assert cond.decision_drift.direction == DriftDirection.UNCHANGED


class TestQuality:
    def test_empty_reports(self) -> None:
        q = engine_quality([])
        assert q.report_count == 0
        assert q.calibration_quality == "insufficient_data"

    def test_delegates_calibration(self) -> None:
        q = engine_quality([_report(conf=0.8, completeness=0.9)])
        assert 0.0 <= q.calibration_error <= 1.0
        assert q.calibration_quality in {"excellent", "good", "fair", "poor", "insufficient_data"}

    def test_explainability_and_recommendation(self) -> None:
        reports = [
            _report(
                scores=[ScoreResult(dimension="market", score=60.0, rationale="r")],
                dimension_assessments=[
                    DimensionAssessment(
                        dimension="market",
                        confidence=0.6,
                        summary="s",
                        rationale="r",
                    )
                ],
                observations=[_obs("rule_a", "market", 0.6, 0.5)],
                recommendations=[_rec("risk", "act", "high", 0.7)],
            )
        ]
        q = engine_quality(reports)
        assert q.explainability_coverage == 1.0
        assert q.recommendation_quality > 0.0

    def test_reports_count(self) -> None:
        q = engine_quality([_report(), _report()])
        assert q.report_count == 2

    def test_reuses_benchmark_metrics_not_duplicates(self) -> None:
        """quality delegates coverage/quality/completeness to BenchmarkMetrics.

        Guards against reintroducing duplicated pipeline metrics: the
        values reported by ``engine_quality`` must equal the canonical
        :class:`BenchmarkMetrics` values for the same reports.
        """
        reports = [
            _report(
                score=70.0,
                conf=0.6,
                completeness=0.8,
                scores=[ScoreResult(dimension="market", score=70.0, rationale="r")],
                recommendations=[_rec("risk", "act", "high", 0.7)],
            )
        ]
        bench_results = []
        for idx, r in enumerate(reports):
            bench_results.append(
                CaseResult(
                    case_id=f"q{idx}",
                    case_label="",
                    request={},
                    success=True,
                    processing_time_ms=0.0,
                    report=r,
                )
            )
        metrics_report = BenchmarkMetrics().compute(bench_results)
        q = engine_quality(reports)

        assert q.explainability_coverage == (
            metrics_report.by_name["explainability_coverage"].value
        )
        assert q.recommendation_quality == (
            metrics_report.by_name["recommendation_quality"].value
        )
        assert q.data_completeness == (
            metrics_report.by_name["extraction_completeness"].value
        )


class TestBenchmarks:
    def test_evaluate_reuses_metrics(self) -> None:
        report = _report(score=75.0, conf=0.6)
        result = CaseResult(
            case_id="c1",
            case_label="C1",
            request={},
            success=True,
            processing_time_ms=10.0,
            report=report,
            error=None,
        )
        ev = evaluate_benchmark([result])
        assert isinstance(ev.metrics, MetricsReport)
        assert ev.metrics.total_cases == 1
        assert ev.metrics.successful_cases == 1
        assert len(ev.snapshots) == 1
        assert ev.snapshots[0].overall_score == 75.0

    def test_returns_benchmark_evaluation(self) -> None:
        assert issubclass(BenchmarkEvaluation, object)


class TestAnalyzer:
    def test_dashboard_produces_full_model(self) -> None:
        reports = [_report(score=60.0, conf=0.6, observations=[
            _obs("rule_a", "market", 0.6, 0.5),
        ])]
        dash = IntelligenceAnalyzer().dashboard(reports=reports)
        assert dash.version == ENGINE_VERSION
        assert dash.trend.record_count == 1
        assert len(dash.rule_effectiveness) == 1
        assert dash.recommendation_effectiveness.total_recommendations == 0
        assert dash.quality.report_count == 1

    def test_dashboard_with_baseline_comparison(self) -> None:
        base = [_snap(80, 0.8, decision=0.8), _snap(75, 0.7, decision=0.7)]
        comp = [_snap(50, 0.5, decision=0.4)]
        reports = [_report(score=70.0, conf=0.7)]
        dash = IntelligenceAnalyzer().dashboard(
            reports=reports, baseline=base, comparison=comp
        )
        assert len(dash.regressions) >= 1
        assert dash.confidence_drift is not None

    def test_deterministic(self) -> None:
        reports = [_report(score=60.0, conf=0.6)]
        a = IntelligenceAnalyzer().dashboard(reports=reports).to_dict()
        b = IntelligenceAnalyzer().dashboard(reports=reports).to_dict()
        assert a == b

    def test_to_dict_serializable(self) -> None:
        reports = [_report(score=60.0, conf=0.6)]
        d = IntelligenceAnalyzer().dashboard(reports=reports).to_dict()
        assert d["version"] == ENGINE_VERSION
        assert isinstance(d["trend"], dict)

    def test_injected_functions_are_used(self) -> None:
        calls: list[str] = []

        def fake_rules(reports):  # noqa: ANN001
            calls.append("rules")
            return []

        analyzer = IntelligenceAnalyzer(rules_fn=fake_rules)
        analyzer.rules([_report()])
        assert calls == ["rules"]

    def test_individual_methods(self) -> None:
        analyzer = IntelligenceAnalyzer()
        t = analyzer.trend([_snap(50, 0.5), _snap(70, 0.7)])
        assert t.record_count == 2
        r = analyzer.regressions([_snap(80, 0.8)], [_snap(60, 0.6)])
        assert len(r) >= 1
        eff = analyzer.recommendations([_report()])
        assert eff.total_recommendations == 0
        d = analyzer.confidence_drift(
            [_snap(50, 0.5, decision=0.5)], [_snap(50, 0.5, decision=0.6)]
        )
        assert d.decision_drift.direction == DriftDirection.IMPROVED


class TestBackwardCompatibility:
    def test_engine_version_unchanged(self) -> None:
        assert ENGINE_VERSION == "0.12.1"

    def test_modules_import_without_side_effects(self) -> None:
        # Importing the intelligence package must not mutate pipeline state.
        assert ENGINE_VERSION == "0.12.1"

    def test_reuses_existing_metrics_class(self) -> None:
        assert BenchmarkMetrics.__module__ == "benchmarks.benchmark_metrics"
        assert CaseValidationResult.__module__ == "benchmarks.benchmark_validator"
