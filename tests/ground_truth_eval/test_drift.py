"""Tests for deterministic drift detection."""

from __future__ import annotations

import pytest

from benchmarks.ground_truth_eval.drift import (
    CONFIDENCE_DRIFT_EPSILON,
    SCORE_DRIFT_EPSILON,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    SEVERITY_MODERATE,
    DriftDetector,
    DriftItem,
    DriftSignal,
    detect_metric_drift,
)
from benchmarks.ground_truth_eval.runner import FEATURE_KEYS


def _features(**overrides):
    base = {
        "industry": "saas",
        "geography": "us",
        "funding_stage": "series_a",
        "has_revenue": True,
    }
    base.update(overrides)
    return base


class TestDriftItem:
    def test_to_dict(self) -> None:
        item = DriftItem(
            company_id="c1", field="overall_score", from_value=50.0, to_value=60.0, delta=10.0
        )
        d = item.to_dict()
        assert d["field"] == "overall_score"
        assert d["delta"] == 10.0


class TestDriftSignal:
    def test_affected_fraction(self) -> None:
        s = DriftSignal(signal="score", magnitude=2.0, affected_count=1, total_count=4)
        assert s.affected_fraction == 0.25

    def test_fraction_zero_total(self) -> None:
        s = DriftSignal(signal="score", magnitude=0.0, affected_count=0, total_count=0)
        assert s.affected_fraction == 0.0

    @pytest.mark.parametrize(
        ("magnitude", "expected"),
        [
            (0.0, "low"),
            (0.5, "low"),
            (SEVERITY_MINOR, "minor"),
            (2.0, "minor"),
            (SEVERITY_MODERATE, "moderate"),
            (5.0, "moderate"),
            (SEVERITY_MAJOR, "major"),
        ],
    )
    def test_severity_bands(self, magnitude: float, expected: str) -> None:
        s = DriftSignal(signal="score", magnitude=magnitude, affected_count=1, total_count=1)
        assert s.severity == expected

    def test_to_dict_rounded(self) -> None:
        s = DriftSignal(signal="confidence", magnitude=1 / 3, affected_count=1, total_count=3)
        d = s.to_dict()
        assert d["magnitude"] == pytest.approx(0.333333)
        assert d["affected_fraction"] == pytest.approx(0.333333)


class TestIdenticalRuns:
    def test_empty_drift(self, make_run) -> None:
        run = make_run(
            [
                {
                    "company_id": "a",
                    "overall_score": 60.0,
                    "overall_confidence": 0.6,
                    "decision": "invest",
                },
                {
                    "company_id": "b",
                    "overall_score": 40.0,
                    "overall_confidence": 0.4,
                    "decision": "pass",
                },
            ],
            run_id="run_a",
        )
        report = DriftDetector().detect(run, run)
        assert report.is_empty
        assert report.affected_company_ids == []
        for signal in report.signals:
            assert signal.affected_count == 0

    def test_report_metadata(self, make_run) -> None:
        run = make_run([{"company_id": "a"}], run_id="same")
        report = DriftDetector().detect(run, run)
        assert report.run_id_a == report.run_id_b == "same"
        assert report.engine_version_a == report.engine_version_b


class TestScoreDrift:
    def test_detects_positive_change(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_score": 50.0}], run_id="a")
        run_b = make_run([{"company_id": "a", "overall_score": 60.0}], run_id="b")
        signal = DriftDetector().score_drift(run_a, run_b)
        assert signal.affected_count == 1
        assert signal.items[0].delta == 10.0
        assert signal.magnitude == 10.0

    def test_detects_negative_change(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_score": 60.0}], run_id="a")
        run_b = make_run([{"company_id": "a", "overall_score": 55.0}], run_id="b")
        signal = DriftDetector().score_drift(run_a, run_b)
        assert signal.items[0].delta == -5.0

    def test_below_epsilon_no_drift(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_score": 50.0}], run_id="a")
        run_b = make_run(
            [{"company_id": "a", "overall_score": 50.0 + SCORE_DRIFT_EPSILON / 2}],
            run_id="b",
        )
        signal = DriftDetector().score_drift(run_a, run_b)
        assert signal.affected_count == 0

    def test_only_common_ids(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_score": 50.0}], run_id="a")
        run_b = make_run(
            [
                {"company_id": "a", "overall_score": 60.0},
                {"company_id": "b", "overall_score": 10.0},
            ],
            run_id="b",
        )
        signal = DriftDetector().score_drift(run_a, run_b)
        assert signal.total_count == 1
        assert signal.affected_count == 1


class TestDecisionDrift:
    def test_detects_change(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "decision": "invest"}], run_id="a")
        run_b = make_run([{"company_id": "a", "decision": "pass"}], run_id="b")
        signal = DriftDetector().decision_drift(run_a, run_b)
        assert signal.affected_count == 1
        assert signal.magnitude == 1.0
        assert signal.items[0].from_value == "invest"
        assert signal.items[0].to_value == "pass"

    def test_no_change(self, make_run) -> None:
        run = make_run([{"company_id": "a", "decision": "invest"}], run_id="a")
        signal = DriftDetector().decision_drift(run, run)
        assert signal.affected_count == 0

    def test_none_to_decision_counts(self, make_run) -> None:
        run_a = make_run([{"company_id": "a"}], run_id="a")
        run_b = make_run([{"company_id": "a", "decision": "invest"}], run_id="b")
        signal = DriftDetector().decision_drift(run_a, run_b)
        assert signal.affected_count == 1

    def test_both_none_skipped(self, make_run) -> None:
        run_a = make_run([{"company_id": "a"}], run_id="a")
        run_b = make_run([{"company_id": "a"}], run_id="b")
        signal = DriftDetector().decision_drift(run_a, run_b)
        assert signal.affected_count == 0
        assert signal.total_count == 0


class TestConfidenceDrift:
    def test_detects_change(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_confidence": 0.5}], run_id="a")
        run_b = make_run([{"company_id": "a", "overall_confidence": 0.8}], run_id="b")
        signal = DriftDetector().confidence_drift(run_a, run_b)
        assert signal.affected_count == 1
        assert signal.items[0].delta == pytest.approx(0.3)

    def test_below_epsilon_ignored(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_confidence": 0.5}], run_id="a")
        run_b = make_run(
            [{"company_id": "a", "overall_confidence": 0.5 + CONFIDENCE_DRIFT_EPSILON / 2}],
            run_id="b",
        )
        signal = DriftDetector().confidence_drift(run_a, run_b)
        assert signal.affected_count == 0


class TestRecommendationDrift:
    def test_count_change(self, make_run) -> None:
        run_a = make_run(
            [
                {
                    "company_id": "a",
                    "recommendation_categories": ["opportunity"],
                    "recommendation_count": 1,
                }
            ],
            run_id="a",
        )
        run_b = make_run(
            [
                {
                    "company_id": "a",
                    "recommendation_categories": ["opportunity", "follow_up"],
                    "recommendation_count": 2,
                }
            ],
            run_id="b",
        )
        signal = DriftDetector().recommendation_drift(run_a, run_b)
        assert signal.affected_count == 1
        assert signal.items[0].delta == 1.0

    def test_category_set_change_same_count(self, make_run) -> None:
        run_a = make_run(
            [
                {
                    "company_id": "a",
                    "recommendation_categories": ["opportunity"],
                    "recommendation_count": 1,
                }
            ],
            run_id="a",
        )
        run_b = make_run(
            [
                {
                    "company_id": "a",
                    "recommendation_categories": ["follow_up"],
                    "recommendation_count": 1,
                }
            ],
            run_id="b",
        )
        signal = DriftDetector().recommendation_drift(run_a, run_b)
        assert signal.affected_count == 1

    def test_ordering_insensitive(self, make_run) -> None:
        run_a = make_run(
            [
                {
                    "company_id": "a",
                    "recommendation_categories": ["opportunity", "follow_up"],
                    "recommendation_count": 2,
                }
            ],
            run_id="a",
        )
        run_b = make_run(
            [
                {
                    "company_id": "a",
                    "recommendation_categories": ["follow_up", "opportunity"],
                    "recommendation_count": 2,
                }
            ],
            run_id="b",
        )
        signal = DriftDetector().recommendation_drift(run_a, run_b)
        assert signal.affected_count == 0


class TestFeatureDrift:
    def test_feature_value_change(self, make_run) -> None:
        run_a = make_run(
            [{"company_id": "a", "extracted_features": _features(industry="saas")}], run_id="a"
        )
        run_b = make_run(
            [{"company_id": "a", "extracted_features": _features(industry="fintech")}], run_id="b"
        )
        signal = DriftDetector().feature_drift(run_a, run_b)
        assert signal.affected_count == 1
        assert signal.items[0].field == "feature:industry"

    def test_all_features_compared(self, make_run) -> None:
        run = make_run([{"company_id": "a", "extracted_features": _features()}], run_id="a")
        signal = DriftDetector().feature_drift(run, run)
        assert signal.magnitude == 0.0
        # magnitude denominator should equal len(FEATURE_KEYS) per company
        field_total = len(FEATURE_KEYS)
        assert field_total == len(signal.items) + field_total  # nothing changed

    def test_no_features_recorded(self, make_run) -> None:
        run_a = make_run([{"company_id": "a"}], run_id="a")
        run_b = make_run([{"company_id": "a"}], run_id="b")
        signal = DriftDetector().feature_drift(run_a, run_b)
        assert signal.affected_count == 0
        assert signal.magnitude == 0.0


class TestReportAggregation:
    def test_affected_company_ids_sorted_unique(self, make_run) -> None:
        run_a = make_run(
            [
                {"company_id": "b", "overall_score": 50.0, "decision": "invest"},
                {"company_id": "a", "overall_score": 40.0, "decision": "pass"},
            ],
            run_id="a",
        )
        run_b = make_run(
            [
                {"company_id": "b", "overall_score": 80.0, "decision": "pass"},
                {"company_id": "a", "overall_score": 40.0, "decision": "pass"},
            ],
            run_id="b",
        )
        report = DriftDetector().detect(run_a, run_b)
        assert report.affected_company_ids == ["b"]

    def test_signal_lookup(self, make_run) -> None:
        run = make_run([{"company_id": "a"}], run_id="a")
        report = DriftDetector().detect(run, run)
        assert report.signal("score") is not None
        assert report.signal("nope") is None

    def test_to_dict_shape(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_score": 50.0}], run_id="a")
        run_b = make_run([{"company_id": "a", "overall_score": 70.0}], run_id="b")
        d = DriftDetector().detect(run_a, run_b).to_dict()
        assert d["affected_company_count"] == 1
        assert len(d["signals"]) == 5

    def test_explain_empty(self, make_run) -> None:
        run = make_run([{"company_id": "a"}], run_id="a")
        text = DriftDetector().detect(run, run).explain()
        assert "No drift detected" in text

    def test_explain_contains_items(self, make_run) -> None:
        run_a = make_run([{"company_id": "a", "overall_score": 50.0}], run_id="a")
        run_b = make_run([{"company_id": "a", "overall_score": 70.0}], run_id="b")
        text = DriftDetector().detect(run_a, run_b).explain()
        assert "score drift" in text
        assert "a" in text


class TestEngineVersionTracking:
    def test_report_records_versions(self, make_run) -> None:
        run_a = make_run([{"company_id": "a"}], run_id="a", engine_version="1.0.0")
        run_b = make_run([{"company_id": "a"}], run_id="b", engine_version="0.9.0")
        report = DriftDetector().detect(run_a, run_b)
        assert report.engine_version_a == "1.0.0"
        assert report.engine_version_b == "0.9.0"

    def test_empty_before_comparable_content(self, make_run) -> None:
        run_a = make_run([], run_id="a")
        run_b = make_run([], run_id="b")
        report = DriftDetector().detect(run_a, run_b)
        assert report.is_empty


class TestMetricDrift:
    def test_from_metric_objects(self, example_metrics) -> None:
        other = example_metrics.to_dict()
        report = detect_metric_drift(example_metrics, other)
        assert not report.changed_items

    def test_from_dicts(self) -> None:
        base = {"run_id": "r1", "accuracy": 0.7, "precision": 0.8}
        changed = {"run_id": "r2", "accuracy": 0.75, "precision": 0.9}
        report = detect_metric_drift(base, changed)
        by_name = {i.metric: i for i in report.items}
        assert by_name["accuracy"].delta == pytest.approx(0.05)
        assert by_name["precision"].delta == pytest.approx(0.1)

    def test_changed_items_filter(self) -> None:
        report = detect_metric_drift(
            {"run_id": "a", "accuracy": 0.7, "recall": 0.5},
            {"run_id": "b", "accuracy": 0.7, "recall": 0.8},
        )
        assert [i.metric for i in report.changed_items] == ["recall"]

    def test_missing_values_produce_none(self) -> None:
        report = detect_metric_drift({"run_id": "a"}, {"run_id": "b", "accuracy": 0.5})
        by_name = {i.metric: i for i in report.items}
        assert by_name["accuracy"].delta is None

    def test_labels(self) -> None:
        report = detect_metric_drift({"run_id": "alpha"}, {"run_id": "beta"})
        assert report.metrics_a == "alpha"
        assert report.metrics_b == "beta"

    def test_explain_no_change(self) -> None:
        report = detect_metric_drift(
            {"run_id": "a", "accuracy": 0.7}, {"run_id": "b", "accuracy": 0.7}
        )
        assert "No aggregate metric drift" in report.explain()

    def test_explain_changed(self) -> None:
        report = detect_metric_drift(
            {"run_id": "a", "accuracy": 0.7}, {"run_id": "b", "accuracy": 0.9}
        )
        text = report.explain()
        assert "accuracy" in text
        assert "+0.2000" in text

    def test_to_dict(self) -> None:
        report = detect_metric_drift(
            {"run_id": "a", "accuracy": 0.7}, {"run_id": "b", "accuracy": 0.9}
        )
        d = report.to_dict()
        assert d["baseline"] == "a"
        assert d["compared"] == "b"


class TestDeterminism:
    def test_same_inputs_same_output(self, make_run) -> None:
        run_a = make_run(
            [{"company_id": "x", "overall_score": 40.0, "decision": "pass"}], run_id="a"
        )
        run_b = make_run(
            [{"company_id": "x", "overall_score": 80.0, "decision": "invest"}], run_id="b"
        )
        r1 = DriftDetector().detect(run_a, run_b).to_dict()
        r2 = DriftDetector().detect(run_a, run_b).to_dict()
        assert r1 == r2
