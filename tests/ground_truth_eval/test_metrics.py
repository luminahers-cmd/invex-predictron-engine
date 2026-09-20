"""Tests for ground-truth evaluation metrics determinism and correctness."""

from __future__ import annotations

import math

import pytest

from benchmarks.ground_truth_eval.metrics import (
    CoverageMetrics,
    ScoredSample,
    brier_score,
    build_samples,
    calibration_curves,
    calibration_metrics,
    compute_metrics,
    confusion_metrics,
    country_accuracy,
    coverage_metrics,
    false_negative_rate,
    false_positive_rate,
    grouped_accuracy,
    investment_hit_rate,
    metrics_from_dict,
    precision_at_k,
    recall_at_k,
    scoreable,
    sector_accuracy,
    stage_accuracy,
    top_decile_precision,
)


def _sample(
    company_id: str,
    *,
    pred: bool | None = None,
    actual: bool | None = None,
    score: float | None = 70.0,
    confidence: float | None = 0.7,
    decision: str | None = None,
    sector: str | None = None,
    stage: str | None = None,
    country: str | None = None,
) -> ScoredSample:
    if pred is not None and decision is None:
        decision = "invest" if pred else "pass"
    return ScoredSample(
        company_id=company_id,
        predicted_positive=pred,
        actual_positive=actual,
        score=score,
        confidence=confidence,
        decision=decision,
        sector=sector,
        stage=stage,
        country=country,
    )


def _all_tp() -> list[ScoredSample]:
    return [
        _sample("a", pred=True, actual=True),
        _sample("b", pred=True, actual=True),
    ]


class TestScoreable:
    def test_filters_unscoreable(self) -> None:
        samples = [
            _sample("a", pred=True, actual=True),
            _sample("b", pred=True, actual=None),
        ]
        assert [s.company_id for s in scoreable(samples)] == ["a"]

    def test_all_scoreable(self) -> None:
        assert len(scoreable(_all_tp())) == 2


class TestConfusion:
    @pytest.mark.parametrize(
        ("pred", "actual", "tp", "tn", "fp", "fn"),
        [
            (True, True, 1, 0, 0, 0),
            (False, False, 0, 1, 0, 0),
            (True, False, 0, 0, 1, 0),
            (False, True, 0, 0, 0, 1),
        ],
    )
    def test_each_confusion_cell(self, pred, actual, tp, tn, fp, fn) -> None:
        fm = confusion_metrics([_sample("x", pred=pred, actual=actual)])
        assert (fm.true_positives, fm.true_negatives, fm.false_positives, fm.false_negatives) == (
            tp,
            tn,
            fp,
            fn,
        )

    def test_neutral_prediction_excluded(self) -> None:
        samples = [
            _sample("a", pred=None, actual=True),
            _sample("b", pred=True, actual=True),
        ]
        fm = confusion_metrics(samples)
        assert fm.true_positives == 1
        assert fm.scoreable == 2

    def test_unscoreable_excluded(self) -> None:
        samples = [
            _sample("a", pred=True, actual=None),
            _sample("b", pred=True, actual=True),
        ]
        fm = confusion_metrics(samples)
        assert fm.total == 2
        assert fm.scoreable == 1
        assert fm.true_positives == 1

    def test_empty(self) -> None:
        fm = confusion_metrics([])
        assert fm.total == 0
        assert fm.scoreable == 0
        assert fm.accuracy is None
        assert fm.precision is None


class TestDerivedRates:
    @pytest.mark.parametrize(
        ("samples", "attr", "expected"),
        [
            (_all_tp(), "accuracy", 1.0),
            (_all_tp(), "precision", 1.0),
            (_all_tp(), "recall", 1.0),
            (_all_tp(), "specificity", None),
            (_all_tp(), "f1", 1.0),
            (_all_tp(), "balanced_accuracy", None),
            (_all_tp(), "false_positive_rate", None),
            (_all_tp(), "false_negative_rate", 0.0),
        ],
    )
    def test_perfect_classifier(self, samples, attr, expected) -> None:
        assert getattr(confusion_metrics(samples), attr) == expected

    def test_mixed_formulas(self) -> None:
        samples = [
            _sample("tp", pred=True, actual=True),
            _sample("fp", pred=True, actual=False),
            _sample("fn", pred=False, actual=True),
            _sample("tn", pred=False, actual=False),
        ]
        fm = confusion_metrics(samples)
        assert fm.accuracy == 0.5
        assert fm.precision == 0.5
        assert fm.recall == 0.5
        assert fm.specificity == 0.5
        assert fm.f1 == 0.5
        assert fm.balanced_accuracy == 0.5
        assert fm.false_positive_rate == 0.5
        assert fm.false_negative_rate == 0.5

    def test_wrapper_functions(self) -> None:
        samples = [_sample("fp", pred=True, actual=False)]
        assert false_positive_rate(samples) == 1.0
        assert false_negative_rate(samples) is None

    def test_precision_undefined_with_no_positives(self) -> None:
        samples = [_sample("a", pred=False, actual=False)]
        fm = confusion_metrics(samples)
        assert fm.precision is None
        assert fm.accuracy == 1.0


class TestBrier:
    def test_perfect(self) -> None:
        samples = [_sample("a", pred=True, actual=True, confidence=1.0)]
        assert brier_score(samples) == 0.0

    def test_wrong_with_high_confidence(self) -> None:
        samples = [_sample("a", pred=True, actual=False, confidence=1.0)]
        assert brier_score(samples) == 1.0

    def test_half_points(self) -> None:
        samples = [
            _sample("a", pred=True, actual=True, confidence=0.5),
            _sample("b", pred=True, actual=False, confidence=0.5),
        ]
        assert brier_score(samples) == 0.25

    def test_empty(self) -> None:
        assert brier_score([]) is None

    def test_excludes_unscoreable(self) -> None:
        samples = [_sample("a", pred=True, actual=None, confidence=0.9)]
        assert brier_score(samples) is None


class TestCalibration:
    def test_perfect_calibration_is_zero(self) -> None:
        samples = [
            # Benign bin: mean conf 0.5 == accuracy 0.5
            _sample("a", actual=True, confidence=0.5),
            _sample("b", actual=True, confidence=0.5),
            _sample("c", actual=False, confidence=0.5),
            _sample("d", actual=False, confidence=0.5),
            # Benign bin: mean conf 0.25 == accuracy 0.25
            _sample("e", actual=True, confidence=0.25),
            _sample("f", actual=False, confidence=0.25),
            _sample("g", actual=False, confidence=0.25),
            _sample("h", actual=False, confidence=0.25),
        ]
        cal = calibration_metrics(samples, num_bins=10)
        assert cal.expected_calibration_error == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_error(self) -> None:
        samples = [
            _sample("a", actual=False, confidence=1.0),
            _sample("b", actual=True, confidence=0.0),
        ]
        cal = calibration_metrics(samples, num_bins=10)
        assert cal.expected_calibration_error == pytest.approx(1.0, abs=1e-6)

    def test_empty(self) -> None:
        cal = calibration_metrics([], num_bins=10)
        assert cal.expected_calibration_error == 0.0
        assert cal.total_samples == 0

    def test_bins_reported(self) -> None:
        samples = [
            _sample("a", actual=True, confidence=0.9),
            _sample("b", actual=False, confidence=0.1),
        ]
        cal = calibration_metrics(samples, num_bins=10)
        assert len(cal.bins) >= 1
        assert cal.total_samples == 2

    def test_curves_are_dicts(self) -> None:
        samples = [_sample("a", actual=True, confidence=0.5)]
        curves = calibration_curves(samples, num_bins=5)
        assert all(isinstance(c, dict) for c in curves)
        assert all("bin_lower" in c for c in curves)

    def test_overconfidence_flag(self) -> None:
        samples = [
            _sample("a", actual=False, confidence=0.95),
            _sample("b", actual=False, confidence=0.95),
        ]
        cal = calibration_metrics(samples, num_bins=3)
        assert cal.overconfidence_detected is True


class TestRankingMetrics:
    def _samples(self) -> list[ScoredSample]:
        return [
            _sample("high_win", pred=True, actual=True, score=90.0),
            _sample("mid_win", pred=True, actual=True, score=70.0),
            _sample("low_loss", pred=False, actual=False, score=40.0),
            _sample("unscoreable", pred=True, actual=None, score=95.0),
        ]

    def test_precision_at_k_top1(self) -> None:
        assert precision_at_k(self._samples(), 1) == 1.0

    def test_precision_at_k_top2(self) -> None:
        assert precision_at_k(self._samples(), 2) == 1.0

    def test_precision_at_k_top3(self) -> None:
        assert precision_at_k(self._samples(), 3) == pytest.approx(2 / 3)

    def test_recall_at_k(self) -> None:
        samples = [
            _sample("a", pred=True, actual=True, score=90.0),
            _sample("b", pred=False, actual=True, score=10.0),
        ]
        assert recall_at_k(samples, 1) == pytest.approx(0.5)
        assert recall_at_k(samples, 2) == 1.0

    def test_k_zero_or_negative(self) -> None:
        assert precision_at_k(self._samples(), 0) is None
        assert recall_at_k(self._samples(), -1) is None

    def test_k_beyond_sample_count_clamps(self) -> None:
        assert precision_at_k(self._samples(), 100) == pytest.approx(2 / 3)

    def test_top_decile(self) -> None:
        samples = [_sample(f"c{i}", pred=True, actual=True, score=float(i)) for i in range(11)]
        k = max(1, math.ceil(11 / 10))
        assert top_decile_precision(samples) == precision_at_k(samples, k)

    def test_ties_broken_by_company_id(self) -> None:
        samples = [
            _sample("b_win", pred=True, actual=True, score=50.0),
            _sample("a_loss", pred=False, actual=False, score=50.0),
        ]
        top = precision_at_k(samples, 1)
        # "a_loss" sorts first on tie (ascending company_id) -> precision 0.
        assert top == 0.0


class TestInvestmentHitRate:
    def _samples(self) -> list[ScoredSample]:
        return [
            _sample("a", pred=True, actual=True, score=80.0),
            _sample("b", pred=True, actual=False, score=70.0),
            _sample("c", pred=False, actual=True, score=50.0),
        ]

    def test_decision_based(self) -> None:
        assert investment_hit_rate(self._samples()) == pytest.approx(0.5)

    def test_threshold_based(self) -> None:
        assert investment_hit_rate(self._samples(), threshold=60.0) == pytest.approx(0.5)

    def test_threshold_excludes(self) -> None:
        assert investment_hit_rate(self._samples(), threshold=90.0) is None

    def test_no_investments(self) -> None:
        samples = [_sample("a", pred=None, actual=True), _sample("b", pred=False, actual=False)]
        assert investment_hit_rate(samples) is None


class TestGroupedAccuracy:
    def _multi_sector(self) -> list[ScoredSample]:
        return [
            _sample("a", pred=True, actual=True, sector="saas"),
            _sample("b", pred=True, actual=False, sector="saas"),
            _sample("c", pred=True, actual=True, sector="fintech"),
        ]

    def test_sector_accuracy(self) -> None:
        acc = sector_accuracy(self._multi_sector())
        assert acc["saas"] == pytest.approx(0.5)
        assert acc["fintech"] == 1.0

    def test_stage_accuracy(self) -> None:
        samples = [
            _sample("a", pred=True, actual=True, stage="Series A"),
            _sample("b", pred=True, actual=False, stage="Series A"),
        ]
        assert stage_accuracy(samples) == {"Series A": 0.5}

    def test_country_accuracy(self) -> None:
        samples = [_sample("a", pred=True, actual=True, country="us")]
        assert country_accuracy(samples) == {"us": 1.0}

    def test_groups_without_scoreable_samples_omitted(self) -> None:
        samples = [
            _sample("a", pred=True, actual=None, sector="ghost"),
            _sample("b", pred=True, actual=True, sector="real"),
        ]
        assert sector_accuracy(samples) == {"real": 1.0}

    def test_grouped_sorted_output(self) -> None:
        samples = [
            _sample("a", pred=True, actual=True, sector="z"),
            _sample("b", pred=True, actual=True, sector="a"),
        ]
        assert list(grouped_accuracy(samples, lambda s: s.sector)) == ["a", "z"]


class TestCoverage:
    def test_coverage_counts(self, entry_builder, make_run) -> None:
        from benchmarks.ground_truth_eval.models import GoldenDataset

        entries = [
            entry_builder(company_id="a", status="acquired", exit_value_usd=1_000_000),
            entry_builder(company_id="b", status="shutdown"),
            entry_builder(company_id="c", status="operating", arr_usd=None),
        ]
        dataset = GoldenDataset(dataset_name="cov", benchmark_version="1.0", entries=entries)
        run = make_run(
            [
                {"company_id": "a", "decision": "invest", "overall_score": 80.0},
                {"company_id": "b", "decision": "pass", "overall_score": 10.0},
                {"company_id": "c", "success": False, "error": "x"},
            ],
            run_id="cov_run",
        )
        cov = coverage_metrics(dataset, run)
        assert cov.dataset_entries == 3
        assert cov.replayed_entries == 2
        assert cov.failed_entries == 1
        assert cov.scoreable_entries == 2
        assert cov.unscoreable_entries == 1
        assert cov.replay_coverage == pytest.approx(2 / 3)
        assert cov.scoreability == pytest.approx(2 / 2)
        assert cov.sectors["Test Sector"] == 2

    def test_empty_coverage_properties(self) -> None:
        cov = CoverageMetrics()
        assert cov.replay_coverage is None
        assert cov.scoreability is None

    def test_coverage_to_dict_roundtrip(self) -> None:
        cov = CoverageMetrics(dataset_entries=4, replayed_entries=2)
        as_dict = cov.to_dict()
        assert as_dict["replay_coverage"] == pytest.approx(0.5)


class TestBuildSamples:
    def test_pairing(self, entry_builder, make_run) -> None:
        from benchmarks.ground_truth_eval.models import GoldenDataset

        entries = [
            entry_builder(company_id="a", status="acquired", exit_value_usd=5_000_000),
            entry_builder(company_id="b", status="shutdown"),
            entry_builder(company_id="c", status="operating", arr_usd=None),
        ]
        dataset = GoldenDataset(dataset_name="samples", benchmark_version="1.0", entries=entries)
        run = make_run(
            [
                {"company_id": "a", "decision": "invest", "overall_score": 80.0},
                {"company_id": "b", "decision": "pass", "overall_score": 10.0},
                {"company_id": "c", "decision": "watch", "overall_score": 45.0},
            ],
            run_id="samples_run",
        )
        samples = build_samples(dataset, run)
        by_id = {s.company_id: s for s in samples}
        assert by_id["a"].predicted_positive is True
        assert by_id["a"].actual_positive == 1
        assert by_id["b"].predicted_positive is False
        assert by_id["b"].actual_positive == 0
        assert by_id["c"].predicted_positive is None
        assert by_id["c"].actual_positive is None

    def test_failed_entries_skipped(self, entry_builder, make_run) -> None:
        from benchmarks.ground_truth_eval.models import GoldenDataset

        dataset = GoldenDataset(
            dataset_name="s",
            benchmark_version="1.0",
            entries=[entry_builder(company_id="a")],
        )
        run = make_run([{"company_id": "a", "success": False, "error": "boom"}])
        assert build_samples(dataset, run) == []

    def test_score_fallback_prediction(self, make_run, entry_builder) -> None:
        from benchmarks.ground_truth_eval.models import GoldenDataset

        dataset = GoldenDataset(
            dataset_name="s",
            benchmark_version="1.0",
            entries=[entry_builder(company_id="a", status="shutdown")],
        )
        run = make_run([{"company_id": "a", "overall_score": 70.0}])
        s = build_samples(dataset, run)[0]
        assert s.predicted_positive is True


class TestComputeMetrics:
    def test_blob_shape(self, example_dataset, example_run, example_metrics) -> None:
        blob = example_metrics.to_dict()
        assert blob["run_id"] == "pytest_example_run"
        assert blob["confusion"]["total"] == 13
        assert blob["confusion"]["scoreable"] == 11
        assert blob["coverage"]["dataset_entries"] == 13

    def test_deterministic(self, example_dataset, example_run) -> None:
        m1 = compute_metrics(example_dataset, example_run)
        m2 = compute_metrics(example_dataset, example_run)
        assert m1.to_dict() == m2.to_dict()

    def test_golden_known_values(self, example_dataset, example_run, example_metrics) -> None:
        cm = example_metrics.confusion
        assert (cm.true_positives, cm.false_positives) == (7, 3)
        assert cm.true_negatives == 0
        assert cm.false_negatives == 0
        assert cm.accuracy == pytest.approx(0.7, abs=1e-6)
        assert cm.recall == 1.0

    def test_metric_value_lookup(self, example_metrics) -> None:
        assert example_metrics.metric_value("accuracy") == pytest.approx(0.7, abs=1e-6)
        assert example_metrics.metric_value("brier_score") is not None
        assert example_metrics.metric_value("nonsense") is None

    def test_roundtrip_from_dict(self, example_metrics) -> None:
        restored = metrics_from_dict(example_metrics.to_dict())
        assert restored.to_dict() == example_metrics.to_dict()
        assert restored.confusion.scoreable == 11

    def test_roundtrip_derived_props(self, example_metrics) -> None:
        restored = metrics_from_dict(example_metrics.to_dict())
        assert restored.coverage.replay_coverage == example_metrics.coverage.replay_coverage

    def test_from_dict_empty(self) -> None:
        restored = metrics_from_dict({})
        assert restored.run_id == ""
        assert restored.confusion.total == 0

    def test_custom_k_values(self, example_dataset, example_run) -> None:
        m = compute_metrics(example_dataset, example_run, k_values=(2, 4))
        assert set(m.precision_at_k) == {2, 4}


class TestCoverageMetricsClass:
    def test_defaults(self) -> None:
        cm = CoverageMetrics()
        assert cm.to_dict()["dataset_entries"] == 0

    def test_confusion_to_dict_keys(self, example_metrics) -> None:
        d = example_metrics.confusion.to_dict()
        assert set(d) == {
            "total",
            "scoreable",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "balanced_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "f0_5",
        }
