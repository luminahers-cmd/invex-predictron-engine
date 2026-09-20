"""Tests for the extended ground-truth metrics (Milestone V1.4).

Covers F0.5 in the confusion block, observed base rates, ROC-AUC and
average precision, and the to_dict / metrics_from_dict round trip that the
read-only benchmark accuracy view relies on.
"""

from __future__ import annotations

from benchmarks.ground_truth_eval.metrics import (
    BaseRates,
    ConfusionMetrics,
    GroundTruthMetrics,
    ScoredSample,
    average_precision,
    base_rates,
    confusion_metrics,
    metrics_from_dict,
    roc_auc,
)


def _sample(
    company_id: str,
    predicted: bool | None,
    actual: bool | None,
    *,
    score: float = 50.0,
    decision: str = "invest",
) -> ScoredSample:
    return ScoredSample(
        company_id=company_id,
        predicted_positive=predicted,
        actual_positive=actual,
        score=score,
        confidence=0.6,
        decision=decision,
    )


class TestConfusionF05:
    def test_f05_value(self) -> None:
        samples = [
            _sample("tp1", True, True, score=90.0),
            _sample("tp2", True, True, score=80.0),
            _sample("fp1", True, False, score=70.0),
            _sample("tn1", False, False, score=60.0),
            _sample("fn1", False, True, score=50.0),
        ]
        fm = confusion_metrics(samples)
        assert (fm.true_positives, fm.false_positives) == (2, 1)
        assert (fm.true_negatives, fm.false_negatives) == (1, 1)
        assert round(fm.precision, 6) == round(2 / 3, 6)
        assert round(fm.recall, 6) == round(2 / 3, 6)
        # F0.5 with precision == recall collapses to the common value.
        assert round(float(fm.f05), 4) == round(2 / 3, 4)

    def test_f05_weights_precision(self) -> None:
        # High recall, low precision => F0.5 < F1.
        samples = [
            _sample("tp1", True, True, score=90.0),
            _sample("fp1", True, False, score=70.0),
            _sample("fp2", True, False, score=60.0),
            _sample("fp3", True, False, score=50.0),
        ]
        fm = confusion_metrics(samples)
        assert fm.f05 < fm.f1

    def test_to_dict_key_is_f0_5(self) -> None:
        samples = [_sample("a", True, True, score=90.0)]
        block = confusion_metrics(samples).to_dict()
        assert "f0_5" in block
        assert block["f0_5"] is not None


class TestBaseRates:
    def test_prevalence(self) -> None:
        samples = [
            _sample("a", True, True, score=90.0),
            _sample("b", True, True, score=80.0),
            _sample("c", True, True, score=70.0),
            _sample("d", False, False, score=60.0),
            _sample("e", False, False, score=50.0),
        ]
        br = base_rates(samples)
        assert br.total == 5
        assert br.positives == 3
        assert br.negatives == 2
        assert br.positive_rate == 0.6
        assert br.negative_rate == 0.4

    def test_no_scoreable_samples_stays_unknown(self) -> None:
        br = base_rates([])
        assert br.total == 0
        assert br.positive_rate is None
        assert br.negative_rate is None

    def test_ignores_unscoreable(self) -> None:
        samples = [
            _sample("a", True, None, score=90.0),
            _sample("b", True, True, score=80.0),
            _sample("c", False, False, score=70.0),
        ]
        br = base_rates(samples)
        assert br.total == 2
        assert br.positive_rate == 0.5


class TestRankingMetrics:
    def test_perfect_rank_auc_is_one(self) -> None:
        samples = [
            _sample("pos1", True, True, score=90.0),
            _sample("pos2", True, True, score=80.0),
            _sample("neg1", False, False, score=70.0),
            _sample("neg2", False, False, score=60.0),
        ]
        assert roc_auc(samples) == 1.0

    def test_reversed_rank_auc_is_zero(self) -> None:
        samples = [
            _sample("pos1", True, True, score=60.0),
            _sample("neg1", False, False, score=90.0),
        ]
        assert roc_auc(samples) == 0.0

    def test_auc_undefined_without_two_classes(self) -> None:
        samples = [_sample("p", True, True, score=90.0)]
        assert roc_auc(samples) is None

    def test_average_precision_perfect(self) -> None:
        samples = [
            _sample("pos1", True, True, score=90.0),
            _sample("pos2", True, True, score=80.0),
            _sample("neg1", False, False, score=70.0),
        ]
        assert average_precision(samples) == 1.0

    def test_average_precision_undefined_without_positives(self) -> None:
        samples = [_sample("n", False, False, score=70.0)]
        assert average_precision(samples) is None


class TestMetricsRoundTrip:
    def test_round_trip_preserves_extensions(self) -> None:
        metrics = GroundTruthMetrics(
            run_id="run_x",
            confusion=ConfusionMetrics(total=5, scoreable=5, true_positives=4),
            base_rates=BaseRates(total=5, positives=4, positive_rate=0.8),
            roc_auc=0.8125,
            average_precision=0.9,
        )
        revived = metrics_from_dict(metrics.to_dict())
        assert revived.run_id == "run_x"
        assert revived.confusion.f05 == metrics.confusion.f05
        assert revived.base_rates.positive_rate == 0.8
        assert revived.roc_auc == 0.8125
        assert revived.average_precision == 0.9

    def test_round_trip_fills_absent_fields(self) -> None:
        revived = metrics_from_dict({"run_id": "run_y"})
        assert revived.run_id == "run_y"
        assert revived.roc_auc is None
        assert revived.average_precision is None
        assert revived.base_rates.positive_rate is None

    def test_metric_value_headlines(self) -> None:
        samples = [
            _sample("a", True, True, score=90.0),
            _sample("b", True, True, score=80.0),
            _sample("c", False, False, score=70.0),
        ]
        metrics = GroundTruthMetrics(
            confusion=confusion_metrics(samples),
            roc_auc=1.0,
            average_precision=1.0,
        )
        assert metrics.metric_value("accuracy") == 1.0
        assert metrics.metric_value("roc_auc") == 1.0
        assert metrics.metric_value("average_precision") == 1.0
