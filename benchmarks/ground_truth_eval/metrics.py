"""Ground-truth evaluation metrics (Project E5).

Deterministic metrics measuring how well the engine's *fresh* predictions
(``BenchmarkRun``) perform against verified outcomes (``GoldenDataset``).

Principles
----------
* Everything is deterministic: identical inputs always produce identical
  numbers.  Ties are broken by ``company_id`` ordering.
* No fabricated labels: samples whose verified outcome is un-scoreable
  (``actual_positive is None``) are excluded from every binary metric.
  Undefined metrics (e.g. precision with no predicted positives) are
  ``None``, never ``NaN`` or a made-up zero.
* Calibration reuses the engine's existing reliability-diagram machinery
  (:func:`predictron_engine.decision.calibration_sprint8.compute_calibration_error`)
  so the platform does not duplicate calibration logic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from benchmarks.ground_truth_eval.dataset import dataset_hash
from benchmarks.ground_truth_eval.models import GoldenDataset, predicted_positive
from benchmarks.ground_truth_eval.runner import BenchmarkRun, RunEntryOutput
from predictron_engine.decision.calibration_sprint8 import (
    CalibrationBin,
    compute_calibration_error,
    detect_overconfidence,
)


@dataclass
class ScoredSample:
    """One company paired with its fresh prediction and verified outcome.

    ``actual_positive`` is ``None`` when the verified outcome is
    un-scoreable; the sample is then excluded from binary metrics.
    ``predicted_positive`` is ``None`` for neutral decisions (watch /
    investigate further), which cannot be coerced into an invest/pass call.
    """

    company_id: str
    predicted_positive: bool | None
    actual_positive: bool | None
    score: float | None
    confidence: float | None
    decision: str | None
    sector: str | None = None
    stage: str | None = None
    country: str | None = None


def build_samples(
    dataset: GoldenDataset,
    run: BenchmarkRun,
) -> list[ScoredSample]:
    """Pair dataset entries with their run outputs into scored samples."""
    samples: list[ScoredSample] = []
    for entry in dataset.entries:
        output = run.entry_by_id(entry.company_id)
        if output is None or not output.success:
            continue
        predicted = _predicted_from_output(output)
        outcome = entry.verified_outcome.binary_outcome()
        samples.append(
            ScoredSample(
                company_id=entry.company_id,
                predicted_positive=predicted,
                actual_positive=(outcome == 1) if outcome is not None else None,
                score=output.overall_score,
                confidence=output.overall_confidence,
                decision=output.decision,
                sector=entry.sector,
                stage=entry.stage,
                country=entry.country,
            )
        )
    return samples


def _predicted_from_output(output: RunEntryOutput) -> bool | None:
    if output.decision is not None:
        return predicted_positive(output.decision)
    if output.overall_score is not None:
        # Deterministic fallback for engine outputs without a decision:
        # mirror the engine threshold mapping (score >= threshold).
        if output.overall_score >= 60.0:
            return True
        if output.overall_score < 45.0:
            return False
        return None
    return None


def scoreable(samples: list[ScoredSample]) -> list[ScoredSample]:
    """Return samples with a verified binary ground truth."""
    return [s for s in samples if s.actual_positive is not None]


def _confusion(samples: list[ScoredSample]) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for s in scoreable(samples):
        pred, actual = s.predicted_positive, bool(s.actual_positive)
        if pred is None:
            continue
        if pred is True and actual is True:
            tp += 1
        elif pred is True and actual is False:
            fp += 1
        elif pred is False and actual is True:
            fn += 1
        elif pred is False and actual is False:
            tn += 1
    return tp, tn, fp, fn


@dataclass
class ConfusionMetrics:
    """Binary classification metrics over verified samples."""

    total: int = 0
    scoreable: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    f05: float | None = None
    balanced_accuracy: float | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total": self.total,
            "scoreable": self.scoreable,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": _round_or_none(self.accuracy),
            "precision": _round_or_none(self.precision),
            "recall": _round_or_none(self.recall),
            "specificity": _round_or_none(self.specificity),
            "f1": _round_or_none(self.f1),
            "f0_5": _round_or_none(self.f05),
            "balanced_accuracy": _round_or_none(self.balanced_accuracy),
            "false_positive_rate": _round_or_none(self.false_positive_rate),
            "false_negative_rate": _round_or_none(self.false_negative_rate),
        }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def confusion_metrics(samples: list[ScoredSample]) -> ConfusionMetrics:
    """Compute binary confusion metrics over verified samples."""
    tp, tn, fp, fn = _confusion(samples)
    scoreable_count = len(scoreable(samples))
    metrics = ConfusionMetrics(
        total=len(samples),
        scoreable=scoreable_count,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )
    denom = tp + tn + fp + fn
    if denom > 0:
        metrics.accuracy = (tp + tn) / denom
    if tp + fp > 0:
        metrics.precision = tp / (tp + fp)
    if tp + fn > 0:
        metrics.recall = tp / (tp + fn)
        metrics.false_negative_rate = fn / (tp + fn)
    if tn + fp > 0:
        metrics.specificity = tn / (tn + fp)
        metrics.false_positive_rate = fp / (tn + fp)
    if metrics.precision is not None and metrics.recall is not None:
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = (
                2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            )
            metrics.f05 = (
                1.25
                * metrics.precision
                * metrics.recall
                / (0.25 * metrics.precision + metrics.recall)
            )
    if metrics.recall is not None and metrics.specificity is not None:
        metrics.balanced_accuracy = (metrics.recall + metrics.specificity) / 2
    return metrics


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMetrics:
    """Expected/max calibration error plus reliability diagram bins."""

    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    overconfidence_detected: bool = False
    bins: tuple[CalibrationBin, ...] = ()
    total_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "maximum_calibration_error": round(self.maximum_calibration_error, 4),
            "overconfidence_detected": self.overconfidence_detected,
            "overconfident_bins_count": len([b for b in self.bins if b.overconfident]),
            "bins": [b.to_dict() for b in self.bins],
            "total_samples": self.total_samples,
        }


def calibration_metrics(
    samples: list[ScoredSample],
    *,
    num_bins: int = 10,
) -> CalibrationMetrics:
    """Compute expected calibration error over *scoreable* samples.

    Uses the engine's existing reliability-diagram computation, pairing
    each prediction's overall confidence with its binary outcome.
    """
    sc = scoreable(samples)
    predictions = [round(float(s.confidence or 0.0), 6) for s in sc]
    outcomes = [int(bool(s.actual_positive)) for s in sc]
    if not predictions:
        return CalibrationMetrics()
    ece, max_ce, bins = compute_calibration_error(predictions, outcomes, num_bins=num_bins)
    overconfident, _ = detect_overconfidence(bins)
    return CalibrationMetrics(
        expected_calibration_error=ece,
        maximum_calibration_error=max_ce,
        overconfidence_detected=overconfident,
        bins=bins,
        total_samples=len(predictions),
    )


def brier_score(samples: list[ScoredSample]) -> float | None:
    """Mean Brier score over scoreable samples: mean of (conf - outcome)^2."""
    sc = scoreable(samples)
    if not sc:
        return None
    squared = [(float(s.confidence or 0.0) - int(bool(s.actual_positive))) ** 2 for s in sc]
    return round(sum(squared) / len(squared), 6)


def calibration_curves(
    samples: list[ScoredSample],
    *,
    num_bins: int = 10,
) -> list[dict[str, object]]:
    """Reliability-diagram curve points (bin_lower, mean_conf, accuracy)."""
    sc = scoreable(samples)
    predictions = [float(s.confidence or 0.0) for s in sc]
    outcomes = [int(bool(s.actual_positive)) for s in sc]
    _, _, bins = compute_calibration_error(predictions, outcomes, num_bins=num_bins)
    return [b.to_dict() for b in bins]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def _ranked(samples: list[ScoredSample]) -> list[ScoredSample]:
    """Scoreable samples sorted by overall score desc, ties by company_id."""
    sc = scoreable(samples)
    return sorted(sc, key=lambda s: (-float(s.score or 0.0), s.company_id))


def precision_at_k(samples: list[ScoredSample], k: int) -> float | None:
    """Precision among the top-``k`` companies ranked by score."""
    if k <= 0:
        return None
    top = _ranked(samples)[:k]
    if not top:
        return None
    relevant = sum(
        1 for s in top if s.actual_positive is not None and bool(s.actual_positive) is True
    )
    return round(relevant / len(top), 6)


def recall_at_k(samples: list[ScoredSample], k: int) -> float | None:
    """Recall among the top-``k`` companies: TP@K / total positives."""
    if k <= 0:
        return None
    top = _ranked(samples)[:k]
    positives = sum(1 for s in scoreable(samples) if bool(s.actual_positive) is True)
    if positives == 0:
        return None
    tp = sum(1 for s in top if s.actual_positive is not None and bool(s.actual_positive) is True)
    return round(tp / positives, 6)


def top_decile_precision(samples: list[ScoredSample]) -> float | None:
    """Precision of the top decile of scoreable companies (ceil(n/10))."""
    sc = scoreable(samples)
    if not sc:
        return None
    k = max(1, math.ceil(len(sc) / 10))
    return precision_at_k(sc, k)


def investment_hit_rate(
    samples: list[ScoredSample],
    *,
    threshold: float | None = None,
) -> float | None:
    """Fraction of invested companies that succeeded.

    "Invested" is defined by the decision (strong_invest/invest) when
    ``threshold`` is None, or by ``score >= threshold`` when given.
    """
    sc = scoreable(samples)
    invested: list[ScoredSample] = []
    for s in sc:
        if threshold is not None:
            if s.score is not None and s.score >= threshold:
                invested.append(s)
        elif s.predicted_positive is True:
            invested.append(s)
    if not invested:
        return None
    hits = sum(1 for s in invested if bool(s.actual_positive) is True)
    return round(hits / len(invested), 6)


def false_positive_rate(samples: list[ScoredSample]) -> float | None:
    """FP / (FP + TN) — fraction of true negatives wrongly flagged positive."""
    fm = confusion_metrics(samples)
    return fm.false_positive_rate


def false_negative_rate(samples: list[ScoredSample]) -> float | None:
    """FN / (FN + TP) — fraction of true positives missed."""
    fm = confusion_metrics(samples)
    return fm.false_negative_rate


# ---------------------------------------------------------------------------
# Base rates & ranking discrimination
# ---------------------------------------------------------------------------


@dataclass
class BaseRates:
    """Observed success/failure prevalence over verified samples.

    Computed from verified outcomes only — never the engine's label.
    ``positive_rate`` / ``negative_rate`` are ``None`` when no sample is
    scoreable (no rate is fabricated).
    """

    total: int = 0
    positives: int = 0
    negatives: int = 0
    positive_rate: float | None = None
    negative_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "positives": self.positives,
            "negatives": self.negatives,
            "positive_rate": _round_or_none(self.positive_rate),
            "negative_rate": _round_or_none(self.negative_rate),
        }


def base_rates(samples: list[ScoredSample]) -> BaseRates:
    """Success/failure prevalence over verified (scoreable) samples."""
    sc = scoreable(samples)
    positives = sum(1 for s in sc if bool(s.actual_positive) is True)
    total = len(sc)
    if total == 0:
        return BaseRates()
    positive_rate = positives / total
    return BaseRates(
        total=total,
        positives=positives,
        negatives=total - positives,
        positive_rate=positive_rate,
        negative_rate=1.0 - positive_rate,
    )


def _scoreable_ranked_pairs(samples: list[ScoredSample]) -> list[tuple[float, int]]:
    """``(score, actual)`` pairs for scoreable samples with a score, sorted
    ascending by score.  Ties keep the input (company_id-driven) order."""
    pairs = [
        (float(s.score or 0.0), int(bool(s.actual_positive)))
        for s in scoreable(samples)
        if s.score is not None
    ]
    return sorted(pairs, key=lambda pair: pair[0])


def roc_auc(samples: list[ScoredSample]) -> float | None:
    """Area under the ROC curve over (overall score, verified outcome).

    ``None`` when fewer than two distinct thresholds, or when the cohort has
    no positives or no negatives (the statistic is undefined).  Ties are
    grouped so equal scores produce a single operating point.
    """
    pairs = _scoreable_ranked_pairs(samples)
    if len(pairs) < 2:
        return None
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    descending = sorted(pairs, key=lambda pair: (pair[0], -pair[1]), reverse=True)
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    i = 0
    total = len(descending)
    while i < total:
        score = descending[i][0]
        while i < total and descending[i][0] == score:
            _, y = descending[i]
            if y:
                tp += 1
            else:
                fp += 1
            i += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return round(auc, 6)


def average_precision(samples: list[ScoredSample]) -> float | None:
    """Average precision (area under the PR curve) over verified samples.

    ``None`` when no scoreable sample has a score or the cohort has no
    positives.  Ties are ordered with positives first so ties produce the
    deterministic best-case operating point.
    """
    pairs = _scoreable_ranked_pairs(samples)
    if not pairs:
        return None
    n_pos = sum(y for _, y in pairs)
    if n_pos == 0:
        return None
    descending = sorted(pairs, key=lambda pair: (pair[0], -pair[1]), reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, y in descending:
        if y:
            tp += 1
        else:
            fp += 1
        if y:
            recall = tp / n_pos
            precision = tp / (tp + fp)
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return round(ap, 6)


# ---------------------------------------------------------------------------
# Grouped accuracy
# ---------------------------------------------------------------------------


def grouped_accuracy(
    samples: list[ScoredSample],
    group: Callable[[ScoredSample], str | None],
) -> dict[str, float]:
    """Per-group accuracy over verified samples.

    Groups are emitted in sorted order; groups with zero scoreable samples
    are omitted (no fabricated rates).
    """
    buckets: dict[str, list[ScoredSample]] = {}
    for s in samples:
        key = group(s)
        if key is None:
            continue
        buckets.setdefault(key, []).append(s)
    result: dict[str, float] = {}
    for key in sorted(buckets):
        fm = confusion_metrics(buckets[key])
        if fm.accuracy is not None:
            result[key] = round(fm.accuracy, 6)
    return result


def sector_accuracy(
    samples: list[ScoredSample],
) -> dict[str, float]:
    """Per-sector accuracy."""
    return grouped_accuracy(samples, lambda s: s.sector)


def stage_accuracy(
    samples: list[ScoredSample],
) -> dict[str, float]:
    """Per-stage accuracy."""
    return grouped_accuracy(samples, lambda s: s.stage)


def country_accuracy(
    samples: list[ScoredSample],
) -> dict[str, float]:
    """Per-country accuracy."""
    return grouped_accuracy(samples, lambda s: s.country)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@dataclass
class CoverageMetrics:
    """Dataset/run coverage statistics."""

    dataset_entries: int = 0
    replayed_entries: int = 0
    failed_entries: int = 0
    scoreable_entries: int = 0
    unscoreable_entries: int = 0
    sectors: dict[str, int] = field(default_factory=dict)
    stages: dict[str, int] = field(default_factory=dict)
    countries: dict[str, int] = field(default_factory=dict)

    @property
    def replay_coverage(self) -> float | None:
        if self.dataset_entries == 0:
            return None
        return round(self.replayed_entries / self.dataset_entries, 6)

    @property
    def scoreability(self) -> float | None:
        if self.replayed_entries == 0:
            return None
        return round(self.scoreable_entries / self.replayed_entries, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_entries": self.dataset_entries,
            "replayed_entries": self.replayed_entries,
            "failed_entries": self.failed_entries,
            "scoreable_entries": self.scoreable_entries,
            "unscoreable_entries": self.unscoreable_entries,
            "replay_coverage": _round_or_none(self.replay_coverage),
            "scoreability": _round_or_none(self.scoreability),
            "sectors": self.sectors,
            "stages": self.stages,
            "countries": self.countries,
        }


def coverage_metrics(
    dataset: GoldenDataset,
    run: BenchmarkRun,
) -> CoverageMetrics:
    """Compute coverage over the dataset/run pair."""
    samples = build_samples(dataset, run)
    failed = sum(1 for e in run.entries if not e.success)
    replayed = sum(1 for e in run.entries if e.success)
    unscoreable = sum(1 for e in dataset.entries if e.verified_outcome.binary_outcome() is None)
    scoreable_count = dataset.scoreable_entry_count()
    metrics = CoverageMetrics(
        dataset_entries=dataset.entry_count,
        replayed_entries=replayed,
        failed_entries=failed,
        scoreable_entries=scoreable_count,
        unscoreable_entries=unscoreable,
    )
    for s in samples:
        if s.sector is not None:
            metrics.sectors[s.sector] = metrics.sectors.get(s.sector, 0) + 1
        if s.stage is not None:
            metrics.stages[s.stage] = metrics.stages.get(s.stage, 0) + 1
        if s.country is not None:
            metrics.countries[s.country] = metrics.countries.get(s.country, 0) + 1
    return metrics


# ---------------------------------------------------------------------------
# Aggregate metrics object
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthMetrics:
    """Complete deterministic metric block for one benchmark run."""

    run_id: str = ""
    engine_version: str = ""
    benchmark_version: str = ""
    dataset_name: str = ""
    dataset_hash: str = ""
    confusion: ConfusionMetrics = field(default_factory=ConfusionMetrics)
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    brier: float | None = None
    precision_at_k: dict[int, float | None] = field(default_factory=dict)
    recall_at_k: dict[int, float | None] = field(default_factory=dict)
    top_decile_precision: float | None = None
    investment_hit_rate: float | None = None
    investment_hit_rate_at_60: float | None = None
    base_rates: BaseRates = field(default_factory=BaseRates)
    roc_auc: float | None = None
    average_precision: float | None = None
    coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    sectors: dict[str, float] = field(default_factory=dict)
    stages: dict[str, float] = field(default_factory=dict)
    countries: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "benchmark_version": self.benchmark_version,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "confusion": self.confusion.to_dict(),
            "calibration": self.calibration.to_dict(),
            "brier_score": _round_or_none(self.brier),
            "precision_at_k": {
                str(k): _round_or_none(v) for k, v in sorted(self.precision_at_k.items())
            },
            "recall_at_k": {str(k): _round_or_none(v) for k, v in sorted(self.recall_at_k.items())},
            "top_decile_precision": _round_or_none(self.top_decile_precision),
            "investment_hit_rate": _round_or_none(self.investment_hit_rate),
            "investment_hit_rate_at_60": _round_or_none(self.investment_hit_rate_at_60),
            "base_rates": self.base_rates.to_dict(),
            "roc_auc": _round_or_none(self.roc_auc),
            "average_precision": _round_or_none(self.average_precision),
            "coverage": self.coverage.to_dict(),
            "sector_accuracy": self.sectors,
            "stage_accuracy": self.stages,
            "country_accuracy": self.countries,
        }

    def metric_value(self, name: str) -> float | None:
        """Return a top-level float metric by name for drift/trend reporting."""
        values: dict[str, float | None] = {
            "accuracy": self.confusion.accuracy,
            "precision": self.confusion.precision,
            "recall": self.confusion.recall,
            "specificity": self.confusion.specificity,
            "f1": self.confusion.f1,
            "balanced_accuracy": self.confusion.balanced_accuracy,
            "false_positive_rate": self.confusion.false_positive_rate,
            "false_negative_rate": self.confusion.false_negative_rate,
            "expected_calibration_error": self.calibration.expected_calibration_error,
            "maximum_calibration_error": self.calibration.maximum_calibration_error,
            "brier_score": self.brier,
            "top_decile_precision": self.top_decile_precision,
            "investment_hit_rate": self.investment_hit_rate,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
        }
        return values.get(name)


def metrics_from_dict(block: dict[str, Any]) -> GroundTruthMetrics:
    """Reconstruct a metric block from its ``to_dict()`` representation.

    Used to resurrect a stored metric block from benchmark history without
    re-running the engine.  Absent fields default to empty/None rather than
    being fabricated.
    """
    confusion = _confusion_from_dict(block.get("confusion") or {})
    calibration = _calibration_from_dict(block.get("calibration") or {})
    coverage = _coverage_from_dict(block.get("coverage") or {})
    return GroundTruthMetrics(
        run_id=str(block.get("run_id", "")),
        engine_version=str(block.get("engine_version", "")),
        benchmark_version=str(block.get("benchmark_version", "")),
        dataset_name=str(block.get("dataset_name", "")),
        dataset_hash=str(block.get("dataset_hash", "")),
        confusion=confusion,
        calibration=calibration,
        brier=_parse_float(block.get("brier_score")),
        precision_at_k=_parse_int_float_map(block.get("precision_at_k")),
        recall_at_k=_parse_int_float_map(block.get("recall_at_k")),
        top_decile_precision=_parse_float(block.get("top_decile_precision")),
        investment_hit_rate=_parse_float(block.get("investment_hit_rate")),
        investment_hit_rate_at_60=_parse_float(block.get("investment_hit_rate_at_60")),
        base_rates=_base_rates_from_dict(block.get("base_rates") or {}),
        roc_auc=_parse_float(block.get("roc_auc")),
        average_precision=_parse_float(block.get("average_precision")),
        coverage=coverage,
        sectors=_parse_string_float_map(block.get("sector_accuracy")),
        stages=_parse_string_float_map(block.get("stage_accuracy")),
        countries=_parse_string_float_map(block.get("country_accuracy")),
    )


def _confusion_from_dict(block: dict[str, Any]) -> ConfusionMetrics:
    """Rebuild a ConfusionMetrics from its ``to_dict()`` repr."""

    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    return ConfusionMetrics(
        total=_int(block.get("total")),
        scoreable=_int(block.get("scoreable")),
        true_positives=_int(block.get("true_positives")),
        true_negatives=_int(block.get("true_negatives")),
        false_positives=_int(block.get("false_positives")),
        false_negatives=_int(block.get("false_negatives")),
        accuracy=_parse_float(block.get("accuracy")),
        precision=_parse_float(block.get("precision")),
        recall=_parse_float(block.get("recall")),
        specificity=_parse_float(block.get("specificity")),
        f1=_parse_float(block.get("f1")),
        f05=_parse_float(block.get("f0_5")),
        balanced_accuracy=_parse_float(block.get("balanced_accuracy")),
        false_positive_rate=_parse_float(block.get("false_positive_rate")),
        false_negative_rate=_parse_float(block.get("false_negative_rate")),
    )


def _base_rates_from_dict(block: dict[str, Any]) -> BaseRates:
    """Rebuild a BaseRates from its ``to_dict()`` repr."""
    return BaseRates(
        total=_int_or_zero(block.get("total")),
        positives=_int_or_zero(block.get("positives")),
        negatives=_int_or_zero(block.get("negatives")),
        positive_rate=_parse_float(block.get("positive_rate")),
        negative_rate=_parse_float(block.get("negative_rate")),
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _pick(block: dict[str, Any], key: str, default: Any) -> dict[str, Any]:
    value = block.get(key)
    return value if isinstance(value, dict) else default


def _parse_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int_float_map(value: Any) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    if not isinstance(value, dict):
        return result
    for key, val in value.items():
        try:
            result[int(key)] = _parse_float(val)
        except (TypeError, ValueError):
            continue
    return result


def _parse_string_float_map(value: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(value, dict):
        return result
    for key, val in value.items():
        parsed = _parse_float(val)
        if parsed is not None:
            result[str(key)] = parsed
    return result


def _coverage_from_dict(block: dict[str, Any]) -> CoverageMetrics:
    """Rebuild a CoverageMetrics from its ``to_dict()`` repr.  Derived
    properties (``replay_coverage``/``scoreability``) are recomputed."""

    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    return CoverageMetrics(
        dataset_entries=_int(block.get("dataset_entries")),
        replayed_entries=_int(block.get("replayed_entries")),
        failed_entries=_int(block.get("failed_entries")),
        scoreable_entries=_int(block.get("scoreable_entries")),
        unscoreable_entries=_int(block.get("unscoreable_entries")),
        sectors=_parse_string_int_map(block.get("sectors")),
        stages=_parse_string_int_map(block.get("stages")),
        countries=_parse_string_int_map(block.get("countries")),
    )


def _parse_string_int_map(value: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(value, dict):
        return result
    for key, val in value.items():
        try:
            result[str(key)] = int(val)
        except (TypeError, ValueError):
            continue
    return result


def _calibration_from_dict(block: dict[str, Any]) -> CalibrationMetrics:
    bins: list[CalibrationBin] = []
    for item in block.get("bins") or []:
        if not isinstance(item, dict):
            continue
        try:
            bins.append(
                CalibrationBin(
                    bin_lower=float(item["bin_lower"]),
                    bin_upper=float(item["bin_upper"]),
                    sample_count=int(item["sample_count"]),
                    mean_predicted_confidence=float(item["mean_predicted_confidence"]),
                    actual_accuracy=float(item["actual_accuracy"]),
                    overconfident=bool(item.get("overconfident", False)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return CalibrationMetrics(
        expected_calibration_error=_parse_float(block.get("expected_calibration_error")) or 0.0,
        maximum_calibration_error=_parse_float(block.get("maximum_calibration_error")) or 0.0,
        overconfidence_detected=bool(block.get("overconfidence_detected", False)),
        bins=tuple(bins),
        total_samples=int(block.get("total_samples", 0) or 0),
    )


def compute_metrics(
    dataset: GoldenDataset,
    run: BenchmarkRun,
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    num_bins: int = 10,
) -> GroundTruthMetrics:
    """Compute the full deterministic metric block for a run."""
    samples = build_samples(dataset, run)
    confusion = confusion_metrics(samples)
    calibration = calibration_metrics(samples, num_bins=num_bins)
    precision_map: dict[int, float | None] = {k: precision_at_k(samples, k) for k in k_values}
    recall_map: dict[int, float | None] = {k: recall_at_k(samples, k) for k in k_values}
    return GroundTruthMetrics(
        run_id=run.run_id,
        engine_version=run.engine_version,
        benchmark_version=run.benchmark_version,
        dataset_name=run.dataset_name,
        dataset_hash=run.dataset_hash if run.dataset_hash else dataset_hash(dataset),
        confusion=confusion,
        calibration=calibration,
        brier=brier_score(samples),
        precision_at_k=precision_map,
        recall_at_k=recall_map,
        top_decile_precision=top_decile_precision(samples),
        investment_hit_rate=investment_hit_rate(samples),
        investment_hit_rate_at_60=investment_hit_rate(samples, threshold=60.0),
        base_rates=base_rates(samples),
        roc_auc=roc_auc(samples),
        average_precision=average_precision(samples),
        coverage=coverage_metrics(dataset, run),
        sectors=sector_accuracy(samples),
        stages=stage_accuracy(samples),
        countries=country_accuracy(samples),
    )


__all__ = [
    "ScoredSample",
    "build_samples",
    "scoreable",
    "ConfusionMetrics",
    "confusion_metrics",
    "CalibrationMetrics",
    "calibration_metrics",
    "brier_score",
    "calibration_curves",
    "precision_at_k",
    "recall_at_k",
    "top_decile_precision",
    "investment_hit_rate",
    "false_positive_rate",
    "false_negative_rate",
    "BaseRates",
    "base_rates",
    "roc_auc",
    "average_precision",
    "grouped_accuracy",
    "sector_accuracy",
    "stage_accuracy",
    "country_accuracy",
    "CoverageMetrics",
    "coverage_metrics",
    "GroundTruthMetrics",
    "compute_metrics",
    "metrics_from_dict",
]
