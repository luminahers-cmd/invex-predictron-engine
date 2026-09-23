"""Phase 7 — deterministic learning-snapshot builder.

Aggregates :class:`LearningSample` rows (resolved evaluations) into a
:class:`LearningSnapshot`:

* per-dimension cohort patterns and canned observations;
* canonical population metrics (reusing the repository evaluation metrics
  for the binary-scoreable cohort);
* equal-width calibration digest;
* confidence statistics;
* rule-based recommendations.

Everything here is pure and deterministic — no IO, no latency-insensitive
state, no generated prose.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import UTC, date, datetime
from statistics import fmean, pstdev

from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.learning.models import (
    ATTR_UNKNOWN,
    CALIBRATION_BIN_COUNT,
    CONFIDENCE_STABLE_STD,
    LEARNING_DIMENSIONS,
    LEARNING_SNAPSHOT_SCHEMA_VERSION,
    OBSERVATION_ACCURACY_DELTA,
    OBSERVATION_BIAS_THRESHOLD,
    OBSERVATION_MIN_SAMPLES,
    RECOMMENDATION_ACCURACY_DELTA,
    RECOMMENDATION_MAX_CONFIDENCE_BIAS,
    RECOMMENDATION_MAX_ECE,
    RECOMMENDATION_MIN_SAMPLES,
    LearningCalibration,
    LearningConfidence,
    LearningDigest,
    LearningDimension,
    LearningObservation,
    LearningObservationCategory,
    LearningPattern,
    LearningPeriod,
    LearningPeriodKind,
    LearningRecommendation,
    LearningSample,
    LearningSnapshot,
    TrendDirection,
)

_MAX_SAMPLES_TRUNCATE = 100


def _now() -> datetime:
    return datetime.now(UTC)


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _snapshot_id(scope: str, period_kind: LearningPeriodKind, anchor_date: date) -> str:
    material = "|".join((scope, period_kind.value, anchor_date.isoformat()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _confusion(
    samples: list[LearningSample],
) -> tuple[int, int, int, int, int]:
    """Return ``(scoreable, tp, tn, fp, fn)`` over binary-scoreable samples."""
    scoreable = tp = tn = fp = fn = 0
    for sample in samples:
        if sample.verdict not in (
            EvaluationVerdict.CORRECT,
            EvaluationVerdict.INCORRECT,
        ):
            continue
        if sample.actual_positive is None:
            continue
        correct = sample.verdict == EvaluationVerdict.CORRECT
        if sample.actual_positive:
            if correct:
                tp += 1
            else:
                fn += 1
        else:
            if correct:
                tn += 1
            else:
                fp += 1
        scoreable += 1
    return scoreable, tp, tn, fp, fn


def _rates(
    scoreable: int, tp: int, tn: int, fp: int, fn: int
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return ``(accuracy, precision, fp_rate, fn_rate)`` (None when unset)."""
    accuracy: float | None = None
    if scoreable:
        accuracy = (tp + tn) / scoreable
    precision: float | None = None
    if tp + fp:
        precision = tp / (tp + fp)
    fp_rate: float | None = None
    if fp + tn:
        fp_rate = fp / (fp + tn)
    fn_rate: float | None = None
    if fn + tp:
        fn_rate = fn / (fn + tp)
    return accuracy, precision, fp_rate, fn_rate


def _pattern_category(
    accuracy: float | None,
    fp_rate: float | None,
    fn_rate: float | None,
) -> LearningObservationCategory | None:
    if accuracy is not None and accuracy >= 0.7:
        return LearningObservationCategory.STRONG_ACCURACY
    if fp_rate is not None and fp_rate >= 0.35:
        return LearningObservationCategory.HIGH_FALSE_POSITIVE_RATE
    if fn_rate is not None and fn_rate >= 0.35:
        return LearningObservationCategory.HIGH_FALSE_NEGATIVE_RATE
    return None


def _pattern_recommendation(
    pattern: LearningPattern,
    population_accuracy: float | None,
) -> str:
    """Canned rule-engine guidance for one dimension bucket."""
    if pattern.scoreable < RECOMMENDATION_MIN_SAMPLES:
        return (
            f"await more evidence before weighting {pattern.dimension.value} "
            f"category {pattern.value!r} (only {pattern.scoreable} scoreable samples)"
        )
    if pattern.accuracy is not None and population_accuracy is not None:
        if pattern.accuracy >= population_accuracy + RECOMMENDATION_ACCURACY_DELTA:
            return (
                f"increase confidence weighting for {pattern.dimension.value} "
                f"category {pattern.value!r}"
            )
        if pattern.accuracy <= population_accuracy - RECOMMENDATION_ACCURACY_DELTA:
            return (
                f"decrease confidence weighting for {pattern.dimension.value} "
                f"category {pattern.value!r}"
            )
    if pattern.confidence_bias is not None:
        if pattern.confidence_bias >= OBSERVATION_BIAS_THRESHOLD:
            return (
                f"calibrate confidence downward for {pattern.dimension.value} "
                f"category {pattern.value!r}"
            )
        if pattern.confidence_bias <= -OBSERVATION_BIAS_THRESHOLD:
            return (
                f"calibrate confidence upward for {pattern.dimension.value} "
                f"category {pattern.value!r}"
            )
    return (
        f"maintain current confidence weighting for {pattern.dimension.value} "
        f"category {pattern.value!r}"
    )


def _build_patterns(
    samples: list[LearningSample],
    population_accuracy: float | None,
) -> list[LearningPattern]:
    """Deterministic per-dimension cohort patterns (sorted output)."""
    patterns: list[LearningPattern] = []
    for dimension_value in LEARNING_DIMENSIONS:
        buckets: dict[str, list[LearningSample]] = defaultdict(list)
        for sample in samples:
            buckets[sample.resolved.get(dimension_value, ATTR_UNKNOWN)].append(sample)
        for value in sorted(buckets):
            cohort = buckets[value]
            scoreable, tp, tn, fp, fn = _confusion(cohort)
            accuracy, precision, fp_rate, fn_rate = _rates(
                scoreable, tp, tn, fp, fn
            )
            confidences = [sample.confidence for sample in cohort]
            mean_conf = fmean(confidences) if confidences else 0.0
            positives = [s for s in cohort if s.actual_positive is not None]
            base_rate = (
                fmean(1.0 if s.actual_positive else 0.0 for s in positives)
                if positives
                else None
            )
            bias = round(mean_conf - base_rate, 4) if base_rate is not None else None
            pattern = LearningPattern(
                dimension=LearningDimension(dimension_value),
                value=value,
                samples=len(cohort),
                scoreable=scoreable,
                true_positive=tp,
                true_negative=tn,
                false_positive=fp,
                false_negative=fn,
                accuracy=_round4(accuracy),
                precision=_round4(precision),
                false_positive_rate=_round4(fp_rate),
                false_negative_rate=_round4(fn_rate),
                confidence=_round4(mean_conf) or 0.0,
                confidence_bias=_round4(bias),
            )
            pattern.recommendation = _pattern_recommendation(
                pattern, population_accuracy
            )
            patterns.append(pattern)
    return patterns


def _pattern_map(
    patterns: list[LearningPattern],
) -> dict[LearningDimension, dict[str, LearningPattern]]:
    mapping: dict[LearningDimension, dict[str, LearningPattern]] = defaultdict(dict)
    for pattern in patterns:
        mapping[pattern.dimension][pattern.value] = pattern
    return mapping


def _truncate(value: str, limit: int = _MAX_SAMPLES_TRUNCATE) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _build_observations(
    patterns: list[LearningPattern],
    population_accuracy: float | None,
) -> list[LearningObservation]:
    """Canonical observations from per-dimension pattern cohorts."""
    observations: list[LearningObservation] = []
    for pattern in patterns:
        if pattern.scoreable < OBSERVATION_MIN_SAMPLES:
            observations.append(
                LearningObservation(
                    category=LearningObservationCategory.LOW_SUPPORT,
                    dimension=pattern.dimension,
                    value=pattern.value,
                    metric="sample_size",
                    metric_value=float(pattern.scoreable),
                    baseline=float(OBSERVATION_MIN_SAMPLES),
                    direction=TrendDirection.FLAT,
                    sample_size=pattern.scoreable,
                    summary=(
                        f"{pattern.dimension.value} category {_truncate(pattern.value)} "
                        f"has only {pattern.scoreable} scoreable samples"
                    ),
                )
            )
            continue
        category = _pattern_category(
            pattern.accuracy, pattern.false_positive_rate, pattern.false_negative_rate
        )
        if category is not None:
            if category == LearningObservationCategory.STRONG_ACCURACY:
                metric = "accuracy"
                value = pattern.accuracy
            elif category == LearningObservationCategory.HIGH_FALSE_POSITIVE_RATE:
                metric = "false_positive_rate"
                value = pattern.false_positive_rate
            else:  # HIGH_FALSE_NEGATIVE_RATE
                metric = "false_negative_rate"
                value = pattern.false_negative_rate
            display = f"{value:.2f}" if value is not None else "n/a"
            observations.append(
                LearningObservation(
                    category=category,
                    dimension=pattern.dimension,
                    value=pattern.value,
                    metric=metric,
                    metric_value=_round4(value),
                    delta=None,
                    direction=TrendDirection.FLAT,
                    baseline=None,
                    sample_size=pattern.scoreable,
                    summary=(
                        f"{pattern.dimension.value} category {_truncate(pattern.value)} "
                        f"shows {category.value.replace('_', ' ')} "
                        f"({metric}={display})"
                    ),
                )
            )
        if pattern.confidence_bias is not None:
            if pattern.confidence_bias >= OBSERVATION_BIAS_THRESHOLD:
                observations.append(
                    LearningObservation(
                        category=LearningObservationCategory.OVERCONFIDENCE,
                        dimension=pattern.dimension,
                        value=pattern.value,
                        metric="confidence_bias",
                        metric_value=pattern.confidence_bias,
                        baseline=float(OBSERVATION_BIAS_THRESHOLD),
                        direction=TrendDirection.UP,
                        sample_size=pattern.scoreable,
                        summary=(
                            f"{pattern.dimension.value} category {_truncate(pattern.value)} "
                            f"is overconfident by {pattern.confidence_bias:.4f}"
                        ),
                    )
                )
            elif pattern.confidence_bias <= -OBSERVATION_BIAS_THRESHOLD:
                observations.append(
                    LearningObservation(
                        category=LearningObservationCategory.UNDERCONFIDENCE,
                        dimension=pattern.dimension,
                        value=pattern.value,
                        metric="confidence_bias",
                        metric_value=pattern.confidence_bias,
                        baseline=float(-OBSERVATION_BIAS_THRESHOLD),
                        direction=TrendDirection.DOWN,
                        sample_size=pattern.scoreable,
                        summary=(
                            f"{pattern.dimension.value} category {_truncate(pattern.value)} "
                            f"is underconfident by {abs(pattern.confidence_bias):.4f}"
                        ),
                    )
                )
            if pattern.accuracy is not None and population_accuracy is not None:
                difference = pattern.accuracy - population_accuracy
                if difference >= OBSERVATION_ACCURACY_DELTA:
                    observations.append(
                        LearningObservation(
                            category=LearningObservationCategory.ACCURACY_EXCESS,
                            dimension=pattern.dimension,
                            value=pattern.value,
                            metric="accuracy",
                            metric_value=_round4(pattern.accuracy),
                            delta=_round4(difference),
                            direction=TrendDirection.UP,
                            baseline=_round4(population_accuracy),
                            sample_size=pattern.scoreable,
                            summary=(
                                f"{pattern.dimension.value} category "
                                f"{_truncate(pattern.value)} exceeds platform accuracy "
                                f"by {difference:.4f}"
                            ),
                        )
                    )
                elif difference <= -OBSERVATION_ACCURACY_DELTA:
                    observations.append(
                        LearningObservation(
                            category=LearningObservationCategory.ACCURACY_DEFICIT,
                            dimension=pattern.dimension,
                            value=pattern.value,
                            metric="accuracy",
                            metric_value=_round4(pattern.accuracy),
                            delta=_round4(difference),
                            direction=TrendDirection.DOWN,
                            baseline=_round4(population_accuracy),
                            sample_size=pattern.scoreable,
                            summary=(
                                f"{pattern.dimension.value} category "
                                f"{_truncate(pattern.value)} trails platform accuracy "
                                f"by {abs(difference):.4f}"
                            ),
                        )
                    )
    return observations


def _build_knowledge(
    patterns: list[LearningPattern],
) -> dict[str, list[LearningPeriod]]:
    """Per-dimension knowledge views over the cohort patterns."""
    knowledge: dict[str, list[LearningPeriod]] = {dim: [] for dim in LEARNING_DIMENSIONS}
    for pattern in patterns:
        confidence_bias = pattern.confidence_bias
        knowledge[pattern.dimension.value].append(
            LearningPeriod(
                dimension=pattern.dimension,
                value=pattern.value,
                sample_size=pattern.scoreable,
                accuracy=_round4(pattern.accuracy),
                confidence_bias=_round4(confidence_bias),
                false_positive_rate=_round4(pattern.false_positive_rate),
                false_negative_rate=_round4(pattern.false_negative_rate),
                recommendation=pattern.recommendation,
            )
        )
    for dim_entries in knowledge.values():
        dim_entries.sort(key=lambda entry: (entry.value, entry.sample_size))
    return knowledge


def _distribution(
    samples: list[LearningSample], key: str, unknown: str = ATTR_UNKNOWN
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.resolved.get(key, unknown)] += 1
    return dict(sorted(counts.items()))


def compute_calibration(
    samples: list[LearningSample],
) -> LearningCalibration:
    """Equal-width binned ECE over binary-outcome samples.

    Each bin carries ``(confidence_mean, accuracy)``; ECE is the support
    weighted mean of ``|confidence_mean - accuracy|``.  A bin is
    ``overconfident`` when confidence exceeds accuracy by more than the
    bias threshold.
    """
    labeled = [s for s in samples if s.actual_positive is not None]
    if not labeled:
        return LearningCalibration()
    widths: list[list[LearningSample]] = [[] for _ in range(CALIBRATION_BIN_COUNT)]
    total = len(labeled)
    for sample in labeled:
        index = min(CALIBRATION_BIN_COUNT - 1, int(sample.confidence * CALIBRATION_BIN_COUNT))
        widths[index].append(sample)

    bins: list[dict[str, object]] = []
    ece_total = 0.0
    overconfident = 0
    for index, cohort in enumerate(widths):
        if not cohort:
            continue
        conf_mean = fmean(sample.confidence for sample in cohort)
        acc = fmean(1.0 if sample.actual_positive else 0.0 for sample in cohort)
        weight = len(cohort) / total
        gap = abs(conf_mean - acc)
        ece_total += weight * gap
        if conf_mean - acc >= OBSERVATION_BIAS_THRESHOLD:
            overconfident += 1
        bins.append(
            {
                "bin": index,
                "samples": len(cohort),
                "confidence_mean": round(conf_mean, 4),
                "accuracy": round(acc, 4),
                "gap": round(gap, 4),
            }
        )
    detected = overconfident > 0
    return LearningCalibration(
        expected_calibration_error=round(ece_total, 4),
        overconfidence_detected=detected,
        overconfident_bins=overconfident,
        total_samples=total,
        bins=bins,
    )


def _snapshot_recommendations(
    *,
    population_accuracy: float | None,
    digest: LearningDigest,
    calibration: LearningCalibration,
    confidence: LearningConfidence,
) -> list[LearningRecommendation]:
    """Rule-based, canonical, deterministic recommendations."""
    recommendations: list[LearningRecommendation] = []

    if calibration.total_samples >= OBSERVATION_MIN_SAMPLES:
        if calibration.expected_calibration_error > RECOMMENDATION_MAX_ECE:
            recommendations.append(
                LearningRecommendation(
                    kind="recalibrate_confidence",
                    message=(
                        "Expected calibration error exceeds the tolerance "
                        "threshold; recalibrate confidence banding before the "
                        "next forecasting cycle."
                    ),
                    severity="warning",
                    parameters={
                        "expected_calibration_error": calibration.expected_calibration_error,
                        "threshold": RECOMMENDATION_MAX_ECE,
                    },
                )
            )
        elif confidence.bias >= RECOMMENDATION_MAX_CONFIDENCE_BIAS:
            recommendations.append(
                LearningRecommendation(
                    kind="reduce_confidence_optimism",
                    message=(
                        "Overall predicted confidence systematically exceeds "
                        "observed success frequency; reduce confidence optimism "
                        "on new forecasts."
                    ),
                    severity="warning",
                    parameters={
                        "confidence_bias": confidence.bias,
                        "threshold": RECOMMENDATION_MAX_CONFIDENCE_BIAS,
                    },
                )
            )
    if population_accuracy is not None and population_accuracy >= 0.6:
        recommendations.append(
            LearningRecommendation(
                kind="maintain_confidence_weighting",
                message=(
                    "Platform accuracy supports the current confidence "
                    "weighting; no change recommended."
                ),
                severity="info",
                parameters={"accuracy": population_accuracy},
            )
        )
    if confidence.std_dev is not None and confidence.std_dev < CONFIDENCE_STABLE_STD:
        recommendations.append(
            LearningRecommendation(
                kind="concentrated_confidence",
                message=(
                    "Predicted-confidence distribution is concentrated; "
                    "watch for overconfidence in the highest bracket."
                ),
                severity="info",
                parameters={"std_dev": confidence.std_dev},
            )
        )
    return recommendations


def build_learning_snapshot(
    samples: list[LearningSample],
    *,
    anchor_date: date,
    period_kind: LearningPeriodKind = LearningPeriodKind.DAILY,
    scope: str = "repository",
    engine_version: str = "7.0.0",
    snapshot_id: str | None = None,
    recorded_at: datetime | None = None,
    as_of: datetime | None = None,
) -> LearningSnapshot:
    """Full-population learning snapshot over resolved evaluation samples."""

    resolved_id = snapshot_id or _snapshot_id(scope, period_kind, anchor_date)
    analysis_at = as_of or _now()

    scoreable, tp, tn, fp, fn = _confusion(samples)
    accuracy, precision, fp_rate, fn_rate = _rates(scoreable, tp, tn, fp, fn)
    evaluation_count = len(samples)
    sample_count = len(samples)

    verdict_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        verdict_counts[sample.verdict.value] += 1

    digest = LearningDigest(
        evaluation_count=evaluation_count,
        sample_count=sample_count,
        scoreable=scoreable,
        accuracy=_round4(accuracy),
        precision=_round4(precision),
        recall=_round4(tp / (tp + fn) if tp + fn else None),
        false_positive_rate=_round4(fp_rate),
        false_negative_rate=_round4(fn_rate),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        verdict_counts=dict(sorted(verdict_counts.items())),
    )
    calibration = compute_calibration(samples)
    confidences = [sample.confidence for sample in samples]
    mean_conf = fmean(confidences) if confidences else None
    std_conf = pstdev(confidences) if len(confidences) > 1 else None
    labeled = [s for s in samples if s.actual_positive is not None]
    base_rate = (
        fmean(1.0 if s.actual_positive else 0.0 for s in labeled)
        if labeled
        else None
    )
    bias = round((mean_conf or 0.0) - (base_rate or 0.0), 4) if mean_conf is not None else 0.0
    confidence = LearningConfidence(
        count=len(confidences),
        mean=_round4(mean_conf),
        std_dev=_round4(std_conf),
        bias=bias,
        calibrated=not calibration.overconfidence_detected,
    )

    patterns = _build_patterns(samples, digest.accuracy)
    knowledge = _build_knowledge(patterns)
    observations = _build_observations(patterns, digest.accuracy)
    recommendations = _snapshot_recommendations(
        population_accuracy=digest.accuracy,
        digest=digest,
        calibration=calibration,
        confidence=confidence,
    )

    counts: dict[str, int] = {
        "evaluations": evaluation_count,
        "samples": sample_count,
        "scoreable": scoreable,
    }
    metrics: dict[str, float | None] = {
        "accuracy": digest.accuracy,
        "precision": digest.precision,
        "recall": digest.recall,
        "false_positive_rate": digest.false_positive_rate,
        "false_negative_rate": digest.false_negative_rate,
        "ece": calibration.expected_calibration_error,
        "overconfidence": 1.0 if calibration.overconfidence_detected else 0.0,
        "confidence.mean": _round4(mean_conf),
        "confidence.bias": round(bias, 4),
    }

    snapshot = LearningSnapshot(
        scope=scope,
        snapshot_id=resolved_id,
        period_kind=period_kind,
        anchor_date=anchor_date,
        engine_version=engine_version,
        recorded_at=recorded_at or analysis_at,
        schema_version=LEARNING_SNAPSHOT_SCHEMA_VERSION,
        content_hash="",
        counts=counts,
        metrics=metrics,
        digest=digest,
        calibration=calibration,
        confidence=confidence,
        distributions={
            dimension: _distribution(samples, dimension)
            for dimension in LEARNING_DIMENSIONS
        },
        knowledge=knowledge,
        patterns=patterns,
        observations=observations,
        recommendations=recommendations,
        meta={"schema_version": LEARNING_SNAPSHOT_SCHEMA_VERSION},
    )
    snapshot.content_hash = snapshot.content_fingerprint()
    return snapshot


__all__ = [
    "build_learning_snapshot",
    "compute_calibration",
]
