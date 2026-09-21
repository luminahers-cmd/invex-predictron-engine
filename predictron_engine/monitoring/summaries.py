"""Phase 6 — deterministic monitoring summaries over the live ledger.

Every rate here is computed through the **canonical** metric machinery —
never re-derived by hand:

* aggregate rates  -> ``predictron_engine.dataset.metrics.compute_evaluation_metrics``
* calibration      -> ``app.services.company_evaluation.calibration_block`` is the
  app-side wrapper; the engine uses the same Sprint 8 report builder directly
  (``predictron_engine.decision.calibration_sprint8.build_calibration_report``)
  over the *same* binary-labelled samples ``binary_label`` produces.
* mean confidence  -> arithmetic mean of ``ForecastRecord.confidence``.
* distributions    -> frequency tables over the existing label universes
  (:class:`DecisionLabel`, :class:`OutcomeVerdict`, :class:`EvaluationVerdict`).

Mathematical definitions
------------------------
* ``mean_confidence(records) = sum(conf) / n`` (``None`` when empty),
  rounded to 4 decimal places.
* ``rolling_evaluation_metrics(evaluations, window)``: sort by
  ``(created_at, evaluation_id)`` descending, take the first ``window``,
  then delegate to ``compute_evaluation_metrics`` on that slice.  With fewer
  than ``window`` samples the whole collection is used (documented behaviour,
  no padding).
* ``rolling_mean_confidence(records, window)``: the same windowed slice
  semantics over ``ForecastRecord`` ordered by ``analysis_timestamp``.
* ``sector_breakdown``: a sector is one key of a frozen ``dimension_scores``
  payload; per sector the evaluation population is every evaluation whose
  frozen prediction carries that key.  The ``overall`` bucket is the entire
  population.  (This mirrors the approach of
  ``benchmarks.ground_truth_eval.metrics.grouped_accuracy``/``sector_accuracy``
  but reuses ``compute_evaluation_metrics`` for the live-ledger population —
  the benchmark functions operate on benchmark-run samples and are the
  canonical source there.)
* ``horizon_breakdown``: an evaluation is attributed to every horizon that
  forecasts its snapshot (a multi-horizon snapshot is evaluated against each
  of its horizons; documented over-counting on multi-horizon snapshots).

No heuristic or hidden threshold exists.
"""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.evaluation import (
    EvaluationVerdict,
    PredictionEvaluation,
)
from predictron_engine.dataset.metrics import (
    EvaluationMetrics,
    binary_label,
    compute_evaluation_metrics,
)
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.outcomes import OutcomeVerdict
from predictron_engine.decision.calibration_sprint8 import (
    _classify_calibration_quality,
    compute_calibration_error,
    detect_overconfidence,
)
from predictron_engine.monitoring.models import (
    Distribution,
    ForecastRecord,
    HorizonPerformance,
    SectorPerformance,
)

_SENTINEL = datetime.min.replace(tzinfo=UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _evaluation_universe() -> list[str]:
    return sorted(e.value for e in EvaluationVerdict)


def _outcome_universe() -> list[str]:
    return sorted(e.value for e in OutcomeVerdict)


def _decision_universe() -> list[str]:
    return sorted(e.value for e in DecisionLabel)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def evaluation_verdict_distribution(
    evaluations: list[PredictionEvaluation],
) -> Distribution:
    """Frequency table over the ``EvaluationVerdict`` universe.

    Counts reuse ``compute_evaluation_metrics(...).verdict_counts`` (the
    canonical verdict tally) so the distribution can never disagree with the
    reported accuracy.
    """
    counts = compute_evaluation_metrics(evaluations).verdict_counts
    return Distribution(
        label="evaluation_verdict",
        universe=_evaluation_universe(),
        counts={key: counts.get(key, 0) for key in _evaluation_universe()},
    )


def outcome_verdict_distribution(
    evaluations: list[PredictionEvaluation],
) -> Distribution:
    """Frequency table over the ``OutcomeVerdict`` universe.

    Derived from the outcome records each evaluation was compared against
    (observed facts, never fabricated).
    """
    counts: dict[str, int] = {
        key: 0 for key in _outcome_universe()
    }
    for evaluation in evaluations:
        counts[evaluation.outcome_record.verdict.value] = (
            counts.get(evaluation.outcome_record.verdict.value, 0) + 1
        )
    return Distribution(
        label="outcome_verdict",
        universe=_outcome_universe(),
        counts=counts,
    )


def decision_distribution(records: list[ForecastRecord]) -> Distribution:
    """Frequency table over the ``DecisionLabel`` universe of frozen forecasts."""
    counts: dict[str, int] = {key: 0 for key in _decision_universe()}
    for record in records:
        counts[record.decision] = counts.get(record.decision, 0) + 1
    return Distribution(
        label="decision", universe=_decision_universe(), counts=counts
    )


# ---------------------------------------------------------------------------
# Population statistics
# ---------------------------------------------------------------------------


def mean_confidence(records: list[ForecastRecord]) -> float | None:
    """Population mean of frozen forecast confidence (``None`` when empty)."""
    if not records:
        return None
    return round(sum(float(r.confidence) for r in records) / len(records), 4)


def calibration_summary(
    evaluations: list[PredictionEvaluation],
) -> dict[str, float | int | str]:
    """Calibration summary from the canonical Sprint 8 calibration machinery.

    Only binary-labelled outcome samples contribute (``binary_label`` —
    success/failure outcomes).  ECE / MCE and the overconfidence flags are
    produced verbatim by ``compute_calibration_error`` +
    ``detect_overconfidence`` (the functions ``build_calibration_report``
    itself delegates to), so the monitor can never disagree with the Sprint
    8 report on the same samples.
    """
    confidences: list[float] = []
    outcomes: list[int] = []
    for evaluation in evaluations:
        label = binary_label(evaluation)
        if label.actual_positive is None:
            continue
        confidences.append(float(evaluation.prediction.confidence))
        outcomes.append(1 if label.actual_positive else 0)

    ece, max_ece, bins = compute_calibration_error(confidences, outcomes)
    overconfident, overconfident_bins = detect_overconfidence(bins)
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(max_ece),
        "overconfidence_detected": overconfident,
        "total_samples": len(confidences),
        "calibration_quality": _classify_calibration_quality(ece),
        "overconfident_bins_count": len(overconfident_bins),
    }


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------


def _rolling_evaluation_window(
    evaluations: list[PredictionEvaluation], window: int
) -> list[PredictionEvaluation]:
    """The ``window`` most recent evaluations (order documented above)."""
    ordered = sorted(
        evaluations,
        key=lambda evaluation: (
            _as_utc(evaluation.created_at) or _SENTINEL,
            evaluation.evaluation_id,
        ),
        reverse=True,
    )
    return ordered[:window]


def rolling_metrics(
    evaluations: list[PredictionEvaluation],
    window: int,
) -> EvaluationMetrics:
    """Aggregate metrics over the ``window`` most recent evaluations.

    The window is the tail of the collection ordered by
    ``(created_at, evaluation_id)`` descending; the metrics themselves are
    produced verbatim by ``compute_evaluation_metrics`` on the canonical
    metric machinery.  Returns the ``EvaluationMetrics`` object.
    """
    return compute_evaluation_metrics(_rolling_evaluation_window(evaluations, window))


def rolling_verdict_distribution(
    evaluations: list[PredictionEvaluation], window: int
) -> Distribution:
    """Verdict distribution over the ``window`` most recent evaluations."""
    return evaluation_verdict_distribution(
        _rolling_evaluation_window(evaluations, window)
    )


def rolling_mean_confidence(
    records: list[ForecastRecord], window: int
) -> float | None:
    """Mean predicted confidence over the ``window`` most recent forecasts.

    The window is the tail of ``records`` ordered by ``(analysis_timestamp,
    id)`` descending; the mean reuses :func:`mean_confidence`.
    """
    ordered = sorted(
        records,
        key=lambda record: (_as_utc(record.analysis_timestamp) or _SENTINEL, record.id),
        reverse=True,
    )
    return mean_confidence(ordered[:window])


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------


def sector_breakdown(
    evaluations: list[PredictionEvaluation],
) -> dict[str, SectorPerformance]:
    """Per-dimension (sector) accuracy over frozen predictions.

    The ``overall`` bucket aggregates the whole population; each other
    bucket holds evaluations whose frozen prediction carries that dimension
    score key.  Rates reuse ``compute_evaluation_metrics``.
    """
    grouped: dict[str, list[PredictionEvaluation]] = {"overall": list(evaluations)}
    for evaluation in evaluations:
        for sector in evaluation.prediction.dimension_scores:
            grouped.setdefault(sector, []).append(evaluation)

    result: dict[str, SectorPerformance] = {}
    for sector, population in sorted(grouped.items()):
        metrics = compute_evaluation_metrics(population)
        result[sector] = SectorPerformance(
            sector=sector,
            evaluations=metrics.total,
            scoreable=metrics.scoreable,
            accuracy=metrics.accuracy,
        )
    return result


def horizon_breakdown(
    evaluations: list[PredictionEvaluation],
    forecasts: list[ForecastRecord],
) -> dict[int, HorizonPerformance]:
    """Per-horizon performance, evaluations attributed to their forecast horizons.

    An evaluation is attributed to every horizon that forecasts its
    snapshot; multi-horizon snapshots therefore appear in more than one
    bucket (documented).  Rates reuse ``compute_evaluation_metrics``.
    """
    horizons: dict[int, set[str]] = {}
    forecast_count: dict[int, int] = {}
    for forecast in forecasts:
        bucket = horizons.setdefault(forecast.horizon_days, set())
        bucket.add(forecast.snapshot_id)
        forecast_count[forecast.horizon_days] = (
            forecast_count.get(forecast.horizon_days, 0) + 1
        )
    bucket_populations: dict[int, list[PredictionEvaluation]] = {
        days: [] for days in horizons
    }
    evaluated_snapshots: dict[int, set[str]] = {
        days: set() for days in horizons
    }
    for evaluation in evaluations:
        for days, snapshots in horizons.items():
            if evaluation.record_id in snapshots:
                bucket_populations[days].append(evaluation)
                evaluated_snapshots[days].add(evaluation.record_id)

    result: dict[int, HorizonPerformance] = {}
    for days in sorted(horizons):
        population = bucket_populations[days]
        metrics = compute_evaluation_metrics(population)
        result[days] = HorizonPerformance(
            horizon_days=days,
            forecasts=forecast_count.get(days, 0),
            evaluated=len(evaluated_snapshots[days]),
            scoreable=metrics.scoreable,
            accuracy=metrics.accuracy,
        )
    return result


def status_distribution(records: list[ForecastRecord]) -> dict[str, int]:
    """Persisted lifecycle status counts over the forecast population."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "calibration_summary",
    "decision_distribution",
    "evaluation_verdict_distribution",
    "horizon_breakdown",
    "mean_confidence",
    "outcome_verdict_distribution",
    "rolling_metrics",
    "rolling_mean_confidence",
    "rolling_verdict_distribution",
    "sector_breakdown",
    "status_distribution",
]
