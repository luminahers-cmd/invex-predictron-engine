"""Deterministic ground-truth evaluation reports (Project E5).

Ten report kinds are available, each producing a plain dict that is
deterministic in its *content* (every table/ranking/breakdown is sorted;
metrics are recomputed with fixed rounding).  ``generated_at`` is the only
wall-clock value and is explicitly informational.

Report kinds
------------
* ``executive``    — headline numbers and a qualitative assessment.
* ``accuracy``     — confusion matrix + calibration + ranking panel.
* ``calibration``  — reliability diagram, ECE, max CE, overconfidence.
* ``sectors``      — sector-level accuracy breakdown.
* ``countries``    — country-level accuracy breakdown.
* ``stages``       — stage-level accuracy breakdown.
* ``trend``        — per-run metric history for one dataset.
* ``engine_drift`` — score/decision/confidence/recommendation/feature drift.
* ``leaderboard``  — companies ranked by fresh score vs verified outcome.
* ``coverage``     — replay/scoreability coverage and by-group counts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from benchmarks.ground_truth_eval.drift import DriftDetector, DriftReport
from benchmarks.ground_truth_eval.history import BenchmarkHistory
from benchmarks.ground_truth_eval.metrics import (
    GroundTruthMetrics,
    ScoredSample,
    build_samples,
    compute_metrics,
)
from benchmarks.ground_truth_eval.models import GoldenDataset
from benchmarks.ground_truth_eval.runner import BenchmarkRun

REPORT_KINDS: tuple[str, ...] = (
    "executive",
    "accuracy",
    "calibration",
    "sectors",
    "countries",
    "stages",
    "trend",
    "engine_drift",
    "leaderboard",
    "coverage",
)


def build_report(
    kind: str,
    *,
    dataset: GoldenDataset | None = None,
    run: BenchmarkRun | None = None,
    metrics: GroundTruthMetrics | None = None,
    history: BenchmarkHistory | None = None,
    drift: DriftReport | None = None,
    run_a: BenchmarkRun | None = None,
    run_b: BenchmarkRun | None = None,
    dataset_name: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build one deterministic report by kind.

    ``kind`` must be one of :data:`REPORT_KINDS`.  ``metrics`` is computed
    lazily from ``dataset``/``run`` when needed and not supplied.  The
    report is a plain dict, fully JSON-serialisable.
    """
    if kind not in REPORT_KINDS:
        raise ValueError(f"unknown report kind {kind!r}; expected one of {REPORT_KINDS}")
    stamp = generated_at or datetime.now(UTC)

    metric_kinds = (
        "executive",
        "accuracy",
        "calibration",
        "sectors",
        "countries",
        "stages",
        "coverage",
    )
    if kind in metric_kinds:
        metrics_obj = metrics or _require_metrics(dataset, run)
        content = _body_for(kind, metrics_obj)
        return _envelope(kind, run=run, generated_at=stamp, content=content)

    if kind == "trend":
        history_obj = history or _require_history()
        content = _body_for(kind, None, history=history_obj, dataset_name=dataset_name)
        return _envelope(kind, run=None, generated_at=stamp, content=content)

    if kind == "engine_drift":
        drift_obj = drift or _require_drift(run_a, run_b)
        content = _body_for(kind, None, drift=drift_obj)
        return _envelope(kind, run=run_a or run_b, generated_at=stamp, content=content)

    if kind == "leaderboard":
        content = _leaderboard_content(dataset, run)
        return _envelope(kind, run=run, generated_at=stamp, content=content)

    raise AssertionError(f"unhandled report kind {kind!r}")


def _require_metrics(
    dataset: GoldenDataset | None,
    run: BenchmarkRun | None,
) -> GroundTruthMetrics:
    if dataset is None or run is None:
        raise ValueError("this report kind requires both dataset= and run=")
    return compute_metrics(dataset, run)


def _require_history() -> BenchmarkHistory:
    raise ValueError("report kind 'trend' requires history=")


def _require_drift(run_a: BenchmarkRun | None, run_b: BenchmarkRun | None) -> DriftReport:
    if run_a is None or run_b is None:
        raise ValueError("report kind 'engine_drift' requires run_a= and run_b=")
    return DriftDetector().detect(run_a, run_b)


def _envelope(
    kind: str,
    *,
    run: BenchmarkRun | None,
    generated_at: datetime,
    content: dict[str, Any],
) -> dict[str, Any]:
    header: dict[str, Any] = {
        "report_type": kind,
        "generated_at": generated_at.isoformat(),
        "content": content,
    }
    if run is not None:
        header["run_id"] = run.run_id
        header["engine_version"] = run.engine_version
        header["benchmark_version"] = run.benchmark_version
        header["dataset_name"] = run.dataset_name
        header["dataset_hash"] = run.dataset_hash
    return header


def _body_for(
    kind: str,
    metrics: GroundTruthMetrics | None,
    *,
    history: BenchmarkHistory | None = None,
    dataset_name: str | None = None,
    drift: DriftReport | None = None,
) -> dict[str, Any]:
    metric_kinds = (
        "executive",
        "accuracy",
        "calibration",
        "sectors",
        "countries",
        "stages",
        "coverage",
    )
    if kind in metric_kinds:
        m = metrics
        assert m is not None
        if kind == "executive":
            return _executive_summary(m)
        if kind == "accuracy":
            return _accuracy_dashboard(m)
        if kind == "calibration":
            return _calibration_report(m)
        if kind == "sectors":
            return _grouped_breakdown(m, "sectors")
        if kind == "countries":
            return _grouped_breakdown(m, "countries")
        if kind == "stages":
            return _grouped_breakdown(m, "stages")
        if kind == "coverage":
            return _coverage_report(m)
    if kind == "trend":
        return _historical_trend(history, dataset_name)
    if kind == "engine_drift":
        return _engine_drift_content(drift)
    raise AssertionError(f"unhandled report kind {kind!r}")


# ---------------------------------------------------------------------------
# Metric-driven report bodies
# ---------------------------------------------------------------------------


def _executive_summary(m: GroundTruthMetrics) -> dict[str, Any]:
    verdict = _executive_verdict(m)
    return {
        "headline": {
            "overall_accuracy": m.confusion.accuracy,
            "precision": m.confusion.precision,
            "recall": m.confusion.recall,
            "balanced_accuracy": m.confusion.balanced_accuracy,
            "expected_calibration_error": m.calibration.expected_calibration_error,
            "brier_score": m.brier,
            "top_decile_precision": m.top_decile_precision,
            "investment_hit_rate": m.investment_hit_rate,
            "scoreable_samples": m.confusion.scoreable,
            "dataset_entries": m.coverage.dataset_entries,
        },
        "verdict": verdict,
        "key_findings": _key_findings(m),
    }


def _executive_verdict(m: GroundTruthMetrics) -> str:
    acc = m.confusion.accuracy
    ece = m.calibration.expected_calibration_error
    if acc is None:
        return "insufficient_scoreable_outcomes"
    if acc >= 0.75 and ece <= 0.10:
        return "strong"
    if acc >= 0.60 and ece <= 0.20:
        return "promising"
    return "needs_improvement"


def _key_findings(m: GroundTruthMetrics) -> list[str]:
    findings: list[str] = []
    acc = m.confusion.accuracy
    if acc is not None:
        if acc >= 0.75:
            findings.append(f"Strong overall accuracy ({acc:.2%}).")
        elif acc < 0.60:
            findings.append(f"Overall accuracy below 60% ({acc:.2%}).")
    ece = m.calibration.expected_calibration_error
    finding = (
        f"Expected calibration error is {ece:.4f} "
        f"({'within tolerance' if ece <= 0.10 else 'above 0.10 tolerance'})."
    )
    findings.append(finding)
    if m.calibration.overconfidence_detected:
        findings.append(
            "Overconfidence detected in at least one confidence bin; "
            "review high-confidence low-accuracy companies."
        )
    if m.top_decile_precision is not None and m.top_decile_precision >= 0.8:
        findings.append(f"Top-decile precision is strong ({m.top_decile_precision:.2%}).")
    return findings


def _accuracy_dashboard(m: GroundTruthMetrics) -> dict[str, Any]:
    return {
        "confusion": m.confusion.to_dict(),
        "calibration": m.calibration.to_dict(),
        "brier_score": m.brier,
        "precision_at_k": {str(k): v for k, v in sorted(m.precision_at_k.items())},
        "recall_at_k": {str(k): v for k, v in sorted(m.recall_at_k.items())},
        "top_decile_precision": m.top_decile_precision,
        "investment_hit_rate": m.investment_hit_rate,
    }


def _calibration_report(m: GroundTruthMetrics) -> dict[str, Any]:
    c = m.calibration
    return {
        "expected_calibration_error": c.expected_calibration_error,
        "maximum_calibration_error": c.maximum_calibration_error,
        "overconfidence_detected": c.overconfidence_detected,
        "overconfident_bins": [b.to_dict() for b in c.bins if b.overconfident],
        "bins": [b.to_dict() for b in c.bins],
        "total_samples": c.total_samples,
        "curve": [
            {
                "bin_lower": b.bin_lower,
                "mean_predicted_confidence": b.mean_predicted_confidence,
                "actual_accuracy": b.actual_accuracy,
            }
            for b in c.bins
        ],
    }


def _grouped_breakdown(m: GroundTruthMetrics, group: str) -> dict[str, Any]:
    if group == "sectors":
        table, counts, unit = m.sectors, m.coverage.sectors, "sector"
    elif group == "countries":
        table, counts, unit = m.countries, m.coverage.countries, "country"
    elif group == "stages":
        table, counts, unit = m.stages, m.coverage.stages, "stage"
    else:  # pragma: no cover - guarded by caller
        raise AssertionError(f"unhandled group {group!r}")
    rows = [
        {
            "group": name,
            "accuracy": acc,
            "scoreable_count": counts.get(name, 0),
        }
        for name, acc in sorted(table.items())
    ]
    return {"unit": unit, "rows": rows}


def _coverage_report(m: GroundTruthMetrics) -> dict[str, Any]:
    c = m.coverage
    return {
        "dataset_entries": c.dataset_entries,
        "replayed_entries": c.replayed_entries,
        "failed_entries": c.failed_entries,
        "scoreable_entries": c.scoreable_entries,
        "unscoreable_entries": c.unscoreable_entries,
        "replay_coverage": c.replay_coverage,
        "scoreability": c.scoreability,
        "sectors": dict(sorted(c.sectors.items())),
        "stages": dict(sorted(c.stages.items())),
        "countries": dict(sorted(c.countries.items())),
    }


def _leaderboard_content(
    dataset: GoldenDataset | None,
    run: BenchmarkRun | None,
) -> dict[str, Any]:
    if dataset is None or run is None:
        raise ValueError("report kind 'leaderboard' requires dataset= and run=")
    samples = build_samples(dataset, run)
    rows = [
        {
            "company_id": s.company_id,
            "score": s.score,
            "confidence": s.confidence,
            "decision": s.decision,
            "predicted_positive": s.predicted_positive,
            "actual_positive": s.actual_positive,
            "is_correct": _is_correct(s),
            "sector": s.sector,
            "stage": s.stage,
            "country": s.country,
        }
        for s in samples
    ]
    rows.sort(key=lambda r: (-float(r["score"] or 0.0), r["company_id"]))
    return {
        "ranking_by": "overall_score",
        "rows": rows,
        "order": "descending",
    }


def _is_correct(s: ScoredSample) -> bool | None:
    if s.actual_positive is None or s.predicted_positive is None:
        return None
    return s.predicted_positive == s.actual_positive


def _historical_trend(
    history: BenchmarkHistory | None,
    dataset_name: str | None,
) -> dict[str, Any]:
    if history is None:
        return {"error": "history store required", "rows": []}
    summaries = history.list_summaries()
    if dataset_name is not None:
        summaries = [s for s in summaries if s.dataset_name == dataset_name]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        block = history.load_metrics(summary.run_id)
        if block is None:
            continue
        rows.append(
            {
                "run_id": summary.run_id,
                "created_at": summary.created_at.isoformat(),
                "engine_version": summary.engine_version,
                "benchmark_version": summary.benchmark_version,
                "dataset_name": summary.dataset_name,
                "dataset_hash": summary.dataset_hash,
                "accuracy": _nested(block, "confusion", "accuracy"),
                "precision": _nested(block, "confusion", "precision"),
                "recall": _nested(block, "confusion", "recall"),
                "expected_calibration_error": _nested(
                    block, "calibration", "expected_calibration_error"
                ),
                "brier_score": block.get("brier_score"),
                "top_decile_precision": block.get("top_decile_precision"),
            }
        )
    rows.sort(key=lambda r: (r["created_at"], r["run_id"]))
    return {"dataset_name": dataset_name, "rows": rows}


def _nested(block: dict[str, Any], section: str, key: str) -> Any:
    inner = block.get(section)
    if not isinstance(inner, dict):
        return None
    return inner.get(key)


def _engine_drift_content(drift: DriftReport | None) -> dict[str, Any]:
    if drift is None:
        return {"error": "drift report required"}
    return {
        "run_id_a": drift.run_id_a,
        "run_id_b": drift.run_id_b,
        "engine_version_a": drift.engine_version_a,
        "engine_version_b": drift.engine_version_b,
        "benchmark_version_a": drift.benchmark_version_a,
        "benchmark_version_b": drift.benchmark_version_b,
        "affected_company_count": len(drift.affected_company_ids),
        "signals": [s.to_dict() for s in drift.signals],
        "explanation": drift.explain(),
    }


__all__ = [
    "REPORT_KINDS",
    "build_report",
]
