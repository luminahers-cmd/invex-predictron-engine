"""Explainable drift detection (Project E5).

Compares two benchmark runs (or a run against a baseline) and reports how
much the *engine behaviour* drifted across:

* **score drift** — per-company overall score deltas,
* **decision drift** — companies whose decision category changed,
* **confidence drift** — per-company overall-confidence deltas,
* **recommendation drift** — recommendation-count and category changes,
* **feature drift** — extracted-feature value changes.

Every drift signal is deterministic and explainable: it lists the affected
companies, the exact ``from_value``/``to_value``, and a scalar magnitude.
`detect_metric_drift` additionally compares the aggregate metric block of
two runs (e.g. across benchmark *dataset versions*).

Severity thresholds are module-level named constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.ground_truth_eval.runner import FEATURE_KEYS, BenchmarkRun

# Deterministic severity thresholds (from/to deltas are absolute).
SCORE_DRIFT_EPSILON = 0.01
CONFIDENCE_DRIFT_EPSILON = 0.01
SEVERITY_MINOR = 1.0
SEVERITY_MODERATE = 3.0
SEVERITY_MAJOR = 10.0

# Metric names compared by :func:`detect_metric_drift`.
COMPARED_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "top_decile_precision",
    "investment_hit_rate",
)


@dataclass
class DriftItem:
    """One concrete from/to change for a company."""

    company_id: str
    field: str
    from_value: Any
    to_value: Any
    delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "field": self.field,
            "from": self.from_value,
            "to": self.to_value,
            "delta": self.delta,
        }


@dataclass
class DriftSignal:
    """One drift dimension (score, decision, confidence, ...)."""

    signal: str
    magnitude: float
    affected_count: int
    total_count: int
    items: list[DriftItem] = field(default_factory=list)

    @property
    def affected_fraction(self) -> float:
        if self.total_count == 0:
            return 0.0
        return round(self.affected_count / self.total_count, 6)

    @property
    def severity(self) -> str:
        if self.magnitude >= SEVERITY_MAJOR:
            return "major"
        if self.magnitude >= SEVERITY_MODERATE:
            return "moderate"
        if self.magnitude >= SEVERITY_MINOR:
            return "minor"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "magnitude": round(self.magnitude, 6),
            "affected_count": self.affected_count,
            "total_count": self.total_count,
            "affected_fraction": self.affected_fraction,
            "severity": self.severity,
            "items": [i.to_dict() for i in self.items],
        }


@dataclass
class DriftReport:
    """Complete drift comparison between two runs."""

    run_id_a: str
    run_id_b: str
    engine_version_a: str
    engine_version_b: str
    benchmark_version_a: str
    benchmark_version_b: str
    signals: list[DriftSignal] = field(default_factory=list)

    def signal(self, name: str) -> DriftSignal | None:
        for s in self.signals:
            if s.signal == name:
                return s
        return None

    @property
    def affected_company_ids(self) -> list[str]:
        ids: set[str] = set()
        for s in self.signals:
            for item in s.items:
                ids.add(item.company_id)
        return sorted(ids)

    @property
    def is_empty(self) -> bool:
        return all(s.affected_count == 0 for s in self.signals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id_a": self.run_id_a,
            "run_id_b": self.run_id_b,
            "engine_version_a": self.engine_version_a,
            "engine_version_b": self.engine_version_b,
            "benchmark_version_a": self.benchmark_version_a,
            "benchmark_version_b": self.benchmark_version_b,
            "affected_company_count": len(self.affected_company_ids),
            "signals": [s.to_dict() for s in self.signals],
        }

    def explain(self) -> str:
        """Human-readable, deterministic explanation (markdown)."""
        lines = ["# Engine Drift Report", ""]
        lines.append(
            f"- Baseline run: `{self.run_id_a}` "
            f"(engine `{self.engine_version_a}`, benchmark `{self.benchmark_version_a}`)"
        )
        lines.append(
            f"- Compared run: `{self.run_id_b}` "
            f"(engine `{self.engine_version_b}`, benchmark `{self.benchmark_version_b}`)"
        )
        if self.is_empty:
            lines.append("")
            lines.append("No drift detected in any dimension.")
            return "\n".join(lines)
        lines.append("")
        for signal in self.signals:
            if signal.affected_count == 0:
                continue
            lines.append(f"## {signal.signal} drift")
            lines.append(
                f"- Magnitude: {signal.magnitude:.4f} "
                f"(severity: {signal.severity}), "
                f"affected {signal.affected_count}/{signal.total_count} companies"
            )
            for item in signal.items[:20]:
                lines.append(
                    f"  - `{item.company_id}` `{item.field}`: "
                    f"`{item.from_value}` -> `{item.to_value}`"
                )
            if len(signal.items) > 20:
                lines.append(f"  - … {len(signal.items) - 20} more")
            lines.append("")
        return "\n".join(lines)


class DriftDetector:
    """Computes deterministic drift between two benchmark runs."""

    def detect(self, run_a: BenchmarkRun, run_b: BenchmarkRun) -> DriftReport:
        """Detect drift across all five dimensions."""
        return DriftReport(
            run_id_a=run_a.run_id,
            run_id_b=run_b.run_id,
            engine_version_a=run_a.engine_version,
            engine_version_b=run_b.engine_version,
            benchmark_version_a=run_a.benchmark_version,
            benchmark_version_b=run_b.benchmark_version,
            signals=[
                self.score_drift(run_a, run_b),
                self.decision_drift(run_a, run_b),
                self.confidence_drift(run_a, run_b),
                self.recommendation_drift(run_a, run_b),
                self.feature_drift(run_a, run_b),
            ],
        )

    # ------------------------------------------------------------------
    # Individual dimensions
    # ------------------------------------------------------------------

    def score_drift(self, run_a: BenchmarkRun, run_b: BenchmarkRun) -> DriftSignal:
        items: list[DriftItem] = []
        deltas: list[float] = []
        for company_id in _common_failed_safe_ids(run_a, run_b):
            ea = run_a.entry_by_id(company_id)
            eb = run_b.entry_by_id(company_id)
            if ea is None or eb is None or not ea.success or not eb.success:
                continue
            if ea.overall_score is None or eb.overall_score is None:
                continue
            delta = round(float(eb.overall_score) - float(ea.overall_score), 6)
            if abs(delta) > SCORE_DRIFT_EPSILON:
                items.append(
                    DriftItem(
                        company_id=company_id,
                        field="overall_score",
                        from_value=ea.overall_score,
                        to_value=eb.overall_score,
                        delta=delta,
                    )
                )
            deltas.append(abs(delta))
        return DriftSignal(
            signal="score",
            magnitude=round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
            affected_count=len(items),
            total_count=_successful_common_count(run_a, run_b),
            items=items,
        )

    def decision_drift(self, run_a: BenchmarkRun, run_b: BenchmarkRun) -> DriftSignal:
        items: list[DriftItem] = []
        changed = 0
        total = 0
        decisions_a: dict[str, str] = {}
        for company_id in _common_failed_safe_ids(run_a, run_b):
            ea = run_a.entry_by_id(company_id)
            eb = run_b.entry_by_id(company_id)
            if ea is None or eb is None or not ea.success or not eb.success:
                continue
            decision_a = ea.decision
            decision_b = eb.decision
            if decision_a is None and decision_b is None:
                continue
            total += 1
            decisions_a[company_id] = decision_a or "none"
            if decision_a != decision_b:
                changed += 1
                items.append(
                    DriftItem(
                        company_id=company_id,
                        field="decision",
                        from_value=decision_a,
                        to_value=decision_b,
                    )
                )
        magnitude = round(changed / total, 6) if total else 0.0
        return DriftSignal(
            signal="decision",
            magnitude=magnitude,
            affected_count=changed,
            total_count=total,
            items=items,
        )

    def confidence_drift(self, run_a: BenchmarkRun, run_b: BenchmarkRun) -> DriftSignal:
        items: list[DriftItem] = []
        deltas: list[float] = []
        for company_id in _common_failed_safe_ids(run_a, run_b):
            ea = run_a.entry_by_id(company_id)
            eb = run_b.entry_by_id(company_id)
            if ea is None or eb is None or not ea.success or not eb.success:
                continue
            if ea.overall_confidence is None or eb.overall_confidence is None:
                continue
            delta = round(float(eb.overall_confidence) - float(ea.overall_confidence), 6)
            if abs(delta) > CONFIDENCE_DRIFT_EPSILON:
                items.append(
                    DriftItem(
                        company_id=company_id,
                        field="overall_confidence",
                        from_value=ea.overall_confidence,
                        to_value=eb.overall_confidence,
                        delta=delta,
                    )
                )
            deltas.append(abs(delta))
        return DriftSignal(
            signal="confidence",
            magnitude=round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
            affected_count=len(items),
            total_count=_successful_common_count(run_a, run_b),
            items=items,
        )

    def recommendation_drift(self, run_a: BenchmarkRun, run_b: BenchmarkRun) -> DriftSignal:
        items: list[DriftItem] = []
        changed = 0
        total = 0
        for company_id in _common_failed_safe_ids(run_a, run_b):
            ea = run_a.entry_by_id(company_id)
            eb = run_b.entry_by_id(company_id)
            if ea is None or eb is None or not ea.success or not eb.success:
                continue
            total += 1
            count_delta = eb.recommendation_count - ea.recommendation_count
            set_delta = sorted(
                set(eb.recommendation_categories) ^ set(ea.recommendation_categories)
            )
            if count_delta != 0 or set_delta:
                changed += 1
                items.append(
                    DriftItem(
                        company_id=company_id,
                        field="recommendations",
                        from_value={
                            "count": ea.recommendation_count,
                            "categories": sorted(ea.recommendation_categories),
                        },
                        to_value={
                            "count": eb.recommendation_count,
                            "categories": sorted(eb.recommendation_categories),
                        },
                        delta=float(count_delta),
                    )
                )
        return DriftSignal(
            signal="recommendation",
            magnitude=round(changed / total, 6) if total else 0.0,
            affected_count=changed,
            total_count=total,
            items=items,
        )

    def feature_drift(self, run_a: BenchmarkRun, run_b: BenchmarkRun) -> DriftSignal:
        items: list[DriftItem] = []
        changed_count = 0
        field_total = 0
        for company_id in _common_failed_safe_ids(run_a, run_b):
            ea = run_a.entry_by_id(company_id)
            eb = run_b.entry_by_id(company_id)
            if ea is None or eb is None or not ea.success or not eb.success:
                continue
            for key in FEATURE_KEYS:
                value_a = ea.extracted_features.get(key)
                value_b = eb.extracted_features.get(key)
                field_total += 1
                if value_a != value_b:
                    changed_count += 1
                    items.append(
                        DriftItem(
                            company_id=company_id,
                            field=f"feature:{key}",
                            from_value=value_a,
                            to_value=value_b,
                        )
                    )
        magnitude = round(changed_count / field_total, 6) if field_total else 0.0
        return DriftSignal(
            signal="feature",
            magnitude=magnitude,
            affected_count=len({i.company_id for i in items}),
            total_count=_successful_common_count(run_a, run_b),
            items=items,
        )


def _common_failed_safe_ids(run_a: BenchmarkRun, run_b: BenchmarkRun) -> list[str]:
    """Common company ids across both runs (sorted)."""
    ids_a = {e.company_id for e in run_a.entries}
    ids_b = {e.company_id for e in run_b.entries}
    return sorted(ids_a & ids_b)


def _successful_common_count(run_a: BenchmarkRun, run_b: BenchmarkRun) -> int:
    count = 0
    for company_id in _common_failed_safe_ids(run_a, run_b):
        ea = run_a.entry_by_id(company_id)
        eb = run_b.entry_by_id(company_id)
        if ea is not None and eb is not None and ea.success and eb.success:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Metric drift (across benchmark versions / metrics)
# ---------------------------------------------------------------------------


@dataclass
class MetricDriftItem:
    """Aggregate metric delta between two metric blocks."""

    metric: str
    from_value: float | None
    to_value: float | None
    delta: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "from": self.from_value,
            "to": self.to_value,
            "delta": self.delta,
        }


@dataclass
class MetricDriftReport:
    """Aggregate metric drift between two runs/dataset versions."""

    metrics_a: str = ""
    metrics_b: str = ""
    items: list[MetricDriftItem] = field(default_factory=list)

    @property
    def changed_items(self) -> list[MetricDriftItem]:
        return [i for i in self.items if i.delta is not None and abs(i.delta) > 0.0001]

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.metrics_a,
            "compared": self.metrics_b,
            "changed_metric_count": len(self.changed_items),
            "metrics": [i.to_dict() for i in self.items],
        }

    def explain(self) -> str:
        lines = ["# Aggregate Metric Drift", ""]
        lines.append(f"- Baseline: `{self.metrics_a}`")
        lines.append(f"- Compared: `{self.metrics_b}`")
        if not self.changed_items:
            lines.append("")
            lines.append("No aggregate metric drift detected.")
            return "\n".join(lines)
        lines.append("")
        for item in self.changed_items:
            lines.append(
                f"- `{item.metric}`: `{item.from_value}` -> `{item.to_value}` "
                f"(delta {item.delta:+.4f})"
            )
        return "\n".join(lines)


def detect_metric_drift(
    metrics_a: Any,
    metrics_b: Any,
) -> MetricDriftReport:
    """Compare the aggregate metric blocks of two runs.

    Accepts either :class:`GroundTruthMetrics` dataclasses or plain dicts
    (e.g. report sections or exported JSON), so it works across dataset
    versions and engine versions uniformly.
    """
    values_a = _metric_values(metrics_a)
    values_b = _metric_values(metrics_b)
    items: list[MetricDriftItem] = []
    for name in COMPARED_METRICS:
        va = values_a.get(name)
        vb = values_b.get(name)
        delta = _delta(va, vb)
        items.append(MetricDriftItem(metric=name, from_value=va, to_value=vb, delta=delta))
    return MetricDriftReport(
        metrics_a=_label(metrics_a),
        metrics_b=_label(metrics_b),
        items=items,
    )


def _label(metrics: Any) -> str:
    if isinstance(metrics, dict):
        return str(metrics.get("run_id", metrics.get("benchmark_version", "?")))
    return str(getattr(metrics, "run_id", getattr(metrics, "benchmark_version", "?")))


def _metric_values(metrics: Any) -> dict[str, float | None]:
    if isinstance(metrics, dict):
        return {name: _as_float(metrics.get(name)) for name in COMPARED_METRICS if name in metrics}
    return {
        name: getattr(metrics, "metric_value", lambda n: None)(name) for name in COMPARED_METRICS
    }


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 6)


__all__ = [
    "SCORE_DRIFT_EPSILON",
    "CONFIDENCE_DRIFT_EPSILON",
    "SEVERITY_MINOR",
    "SEVERITY_MODERATE",
    "SEVERITY_MAJOR",
    "COMPARED_METRICS",
    "DriftItem",
    "DriftSignal",
    "DriftReport",
    "DriftDetector",
    "MetricDriftItem",
    "MetricDriftReport",
    "detect_metric_drift",
]
