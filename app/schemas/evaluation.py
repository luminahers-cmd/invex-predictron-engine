"""Pydantic schemas for the company validity-loop read side (CIH Phase 4).

Additive response schemas for prediction evaluation: aggregate metrics,
calibration, prediction/outcome alignment, confidence-versus-verdict, and
per-snapshot evaluation history. All values are derived from the existing
benchmark machinery (``compute_evaluation_metrics`` and
``build_calibration_report``) — nothing is recomputed by hand.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EvaluationMetricsBlock(BaseModel):
    """Aggregate evaluation metrics over a set of prediction evaluations."""

    total: int = Field(default=0, ge=0, description="Evaluations examined")
    scoreable: int = Field(
        default=0, ge=0, description="Evaluations with sufficient binary ground truth"
    )
    insufficient_ground_truth: int = Field(
        default=0, ge=0, description="Evaluations excluded for insufficient ground truth"
    )
    true_positives: int = Field(default=0, ge=0)
    true_negatives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)
    precision: float | None = Field(default=None)
    recall: float | None = Field(default=None)
    specificity: float | None = Field(default=None)
    f1: float | None = Field(default=None)
    balanced_accuracy: float | None = Field(default=None)
    coverage: float | None = Field(default=None)
    insufficient_ground_truth_rate: float | None = Field(default=None)
    verdict_counts: dict[str, int] = Field(
        default_factory=dict, description="Evaluation verdict distribution"
    )


class CalibrationBlock(BaseModel):
    """Reliability-diagram calibration summary for one evaluation set."""

    expected_calibration_error: float = Field(default=0.0)
    maximum_calibration_error: float = Field(default=0.0)
    overconfidence_detected: bool = Field(default=False)
    overconfident_bins_count: int = Field(default=0, ge=0)
    bins: list[dict[str, object]] = Field(
        default_factory=list, description="Per-bin reliability diagram rows"
    )
    total_samples: int = Field(default=0, ge=0)
    calibration_quality: str = Field(default="unknown")


class EvaluationHistoryEntry(BaseModel):
    """One persisted prediction evaluation, as stored (append-only)."""

    id: str = Field(..., description="Deterministic evaluation identifier")
    snapshot_id: str = Field(..., description="Evaluated prediction snapshot")
    outcome_id: str = Field(..., description="Outcome the snapshot was compared against")
    snapshot_created_at: datetime = Field(
        ..., description="When the evaluated snapshot was created"
    )
    decision: str | None = Field(default=None, description="Engine decision at snapshot time")
    confidence: float | None = Field(default=None, description="Snapshot confidence")
    verdict: str = Field(..., description="Evaluation verdict")
    alignment: str = Field(..., description="Prediction/outcome direction alignment")
    outcome_verdict: str = Field(..., description="Derived outcome verdict")
    decision_match: bool | None = Field(
        default=None, description="Whether the engine decision was directionally correct"
    )
    created_at: datetime = Field(..., description="When the evaluation was recorded")


class CompanyPerformanceResponse(BaseModel):
    """A company's validity-loop performance over all its evaluations."""

    company_id: str = Field(..., description="Company identifier")
    outcome_count: int = Field(default=0, ge=0, description="Outcomes recorded")
    evaluation_count: int = Field(default=0, ge=0, description="Evaluations recorded")
    metrics: EvaluationMetricsBlock = Field(..., description="Aggregate metrics")
    calibration: CalibrationBlock = Field(..., description="Calibration summary")
    alignment: dict[str, int] = Field(
        default_factory=dict, description="Alignment distribution"
    )
    confidence_vs_outcome: dict[str, float | None] = Field(
        default_factory=dict,
        description="Mean snapshot confidence per evaluation verdict",
    )
    history: list[EvaluationHistoryEntry] = Field(
        default_factory=list, description="Evaluation history (newest-first, paginated)"
    )
    generated_at: datetime = Field(..., description="When this view was generated")


class EvaluationCalibrationResponse(BaseModel):
    """Calibration across every company in the caller's scope."""

    company_count: int = Field(default=0, ge=0)
    evaluation_count: int = Field(default=0, ge=0)
    scoreable_count: int = Field(default=0, ge=0)
    calibration: CalibrationBlock = Field(..., description="Calibration summary")
    alignment: dict[str, int] = Field(default_factory=dict)
    confidence_vs_outcome: dict[str, float | None] = Field(default_factory=dict)
    generated_at: datetime = Field(..., description="When this view was generated")


class CompanyPerformanceEntry(BaseModel):
    """Per-company aggregate row inside the global evaluation summary."""

    company_id: str = Field(..., description="Company identifier")
    outcome_count: int = Field(default=0, ge=0)
    evaluation_count: int = Field(default=0, ge=0)
    scoreable: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)
    precision: float | None = Field(default=None)
    recall: float | None = Field(default=None)
    f1: float | None = Field(default=None)
    coverage: float | None = Field(default=None)


class EvaluationSummaryResponse(BaseModel):
    """Global validity-loop summary plus per-company breakdown."""

    company_count: int = Field(default=0, ge=0)
    outcome_count: int = Field(default=0, ge=0)
    evaluation_count: int = Field(default=0, ge=0)
    metrics: EvaluationMetricsBlock = Field(..., description="Aggregate metrics")
    calibration: CalibrationBlock = Field(..., description="Calibration summary")
    alignment: dict[str, int] = Field(default_factory=dict)
    confidence_vs_outcome: dict[str, float | None] = Field(default_factory=dict)
    by_company: list[CompanyPerformanceEntry] = Field(default_factory=list)
    generated_at: datetime = Field(..., description="When this view was generated")


class BenchmarkCompanyEntry(BaseModel):
    """One company's engine replay result inside a benchmark run."""

    company_id: str = Field(..., description="Company identifier in the dataset")
    success: bool = Field(..., description="Whether engine replay produced a report")
    decision: str | None = Field(default=None, description="Engine decision at replay time")
    overall_score: float | None = Field(default=None)
    overall_confidence: float | None = Field(default=None)
    error: str | None = Field(default=None)


class BenchmarkMetricsSummary(BaseModel):
    """Headline accuracy metrics of the latest benchmark run (V1.4)."""

    total: int = Field(default=0, ge=0, description="Dataset entries examined")
    scoreable: int = Field(default=0, ge=0, description="Entries with verified binary ground truth")
    insufficient_ground_truth: int = Field(default=0, ge=0)
    true_positives: int = Field(default=0, ge=0)
    true_negatives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)
    precision: float | None = Field(default=None)
    recall: float | None = Field(default=None)
    specificity: float | None = Field(default=None)
    f1: float | None = Field(default=None)
    f0_5: float | None = Field(default=None)
    balanced_accuracy: float | None = Field(default=None)
    false_positive_rate: float | None = Field(default=None)
    false_negative_rate: float | None = Field(default=None)
    base_rate_positive: float | None = Field(
        default=None, description="Observed success prevalence"
    )
    base_rate_negative: float | None = Field(
        default=None, description="Observed failure prevalence"
    )
    roc_auc: float | None = Field(default=None)
    average_precision: float | None = Field(default=None)
    brier_score: float | None = Field(default=None)
    expected_calibration_error: float = Field(default=0.0)
    overconfidence_detected: bool = Field(default=False)
    coverage: float | None = Field(
        default=None, description="Fraction of entries with ground truth"
    )


class BenchmarkAccuracyResponse(BaseModel):
    """Read-only accuracy view over the latest recorded benchmark run."""

    run_id: str = Field(..., description="Latest benchmark run identifier")
    dataset_name: str = Field(..., description="Golden dataset benchmarked")
    benchmark_version: str = Field(..., description="Dataset version string")
    engine_version: str = Field(..., description="Engine version that produced the run")
    run_created_at: datetime = Field(..., description="When the run was recorded")
    dataset_hash: str = Field(..., description="Golden dataset fingerprint")
    result_hash: str = Field(..., description="Run result fingerprint (integrity-verified)")
    entry_count: int = Field(default=0, ge=0)
    successful_count: int = Field(default=0, ge=0)
    has_metrics: bool = Field(default=False, description="Whether a metric block was stored")
    metrics: BenchmarkMetricsSummary | None = Field(
        default=None, description="Headline metrics (None unless the run stored a metric block)"
    )
    companies: list[BenchmarkCompanyEntry] = Field(
        default_factory=list, description="Per-company engine replay results"
    )
    generated_at: datetime = Field(..., description="When this view was generated")


__all__ = [
    "BenchmarkAccuracyResponse",
    "BenchmarkCompanyEntry",
    "BenchmarkMetricsSummary",
    "CalibrationBlock",
    "CompanyPerformanceEntry",
    "CompanyPerformanceResponse",
    "EvaluationCalibrationResponse",
    "EvaluationHistoryEntry",
    "EvaluationMetricsBlock",
    "EvaluationSummaryResponse",
]
