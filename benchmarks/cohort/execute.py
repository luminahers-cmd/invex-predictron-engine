"""Cohort prediction-execution runner (Phase 4 prediction validation).

Runs a supplied cohort manifest (JSON or CSV — see
:mod:`benchmarks.cohort.manifest_csv`) end to end:

1. **Freeze** — for every manifest record, run one time-pinned, time-scoped
   engine analysis (:func:`record_pinned_prediction`) and persist the frozen
   :class:`TimeScopedPrediction` into an append-only
   :class:`~predictron_engine.dataset.prediction_store.PredictionStore`
   under a deterministic id, so re-running the cohort is idempotent.
2. **Evaluate** — for records with a documented outcome, derive the engine's
   :class:`OutcomeRecord` from the manifest facts (never fabricated) and
   compare it against the frozen prediction using the engine's own
   :class:`PredictionEvaluation` machinery.
3. **Aggregate** — compute evaluation metrics
   (:func:`compute_evaluation_metrics`) and a calibration report
   (:func:`build_calibration_report`) over the cohort.

Artefacts are generated for reports by :mod:`benchmarks.cohort.report` and
exposed to the CLI through :mod:`predictron_engine.dataset.cli_cohort`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.schemas.analysis import StartupAnalysisRequest
from benchmarks.cohort.manifest import (
    CohortManifest,
    ManifestOutcome,
    ManifestSourceRecord,
    manifest_hash,
)
from benchmarks.ground_truth.schema import OutcomeEventKind, StartupStatus
from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.forecast import dataset_record_from_prediction
from predictron_engine.dataset.metrics import (
    EvaluationMetrics,
    binary_label,
    compute_evaluation_metrics,
)
from predictron_engine.dataset.outcomes import (
    FundingEvent,
    OutcomeRecord,
    OutcomeStatus,
    OutcomeVerdict,
    StartupOutcome,
)
from predictron_engine.dataset.prediction import (
    PredictionScopeError,
    TimeScopedPrediction,
    record_pinned_prediction,
    scoped_evidence,
)
from predictron_engine.dataset.prediction_store import (
    PredictionStore,
    stable_prediction_id,
)
from predictron_engine.decision.calibration_sprint8 import build_calibration_report

STATUS_FROZEN = "frozen"
STATUS_PENDING = "pending_ground_truth"
STATUS_FAILED = "failed"


class CohortExecutionError(RuntimeError):
    """Raised when a cohort cannot be executed from its manifest."""


@dataclass
class CohortRecordResult:
    """Outcome of executing one manifest record (freeze + optional evaluation)."""

    company_id: str
    status: str = STATUS_FAILED
    frozen: bool = False
    prediction_id: str | None = None
    prediction: TimeScopedPrediction | None = None
    outcome_record: OutcomeRecord | None = None
    evaluation: PredictionEvaluation | None = None
    error: str | None = None

    @property
    def blocked_lookahead(self) -> bool:
        scope = self.prediction.scope_stats if self.prediction is not None else None
        return bool(scope is not None and scope.blocked_lookahead)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "company_id": self.company_id,
            "status": self.status,
            "frozen": self.frozen,
            "prediction_id": self.prediction_id,
            "error": self.error,
            "blocked_lookahead": self.blocked_lookahead,
        }
        if self.prediction is not None and self.prediction.scope_stats is not None:
            data["evidence_scope"] = self.prediction.scope_stats.model_dump(mode="json")
        if self.evaluation is not None:
            data["evaluation"] = {
                "evaluation_id": self.evaluation.evaluation_id,
                "verdict": self.evaluation.verdict.value,
                "alignment": self.evaluation.alignment.value,
                "decision_match": self.evaluation.decision_match,
            }
        return data


@dataclass
class CohortExecutionResult:
    """Result of executing a cohort manifest (frozen predictions + evaluations)."""

    dataset_name: str
    benchmark_version: str
    manifest_hash: str
    engine_version: str
    evaluation_horizon_days: int | None
    company_results: dict[str, CohortRecordResult] = field(default_factory=dict)
    frozen_predictions: list[TimeScopedPrediction] = field(default_factory=list)
    evaluations: list[PredictionEvaluation] = field(default_factory=list)
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    calibration: dict[str, Any] = field(default_factory=dict)
    freeze_store_root: Path | None = field(default=None)

    @property
    def frozen_company_ids(self) -> list[str]:
        return sorted(
            cid for cid, res in self.company_results.items() if res.status == STATUS_FROZEN
        )

    @property
    def pending_company_ids(self) -> list[str]:
        return sorted(
            cid
            for cid, res in self.company_results.items()
            if res.status == STATUS_PENDING
        )

    @property
    def failed_company_ids(self) -> list[str]:
        return sorted(
            cid for cid, res in self.company_results.items() if res.status == STATUS_FAILED
        )

    def to_report(self) -> dict[str, Any]:
        """Deterministic JSON report document (see :mod:`benchmarks.cohort.report`)."""
        from benchmarks.cohort.report import build_cohort_report

        return build_cohort_report(self)


def run_cohort_execution(
    manifest: CohortManifest,
    *,
    engine: Any | None = None,
    store: PredictionStore | None = None,
    freeze_root: Path | str | None = None,
) -> CohortExecutionResult:
    """Execute a cohort manifest: freeze predictions and evaluate outcomes.

    ``engine`` is injected for tests and defaults to a real
    :class:`predictron_engine.engine.PredictronEngine`.  Frozen predictions
    are persisted into an append-only :class:`PredictionStore`
    (``store`` or a store rooted at ``freeze_root``) under deterministic ids,
    so repeated execution is idempotent.
    """
    from predictron_engine.engine import ENGINE_VERSION, PredictronEngine

    engine_instance = engine if engine is not None else PredictronEngine()
    resolved_store = store if store is not None else _default_store(freeze_root)
    fingerprint = manifest_hash(manifest)
    horizon = manifest.evaluation_horizon_days

    results: dict[str, CohortRecordResult] = {}
    frozen: list[TimeScopedPrediction] = []
    evaluations: list[PredictionEvaluation] = []

    for record in sorted(manifest.source_records, key=lambda r: r.company_id):
        result = _execute_record(
            engine_instance,
            record,
            manifest,
            resolved_store,
            engine_version=ENGINE_VERSION,
            manifest_fingerprint=fingerprint,
            default_horizon=horizon,
        )
        results[record.company_id] = result
        if result.prediction is not None:
            frozen.append(result.prediction)
        if result.evaluation is not None:
            evaluations.append(result.evaluation)

    frozen.sort(key=lambda p: (p.company_id, p.prediction_id))
    evaluations.sort(key=lambda e: (e.record_id, e.evaluation_id))
    metrics = compute_evaluation_metrics(evaluations)
    calibration = _calibration_report(evaluations)

    return CohortExecutionResult(
        dataset_name=manifest.dataset_name,
        benchmark_version=manifest.benchmark_version,
        manifest_hash=fingerprint,
        engine_version=ENGINE_VERSION,
        evaluation_horizon_days=horizon,
        company_results=results,
        frozen_predictions=frozen,
        evaluations=evaluations,
        metrics=metrics,
        calibration=calibration,
        freeze_store_root=resolved_store.root,
    )


def _default_store(freeze_root: Path | str | None) -> PredictionStore:
    if freeze_root is not None:
        return PredictionStore(freeze_root)
    return PredictionStore()


def _execute_record(
    engine: Any,
    record: ManifestSourceRecord,
    manifest: CohortManifest,
    store: PredictionStore,
    *,
    engine_version: str,
    manifest_fingerprint: str,
    default_horizon: int | None,
) -> CohortRecordResult:
    if record.analysis_timestamp.tzinfo is None:
        return CohortRecordResult(
            company_id=record.company_id,
            error="analysis_timestamp must be timezone-aware (UTC)",
        )

    horizon = (
        record.evaluation_horizon_days
        if record.evaluation_horizon_days is not None
        else default_horizon
    )
    prediction_id = stable_prediction_id(
        company_id=record.company_id,
        analysis_timestamp=record.analysis_timestamp,
    )

    if store.has_prediction(prediction_id):
        try:
            existing = store.load(prediction_id)
        except Exception as exc:  # noqa: BLE001 - surface store integrity failures
            return CohortRecordResult(
                company_id=record.company_id,
                status=STATUS_FAILED,
                error=f"stored prediction failed integrity check: {exc}",
            )
        if record.outcome is None:
            return CohortRecordResult(
                company_id=record.company_id,
                status=STATUS_PENDING,
                frozen=True,
                prediction_id=prediction_id,
                prediction=existing,
            )
        outcome_record = outcome_record_from_manifest(
            record.outcome,
            company_id=record.company_id,
            verification_date=record.outcome.verification_date,
            report_horizon_days=horizon,
            analysis_timestamp=record.analysis_timestamp,
        )
        evaluation = _evaluate(existing, outcome_record, record.website)
        return CohortRecordResult(
            company_id=record.company_id,
            status=STATUS_FROZEN,
            frozen=True,
            prediction_id=prediction_id,
            prediction=existing,
            outcome_record=outcome_record,
            evaluation=evaluation,
        )

    try:
        scoped, stats = scoped_evidence(record.evidence_corpus, as_of=record.analysis_timestamp)
    except PredictionScopeError as exc:
        return CohortRecordResult(company_id=record.company_id, error=str(exc))

    try:
        prediction = record_pinned_prediction(
            engine,
            request=_request_from_record(record),
            analysis_timestamp=record.analysis_timestamp,
            company_id=record.company_id,
            company_name=record.company_name,
            evidence_bundle=scoped,
            evidence_reference=record.evidence_corpus,
            evaluation_horizon_days=horizon,
            metadata={
                "sources": list(record.metadata.get("sources") or []),
                "manifest_hash": manifest_fingerprint,
                "engine_version": engine_version,
                "evidence_scope": stats.model_dump(mode="json"),
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface per-company freeze failures
        return CohortRecordResult(
            company_id=record.company_id, error=f"engine analyze failed: {exc}"
        )

    prediction = prediction.model_copy(
        update={"prediction_id": prediction_id}
    )
    _path, persisted = store.save_if_absent(prediction)
    del _path

    if record.outcome is None:
        return CohortRecordResult(
            company_id=record.company_id,
            status=STATUS_PENDING,
            frozen=persisted,
            prediction_id=prediction_id,
            prediction=prediction,
        )

    outcome_record = outcome_record_from_manifest(
        record.outcome,
        company_id=record.company_id,
        verification_date=record.outcome.verification_date,
        report_horizon_days=horizon,
        analysis_timestamp=record.analysis_timestamp,
    )
    evaluation = _evaluate(prediction, outcome_record, record.website)
    return CohortRecordResult(
        company_id=record.company_id,
        status=STATUS_FROZEN,
        frozen=persisted,
        prediction_id=prediction_id,
        prediction=prediction,
        outcome_record=outcome_record,
        evaluation=evaluation,
    )


def _request_from_record(record: ManifestSourceRecord) -> StartupAnalysisRequest:
    request: dict[str, Any] = {
        "startup_name": record.company_name,
        "description": record.description,
    }
    if record.website:
        request["website"] = record.website
    if record.pitch_deck_url:
        request["pitch_deck_url"] = record.pitch_deck_url
    if record.founder_linkedin_urls:
        request["founder_linkedin_urls"] = list(record.founder_linkedin_urls)
    return StartupAnalysisRequest(**request)


def _evaluate(
    prediction: TimeScopedPrediction,
    outcome_record: OutcomeRecord,
    website: str | None,
) -> PredictionEvaluation:
    dataset_record = dataset_record_from_prediction(prediction, website=website)
    evaluation = PredictionEvaluation.from_records(dataset_record, outcome_record)
    evaluation_id = hashlib.sha256(
        _PREDICTION_EVALUATION_SEPARATOR.join(
            [prediction.prediction_id, outcome_record.outcome_id]
        ).encode("utf-8")
    ).hexdigest()
    return evaluation.model_copy(update={"evaluation_id": evaluation_id})


def stable_outcome_id(
    *, company_id: str, outcome: ManifestOutcome
) -> str:
    """Deterministic outcome identity from (company, manifest outcome)."""
    import json

    signature = json.dumps(
        outcome.model_dump(mode="json"), sort_keys=True, default=str
    )
    material = _PREDICTION_EVALUATION_SEPARATOR.join([company_id, signature])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def outcome_record_from_manifest(
    outcome: ManifestOutcome,
    *,
    company_id: str,
    verification_date: date | None,
    report_horizon_days: int | None,
    analysis_timestamp: datetime,
) -> OutcomeRecord:
    """Map built manifest outcome facts onto the engine's OutcomeRecord.

    Only documented facts are translated: terminal events come from the
    manifest status / outcome events, never invented.  The verdict is
    derived through the engine's own ``OutcomeRecord.derive_verdict``.
    """
    startup_outcome = _startup_outcome(outcome)
    horizon_days: int | None = None
    if report_horizon_days is not None:
        horizon_days = report_horizon_days
    elif verification_date is not None:
        delta = (verification_date - analysis_timestamp.date()).days
        if delta >= 0:
            horizon_days = delta
    record = OutcomeRecord(
        outcome_id=stable_outcome_id(company_id=company_id, outcome=outcome),
        record_id=company_id,
        outcome=startup_outcome,
        verdict=OutcomeVerdict.UNKNOWN,
        time_horizon_days=horizon_days,
        notes=outcome.notes,
        metadata={
            "verified": outcome.verified,
            "verified_by": outcome.verified_by,
            "verification_date": (
                verification_date.isoformat() if verification_date is not None else None
            ),
        },
    )
    verdict = record.derive_verdict()
    if verdict == OutcomeVerdict.UNKNOWN and outcome.status != StartupStatus.UNKNOWN:
        verdict = OutcomeVerdict.INCONCLUSIVE
    return record.model_copy(update={"verdict": verdict})


def _startup_outcome(outcome: ManifestOutcome) -> StartupOutcome:
    """Translate manifest outcome facts into engine StartupOutcome facts."""
    if outcome.status == StartupStatus.UNKNOWN:
        status = OutcomeStatus.UNKNOWN
    else:
        status = (
            OutcomeStatus.FULLY_VERIFIED
            if outcome.verified
            else OutcomeStatus.PARTIALLY_VERIFIED
        )
    so = StartupOutcome(status=status)
    verification = datetime.combine(outcome.verification_date, datetime.min.time(), tzinfo=UTC)

    for event in outcome.outcome_events:
        if event.kind in (OutcomeEventKind.FUNDING, OutcomeEventKind.FOLLOW_ON_ROUND):
            so.funding_rounds.append(
                FundingEvent(
                    round_type=event.kind.value,
                    amount_usd=event.value_currency_usd,
                    date=datetime.combine(event.occurred_at, datetime.min.time(), tzinfo=UTC),
                    investors=[],
                    valuation_usd=None,
                    source="; ".join(event.sources),
                )
            )
        elif event.kind == OutcomeEventKind.ARR_MILESTONE:
            so.arr_milestones.append(
                {
                    "date": event.occurred_at.isoformat(),
                    "arr_usd": event.amount_nominal_units,
                    "source": "; ".join(event.sources),
                }
            )
        elif event.kind == OutcomeEventKind.INVESTOR_PARTICIPATION:
            so.investors.extend(list(event.sources))
        elif event.kind == OutcomeEventKind.SHUTDOWN:
            so.shutdown = True
            so.shutdown_date = datetime.combine(
                event.occurred_at, datetime.min.time(), tzinfo=UTC
            )
        elif event.kind == OutcomeEventKind.ACQUISITION:
            acquirer = _acquirer_name(event)
            so.acquisition = acquirer
            so.exit_type = "acquisition"
            so.exit_date = datetime.combine(
                event.occurred_at, datetime.min.time(), tzinfo=UTC
            )
            if event.value_currency_usd is not None:
                so.acquisition_price_usd = event.value_currency_usd

    if outcome.status == StartupStatus.SHUTDOWN:
        so.shutdown = True
        so.shutdown_date = so.shutdown_date or verification
    if outcome.status == StartupStatus.ACQUIRED:
        so.exit_type = "acquisition"
        so.exit_date = so.exit_date or verification
        if outcome.exit_value_usd is not None:
            so.acquisition_price_usd = outcome.exit_value_usd

    funding_total = sum(
        round.amount_usd for round in so.funding_rounds if round.amount_usd is not None
    )
    if funding_total > 0:
        so.total_funding_usd = funding_total
    if so.acquisition_price_usd is None and outcome.exit_value_usd is not None:
        so.acquisition_price_usd = outcome.exit_value_usd
    so.latest_verification_date = verification
    return so


def _acquirer_name(event: Any) -> str | None:
    name = str(event.description or "").strip()
    if not name or name.lower() in ("acquired", "acquisition"):
        return None
    return name


def _calibration_report(evaluations: list[PredictionEvaluation]) -> dict[str, Any]:
    """Calibration report from evaluations with verified binary outcomes."""
    confidences: list[float] = []
    outcomes: list[int] = []
    for evaluation in evaluations:
        label = binary_label(evaluation)
        if label.actual_positive is None:
            continue
        confidences.append(float(evaluation.prediction.confidence))
        outcomes.append(1 if label.actual_positive else 0)
    if not confidences:
        return {
            "expected_calibration_error": 0.0,
            "maximum_calibration_error": 0.0,
            "overconfidence_detected": False,
            "overconfident_bins_count": 0,
            "bins": [],
            "total_samples": 0,
            "calibration_quality": "insufficient_data",
        }
    report = build_calibration_report(
        confidences, [0.0] * len(confidences), correct_predictions=outcomes
    )
    return report.to_dict()


_PREDICTION_EVALUATION_SEPARATOR = "\x00"


__all__ = [
    "STATUS_FAILED",
    "STATUS_FROZEN",
    "STATUS_PENDING",
    "CohortExecutionError",
    "CohortExecutionResult",
    "CohortRecordResult",
    "outcome_record_from_manifest",
    "run_cohort_execution",
    "stable_outcome_id",
]
