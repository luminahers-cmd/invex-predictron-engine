"""CompanyEvaluationService — the Phase 4 prediction-evaluation layer.

This service reconciles recorded outcomes against the immutable analysis
snapshots of a company and derives every reported metric through the
existing benchmark machinery:

* :class:`PredictionEvaluation` / :func:`PredictionEvaluation.from_records`
  — verdict and alignment derivation;
* :func:`compute_evaluation_metrics` — accuracy / precision / recall / F1 /
  coverage (``None`` when undefined — never fabricated);
* :func:`build_calibration_report` — Expected Calibration Error and
  overconfidence detection from observed outcomes only.

Evaluations are append-only rows with deterministic ids.  ``reconcile`` is
an idempotent pass: it creates the evaluations that are newly applicable
(for example when a later snapshot becomes applicable to an earlier
outcome) and never rewrites an existing row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import desc, func, select

from app.models.company import Company, CompanySnapshot
from app.models.company_outcome import CompanyEvaluation, CompanyOutcome
from app.schemas.evaluation import (
    CalibrationBlock,
    CompanyPerformanceEntry,
    CompanyPerformanceResponse,
    EvaluationCalibrationResponse,
    EvaluationHistoryEntry,
    EvaluationMetricsBlock,
    EvaluationSummaryResponse,
)
from app.services.company_outcomes import (
    as_utc,
    outcome_record_from_row,
)
from app.services.company_postgres import PostgresCompanyStore
from predictron_engine.dataset.evaluation import (
    EvaluationVerdict,
    PredictionEvaluation,
    PredictionOutcomeAlignment,
)
from predictron_engine.dataset.metrics import (
    EvaluationMetrics,
    binary_label,
    compute_evaluation_metrics,
)
from predictron_engine.dataset.models import (
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.decision.calibration_sprint8 import build_calibration_report

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_EVALUATION_ID_SEPARATOR = "\x00"


def compute_evaluation_id(
    *, company_id: str, snapshot_id: str, outcome_id: str
) -> str:
    """Deterministic evaluation ID derived from (company, snapshot, outcome)."""
    import hashlib

    material = _EVALUATION_ID_SEPARATOR.join(
        [company_id, snapshot_id, outcome_id]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def decision_label(value: str | None) -> DecisionLabel:
    """Best-effort mapping of a stored decision string to a decision label."""
    if value is None:
        return DecisionLabel.WATCH
    try:
        return DecisionLabel(value)
    except ValueError:
        return DecisionLabel.WATCH


def prediction_summary_from_snapshot(snapshot: CompanySnapshot) -> PredictionSummary:
    """Build the frozen prediction summary captured by a snapshot.

    Only stored snapshot fields are used (decision, confidence, composite
    score, dimension scores, readiness). ``recommendation_count`` defaults to
    0 because snapshots do not persist recommendations.
    """
    dimension_scores = {
        str(key): float(value)
        for key, value in (snapshot.dimension_scores or {}).items()
    }
    return PredictionSummary(
        decision=decision_label(snapshot.decision),
        confidence=float(snapshot.confidence or 0.0),
        composite_score=float(snapshot.composite_score or 0.0),
        dimension_scores=dimension_scores,
        investment_readiness_score=(
            float(snapshot.readiness_score)
            if snapshot.readiness_score is not None
            else None
        ),
        recommendation_count=0,
    )


def prediction_evaluation_from_rows(
    evaluation: CompanyEvaluation, outcome: CompanyOutcome
) -> PredictionEvaluation:
    """Rebuild the engine's :class:`PredictionEvaluation` from stored rows."""
    return PredictionEvaluation(
        evaluation_id=evaluation.id,
        record_id=evaluation.snapshot_id,
        prediction=PredictionSummary.model_validate(evaluation.prediction),
        outcome_record=outcome_record_from_row(outcome),
        verdict=EvaluationVerdict(evaluation.verdict),
        alignment=PredictionOutcomeAlignment(evaluation.alignment),
        decision_match=evaluation.decision_match,
        created_at=evaluation.created_at,
    )


def decision_match_from_verdict(verdict: EvaluationVerdict) -> bool | None:
    """Map an evaluation verdict to the engine's decision-correctness flag."""
    if verdict == EvaluationVerdict.CORRECT:
        return True
    if verdict == EvaluationVerdict.INCORRECT:
        return False
    return None


def calibration_block(evaluations: list[PredictionEvaluation]) -> CalibrationBlock:
    """Calibration summary from evaluations with SUCCESS/FAILURE outcomes."""
    confidences: list[float] = []
    outcomes: list[int] = []
    for evaluation in evaluations:
        label = binary_label(evaluation)
        if label.actual_positive is None:
            continue
        confidences.append(float(evaluation.prediction.confidence))
        outcomes.append(1 if label.actual_positive else 0)
    report = build_calibration_report(
        confidences, [0.0] * len(confidences), correct_predictions=outcomes
    )
    return CalibrationBlock(
        expected_calibration_error=report.expected_calibration_error,
        maximum_calibration_error=report.maximum_calibration_error,
        overconfidence_detected=report.overconfidence_detected,
        overconfident_bins_count=len(report.overconfident_bins),
        bins=[b.to_dict() for b in report.bins],
        total_samples=report.total_samples,
        calibration_quality=report.calibration_quality,
    )


def alignment_counts(evaluations: list[PredictionEvaluation]) -> dict[str, int]:
    """Distribution of prediction/outcome alignments."""
    counts: dict[str, int] = {}
    for evaluation in evaluations:
        key = evaluation.alignment.value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def confidence_vs_outcome(
    evaluations: list[PredictionEvaluation],
) -> dict[str, float | None]:
    """Mean predicted confidence per evaluation verdict bucket."""
    buckets: dict[str, list[float]] = {}
    for evaluation in evaluations:
        buckets.setdefault(evaluation.verdict.value, []).append(
            float(evaluation.prediction.confidence)
        )
    return {
        verdict: (round(fmean(values), 4) if values else None)
        for verdict, values in sorted(buckets.items())
    }


def _metrics_block(metrics: EvaluationMetrics) -> EvaluationMetricsBlock:
    return EvaluationMetricsBlock(**cast(dict[str, Any], metrics.summary()))


def _history_entry(evaluation: CompanyEvaluation) -> EvaluationHistoryEntry:
    decision: str | None = None
    if isinstance(evaluation.prediction, dict):
        raw = evaluation.prediction.get("decision")
        if isinstance(raw, str):
            decision = raw
    return EvaluationHistoryEntry(
        id=evaluation.id,
        snapshot_id=evaluation.snapshot_id,
        outcome_id=evaluation.outcome_id,
        snapshot_created_at=evaluation.snapshot_created_at,
        decision=decision,
        confidence=evaluation.snapshot_confidence,
        verdict=evaluation.verdict,
        alignment=evaluation.alignment,
        outcome_verdict=evaluation.outcome_verdict,
        decision_match=evaluation.decision_match,
        created_at=evaluation.created_at,
    )


@dataclass(frozen=True)
class StoredEvaluation:
    """A stored evaluation row paired with the outcome it was compared to."""

    evaluation: CompanyEvaluation
    outcome: CompanyOutcome

    @property
    def prediction_evaluation(self) -> PredictionEvaluation:
        return prediction_evaluation_from_rows(self.evaluation, self.outcome)


def _scope_company(stmt, user_id: str | None):
    """Scope a statement on ``Company`` to the anonymous or one-user namespace."""
    if user_id is None:
        return stmt.where(Company.user_id.is_(None))
    return stmt.where(Company.user_id == user_id)


class CompanyEvaluationService:
    """Reconcile outcomes against snapshots and aggregate evaluation metrics."""

    def __init__(self, *, store: PostgresCompanyStore | None = None) -> None:
        self._store = store or PostgresCompanyStore()

    async def reconcile(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> int:
        """Create newly applicable evaluations for one company (idempotent).

        An outcome evaluates a snapshot when the outcome is pinned to that
        snapshot, or when the snapshot predates the outcome's observation
        time.  Existing ``(snapshot, outcome)`` pairs are never rewritten.
        Returns the number of evaluations created.
        """
        del user_id  # ownership is asserted by the caller before reconciling
        outcomes = await self._outcomes_for_company(session, company_id)
        if not outcomes:
            return 0
        snapshots = await self._snapshots_for_company(session, company_id)
        if not snapshots:
            return 0

        existing = await self._existing_pairs(session, company_id)
        created = 0
        for outcome in sorted(outcomes, key=lambda row: (row.occurred_at, row.id)):
            outcome_record = outcome_record_from_row(outcome)
            outcome_time = as_utc(outcome.occurred_at)
            for snapshot in snapshots:
                if outcome.snapshot_id is not None and snapshot.id != outcome.snapshot_id:
                    continue
                snapshot_time = as_utc(snapshot.created_at)
                if (
                    outcome_time is not None
                    and snapshot_time is not None
                    and snapshot_time > outcome_time
                ):
                    continue
                pair = (snapshot.id, outcome.id)
                if pair in existing:
                    continue
                prediction = prediction_summary_from_snapshot(snapshot)
                derived = PredictionEvaluation.from_records(
                    DatasetRecord.model_construct(
                        record_id=snapshot.id, prediction=prediction
                    ),
                    outcome_record,
                )
                session.add(
                    CompanyEvaluation(
                        id=compute_evaluation_id(
                            company_id=company_id,
                            snapshot_id=snapshot.id,
                            outcome_id=outcome.id,
                        ),
                        company_id=company_id,
                        snapshot_id=snapshot.id,
                        outcome_id=outcome.id,
                        prediction=prediction.model_dump(mode="json"),
                        verdict=derived.verdict.value,
                        alignment=derived.alignment.value,
                        outcome_verdict=outcome.verdict,
                        decision_match=decision_match_from_verdict(derived.verdict),
                        snapshot_confidence=snapshot.confidence,
                        snapshot_composite_score=snapshot.composite_score,
                        snapshot_created_at=snapshot.created_at,
                    )
                )
                existing.add(pair)
                created += 1

        if created:
            await session.flush()
        return created

    async def reconcile_all(
        self, session: AsyncSession, *, user_id: str | None = None
    ) -> int:
        """Reconcile every company in the caller's scope."""
        created = 0
        for company_id in await self._scoped_company_ids(session, user_id):
            created += await self.reconcile(session, company_id)
        return created

    async def performance(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyPerformanceResponse | None:
        """Aggregate validity-loop performance for one company."""
        company = await self._store.get_company(session, company_id, user_id=user_id)
        if company is None:
            return None
        await self.reconcile(session, company_id)
        stored = await self._load_evaluations(session, company_id=company_id)
        evaluations = [row.prediction_evaluation for row in stored]
        metrics = compute_evaluation_metrics(evaluations)
        history = stored[offset : offset + limit]
        return CompanyPerformanceResponse(
            company_id=company_id,
            outcome_count=await self._count_outcomes(session, company_id),
            evaluation_count=len(stored),
            metrics=_metrics_block(metrics),
            calibration=calibration_block(evaluations),
            alignment=alignment_counts(evaluations),
            confidence_vs_outcome=confidence_vs_outcome(evaluations),
            history=[_history_entry(row.evaluation) for row in history],
            generated_at=datetime.now(UTC),
        )

    async def calibration(
        self, session: AsyncSession, *, user_id: str | None = None
    ) -> EvaluationCalibrationResponse:
        """Calibration and alignment across every company in scope."""
        company_ids = await self._scoped_company_ids(session, user_id)
        for company_id in company_ids:
            await self.reconcile(session, company_id)
        stored = await self._load_evaluations(session)
        evaluations = [row.prediction_evaluation for row in stored]
        metrics = compute_evaluation_metrics(evaluations)
        return EvaluationCalibrationResponse(
            company_count=len(company_ids),
            evaluation_count=len(stored),
            scoreable_count=metrics.scoreable,
            calibration=calibration_block(evaluations),
            alignment=alignment_counts(evaluations),
            confidence_vs_outcome=confidence_vs_outcome(evaluations),
            generated_at=datetime.now(UTC),
        )

    async def summary(
        self, session: AsyncSession, *, user_id: str | None = None
    ) -> EvaluationSummaryResponse:
        """Global summary plus per-company breakdown for the caller's scope."""
        company_ids = await self._scoped_company_ids(session, user_id)
        for company_id in company_ids:
            await self.reconcile(session, company_id)
        stored = await self._load_evaluations(session)
        evaluations = [row.prediction_evaluation for row in stored]
        metrics = compute_evaluation_metrics(evaluations)

        grouped: dict[str, list[StoredEvaluation]] = {}
        for row in stored:
            grouped.setdefault(row.evaluation.company_id, []).append(row)

        by_company: list[CompanyPerformanceEntry] = []
        outcome_total = 0
        for company_id in sorted(company_ids):
            rows = grouped.get(company_id, [])
            company_metrics = compute_evaluation_metrics(
                [row.prediction_evaluation for row in rows]
            )
            outcome_count = await self._count_outcomes(session, company_id)
            outcome_total += outcome_count
            by_company.append(
                CompanyPerformanceEntry(
                    company_id=company_id,
                    outcome_count=outcome_count,
                    evaluation_count=len(rows),
                    scoreable=company_metrics.scoreable,
                    accuracy=company_metrics.accuracy,
                    precision=company_metrics.precision,
                    recall=company_metrics.recall,
                    f1=company_metrics.f1,
                    coverage=company_metrics.coverage,
                )
            )

        return EvaluationSummaryResponse(
            company_count=len(company_ids),
            outcome_count=outcome_total,
            evaluation_count=len(stored),
            metrics=_metrics_block(metrics),
            calibration=calibration_block(evaluations),
            alignment=alignment_counts(evaluations),
            confidence_vs_outcome=confidence_vs_outcome(evaluations),
            by_company=by_company,
            generated_at=datetime.now(UTC),
        )

    async def _snapshots_for_company(
        self, session: AsyncSession, company_id: str
    ) -> list[CompanySnapshot]:
        stmt = (
            select(CompanySnapshot)
            .where(CompanySnapshot.company_id == company_id)
            .order_by(CompanySnapshot.created_at, CompanySnapshot.id)
        )
        return list((await session.execute(stmt)).scalars().all())

    async def _outcomes_for_company(
        self, session: AsyncSession, company_id: str
    ) -> list[CompanyOutcome]:
        stmt = (
            select(CompanyOutcome)
            .where(CompanyOutcome.company_id == company_id)
            .order_by(CompanyOutcome.occurred_at, CompanyOutcome.id)
        )
        return list((await session.execute(stmt)).scalars().all())

    async def _existing_pairs(
        self, session: AsyncSession, company_id: str
    ) -> set[tuple[str, str]]:
        stmt = select(
            CompanyEvaluation.snapshot_id, CompanyEvaluation.outcome_id
        ).where(CompanyEvaluation.company_id == company_id)
        rows = (await session.execute(stmt)).all()
        return {(snapshot_id, outcome_id) for snapshot_id, outcome_id in rows}

    async def _load_evaluations(
        self,
        session: AsyncSession,
        *,
        company_id: str | None = None,
    ) -> list[StoredEvaluation]:
        stmt = (
            select(CompanyEvaluation, CompanyOutcome)
            .join(
                CompanyOutcome,
                CompanyOutcome.id == CompanyEvaluation.outcome_id,
            )
            .join(Company, Company.company_id == CompanyEvaluation.company_id)
        )
        if company_id is not None:
            stmt = stmt.where(CompanyEvaluation.company_id == company_id)
        stmt = stmt.order_by(
            desc(CompanyEvaluation.created_at), desc(CompanyEvaluation.id)
        )
        rows = (await session.execute(stmt)).all()
        return [
            StoredEvaluation(evaluation=evaluation, outcome=outcome)
            for evaluation, outcome in rows
        ]

    async def _scoped_company_ids(
        self, session: AsyncSession, user_id: str | None
    ) -> list[str]:
        ids: list[str] = []
        offset = 0
        page_size = 100
        while True:
            page = await self._store.list_companies(
                session, user_id=user_id, offset=offset, limit=page_size
            )
            ids.extend(company.company_id for company in page.companies)
            if len(page.companies) < page_size:
                break
            offset += page_size
        return ids

    async def _count_outcomes(
        self, session: AsyncSession, company_id: str
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(CompanyOutcome)
            .where(CompanyOutcome.company_id == company_id)
        )
        return int((await session.execute(stmt)).scalar_one())


__all__ = [
    "CompanyEvaluationService",
    "StoredEvaluation",
    "alignment_counts",
    "calibration_block",
    "compute_evaluation_id",
    "confidence_vs_outcome",
    "decision_label",
    "decision_match_from_verdict",
    "prediction_evaluation_from_rows",
    "prediction_summary_from_snapshot",
]
