"""LearningService — the Phase 7 continuous-learning service layer.

Read-only projections over the live evaluated prediction ledger plus
deterministic recording of learning snapshots.  All analytics are delegated
to the pure engine builders in ``predictron_engine.learning`` — nothing is
recomputed here.

* attribute resolution — the categorical dimensions (sector, stage, country,
  technology, business model, founder) are resolved deterministically from
  the frozen analysis report ``features`` and the stored evaluation
  prediction, with a documented ``"unknown"`` fallback.  ``features`` were
  captured into ``AnalysisReport.full_report`` at report time by the
  persistence layer, so resolution never invents a category;
* snapshot recording — an operator-facing idempotent record of a
  :class:`LearningSnapshot` into the append-only learning tables so the API
  can consume recorded reports without recomputing them.

``scope`` handling mirrors the monitoring layer: API views are scoped to the
requesting user/namespace, while recording and report views are
repository-wide.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import desc, select

from app.models.company import Company, CompanySnapshot
from app.models.company_outcome import CompanyEvaluation, CompanyOutcome
from app.models.learning import (
    LearningObservationRecord,
    LearningPatternRecord,
    LearningReportRecord,
    LearningSnapshotRecord,
)
from app.schemas.learning import (
    LearningBiasResponse,
    LearningConfidenceResponse,
    LearningKnowledgeResponse,
    LearningObservationResponse,
    LearningPatternResponse,
    LearningRecommendationsResponse,
    LearningReportDetailResponse,
    LearningReportsResponse,
    LearningSummaryResponse,
)
from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.learning.engine import build_learning_snapshot
from predictron_engine.learning.models import (
    ATTR_UNKNOWN,
    LearningDimension,
    LearningPeriodKind,
    LearningSample,
    LearningSnapshot,
)

_ENGINE_VERSION = "7.0.0"

#: Optional feature keys consulted per dimension, highest priority first.
_FEATURE_KEYS: dict[str, tuple[str, ...]] = {
    "sector": ("industry", "sub_industry", "target_market"),
    "stage": ("funding_stage",),
    "country": ("headquarters_region", "geography"),
    "technology": ("primary_technology_domain", "secondary_technology_domain"),
    "business_model": ("business_model", "revenue_model", "monetization_strategy"),
    "founder": ("founder_team_type",),
}

#: Dimension-score keys consulted as a last resort before ``"unknown"``.
_DIMENSION_FALLBACKS: dict[str, tuple[str, ...]] = {
    "sector": ("market_opportunity",),
    "stage": ("traction_signals",),
    "founder": ("founder_quality",),
    "technology": ("product_strength",),
    "country": (),
    "business_model": ("business_model_viability",),
}

_LEARNING_DIMENSIONS = tuple(
    item.value for item in LearningDimension
)


def normalize(value: str | None) -> str:
    """Normalize one resolved attribute value to its canonical key form."""
    if value is None:
        return ATTR_UNKNOWN
    cleaned = value.strip().lower()
    return cleaned if cleaned else ATTR_UNKNOWN


def resolve_attributes(features: dict[str, Any] | None) -> dict[str, str]:
    """Deterministically resolve the six learning dimensions.

    Resolution precedence (documented):
    1. the frozen report ``features`` map (industry, funding stage,
       geography, technology domain, business model, founder team type);
    2. the stored prediction ``dimension_scores`` bucket whenever the
       feature set is unknown (sector → market opportunity, stage →
       traction, founder → founder quality, technology → product strength);
    3. the canonical ``"unknown"`` value — attribute resolution never
       fabricates a category.
    """
    resolved: dict[str, str] = {}
    candidates = features or {}
    for dimension in _LEARNING_DIMENSIONS:
        for key in _FEATURE_KEYS[dimension]:
            candidate = candidates.get(key)
            if isinstance(candidate, str) and candidate.strip():
                resolved[dimension] = normalize(candidate)
                break
        else:
            resolved[dimension] = ATTR_UNKNOWN
    return resolved


def _fallback_dimension_score(
    dimension_scores: dict[str, float], dimension: str
) -> str:
    for key in _DIMENSION_FALLBACKS.get(dimension, ()):
        value = dimension_scores.get(key)
        if value is not None:
            return f"{key}::{value:.2f}"
    return ATTR_UNKNOWN


def _dimension_scores_from_prediction(prediction: dict[str, Any]) -> dict[str, float]:
    raw = prediction.get("dimension_scores")
    if not isinstance(raw, dict):
        return {}
    scores: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            scores[str(key)] = float(value)
    return scores


class LearningService:
    """Derive learning intelligence from the live evaluated ledger."""

    # ------------------------------------------------------------------
    # Live projections
    # ------------------------------------------------------------------

    async def summary(
        self,
        session,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningSummaryResponse:
        """Live learning summary over the caller's evaluated population."""
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningSummaryResponse(
            scope=snapshot.scope,
            period_kind=snapshot.period_kind,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            engine_version=snapshot.engine_version,
            counts=snapshot.counts,
            metrics=snapshot.metrics,
            digest=snapshot.digest,
            calibration=snapshot.calibration,
            confidence=snapshot.confidence,
            distributions=snapshot.distributions,
            knowledge=snapshot.knowledge,
            patterns=snapshot.patterns,
            observations=snapshot.observations,
            recommendations=snapshot.recommendations,
        )

    async def patterns(
        self,
        session,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningPatternResponse:
        """Per-dimension cohort patterns over the evaluated population."""
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningPatternResponse(
            scope=snapshot.scope,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            patterns=snapshot.patterns,
        )

    async def observations(
        self,
        session,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningObservationResponse:
        """Canonical observations about the evaluated population."""
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningObservationResponse(
            scope=snapshot.scope,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            observations=snapshot.observations,
        )

    async def dimension_intelligence(
        self,
        session,
        dimension: str,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningKnowledgeResponse:
        """Per-dimension knowledge view (sector/stage/country/...)."""
        try:
            dim = LearningDimension(dimension)
        except ValueError as exc:
            raise ValueError(
                f"unknown learning dimension {dimension!r}; expected one of "
                f"{sorted(item.value for item in LearningDimension)}"
            ) from exc
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningKnowledgeResponse(
            dimension=dim.value,
            scope=snapshot.scope,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            entries=snapshot.knowledge.get(dim.value, []),
        )

    async def confidence(
        self,
        session,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningConfidenceResponse:
        """Population confidence statistics and calibration digest."""
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningConfidenceResponse(
            scope=snapshot.scope,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            confidence=snapshot.confidence,
            calibration=snapshot.calibration,
        )

    async def bias(
        self,
        session,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningBiasResponse:
        """Confidence-bias view plus the population metric surface."""
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningBiasResponse(
            scope=snapshot.scope,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            metrics=snapshot.metrics,
            knowledge=snapshot.knowledge,
        )

    async def recommendations(
        self,
        session,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        scope_all: bool = False,
    ) -> LearningRecommendationsResponse:
        """Deterministic, rule-based recommendations for the population."""
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(
            session, user_id=user_id, scope_all=scope_all
        )
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=LearningPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return LearningRecommendationsResponse(
            scope=snapshot.scope,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            recommendations=snapshot.recommendations,
        )

    async def snapshot(
        self,
        session,
        *,
        as_of: datetime | None = None,
        period_kind: LearningPeriodKind = LearningPeriodKind.DAILY,
    ) -> LearningSnapshot:
        """Compute and persistently record the learning snapshot.

        Repository-wide (ignores user scoping) and idempotent: the snapshot
        id is a deterministic digest of ``(scope, period_kind, anchor_date)``,
        so re-recording the same anchor overwrites nothing and inserts
        nothing.
        """
        as_of_utc = as_of if as_of is not None else datetime.now(UTC)
        evidence = await self._load_evidence(session, user_id=None, scope_all=True)
        snapshot = build_learning_snapshot(
            evidence.samples,
            anchor_date=as_of_utc.date(),
            period_kind=period_kind,
            scope="repository",
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        await self._record_snapshot(session, snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Recorded history
    # ------------------------------------------------------------------

    async def reports(
        self,
        session,
        *,
        period_kind: LearningPeriodKind = LearningPeriodKind.DAILY,
    ) -> LearningReportsResponse:
        """Header summaries of recorded learning snapshots (latest first)."""
        stmt = (
            select(LearningSnapshotRecord)
            .where(LearningSnapshotRecord.period_kind == period_kind.value)
            .order_by(
                desc(LearningSnapshotRecord.anchor_date),
                desc(LearningSnapshotRecord.created_at),
            )
        )
        rows = list((await session.execute(stmt)).scalars().all())
        return LearningReportsResponse(
            period_kind=period_kind,
            reports=[
                {
                    "snapshot_id": row.id,
                    "scope": row.scope,
                    "period_kind": row.period_kind,
                    "anchor_date": row.anchor_date.date().isoformat(),
                    "engine_version": row.engine_version,
                    "content_hash": row.content_hash,
                    "recorded_at": row.recorded_at.isoformat(),
                }
                for row in rows
            ],
        )

    async def report_detail(
        self,
        session,
        snapshot_id: str,
    ) -> LearningReportDetailResponse | None:
        """Full recorded report payload for one learning snapshot id."""
        stmt = (
            select(LearningReportRecord)
            .where(LearningReportRecord.snapshot_id == snapshot_id)
            .order_by(desc(LearningReportRecord.created_at))
        )
        row = (await session.execute(stmt)).scalars().first()
        if row is None:
            return None
        return LearningReportDetailResponse(
            snapshot_id=row.snapshot_id,
            report_id=row.id,
            content_hash=row.content_hash,
            recorded_at=row.recorded_at,
            payload=row.payload,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _load_evidence(
        self,
        session,
        *,
        user_id: str | None,
        scope_all: bool = False,
    ) -> Evidence:
        """Evaluations + resolved attribute profiles in scope."""
        stmt = (
            select(CompanyEvaluation, CompanyOutcome, Company)
            .join(
                CompanyOutcome,
                CompanyOutcome.id == CompanyEvaluation.outcome_id,
            )
            .join(Company, Company.company_id == CompanyEvaluation.company_id)
            .order_by(desc(CompanyEvaluation.created_at), desc(CompanyEvaluation.id))
        )
        if not scope_all:
            if user_id is None:
                stmt = stmt.where(Company.user_id.is_(None))
            else:
                stmt = stmt.where(Company.user_id == user_id)
        rows = (await session.execute(stmt)).all()

        snapshot_ids = [
            evaluation.snapshot_id for evaluation, _outcome, _company in rows
        ]
        snapshot_features = await self._resolve_attributes_for_snapshots(
            session, snapshot_ids
        )

        samples: list[LearningSample] = []
        for evaluation, _outcome, _company in rows:
            prediction = evaluation.prediction or {}
            confidence = float(_safe_float(
                prediction.get("confidence"), evaluation.snapshot_confidence
            ) or 0.0)
            resolved = snapshot_features.get(evaluation.snapshot_id, {})
            scores = _dimension_scores_from_prediction(prediction)
            for dimension in _LEARNING_DIMENSIONS:
                if resolved.get(dimension) in (None, "", ATTR_UNKNOWN):
                    resolved = {
                        **resolved,
                        dimension: _fallback_dimension_score(scores, dimension),
                    }
            verdict = self._verdict(evaluation.verdict)
            samples.append(
                LearningSample(
                    evaluation_id=evaluation.id,
                    confidence=confidence,
                    verdict=verdict or EvaluationVerdict.INCONCLUSIVE,
                    actual_positive=_actual_positive(verdict),
                    resolved=resolved,
                )
            )
        return Evidence(samples=samples)

    async def _resolve_attributes_for_snapshots(
        self, session, snapshot_ids: list[str]
    ) -> dict[str, dict[str, str]]:
        """Resolve attributes for one batch of snapshot ids.

        Snapshot ids are company-snapshot ids; attributes come from the
        frozen analysis report payload linked through ``analysis_id``.  When
        no report exists for a snapshot the resolution stays ``"unknown"``.
        """
        features_by_snapshot: dict[str, dict[str, Any]] = {}
        if not snapshot_ids:
            return {}
        from app.models.analysis import AnalysisReport

        summaries = (
            await session.execute(
                select(CompanySnapshot.id, CompanySnapshot.analysis_id).where(
                    CompanySnapshot.id.in_(snapshot_ids)
                )
            )
        ).all()
        analysis_ids = [analysis_id for _snapshot_id, analysis_id in summaries]
        if analysis_ids:
            reports = (
                await session.execute(
                    select(AnalysisReport.id, AnalysisReport.full_report).where(
                        AnalysisReport.id.in_(analysis_ids)
                    )
                )
            ).all()
            full_report_by_id = {report_id: full_report for report_id, full_report in reports}
            for snapshot_id, analysis_id in summaries:
                full_report = full_report_by_id.get(analysis_id)
                features = (
                    full_report.get("features") if isinstance(full_report, dict) else None
                )
                if isinstance(features, dict):
                    features_by_snapshot[snapshot_id] = features
        return {
            snapshot_id: resolve_attributes(features_by_snapshot.get(snapshot_id))
            for snapshot_id in snapshot_ids
        }

    @staticmethod
    def _verdict(value: str | None) -> EvaluationVerdict | None:
        if value is None:
            return None
        try:
            return EvaluationVerdict(value)
        except ValueError:
            return None

    async def _record_snapshot(
        self, session, snapshot: LearningSnapshot
    ) -> None:
        """Append-only, idempotent record of a computed learning snapshot."""
        existing = await session.get(LearningSnapshotRecord, snapshot.snapshot_id)
        if existing is not None:
            return
        session.add(
            LearningSnapshotRecord(
                id=snapshot.snapshot_id,
                scope=snapshot.scope,
                period_kind=snapshot.period_kind.value,
                anchor_date=_as_datetime(snapshot.anchor_date),
                engine_version=snapshot.engine_version,
                schema_version=snapshot.schema_version,
                content_hash=snapshot.content_hash,
                counts=snapshot.counts,
                metrics=dict(snapshot.metrics),
                distributions=snapshot.distributions,
                meta={"schema_version": snapshot.schema_version},
                recorded_at=snapshot.recorded_at,
            )
        )
        for index, observation in enumerate(snapshot.observations):
            session.add(
                LearningObservationRecord(
                    id=_observation_id(snapshot.snapshot_id, index),
                    snapshot_id=snapshot.snapshot_id,
                    category=observation.category.value,
                    dimension=observation.dimension.value,
                    value=observation.value,
                    metric=observation.metric,
                    metric_value=observation.metric_value,
                    delta=observation.delta,
                    direction=observation.direction.value,
                    baseline=observation.baseline,
                    sample_size=observation.sample_size,
                    summary=observation.summary,
                )
            )
        for index, pattern in enumerate(snapshot.patterns):
            session.add(
                LearningPatternRecord(
                    id=_pattern_id(snapshot.snapshot_id, index),
                    snapshot_id=snapshot.snapshot_id,
                    dimension=pattern.dimension.value,
                    value=pattern.value,
                    samples=pattern.samples,
                    scoreable=pattern.scoreable,
                    accuracy=pattern.accuracy,
                    precision=pattern.precision,
                    confidence=pattern.confidence,
                    confidence_bias=pattern.confidence_bias,
                    false_positive_rate=pattern.false_positive_rate,
                    false_negative_rate=pattern.false_negative_rate,
                    recommendation=pattern.recommendation,
                )
            )
        session.add(
            LearningReportRecord(
                id=_report_id(snapshot.snapshot_id),
                snapshot_id=snapshot.snapshot_id,
                payload=snapshot.model_dump(mode="json"),
                content_hash=snapshot.content_hash,
                recorded_at=snapshot.recorded_at,
            )
        )
        await session.flush()


class Evidence:
    """Evaluated samples plus the learning-relevant ledger counts."""

    def __init__(self, *, samples: list[LearningSample], forecast_count: int = 0) -> None:
        self.samples = samples
        self.forecast_count = forecast_count


def _denormalize_scope(user_id: str | None) -> str:
    return f"user:{user_id}" if user_id is not None else "anonymous"


def _as_datetime(value: date) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _safe_float(value: object, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return fallback


def _actual_positive(verdict: EvaluationVerdict | None) -> bool | None:
    if verdict == EvaluationVerdict.CORRECT:
        return True
    if verdict == EvaluationVerdict.INCORRECT:
        return False
    return None


def _observation_id(snapshot_id: str, index: int) -> str:
    material = f"observation\x00{snapshot_id}\x00{index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _pattern_id(snapshot_id: str, index: int) -> str:
    material = f"pattern\x00{snapshot_id}\x00{index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _report_id(snapshot_id: str) -> str:
    material = f"report\x00{snapshot_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


__all__ = [
    "Evidence",
    "LearningService",
    "normalize",
    "resolve_attributes",
]
