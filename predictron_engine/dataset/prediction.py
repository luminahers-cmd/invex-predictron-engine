"""Time-scoped prediction recording (Milestone V1.4).

Deterministic machinery that pins an engine analysis to an explicit
``analysis_timestamp`` and enforces a **look-ahead guard**: evidence whose
retrieval timestamp (``EvidenceDocument.fetched_at`` /
``EvidenceSource.fetched_at``) is *after* the analysis timestamp is blocked
from the engine run, so a recorded prediction can never be informed by
documents that were not yet knowable at analysis time.

The module is purely additive and deterministic:

* :func:`time_scope_bundle` — filter an :class:`EvidenceBundle` to evidence
  knowable at ``as_of``; identical inputs always produce identical output
  and identical :class:`ScopeStats`.
* :func:`scoped_evidence` — load an offline corpus and time-scope it.
* :func:`record_pinned_prediction` — run the engine once against the
  time-scoped evidence and capture an immutable :class:`TimeScopedPrediction`
  that records exactly what the engine was allowed to see and when.

Nothing here invents outcomes, mutates engine behavior, or touches the
network: corpus loading goes through the offline replay layer
(:mod:`predictron_engine.evidence.replay`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.dataset.analysis_support import extract_prediction_summary
from predictron_engine.dataset.models import PredictionSummary


class PredictionScopeError(RuntimeError):
    """Raised when a time-scoped prediction cannot be produced safely."""


class ScopeStats(BaseModel):
    """Bookkeeping for one evidence time-scope operation.

    ``retained_*`` counts evidence knowable at ``as_of``; ``dropped_*``
    counts evidence that postdates it and was therefore blocked by the
    look-ahead guard.  Everything here is derived, never estimated.
    """

    as_of: datetime = Field(..., description="Analysis timestamp (UTC)")
    retained_documents: int = Field(default=0, ge=0)
    dropped_documents: int = Field(default=0, ge=0, description="Blocked look-ahead documents")
    retained_sources: int = Field(default=0, ge=0)
    dropped_sources: int = Field(default=0, ge=0, description="Blocked look-ahead source attempts")
    latest_retained_fetched_at: datetime | None = Field(default=None)
    earliest_dropped_fetched_at: datetime | None = Field(default=None)

    @property
    def blocked_lookahead(self) -> bool:
        """True when any evidence was blocked for postdating the analysis."""
        return self.dropped_documents > 0 or self.dropped_sources > 0


class TimeScopedPrediction(BaseModel):
    """Immutable artefact of one time-pinned engine analysis.

    Bundles the engine's fresh prediction with the temporal anchor and the
    evidence-scope bookkeeping, so a later evaluation can attest exactly
    which evidence was knowable when the prediction was recorded.
    """

    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str = Field(..., description="Stable company identifier")
    company_name: str = Field(..., description="Company name as submitted")
    analysis_timestamp: datetime = Field(
        ..., description="UTC timestamp the prediction is pinned to"
    )
    engine_version: str = Field(..., description="Engine version that produced the prediction")
    evidence_reference: str | None = Field(
        default=None, description="Offline corpus reference used (if any)"
    )
    prediction: PredictionSummary = Field(
        ..., description="Engine prediction recorded under the time scope"
    )
    scope_stats: ScopeStats | None = Field(
        default=None, description="Evidence-scope bookkeeping for this prediction"
    )
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this artefact was recorded (informational only)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


def _as_utc(value: datetime) -> datetime:
    """Normalize naive datetimes to UTC for comparison (never mutates input)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def time_scope_bundle(
    bundle: Any,
    *,
    as_of: datetime,
) -> tuple[Any, ScopeStats]:
    """Return ``(scoped_bundle, stats)`` with evidence knowable at ``as_of``.

    The look-ahead guard drops every document/source whose ``fetched_at`` is
    strictly after ``as_of``.  Ties at ``fetched_at == as_of`` are retained.
    When nothing is dropped the input bundle is returned verbatim so corpus
    fidelity is byte-identical; when evidence is dropped the aggregate
    enrichment diagnostics (``intelligence`` / ``trust_summary``) are cleared
    because they described the unfiltered document set.

    Output ordering is deterministic: retained documents/sources are sorted
    by ``id``.
    """
    as_of_utc = _as_utc(as_of)
    if as_of_utc.tzinfo is None:  # pragma: no cover - _as_utc attaches UTC
        raise PredictionScopeError("analysis timestamp must be timezone-aware (UTC)")

    docs = list(getattr(bundle, "documents", []) or [])
    sources = list(getattr(bundle, "sources", []) or [])
    retained_documents = sorted(
        (d for d in docs if _as_utc(d.fetched_at) <= as_of_utc), key=lambda d: d.id
    )
    retained_sources = sorted(
        (s for s in sources if _as_utc(s.fetched_at) <= as_of_utc), key=lambda s: str(s.original_url)
    )
    dropped_documents = [d for d in docs if _as_utc(d.fetched_at) > as_of_utc]
    dropped_sources = [s for s in sources if _as_utc(s.fetched_at) > as_of_utc]

    stats = ScopeStats(
        as_of=as_of_utc,
        retained_documents=len(retained_documents),
        dropped_documents=len(dropped_documents),
        retained_sources=len(retained_sources),
        dropped_sources=len(dropped_sources),
        latest_retained_fetched_at=(
            max((_as_utc(d.fetched_at) for d in retained_documents), default=None)
            if retained_documents
            else None
        ),
        earliest_dropped_fetched_at=(
            min((_as_utc(d.fetched_at) for d in dropped_documents), default=None)
            if dropped_documents
            else None
        ),
    )

    if dropped_documents or dropped_sources:
        scoped = bundle.model_copy(
            update={
                "documents": retained_documents,
                "sources": retained_sources,
                "intelligence": None,
                "trust_summary": None,
            }
        )
    else:
        scoped = bundle
    return scoped, stats


def scoped_evidence(
    reference: str,
    *,
    as_of: datetime,
) -> tuple[Any, ScopeStats]:
    """Load an offline corpus and time-scope it to ``as_of``.

    ``reference`` is resolved via the offline replay layer
    (:func:`~predictron_engine.evidence.replay.dataset.resolve_dataset_path`).
    """
    from predictron_engine.evidence.replay.dataset import (
        EvidenceCorpusError,
        load_corpus,
        rebuild_bundle,
        resolve_dataset_path,
    )

    try:
        document = load_corpus(resolve_dataset_path(reference))
        bundle = rebuild_bundle(document)
    except EvidenceCorpusError as exc:
        raise PredictionScopeError(f"evidence corpus unreachable for {reference!r}: {exc}") from exc
    return time_scope_bundle(bundle, as_of=as_of)


def record_pinned_prediction(
    engine: Any,
    *,
    request: StartupAnalysisRequest,
    analysis_timestamp: datetime,
    company_id: str,
    company_name: str,
    evidence_bundle: Any | None = None,
    evidence_reference: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TimeScopedPrediction:
    """Record the engine's prediction pinned to ``analysis_timestamp``.

    At least one of ``evidence_bundle`` / ``evidence_reference`` must be
    supplied; the supplied evidence is time-scoped before the engine runs so
    post-analysis documents can never influence the recorded prediction.
    """
    if analysis_timestamp.tzinfo is None:
        raise PredictionScopeError("analysis_timestamp must be timezone-aware (UTC)")

    if evidence_bundle is not None:
        scoped, stats = time_scope_bundle(evidence_bundle, as_of=analysis_timestamp)
        reference = evidence_reference
    elif evidence_reference is not None:
        scoped, stats = scoped_evidence(evidence_reference, as_of=analysis_timestamp)
        reference = evidence_reference
    else:
        raise PredictionScopeError(
            "record_pinned_prediction requires evidence_bundle or evidence_reference"
        )

    report = engine.analyze(request, evidence_bundle=scoped)
    prediction = extract_prediction_summary(report)

    return TimeScopedPrediction(
        company_id=company_id,
        company_name=company_name,
        analysis_timestamp=analysis_timestamp,
        engine_version=_engine_version(report),
        evidence_reference=reference,
        prediction=prediction,
        scope_stats=stats,
        metadata=metadata or {},
    )


def _engine_version(report: Any) -> str:
    metadata = getattr(report, "analysis_metadata", None)
    if metadata is not None:
        version = getattr(metadata, "engine_version", None)
        if version:
            return str(version)
    return "unknown"


__all__ = [
    "PredictionScopeError",
    "ScopeStats",
    "TimeScopedPrediction",
    "scoped_evidence",
    "record_pinned_prediction",
    "time_scope_bundle",
]