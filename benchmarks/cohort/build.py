"""Deterministic cohort builder (Milestone V1.4).

Turns a :class:`~benchmarks.cohort.manifest.CohortManifest` into a golden
dataset for :mod:`benchmarks.ground_truth_eval`.  For every manifest record
the builder:

1. loads the referenced offline evidence corpus and **time-scopes it** to the
   record's ``analysis_timestamp`` (the look-ahead guard),
2. runs the current engine once against the scoped evidence to capture a real,
   time-pinned **historical prediction** (never fabricated),
3. records the best-known ``verified_outcome`` verbatim with its provenance
   (``is_example`` when not yet independently verified),
4. stamps the golden entry's temporal anchor (``analysis_timestamp``,
   ``evaluation_horizon_days``) so downstream replay and validation enforce
   temporal integrity.

Records without a documented outcome are *pending ground truth* and excluded
from the dataset.  Output is deterministic: identical manifests produce
byte-identical datasets (modulo the recorded wall-clock fields the platform
already excludes from every hash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.analysis import StartupAnalysisRequest
from benchmarks.cohort.manifest import (
    CohortManifest,
    ManifestOutcome,
    ManifestSourceRecord,
    manifest_hash,
)
from benchmarks.ground_truth_eval.dataset import (
    ValidationReport,
    dataset_hash,
    validate_golden_dataset,
)
from benchmarks.ground_truth_eval.models import (
    GoldenDataset,
    GoldenEntry,
    HistoricalEvidence,
    HistoricalPrediction,
    Provenance,
    VerifiedOutcome,
)
from predictron_engine.dataset.analysis_support import extract_prediction_summary
from predictron_engine.dataset.prediction import (
    PredictionScopeError,
    ScopeStats,
    scoped_evidence,
)


class CohortBuildError(RuntimeError):
    """Raised when a cohort cannot be built from its manifest."""


BUILD_STATUS_BUILT = "built"
BUILD_STATUS_PENDING = "pending_ground_truth"
BUILD_STATUS_FAILED = "failed"


@dataclass
class CompanyBuildResult:
    """Outcome of building one manifest record."""

    company_id: str
    status: str = BUILD_STATUS_FAILED
    scope_stats: ScopeStats | None = None
    prediction: Any | None = None
    entry: GoldenEntry | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "status": self.status,
            "error": self.error,
            "blocked_lookahead": (
                bool(self.scope_stats.blocked_lookahead) if self.scope_stats else False
            ),
            "evidence_scope": (
                self.scope_stats.model_dump(mode="json") if self.scope_stats else None
            ),
        }


@dataclass
class CohortBuild:
    """Result of building a cohort from its manifest."""

    dataset: GoldenDataset
    manifest_hash: str
    company_results: dict[str, CompanyBuildResult] = field(default_factory=dict)
    validation: ValidationReport | None = None

    @property
    def built_company_ids(self) -> list[str]:
        return sorted(
            cid
            for cid, res in self.company_results.items()
            if res.status == BUILD_STATUS_BUILT
        )

    @property
    def pending_company_ids(self) -> list[str]:
        return sorted(
            cid for cid, res in self.company_results.items() if res.status == BUILD_STATUS_PENDING
        )

    @property
    def failed_company_ids(self) -> list[str]:
        return sorted(
            cid for cid, res in self.company_results.items() if res.status == BUILD_STATUS_FAILED
        )

    @property
    def lookahead_blocked_company_ids(self) -> list[str]:
        return sorted(
            cid
            for cid, res in self.company_results.items()
            if res.scope_stats is not None and res.scope_stats.blocked_lookahead
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "dataset_name": self.dataset.dataset_name,
            "benchmark_version": self.dataset.benchmark_version,
            "manifest_hash": self.manifest_hash,
            "dataset_hash": dataset_hash(self.dataset),
            "entries_built": len(self.dataset.entries),
            "built_company_ids": self.built_company_ids,
            "pending_ground_truth": self.pending_company_ids,
            "failed": self.failed_company_ids,
            "lookahead_blocked_company_ids": self.lookahead_blocked_company_ids,
            "company_results": {
                cid: res.to_dict() for cid, res in sorted(self.company_results.items())
            },
            "validation": self.validation.to_dict() if self.validation is not None else None,
        }


def build_cohort_dataset(manifest: CohortManifest, *, engine: Any | None = None) -> CohortBuild:
    """Build a golden dataset from a cohort manifest.

    ``engine`` is injected for tests; defaults to a real
    :class:`predictron_engine.engine.PredictronEngine`.  Never writes to
    disk — the caller persists the dataset (see
    :func:`benchmarks.ground_truth_eval.dataset.dump_golden_dataset`).
    """
    from predictron_engine.engine import ENGINE_VERSION, PredictronEngine

    engine = engine if engine is not None else PredictronEngine()
    engine_version = ENGINE_VERSION
    fingerprint = manifest_hash(manifest)

    entries: list[GoldenEntry] = []
    results: dict[str, CompanyBuildResult] = {}
    for record in sorted(manifest.source_records, key=lambda r: r.company_id):
        result = _build_entry(engine, record, manifest, engine_version, fingerprint)
        results[record.company_id] = result
        if result.status == BUILD_STATUS_BUILT and result.entry is not None:
            entries.append(result.entry)

    dataset = GoldenDataset(
        dataset_name=manifest.dataset_name,
        benchmark_version=manifest.benchmark_version,
        created_at=manifest.created_at,
        entries=entries,
    )
    validation = validate_golden_dataset(dataset)
    return CohortBuild(
        dataset=dataset,
        manifest_hash=fingerprint,
        company_results=results,
        validation=validation,
    )


def _build_entry(
    engine: Any,
    record: ManifestSourceRecord,
    manifest: CohortManifest,
    engine_version: str,
    manifest_fingerprint: str,
) -> CompanyBuildResult:
    if record.outcome is None:
        return CompanyBuildResult(
            company_id=record.company_id,
            status=BUILD_STATUS_PENDING,
            error="no outcome recorded",
        )
    try:
        bundle, stats = scoped_evidence(record.evidence_corpus, as_of=record.analysis_timestamp)
    except PredictionScopeError as exc:
        return CompanyBuildResult(company_id=record.company_id, error=str(exc))
    try:
        report = engine.analyze(_request_from_record(record), evidence_bundle=bundle)
    except Exception as exc:  # noqa: BLE001 - surface per-company build failures
        return CompanyBuildResult(
            company_id=record.company_id, error=f"engine analyze failed: {exc}"
        )

    prediction = extract_prediction_summary(report)
    entry = GoldenEntry(
        company_id=record.company_id,
        company_name=record.company_name,
        request=_request_dict(record),
        sector=record.sector,
        country=record.country,
        stage=record.stage,
        historical_prediction=_historical_prediction(prediction, report),
        historical_evidence=HistoricalEvidence(
            corpus_name=record.evidence_corpus,
            sources=_record_source_labels(record),
        ),
        verified_outcome=_verified_outcome(record.outcome),
        provenance=_provenance(record),
        analysis_timestamp=record.analysis_timestamp,
        evaluation_horizon_days=(
            record.evaluation_horizon_days
            if record.evaluation_horizon_days is not None
            else manifest.evaluation_horizon_days
        ),
        metadata={
            "builder": "scripts/build_cohort.py",
            "manifest_hash": manifest_fingerprint,
            "evidence_scope": stats.model_dump(mode="json"),
        },
    )
    return CompanyBuildResult(
        company_id=record.company_id,
        status=BUILD_STATUS_BUILT,
        scope_stats=stats,
        prediction=prediction,
        entry=entry,
    )


def _request_dict(record: ManifestSourceRecord) -> dict[str, Any]:
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
    return request


def _request_from_record(record: ManifestSourceRecord) -> StartupAnalysisRequest:
    return StartupAnalysisRequest(**_request_dict(record))


def _historical_prediction(prediction: Any, report: Any) -> HistoricalPrediction:
    """Map the pinned engine run onto a frozen HistoricalPrediction."""
    overall_score = float(prediction.composite_score or 0.0)
    recommendation_categories = sorted({str(r.category) for r in report.recommendations})
    return HistoricalPrediction(
        overall_score=round(overall_score, 4),
        overall_confidence=round(float(prediction.confidence), 4),
        decision=prediction.decision.value,
        composite_score=round(overall_score, 4),
        recommendation_categories=recommendation_categories,
        dimension_scores={k: float(v) for k, v in prediction.dimension_scores.items()},
    )


def _verified_outcome(outcome: ManifestOutcome) -> VerifiedOutcome:
    return VerifiedOutcome(
        status=outcome.status,
        verification_date=outcome.verification_date,
        outcome_events=list(outcome.outcome_events),
        exit_value_usd=outcome.exit_value_usd,
        sources=list(outcome.sources),
    )


def _provenance(record: ManifestSourceRecord) -> Provenance:
    outcome = record.outcome
    notes = outcome.notes if outcome is not None else ""
    if outcome is not None and not outcome.verified and not notes:
        notes = "Outcome recorded verbatim but not yet independently verified (is_example)."
    return Provenance(
        sources=_record_source_labels(record),
        collector="scripts/build_cohort.py",
        verified_by=outcome.verified_by if outcome is not None and outcome.verified else None,
        is_example=outcome is None or not outcome.verified,
        notes=notes,
    )


def _record_source_labels(record: ManifestSourceRecord) -> list[str]:
    labels = list(record.metadata.get("sources") or []) if isinstance(record.metadata, dict) else []
    labels.append(record.evidence_corpus)
    return labels


__all__ = [
    "BUILD_STATUS_BUILT",
    "BUILD_STATUS_PENDING",
    "BUILD_STATUS_FAILED",
    "CompanyBuildResult",
    "CohortBuild",
    "CohortBuildError",
    "build_cohort_dataset",
]
