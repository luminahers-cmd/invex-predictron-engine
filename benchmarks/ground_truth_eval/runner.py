"""Deterministic benchmark runner (Project E5).

Executes the *current* engine over a golden dataset and stores the fresh
predictions — scores, confidence, decisions, recommendations — in a
:class:`BenchmarkRun` without ever modifying the dataset's historical
records.

Determinism comes from offline evidence replay: every entry references a
committed evidence corpus (``benchmarks/offline_evidence/*.json``) that is
rebuilt into an ``EvidenceBundle`` and injected into
:meth:`PredictronEngine.analyze`, which skips website collection entirely.
No network, no randomness, no shared mutable state.  An entry whose corpus
is missing is recorded as a *failed* entry with a clear error — the runner
never falls back to live collection.

The runner reuses the production ``engine.analyze`` pipeline (identical
code path to production) and the existing offline-corpus tooling
(:func:`predictron_engine.evidence.replay.dataset.load_corpus` /
``rebuild_bundle``), so it adds no duplicate pipeline logic.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.analysis import StartupAnalysisRequest
from benchmarks.ground_truth_eval.dataset import dataset_hash
from benchmarks.ground_truth_eval.models import GoldenDataset, GoldenEntry
from predictron_engine.dataset.prediction import ScopeStats, time_scope_bundle
from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.evidence.replay.dataset import (
    EvidenceCorpusError,
    load_corpus,
    rebuild_bundle,
    resolve_dataset_path,
)
from predictron_engine.models.report import Report


class RunnerError(RuntimeError):
    """Raised when a benchmark run cannot be produced."""


def build_entry_bundle(entry: GoldenEntry) -> EvidenceBundle:
    """Rebuild the deterministic evidence bundle for a golden entry.

    When the entry carries an ``analysis_timestamp`` the bundle is time-scoped
    first so the engine run enforces the look-ahead guard (evidence fetched
    after the analysis timestamp is blocked).  Entries without an anchor are
    replayed verbatim, preserving legacy behavior.

    Raises :class:`RunnerError` when the entry has no evidence reference or
    the referenced corpus cannot be loaded, so callers can record the entry
    as failed instead of touching the network.
    """
    bundle = _load_corpus_bundle(entry)
    if entry.analysis_timestamp is not None:
        scoped, _unused = time_scope_bundle(bundle, as_of=entry.analysis_timestamp)
        assert isinstance(scoped, EvidenceBundle)
        return scoped
    return bundle


def entry_evidence_scope(entry: GoldenEntry) -> ScopeStats | None:
    """Return the look-ahead-guard scope stats for a pinned entry.

    ``None`` when the entry has no ``analysis_timestamp`` (no guard applied).
    """
    if entry.analysis_timestamp is None:
        return None
    bundle = _load_corpus_bundle(entry)
    _, stats = time_scope_bundle(bundle, as_of=entry.analysis_timestamp)
    return stats


def _load_corpus_bundle(entry: GoldenEntry) -> EvidenceBundle:
    reference = entry.historical_evidence.reference
    if reference is None:
        raise RunnerError(f"company {entry.company_id!r} has no evidence corpus reference")
    try:
        path = resolve_dataset_path(reference)
        document = load_corpus(path)
        return rebuild_bundle(document)
    except EvidenceCorpusError as exc:
        raise RunnerError(
            f"company {entry.company_id!r} evidence corpus unavailable: {exc}"
        ) from exc


def build_request(entry: GoldenEntry) -> StartupAnalysisRequest:
    """Build a ``StartupAnalysisRequest`` from a golden entry's request dict."""
    raw = entry.request
    kwargs: dict[str, Any] = {}
    for key in ("startup_name", "website", "description"):
        value = raw.get(key)
        if value is not None:
            kwargs[key] = value
    if raw.get("pitch_deck_url"):
        kwargs["pitch_deck_url"] = raw["pitch_deck_url"]
    if raw.get("founder_linkedin_urls"):
        kwargs["founder_linkedin_urls"] = raw["founder_linkedin_urls"]
    return StartupAnalysisRequest(**kwargs)


class RunEntryOutput(BaseModel):
    """Captured fresh prediction for one company in a benchmark run.

    Mirrors the engine report's relevant surfaces.  ``success`` is False
    and ``error`` populated when replay failed (missing corpus, malformed
    request, engine exception); the entry is then excluded from metrics but
    preserved verbatim for explainability.
    """

    company_id: str = Field(..., description="Company identifier in the dataset")
    success: bool = Field(..., description="Whether replay produced a report")
    error: str | None = Field(default=None)
    overall_score: float | None = Field(default=None, ge=0.0, le=100.0)
    overall_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: str | None = Field(default=None)
    decision_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    recommendation_categories: list[str] = Field(default_factory=list)
    recommendation_count: int = Field(default=0, ge=0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    extracted_features: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(default=0.0)

    def sorted(self) -> RunEntryOutput:
        """Return a copy with ordered containers (canonical serialization)."""
        return RunEntryOutput(
            company_id=self.company_id,
            success=self.success,
            error=self.error,
            overall_score=self.overall_score,
            overall_confidence=self.overall_confidence,
            decision=self.decision,
            decision_confidence=self.decision_confidence,
            composite_score=self.composite_score,
            recommendation_categories=sorted(self.recommendation_categories),
            recommendation_count=self.recommendation_count,
            dimension_scores=dict(sorted(self.dimension_scores.items())),
            extracted_features=_sort_value(self.extracted_features),
            evidence_count=self.evidence_count,
            observation_count=self.observation_count,
            processing_time_ms=self.processing_time_ms,
        )


def _sort_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_value(v) for v in value]
    return value


class BenchmarkRun(BaseModel):
    """Immutable record of one benchmark execution.

    Each run is fingerprinted by ``dataset_hash`` (of the input dataset)
    and ``result_hash`` (of the captured outputs), so identical inputs and
    engine versions always produce identical hashes.  ``created_at`` is the
    UTC wall-clock execution time; it is informational only — it never
    affects any metric or comparison.
    """

    run_id: str = Field(..., description="Unique run identifier")
    created_at: datetime = Field(..., description="UTC execution timestamp")
    engine_version: str = Field(..., description="Engine version that produced this run")
    benchmark_version: str = Field(..., description="Golden dataset version used")
    dataset_name: str = Field(..., description="Name of the dataset used")
    dataset_hash: str = Field(..., description="sha256 fingerprint of the dataset")
    result_hash: str = Field(..., description="sha256 fingerprint of the captured outputs")
    entries: list[RunEntryOutput] = Field(default_factory=list)

    def entry_by_id(self, company_id: str) -> RunEntryOutput | None:
        for entry in self.entries:
            if entry.company_id == company_id:
                return entry
        return None

    @property
    def successful_entries(self) -> list[RunEntryOutput]:
        return [e for e in self.entries if e.success]


def _result_hash(entries: list[RunEntryOutput]) -> str:
    """Canonical sha256 over captured run outputs.

    Deterministic regardless of input ordering: entries are sorted by
    company id and nested containers ordered before hashing.  Wall-clock
    timing (``processing_time_ms``) is excluded so two replays of the same
    dataset always produce the same hash.
    """
    ordered = sorted((e.sorted() for e in entries), key=lambda e: e.company_id)
    payload = json.dumps(
        [
            {k: v for k, v in e.model_dump(mode="json").items() if k != "processing_time_ms"}
            for e in ordered
        ],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_run_id(engine_version: str) -> str:
    """Human-readable, sortable, collision-resistant run identifier."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"run_{engine_version}_{stamp}_{uuid.uuid4().hex[:8]}"


class BenchmarkRunner:
    """Executes golden datasets through the current engine."""

    def __init__(self, engine: Any | None = None) -> None:
        from predictron_engine.engine import ENGINE_VERSION, PredictronEngine

        self._engine = engine if engine is not None else PredictronEngine()
        self._engine_version = ENGINE_VERSION

    @property
    def engine_version(self) -> str:
        return self._engine_version

    def run(
        self,
        dataset: GoldenDataset,
        *,
        run_id: str | None = None,
    ) -> BenchmarkRun:
        """Run the current engine over the dataset and return a run.

        Never writes to the dataset.  To persist, hand the returned run to
        a :class:`~benchmarks.ground_truth_eval.history.BenchmarkHistory`.
        """
        created_at = datetime.now(UTC)
        ident = run_id or generate_run_id(self._engine_version)
        entries: list[RunEntryOutput] = []
        for entry in dataset.entries:
            entries.append(self._run_entry(entry))
        result_hash = _result_hash(entries)
        return BenchmarkRun(
            run_id=ident,
            created_at=created_at,
            engine_version=self._engine_version,
            benchmark_version=dataset.benchmark_version,
            dataset_name=dataset.dataset_name,
            dataset_hash=dataset_hash(dataset),
            result_hash=result_hash,
            entries=entries,
        )

    def _run_entry(self, entry: GoldenEntry) -> RunEntryOutput:
        start = time.perf_counter()
        try:
            request = build_request(entry)
            bundle = build_entry_bundle(entry)
            report = self._engine.analyze(request, evidence_bundle=bundle)
            elapsed = (time.perf_counter() - start) * 1000
            return report_to_output(entry, report, elapsed)
        except Exception as exc:  # noqa: BLE001 - entry failures surface in the run
            elapsed = (time.perf_counter() - start) * 1000
            return RunEntryOutput(
                company_id=entry.company_id,
                success=False,
                error=str(exc),
                processing_time_ms=round(elapsed, 2),
            )


# Extracted-feature keys that matter for drift/breakdown reporting.  The
# engine's ExtractedFeatures model is large; capturing this stable subset
# keeps runs version-portable.
FEATURE_KEYS: tuple[str, ...] = (
    "industry",
    "sub_industry",
    "business_model",
    "funding_stage",
    "geography",
    "customer_type",
    "team_size_indicator",
    "founded_year",
    "has_revenue",
)


def report_to_output(
    entry: GoldenEntry,
    report: Report,
    elapsed_ms: float,
) -> RunEntryOutput:
    """Extract the portable prediction surface from a full engine report."""
    decision: str | None = None
    decision_confidence: float | None = None
    composite_score: float | None = None
    if report.investment_decision is not None:
        decision = report.investment_decision.category.value
        composite_score = round(float(report.investment_decision.composite_score), 6)
    if report.decision_confidence is not None:
        decision_confidence = round(float(report.decision_confidence.confidence), 6)

    features: dict[str, Any] = {}
    for key in FEATURE_KEYS:
        features[key] = getattr(report.features, key, None)

    dimension_scores = {s.dimension: round(float(s.score), 6) for s in report.scores}
    rec_categories = sorted({rec.category for rec in report.recommendations})

    return RunEntryOutput(
        company_id=entry.company_id,
        success=True,
        error=None,
        overall_score=round(float(report.overall_score), 6),
        overall_confidence=round(float(report.overall_confidence), 6),
        decision=decision,
        decision_confidence=decision_confidence,
        composite_score=composite_score,
        recommendation_categories=rec_categories,
        recommendation_count=len(report.recommendations),
        dimension_scores=dimension_scores,
        extracted_features=features,
        evidence_count=len(report.evidence),
        observation_count=len(report.observations),
        processing_time_ms=round(elapsed_ms, 2),
    )


__all__ = [
    "RunnerError",
    "build_entry_bundle",
    "build_request",
    "RunEntryOutput",
    "BenchmarkRun",
    "BenchmarkRunner",
    "report_to_output",
    "FEATURE_KEYS",
    "generate_run_id",
    "entry_evidence_scope",
]
