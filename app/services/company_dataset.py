"""DatasetCompanyStore — read-only adapter over the offline historical dataset.

Phase 2 adapter that exposes the grounded offline stack without modifying it:

* :class:`~predictron_engine.dataset.store.DatasetStore` — dataset records.
* :class:`~predictron_engine.dataset.graph.queries.GraphQueries` — knowledge graph.
* :class:`~predictron_engine.dataset.signals.integration.SignalDatasetManager` —
  company signal timelines.
* :class:`~predictron_engine.feature_store.engine.FeatureEngine` — computed features.

The adapter is strictly read-only and additive: it never writes records,
outcomes, timelines, graph objects, or feature sets. Every result is
deterministic for a fixed ``as_of`` reference timestamp.

Identity bridging
-----------------
Registry ``company_id`` values (SHA-256 digests, Phase 1) differ from the
dataset's graph ``company:<record_ids>`` node ids, so the adapter resolves
offline records from the live company's identity **signals**:

1. exact ``startup_name`` matches (primary name, then canonical name),
2. normalized domain matches via ``extract_domain`` against ``record.website``.

Records are de-duplicated by ``record_id`` and returned in a stable
``(analysis_date, record_id)`` order.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.services.companies import CompanyIdentityResolver
from app.services.company_postgres import compute_snapshot_id
from app.services.company_protocols import (
    CompanyPage,
    CompanyRecord,
    CompanySnapshotPayload,
    CompanySnapshotRecord,
    SnapshotAppendResult,
    SnapshotPage,
)
from predictron_engine.dataset.enrichment import extract_domain
from predictron_engine.dataset.signals.aggregate import momentum_score, recent_activity
from predictron_engine.dataset.signals.integration import SignalDatasetManager
from predictron_engine.dataset.store import DatasetStore
from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.features import ALL_FEATURES
from predictron_engine.feature_store.models import FeatureStatus
from predictron_engine.feature_store.registry import FeatureRegistry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.services.company_protocols import CompanyIdentity
    from predictron_engine.dataset.graph.queries import GraphQueries
    from predictron_engine.dataset.graph.store import KnowledgeGraph
    from predictron_engine.dataset.models import DatasetRecord
    from predictron_engine.dataset.signals.timeline import CompanyTimeline

_UNKNOWN_DECISION_VALUES = frozenset({"", "none", "unknown", "na"})


def placeholder_reasons(record: Any) -> list[str]:
    """Return the deterministic quality-gate failures of one record.

    An empty list means the record is grounded (not a placeholder).  Gates
    are pure and ordered; results never depend on external state.
    """
    reasons: list[str] = []
    website = getattr(record, "website", None)
    if not website:
        reasons.append("missing website")
    if (
        getattr(record, "prediction", None) is None
        or record.prediction.confidence is None
        or record.prediction.confidence == 0.0
    ):
        reasons.append("zero-confidence prediction")
    decision = getattr(record.prediction, "decision", None)
    decision_value = getattr(decision, "value", None)
    if decision_value is None or str(decision_value).strip().casefold() in _UNKNOWN_DECISION_VALUES:
        reasons.append("placeholder decision")
    if (
        record.prediction.composite_score is None
        or record.prediction.composite_score == 0.0
    ):
        reasons.append("zero composite score")
    if getattr(record, "evidence_bundle_reference", None) is None:
        reasons.append("missing evidence")
    if getattr(record.prediction, "recommendation_count", 0) == 0:
        reasons.append("no recommendations")
    if getattr(record, "profile", None) is not None and record.profile.is_empty():
        reasons.append("no structured profile")
    return reasons


class DatasetCompanyStore:
    """Read-only facade combining the offline dataset layers for one profile.

    Instances are stateless and re-usable: heavyweight components (graph,
    graph queries, signal manager, feature engine) are built once and cached.
    """

    def __init__(
        self,
        store: DatasetStore,
        *,
        as_of: datetime | None = None,
    ) -> None:
        self._store = store
        self._as_of = as_of
        self._graph: KnowledgeGraph | None = None
        self._graph_queries: GraphQueries | None = None
        self._signals: SignalDatasetManager | None = None
        self._feature_engine: FeatureEngine | None = None

    # ---- component factories -------------------------------------------------

    def _build_registry(self) -> FeatureRegistry:
        registry = FeatureRegistry()
        registry.register_all(ALL_FEATURES)
        return registry

    def graph(self) -> KnowledgeGraph:
        """The knowledge graph built from all stored records."""
        if self._graph is None:
            from app.services.graph import build_graph

            self._graph = build_graph(self._store)
        return self._graph

    def graph_queries(self) -> GraphQueries:
        """Read-only graph traversal facade over the built graph."""
        if self._graph_queries is None:
            from predictron_engine.dataset.graph.queries import GraphQueries

            self._graph_queries = GraphQueries(self.graph())
        return self._graph_queries

    def signals(self) -> SignalDatasetManager:
        """Signal timeline manager bound to the dataset store."""
        if self._signals is None:
            self._signals = SignalDatasetManager(self._store, as_of=self._as_of)
        return self._signals

    def features(self) -> FeatureEngine:
        """Feature engine with the canonical registered feature set."""
        if self._feature_engine is None:
            self._feature_engine = FeatureEngine(self._build_registry())
        return self._feature_engine

    # ---- identity bridging ---------------------------------------------------

    def records_for(
        self,
        *,
        name: str | None = None,
        website: str | None = None,
        domain: str | None = None,
    ) -> list[DatasetRecord]:
        """Resolve the matching dataset records for a company's identity signals.

        Deterministic: de-duplicated by ``record_id`` and sorted by
        ``(analysis_date, record_id)``.  Returns ``[]`` when nothing matches.
        """
        matched: dict[str, Any] = {}
        for key in _distinct_nonempty(name):
            for record in self._store.find_records_by_startup(key):
                matched[record.record_id] = record

        target_domains = {
            d
            for d in (_distinct_nonempty(domain) + [extract_domain(website) or ""])
            if d
        }
        if target_domains:
            for record_id in self._store.list_records():
                loaded = self._store.load_record(record_id)
                if loaded is None or not loaded.website:
                    continue
                candidate = extract_domain(loaded.website)
                if candidate and candidate in target_domains:
                    matched[loaded.record_id] = loaded

        return sorted(
            matched.values(), key=lambda rec: (rec.analysis_date, rec.record_id)
        )

    def company_node_id(self, records: list[DatasetRecord] | None = None) -> str | None:
        """Resolve a company node id from records via deterministic graph lookup."""
        if records is None:
            records = []
        if not records:
            return None
        latest = records[-1]
        return self.graph_queries().lookup(latest.startup_name or latest.website or "")

    # ---- placeholder gating ---------------------------------------------------

    def placeholder_status(
        self, records: list[DatasetRecord] | None = None
    ) -> tuple[bool, list[str]]:
        """Classify a record set as placeholder using deterministic quality gates.

        Returns ``(is_placeholder, sorted_reasons)``.  ``is_placeholder`` is
        True only when *every* record fails at least one gate.
        """
        if not records:
            return True, []
        reasons: set[str] = set()
        for record in records:
            reasons.update(placeholder_reasons(record))
        all_placeholder = all(bool(placeholder_reasons(record)) for record in records)
        return all_placeholder, sorted(reasons)

    # ---- derived summaries ----------------------------------------------------

    def graph_summary(self, node_id: str | None = None) -> dict[str, Any]:
        """Deterministic graph summary for a company node id."""
        if not node_id:
            return _empty_graph_summary()
        graph = self.graph()
        queries = self.graph_queries()
        neighbors = queries.neighbors(node_id)
        return {
            "node_id": node_id,
            "degree": graph.degree(node_id),
            "neighbor_count": len(neighbors),
            "neighbors": [
                {
                    "node_id": entry["node_id"],
                    "node_type": entry["node"].get("node_type", ""),
                    "label": entry["node"].get("label", ""),
                    "edge_types": entry["edge_types"],
                }
                for entry in neighbors
            ],
        }

    def signal_summary(
        self, company_id: str | None, *, as_of: datetime | None = None
    ) -> dict[str, Any] | None:
        """Deterministic signal-timeline summary for a company node id."""
        if not company_id:
            return None
        timeline = self.signals().timeline(company_id)
        if timeline is None:
            return None
        reference = as_of or self._as_of
        activity = recent_activity(timeline, as_of=reference)
        highlight_bundle = self.signals().profile_highlights(
            company_id, as_of=reference, limit=5
        )
        return {
            "signal_count": timeline.signal_count,
            "type_counts": dict(sorted(timeline.types().items())),
            "span_days": timeline.span_days,
            "first_signal_at": timeline.first.timestamp if timeline.first else None,
            "last_signal_at": timeline.last.timestamp if timeline.last else None,
            "momentum_score": momentum_score(timeline, as_of=reference),
            "is_active": bool(activity.get("is_active", False)),
            "highlights": highlight_bundle,
        }

    def feature_summary(
        self,
        record: DatasetRecord,
        *,
        company_node_id: str | None = None,
        timeline: CompanyTimeline | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        """Deterministic feature summary computed by the feature engine."""
        reference = as_of or self._as_of
        queries = self.graph_queries()
        feature_set = self.features().build_company_features(
            record,
            timeline=timeline,
            knowledge_graph=self.graph(),
            graph_queries=queries,
            company_node_id=company_node_id,
            as_of=reference,
            record_id=record.record_id,
        )
        computed = [
            snap
            for snap in feature_set.features.values()
            if snap.status == FeatureStatus.COMPUTED
        ]
        by_category: dict[str, int] = {}
        for snap in computed:
            by_category[snap.category.value] = by_category.get(snap.category.value, 0) + 1
        return {
            "feature_count": feature_set.feature_count(),
            "computed_count": len(computed),
            "failed_count": feature_set.feature_count() - len(computed),
            "by_category": dict(sorted(by_category.items())),
            "features": [
                {
                    "feature_id": snap.feature_id,
                    "feature_name": snap.feature_name,
                    "category": snap.category.value,
                    "value": snap.value,
                    "value_type": snap.value_type.value,
                    "status": snap.status.value,
                }
                for snap in sorted(
                    feature_set.features.values(), key=lambda snap: snap.feature_id
                )
            ],
        }

    def offline_summary(self, records: list[DatasetRecord]) -> dict[str, Any]:
        """Deterministic summary of the matching offline record set."""
        if not records:
            return _empty_offline_summary()
        is_placeholder, reasons = self.placeholder_status(records)
        return {
            "record_count": len(records),
            "sources": sorted({r.source for r in records if r.source}),
            "placeholder": is_placeholder,
            "placeholder_count": (
                sum(1 for r in records if placeholder_reasons(r)) if is_placeholder else 0
            ),
            "placeholder_reasons": reasons,
            "first_analysis_date": records[0].analysis_date,
            "last_analysis_date": records[-1].analysis_date,
        }

    def benchmark_summary(self, composite_score: float) -> dict[str, Any]:
        """Deterministic benchmark placement of a composite score.

        Computes the percentile rank of ``composite_score`` within the
        composite scores of *every* stored dataset record — the same
        population and formula used by the benchmark comparison endpoints
        (see ``app.services.venture._percentile_rank``).
        """
        scores: list[float] = []
        for record_id in self._store.list_records():
            loaded = self._store.load_record(record_id)
            if loaded is None or loaded.prediction.composite_score is None:
                continue
            scores.append(loaded.prediction.composite_score)
        mean = statistics.mean(scores) if scores else 0.0
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        z_score = (composite_score - mean) / std if std > 0 else 0.0
        percentile = _percentile_rank(composite_score, scores) if scores else 0.0
        return {
            "composite_score": composite_score,
            "benchmark_mean": mean,
            "benchmark_std_dev": std,
            "percentile_rank": percentile,
            "z_score": z_score,
            "sample_size": len(scores),
        }

    def decision_summary(
        self,
        record: DatasetRecord,
        *,
        company_node_id: str | None = None,
        timeline: CompanyTimeline | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        """Deterministic decision-trace summary for a grounded offline record.

        Runs the Decision Intelligence pipeline over the same computed
        feature set exposed by :meth:`feature_summary`, so the profile's
        decision verdict, confidence, and trace counts are read-only joins
        onto one snapshot of features.
        """
        from predictron_engine.decision.service import DecisionIntelligenceService

        reference = as_of or self._as_of
        queries = self.graph_queries()
        trace_report = DecisionIntelligenceService().produce_decision(
            self.features().build_company_features(
                record,
                timeline=timeline,
                knowledge_graph=self.graph(),
                graph_queries=queries,
                company_node_id=company_node_id,
                as_of=reference,
                record_id=record.record_id,
            )
        )
        trace = trace_report.trace
        return {
            "verdict": str(trace_report.decision_summary.get("verdict", "")),
            "confidence": float(trace_report.decision_summary.get("confidence", 0.0)),
            "overall_score": float(
                trace_report.decision_summary.get("overall_score", 0.0)
            ),
            "feature_count": int(
                trace_report.decision_summary.get("feature_count", 0)
            ),
            "trace_id": trace_report.decision_summary.get("trace_id"),
            "node_count": len(trace.nodes) if trace is not None else 0,
            "edge_count": len(trace.edges) if trace is not None else 0,
            "positive_factor_count": len(trace_report.positive_contributions),
            "negative_factor_count": len(trace_report.negative_contributions),
            "top_strengths": trace_report.top_strengths,
            "top_weaknesses": trace_report.top_weaknesses,
            "source_record_id": record.record_id,
        }

    # ---- CompanyStore read protocol (read-only conformance) ------------------

    def _grounded_records(
        self, *, company_id: str | None = None
    ) -> list[tuple[CompanyIdentity, DatasetRecord]]:
        """Chronologically ordered grounded records, optionally scoped to one company.

        Only records passing the CIH quality gates surface through the
        protocol reads; placeholder rows are treated as missing data. Each row
        is paired with its deterministic company identity so the caller never
        recomputes the SHA-256 company id.
        """
        rows: list[tuple[CompanyIdentity, DatasetRecord]] = []
        for record_id in self._store.list_records():
            loaded = self._store.load_record(record_id)
            if loaded is None or placeholder_reasons(loaded):
                continue
            identity = CompanyIdentityResolver().resolve(
                loaded.startup_name or "", loaded.website
            )
            if company_id is not None and identity.company_id != company_id:
                continue
            rows.append((identity, loaded))
        rows.sort(key=lambda row: (row[1].analysis_date, row[1].record_id))
        return rows

    def record_for(
        self, records: list[DatasetRecord]
    ) -> CompanyRecord | None:
        """Assemble a protocol company record from pre-resolved offline records.

        Grounding-gated: returns ``None`` when no record passes the CIH quality
        gates, so offline-only profiles are never built from placeholders.
        ``records`` are expected in the chronological order produced by
        :meth:`records_for`, so the last grounded row is the latest.
        """
        grounded = [record for record in records if not placeholder_reasons(record)]
        if not grounded:
            return None
        latest = grounded[-1]
        identity = CompanyIdentityResolver().resolve(
            latest.startup_name or "", latest.website
        )
        return _company_record_from(identity, grounded)

    async def get_company(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> CompanyRecord | None:
        """Fetch one company by its deterministic id (offline, no ownership scope)."""
        rows = self._grounded_records(company_id=company_id)
        if not rows:
            return None
        return _company_record_from(rows[0][0], [record for _, record in rows])

    async def list_companies(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyPage:
        """List grounded offline companies, newest activity first."""
        by_company: dict[str, list[tuple[CompanyIdentity, DatasetRecord]]] = {}
        for identity, record in self._grounded_records():
            by_company.setdefault(identity.company_id, []).append((identity, record))

        companies = [
            _company_record_from(rows[0][0], [record for _, record in rows])
            for rows in by_company.values()
        ]
        companies.sort(
            key=lambda record: (record.last_seen, record.company_id), reverse=True
        )
        total = len(companies)
        return CompanyPage(companies=companies[offset : offset + limit], total=total)

    async def list_snapshots(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> SnapshotPage:
        """List one company's offline snapshots, newest first."""
        rows = self._grounded_records(company_id=company_id)
        snapshots = [_snapshot_record_from(identity, record) for identity, record in rows]
        snapshots.sort(key=lambda snap: (snap.created_at, snap.id), reverse=True)
        total = len(snapshots)
        return SnapshotPage(snapshots=snapshots[offset : offset + limit], total=total)

    async def upsert_company(
        self,
        session: AsyncSession,
        identity: CompanyIdentity,
        *,
        user_id: str | None = None,
        seen_at: datetime | None = None,
        latest_decision: str | None = None,
        latest_confidence: float | None = None,
        latest_composite_score: float | None = None,
    ) -> CompanyRecord:
        raise NotImplementedError(
            "DatasetCompanyStore is a read-only offline adapter; upserts are unsupported"
        )

    async def append_snapshot(
        self,
        session: AsyncSession,
        payload: CompanySnapshotPayload,
    ) -> SnapshotAppendResult:
        raise NotImplementedError(
            "DatasetCompanyStore is a read-only offline adapter; appends are unsupported"
        )


def _prediction_decision(prediction: Any) -> str | None:
    """Coerce a prediction's decision (enum or plain string) to ``str`` or None."""
    decision = getattr(prediction, "decision", None)
    decision_value = getattr(decision, "value", None)
    if decision_value is None and isinstance(decision, str):
        decision_value = decision
    if decision_value is None:
        return None
    return str(decision_value)


def _company_record_from(
    identity: CompanyIdentity, records: list[DatasetRecord]
) -> CompanyRecord:
    """Assemble the protocol record for one identity from its offline rows."""
    latest = records[-1]
    prediction = latest.prediction
    return CompanyRecord(
        company_id=identity.company_id,
        canonical_name=identity.canonical_name,
        canonical_domain=identity.canonical_domain,
        primary_name=identity.primary_name,
        website=identity.website,
        latest_decision=_prediction_decision(prediction),
        latest_confidence=prediction.confidence,
        latest_composite_score=prediction.composite_score,
        snapshot_count=len(records),
        first_seen=records[0].analysis_date,
        last_seen=records[-1].analysis_date,
        user_id=None,
        canonical_name_key=identity.canonical_name_key,
        fallback_slug=identity.fallback_slug,
    )


def _snapshot_record_from(
    identity: CompanyIdentity, record: DatasetRecord
) -> CompanySnapshotRecord:
    """Assemble one protocol snapshot for an offline record."""
    prediction = record.prediction
    return CompanySnapshotRecord(
        id=compute_snapshot_id(identity.company_id, record.record_id),
        company_id=identity.company_id,
        analysis_id=record.record_id,
        report_id=record.record_id,
        decision=_prediction_decision(prediction),
        confidence=prediction.confidence,
        composite_score=prediction.composite_score,
        readiness_score=None,
        dimension_scores=prediction.dimension_scores or {},
        created_at=record.analysis_date,
    )


def _distinct_nonempty(*values: str | None) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value.strip() and value not in seen:
            seen.append(value.strip())
    return seen


def _percentile_rank(value: float, scores: list[float]) -> float:
    """Percentile rank of ``value`` within ``scores`` (0-100).

    Mirrors ``app.services.venture._percentile_rank`` so benchmark
    placement is identical across every reporting surface.
    """
    if not scores:
        return 0.0
    count_below = sum(1 for s in scores if s < value)
    count_equal = sum(1 for s in scores if s == value)
    return ((count_below + 0.5 * count_equal) / len(scores)) * 100.0


def _empty_graph_summary() -> dict[str, Any]:
    return {
        "node_id": None,
        "degree": 0,
        "neighbor_count": 0,
        "neighbors": [],
    }


def _empty_offline_summary() -> dict[str, Any]:
    return {
        "record_count": 0,
        "sources": [],
        "placeholder": False,
        "placeholder_count": 0,
        "placeholder_reasons": [],
        "first_analysis_date": None,
        "last_analysis_date": None,
    }


__all__ = [
    "DatasetCompanyStore",
    "placeholder_reasons",
]
