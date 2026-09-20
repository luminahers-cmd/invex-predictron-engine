"""Venture Analysis service — orchestrates full pipeline analysis.

Wraps PredictronEngine, DatasetStore, KnowledgeGraph, and Signal timelines
to serve the venture analysis API endpoints.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import TYPE_CHECKING, Any

from app.services.graph import build_graph
from predictron_engine.dataset.store import DatasetStore

if TYPE_CHECKING:
    from predictron_engine.engine import PredictronEngine
    from predictron_engine.models.report import Report

from app.schemas.venture import (
    BenchmarkComparisonResponse,
    DecisionExplainResponse,
    DecisionTraceResponse,
    DimensionScore,
    ExplanationResponse,
    FeatureSnapshotListResponse,
    FeatureSnapshotResponse,
    GraphEdgeResponse,
    GraphNodeResponse,
    KnowledgeGraphSummaryResponse,
    SignalAggregationResponse,
    SignalResponse,
    SignalTimelineResponse,
    SignalTrendsResponse,
    TraceNodeResponse,
    VentureAnalysisResponse,
)


async def run_full_analysis(
    engine: PredictronEngine,
    startup_name: str,
    website_url: str | None = None,
    description: str = "",
    pitch_deck_url: str | None = None,
    founder_linkedin_urls: list[str] | None = None,
) -> tuple[Report, VentureAnalysisResponse]:
    """Run a full venture analysis and return both Report and response."""
    from pydantic import HttpUrl

    from app.schemas.analysis import StartupAnalysisRequest as _Req

    request = _Req(
        startup_name=startup_name,
        website_url=HttpUrl(website_url) if website_url else None,
        description=description,
        pitch_deck_url=HttpUrl(pitch_deck_url) if pitch_deck_url else None,
        founder_linkedin_urls=[
            HttpUrl(u) for u in (founder_linkedin_urls or [])
        ],
    )
    report: Report = await asyncio.to_thread(engine.analyze, request)
    response = _build_venture_response(report)
    return report, response


def _build_venture_response(report: Report) -> VentureAnalysisResponse:
    """Convert a Report into a VentureAnalysisResponse."""
    scores = [
        DimensionScore(
            dimension=s.dimension,
            score=s.score,
            rationale=s.rationale,
        )
        for s in report.scores
    ]

    decision_cat = None
    conviction = None
    if report.investment_decision:
        decision_cat = report.investment_decision.category.value
        conviction = report.investment_decision.conviction.value

    recommendations = [r.action for r in report.recommendations[:5]]

    return VentureAnalysisResponse(
        startup_name=report.startup.name,
        overall_score=report.overall_score,
        overall_confidence=report.overall_confidence,
        dimension_scores=scores,
        decision_category=decision_cat,
        conviction_level=conviction,
        recommendation_count=len(report.recommendations),
        key_recommendations=recommendations,
        processing_time_ms=report.analysis_metadata.processing_time_ms,
        engine_version=report.analysis_metadata.engine_version,
        created_at=report.analysis_metadata.timestamp,
    )


def get_decision_explain(
    report: Report,
    feature_set: Any = None,
) -> DecisionExplainResponse:
    """Build the decision explanation from a report and optional feature set."""
    from predictron_engine.decision.service import DecisionIntelligenceService
    from predictron_engine.feature_store.models import CompanyFeatureSet

    service = DecisionIntelligenceService()

    if feature_set is not None and isinstance(feature_set, CompanyFeatureSet):
        explanation = service.produce_explanation(feature_set)
        strengths = [s for s in explanation.strengths if s]
        weaknesses = [w for w in explanation.weaknesses if w]
    else:
        explanation = None
        strengths = []
        weaknesses = []

    decision = report.investment_decision
    company_id = report.startup.name.lower().replace(" ", "_")

    return DecisionExplainResponse(
        company_id=company_id,
        overall_score=report.overall_score,
        verdict=decision.category.value if decision else "",
        confidence=report.overall_confidence,
        explanation=(
            ExplanationResponse(
                company_id=explanation.company_id,
                headline=explanation.headline,
                strengths=strengths,
                weaknesses=weaknesses,
                neutral_factors=explanation.neutral_factors,
                confidence_factors=explanation.confidence_factors,
                evidence_summary=explanation.evidence_summary,
                recommendation=explanation.recommendation,
            )
            if explanation
            else None
        ),
        top_strengths=strengths[:5],
        top_weaknesses=weaknesses[:5],
    )


def get_decision_trace(
    report: Report,
    feature_set: Any = None,
) -> DecisionTraceResponse:
    """Build the decision trace from a report."""
    from predictron_engine.decision.service import DecisionIntelligenceService
    from predictron_engine.feature_store.models import CompanyFeatureSet

    company_id = report.startup.name.lower().replace(" ", "_")
    decision = report.investment_decision

    service = DecisionIntelligenceService()

    if feature_set is not None and isinstance(feature_set, CompanyFeatureSet):
        trace = service.produce_trace(feature_set, confidence=report.overall_confidence)
        nodes = [
            TraceNodeResponse(
                node_id=n.node_id,
                node_type=n.node_type.value,
                feature_id=n.feature_id,
                label=n.label,
                value=n.value,
                value_type=n.value_type,
                rule=n.rule,
                input_node_ids=n.input_node_ids,
                output_node_ids=n.output_node_ids,
            )
            for n in trace.nodes
        ]
        edges = [[e[0], e[1]] for e in trace.edges]
        return DecisionTraceResponse(
            company_id=trace.company_id,
            trace_id=trace.trace_id,
            overall_score=trace.overall_score,
            verdict=trace.verdict.value,
            confidence=trace.confidence,
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    return DecisionTraceResponse(
        company_id=company_id,
        trace_id="",
        overall_score=report.overall_score,
        verdict=decision.category.value if decision else "",
        confidence=report.overall_confidence,
    )


def get_feature_snapshot(
    report: Report,
    feature_set: Any = None,
) -> FeatureSnapshotListResponse:
    """Build feature snapshot list from a report."""
    from predictron_engine.feature_store.models import CompanyFeatureSet

    company_id = report.startup.name.lower().replace(" ", "_")
    features: list[FeatureSnapshotResponse] = []

    if feature_set is not None and isinstance(feature_set, CompanyFeatureSet):
        for fid, snap in feature_set.features.items():
            features.append(
                FeatureSnapshotResponse(
                    feature_id=snap.feature_id,
                    feature_name=snap.feature_name,
                    category=snap.category.value,
                    value=snap.value,
                    value_type=snap.value_type.value,
                    status=snap.status.value,
                    computed_at=snap.computed_at.isoformat(),
                )
            )

    return FeatureSnapshotListResponse(
        company_id=company_id,
        feature_count=len(features),
        features=features,
    )


def get_knowledge_graph_summary(
    store: DatasetStore | None = None,
) -> KnowledgeGraphSummaryResponse:
    """Build knowledge graph summary from the dataset store."""
    from predictron_engine.dataset.graph.model import EdgeType, NodeType

    graph = build_graph(store)

    node_type_counts: dict[str, int] = {}
    for nt in NodeType:
        cnt = len(graph.nodes_of_type(nt))
        if cnt > 0:
            node_type_counts[nt.value] = cnt

    edge_type_counts: dict[str, int] = {}
    for et in EdgeType:
        cnt = sum(1 for e in graph.edges() if e.edge_type == et)
        if cnt > 0:
            edge_type_counts[et.value] = cnt

    from predictron_engine.dataset.graph.metrics import compute_graph_metrics
    metrics = compute_graph_metrics(graph)

    top_connected: list[dict[str, object]] = []
    degrees = [(n.node_id, graph.degree(n.node_id)) for n in graph.nodes()]
    degrees.sort(key=lambda x: x[1], reverse=True)
    for nid, deg in degrees[:10]:
        node = graph.node(nid)
        if node:
            top_connected.append({
                "node_id": nid,
                "label": node.label,
                "degree": deg,
            })

    return KnowledgeGraphSummaryResponse(
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        node_type_counts=node_type_counts,
        edge_type_counts=edge_type_counts,
        connected_components=metrics.component_count,
        density=metrics.density_undirected,
        nodes=[
            GraphNodeResponse(
                node_id=n.node_id,
                node_type=n.node_type.value,
                label=n.label,
                properties=n.properties,
                sources=n.sources,
            )
            for n in graph.nodes()
        ],
        edges=[
            GraphEdgeResponse(
                edge_type=e.edge_type.value,
                source_id=e.source_id,
                target_id=e.target_id,
                properties=e.properties,
                sources=e.sources,
            )
            for e in graph.edges()
        ],
        top_connected_nodes=top_connected,
    )


def get_signals_timeline(
    company_id: str,
    store: DatasetStore | None = None,
) -> SignalTimelineResponse:
    """Build signal timeline response for a company."""

    if store is None:
        return SignalTimelineResponse(company_id=company_id)

    timeline = store.load_timeline(company_id)
    if timeline is None:
        return SignalTimelineResponse(company_id=company_id)

    type_counts: dict[str, int] = {}
    for sig in timeline.signals:
        type_counts[sig.signal_type.value] = type_counts.get(
            sig.signal_type.value, 0
        ) + 1

    signals = [
        SignalResponse(
            signal_id=s.signal_id,
            company_id=s.company_id,
            signal_type=s.signal_type.value,
            timestamp=s.timestamp.isoformat(),
            source=s.source,
            provenance=s.provenance,
            confidence=s.confidence,
            metadata=dict(s.metadata),
        )
        for s in timeline.signals
    ]

    return SignalTimelineResponse(
        company_id=company_id,
        signal_count=timeline.signal_count,
        signals=signals,
        type_counts=type_counts,
        span_days=timeline.span_days,
        first_signal=timeline.first.timestamp.isoformat() if timeline.first else None,
        last_signal=timeline.last.timestamp.isoformat() if timeline.last else None,
    )


def get_signal_trends(
    company_id: str,
    store: DatasetStore | None = None,
) -> SignalTrendsResponse:
    """Compute signal trend results for a company."""
    from predictron_engine.dataset.signals.trends import TrendEngine

    if store is None:
        return SignalTrendsResponse(company_id=company_id)

    timeline = store.load_timeline(company_id)
    if timeline is None:
        return SignalTrendsResponse(company_id=company_id)

    engine = TrendEngine()
    trends = engine.evaluate(timeline)

    from app.schemas.venture import TrendResultResponse
    result_trends: dict[str, TrendResultResponse] = {}
    for name, tr in trends.items():
        result_trends[name] = TrendResultResponse(
            name=tr.name,
            value=tr.value,
            direction=tr.direction,
            window_days=tr.window_days,
            explanation=tr.explanation,
            available=tr.available,
        )

    return SignalTrendsResponse(
        company_id=company_id,
        trends=result_trends,
    )


def get_signal_aggregation(
    company_id: str,
    store: DatasetStore | None = None,
) -> SignalAggregationResponse:
    """Compute signal aggregation metrics for a company."""
    from predictron_engine.dataset.signals.aggregate import (
        funding_cadence,
        momentum_score,
        recent_activity,
        signal_freshness,
    )

    if store is None:
        return SignalAggregationResponse(company_id=company_id)

    timeline = store.load_timeline(company_id)
    if timeline is None:
        return SignalAggregationResponse(company_id=company_id)

    return SignalAggregationResponse(
        company_id=company_id,
        recent_activity=recent_activity(timeline),
        momentum_score=momentum_score(timeline),
        signal_freshness=signal_freshness(timeline),
        funding_cadence=funding_cadence(timeline),
    )


def get_benchmark_comparison(
    report: Report,
    store: DatasetStore | None = None,
) -> BenchmarkComparisonResponse:
    """Compute benchmark comparison for a report against historical data."""
    scores: list[float] = []
    dimension_scores_by_dim: dict[str, list[float]] = {}

    if store is not None:

        for rid in store.list_records():
            rec = store.load_record(rid)
            if rec is not None:
                scores.append(rec.prediction.composite_score)
                for dim, dscore in rec.prediction.dimension_scores.items():
                    dimension_scores_by_dim.setdefault(dim, []).append(dscore)

    composite = report.overall_score
    mean = statistics.mean(scores) if scores else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    z_score = (composite - mean) / std if std > 0 else 0.0
    percentile = _percentile_rank(composite, scores) if scores else 0.0

    dim_comparisons: dict[str, dict[str, float]] = {}
    report_scores = {s.dimension: s.score for s in report.scores}
    for dim, dim_scores in dimension_scores_by_dim.items():
        if dim_scores:
            dim_mean = statistics.mean(dim_scores)
            dim_std = statistics.stdev(dim_scores) if len(dim_scores) > 1 else 0.0
            dim_val = report_scores.get(dim, 0.0)
            dim_z = (dim_val - dim_mean) / dim_std if dim_std > 0 else 0.0
            dim_comparisons[dim] = {
                "score": dim_val,
                "benchmark_mean": dim_mean,
                "benchmark_std_dev": dim_std,
                "z_score": dim_z,
            }

    category_dist: dict[str, int] = {}
    if store is not None:
        for rid in store.list_records():
            rec = store.load_record(rid)
            if rec:
                cat = rec.prediction.decision.value
                category_dist[cat] = category_dist.get(cat, 0) + 1

    return BenchmarkComparisonResponse(
        startup_name=report.startup.name,
        composite_score=composite,
        benchmark_mean=mean,
        benchmark_std_dev=std,
        percentile_rank=percentile,
        score_z_score=z_score,
        dimension_comparisons=dim_comparisons,
        category_distribution=category_dist,
        sample_size=len(scores),
    )


def _percentile_rank(value: float, scores: list[float]) -> float:
    """Compute percentile rank of value within scores (0-100)."""
    if not scores:
        return 0.0
    count_below = sum(1 for s in scores if s < value)
    count_equal = sum(1 for s in scores if s == value)
    return ((count_below + 0.5 * count_equal) / len(scores)) * 100.0
