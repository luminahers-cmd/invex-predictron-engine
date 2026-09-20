"""Venture Analysis API — endpoints for startup analysis intelligence."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status

from app.auth.jwt import get_current_user_optional
from app.schemas.venture import (
    BenchmarkComparisonResponse,
    DecisionExplainResponse,
    DecisionTraceResponse,
    FeatureSnapshotListResponse,
    KnowledgeGraphSummaryResponse,
    SignalAggregationResponse,
    SignalTimelineResponse,
    SignalTrendsResponse,
    VentureAnalysisRequest,
    VentureAnalysisResponse,
)
from app.services.venture import (
    get_benchmark_comparison,
    get_decision_explain,
    get_decision_trace,
    get_feature_snapshot,
    get_knowledge_graph_summary,
    get_signal_aggregation,
    get_signal_trends,
    get_signals_timeline,
    run_full_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/venture", tags=["venture-analysis"])


@router.post(
    "",
    response_model=VentureAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Full venture analysis",
    description="Run a complete venture intelligence analysis on a startup.",
    responses={
        200: {"description": "Analysis completed", "model": VentureAnalysisResponse},
        422: {"description": "Validation error"},
    },
)
async def venture_analyze(
    request: VentureAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> VentureAnalysisResponse:
    engine = raw_request.app.state.predictron_engine
    _report, response = await run_full_analysis(
        engine=engine,
        startup_name=request.startup_name,
        website_url=request.website_url,
        description=request.description,
        pitch_deck_url=request.pitch_deck_url,
        founder_linkedin_urls=request.founder_linkedin_urls,
    )
    return response


@router.post(
    "/explain",
    response_model=DecisionExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Decision explanation",
    description="Get a human-readable explanation of the investment decision.",
)
async def venture_explain(
    request: VentureAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> DecisionExplainResponse:
    engine = raw_request.app.state.predictron_engine
    report, _ = await run_full_analysis(
        engine=engine,
        startup_name=request.startup_name,
        website_url=request.website_url,
        description=request.description,
        pitch_deck_url=request.pitch_deck_url,
        founder_linkedin_urls=request.founder_linkedin_urls,
    )
    return get_decision_explain(report)


@router.post(
    "/trace",
    response_model=DecisionTraceResponse,
    status_code=status.HTTP_200_OK,
    summary="Decision trace",
    description="Get the full decision reasoning trace as a graph.",
)
async def venture_trace(
    request: VentureAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> DecisionTraceResponse:
    engine = raw_request.app.state.predictron_engine
    report, _ = await run_full_analysis(
        engine=engine,
        startup_name=request.startup_name,
        website_url=request.website_url,
        description=request.description,
        pitch_deck_url=request.pitch_deck_url,
        founder_linkedin_urls=request.founder_linkedin_urls,
    )
    return get_decision_trace(report)


@router.post(
    "/features",
    response_model=FeatureSnapshotListResponse,
    status_code=status.HTTP_200_OK,
    summary="Feature snapshot",
    description="Get the feature snapshot for a startup analysis.",
)
async def venture_features(
    request: VentureAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> FeatureSnapshotListResponse:
    engine = raw_request.app.state.predictron_engine
    report, _ = await run_full_analysis(
        engine=engine,
        startup_name=request.startup_name,
        website_url=request.website_url,
        description=request.description,
        pitch_deck_url=request.pitch_deck_url,
        founder_linkedin_urls=request.founder_linkedin_urls,
    )
    return get_feature_snapshot(report)


@router.get(
    "/knowledge-graph",
    response_model=KnowledgeGraphSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Knowledge graph summary",
    description="Get a summary of the knowledge graph built from the dataset.",
)
async def venture_knowledge_graph(
    raw_request: Request,
) -> KnowledgeGraphSummaryResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return get_knowledge_graph_summary(store=store)


@router.get(
    "/signals/{company_id}",
    response_model=SignalTimelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Signals timeline",
    description="Get the signal timeline for a specific company.",
)
async def venture_signals_timeline(
    company_id: str,
) -> SignalTimelineResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return get_signals_timeline(company_id, store=store)


@router.get(
    "/signals/{company_id}/trends",
    response_model=SignalTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Signal trends",
    description="Get trend analysis results for a company's signals.",
)
async def venture_signal_trends(
    company_id: str,
) -> SignalTrendsResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return get_signal_trends(company_id, store=store)


@router.get(
    "/signals/{company_id}/aggregation",
    response_model=SignalAggregationResponse,
    status_code=status.HTTP_200_OK,
    summary="Signal aggregation",
    description="Get aggregated signal metrics for a company.",
)
async def venture_signal_aggregation(
    company_id: str,
) -> SignalAggregationResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return get_signal_aggregation(company_id, store=store)


@router.post(
    "/benchmark",
    response_model=BenchmarkComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Benchmark comparison",
    description="Compare a startup's analysis against historical benchmarks.",
)
async def venture_benchmark(
    request: VentureAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> BenchmarkComparisonResponse:
    from pathlib import Path

    from predictron_engine.dataset.store import DatasetStore

    engine = raw_request.app.state.predictron_engine
    report, _ = await run_full_analysis(
        engine=engine,
        startup_name=request.startup_name,
        website_url=request.website_url,
        description=request.description,
        pitch_deck_url=request.pitch_deck_url,
        founder_linkedin_urls=request.founder_linkedin_urls,
    )
    data_dir = Path("data") / "dataset"
    store = DatasetStore(str(data_dir)) if data_dir.exists() else None
    return get_benchmark_comparison(report, store=store)
