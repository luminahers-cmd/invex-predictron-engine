"""Company Comparison service — multi-dimension comparison of startups."""

from __future__ import annotations

import asyncio
from typing import Any

from app.schemas.comparison import (
    BenchmarkComparisonDetail,
    ComparisonRequest,
    ContributionDiffItem,
    ContributionDiffResponse,
    DecisionDiffResponse,
    FeatureDiff,
    FeatureDiffResponse,
    FullComparisonResponse,
    KnowledgeGraphDiffResponse,
    SignalDiffResponse,
)


async def compare_companies(
    engine: Any,
    request: ComparisonRequest,
    store: Any = None,
) -> FullComparisonResponse:
    """Run comparison across all dimensions for the requested companies."""
    descriptions = request.descriptions or {}
    website_urls = request.website_urls or {}
    names = list(request.company_names)

    reports: list[Any] = []
    for name in names:
        desc = descriptions.get(name, f"Comparison analysis for {name}")
        url = website_urls.get(name)
        report = await _run_analysis(engine, name, url, desc)
        reports.append(report)

    feature_diffs = _build_feature_diffs(names, reports)
    decision_diffs = _build_decision_diffs(names, reports)
    contribution_diffs = _build_contribution_diffs(names, reports)
    signal_diffs = _build_signal_diffs(names, reports)
    kg_diffs = _build_knowledge_graph_diffs(names, reports)
    benchmark_cmp = _build_benchmark_comparison(names, reports)

    summary: dict[str, object] = {
        "company_count": len(names),
        "avg_score": (
            sum(r.overall_score for r in reports) / len(reports)
            if reports
            else 0.0
        ),
        "score_range": (
            max(r.overall_score for r in reports) - min(r.overall_score for r in reports)
            if len(reports) > 1
            else 0.0
        ),
    }

    return FullComparisonResponse(
        companies=names,
        feature_diffs=feature_diffs,
        decision_diffs=decision_diffs,
        contribution_diffs=contribution_diffs,
        signal_diffs=signal_diffs,
        knowledge_graph_diffs=kg_diffs,
        benchmark_comparison=benchmark_cmp,
        overall_summary=summary,
    )


async def _run_analysis(
    engine: Any, name: str, url: str | None, desc: str
) -> Any:
    """Run a single analysis in a thread."""
    from pydantic import HttpUrl

    from app.schemas.analysis import StartupAnalysisRequest as _Req

    request = _Req(
        startup_name=name,
        website_url=HttpUrl(url) if url else None,
        description=desc,
    )
    return await asyncio.to_thread(engine.analyze, request)


def _build_feature_diffs(
    names: list[str], reports: list[Any]
) -> FeatureDiffResponse:
    """Compare extracted features across companies."""
    all_dims: set[str] = set()
    for r in reports:
        for s in r.scores:
            all_dims.add(s.dimension)

    diffs: list[FeatureDiff] = []
    for dim in sorted(all_dims):
        vals = []
        for r in reports:
            val = next(
                (s.score for s in r.scores if s.dimension == dim), 0.0
            )
            vals.append(val)

        if len(vals) >= 2:
            diff_val = max(vals) - min(vals)
            direction = "equal"
            if vals[0] > vals[1]:
                direction = "positive"
            elif vals[0] < vals[1]:
                direction = "negative"
        else:
            diff_val = 0.0
            direction = "equal"

        diffs.append(
            FeatureDiff(
                feature=dim,
                company_a=vals[0] if vals else 0.0,
                company_b=vals[1] if len(vals) > 1 else 0.0,
                difference=diff_val,
                direction=direction,
            )
        )

    return FeatureDiffResponse(
        companies=names,
        feature_diffs=diffs,
    )


def _build_decision_diffs(
    names: list[str], reports: list[Any]
) -> DecisionDiffResponse:
    """Compare decisions across companies."""
    comparison: list[dict[str, object]] = []
    for r in reports:
        decision = r.investment_decision
        comparison.append({
            "startup_name": r.startup.name,
            "overall_score": r.overall_score,
            "confidence": r.overall_confidence,
            "decision_category": decision.category.value if decision else "unknown",
            "conviction": decision.conviction.value if decision else "unknown",
            "composite_score": (
                decision.composite_score if decision else r.overall_score
            ),
        })

    return DecisionDiffResponse(
        companies=names,
        decision_comparison=comparison,
    )


def _build_contribution_diffs(
    names: list[str], reports: list[Any]
) -> ContributionDiffResponse:
    """Compare dimension contributions across companies."""
    all_dims: set[str] = set()
    for r in reports:
        for s in r.scores:
            all_dims.add(s.dimension)

    diffs: list[ContributionDiffItem] = []
    for dim in sorted(all_dims):
        contribs: dict[str, float] = {}
        for r in reports:
            score = next(
                (s.score for s in r.scores if s.dimension == dim), 0.0
            )
            contribs[r.startup.name] = score

        vals = list(contribs.values())
        diffs.append(
            ContributionDiffItem(
                feature=dim,
                contributions=contribs,
                max_contribution=max(vals) if vals else 0.0,
                min_contribution=min(vals) if vals else 0.0,
            )
        )

    return ContributionDiffResponse(
        companies=names,
        diffs=diffs,
    )


def _build_signal_diffs(
    names: list[str], reports: list[Any]
) -> SignalDiffResponse:
    """Compare signal information across companies."""
    comparison: list[dict[str, object]] = []
    for r in reports:
        comparison.append({
            "startup_name": r.startup.name,
            "evidence_count": len(r.evidence),
            "observation_count": len(r.observations),
            "signal_relationship_count": len(r.signal_relationships),
        })

    return SignalDiffResponse(
        companies=names,
        signal_comparison=comparison,
    )


def _build_knowledge_graph_diffs(
    names: list[str], reports: list[Any]
) -> KnowledgeGraphDiffResponse:
    """Compare knowledge graph structures across companies."""
    graph_summary: dict[str, dict[str, object]] = {}
    for r in reports:
        graph_summary[r.startup.name] = {
            "startup_name": r.startup.name,
            "has_description": bool(r.startup.description),
            "has_website": bool(r.startup.website),
            "has_tech_stack": bool(getattr(r.startup, "technology_stack", None)),
        }

    return KnowledgeGraphDiffResponse(
        companies=names,
        graph_summary=graph_summary,
    )


def _build_benchmark_comparison(
    names: list[str], reports: list[Any]
) -> BenchmarkComparisonDetail:
    """Build benchmark comparison for each company."""
    metrics: list[dict[str, object]] = []
    for r in reports:
        metrics.append({
            "startup_name": r.startup.name,
            "overall_score": r.overall_score,
            "overall_confidence": r.overall_confidence,
            "decision_category": (
                r.investment_decision.category.value
                if r.investment_decision
                else "unknown"
            ),
            "dimension_count": len(r.scores),
        })

    return BenchmarkComparisonDetail(
        companies=names,
        benchmark_metrics=metrics,
    )
