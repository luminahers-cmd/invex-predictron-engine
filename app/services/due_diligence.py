"""Due Diligence Report service — structured report generation."""

from __future__ import annotations

import asyncio
from typing import Any

from app.schemas.due_diligence import (
    BenchmarkContext,
    DecisionTraceEntry,
    DueDiligenceReportResponse,
    EvidenceEntry,
    ExecutiveSummary,
    OpportunityItemResponse,
    RiskItemResponse,
    StrengthItem,
    WeaknessItem,
)


async def generate_due_diligence(
    engine: Any,
    startup_name: str,
    website_url: str | None = None,
    description: str = "",
    pitch_deck_url: str | None = None,
    founder_linkedin_urls: list[str] | None = None,
    store: Any = None,
) -> DueDiligenceReportResponse:
    """Generate a complete due diligence report for a startup."""
    import time

    from pydantic import HttpUrl

    from app.schemas.analysis import StartupAnalysisRequest as _Req

    start = time.monotonic()

    request = _Req(
        startup_name=startup_name,
        website_url=HttpUrl(website_url) if website_url else None,
        description=description,
        pitch_deck_url=HttpUrl(pitch_deck_url) if pitch_deck_url else None,
        founder_linkedin_urls=[
            HttpUrl(u) for u in (founder_linkedin_urls or [])
        ],
    )

    report = await asyncio.to_thread(engine.analyze, request)
    elapsed = (time.monotonic() - start) * 1000

    exec_summary = _build_executive_summary(report)
    strengths = _extract_strengths(report)
    weaknesses = _extract_weaknesses(report)
    opportunities = _extract_opportunities(report)
    risks = _extract_risks(report)
    evidence = _extract_evidence(report)
    decision_trace = _build_decision_trace(report)
    benchmark_ctx = _build_benchmark_context(report, store)
    supporting = _build_supporting_features(report)

    return DueDiligenceReportResponse(
        startup_name=startup_name,
        executive_summary=exec_summary,
        strengths=strengths,
        weaknesses=weaknesses,
        opportunities=opportunities,
        risks=risks,
        evidence=evidence,
        decision_trace=decision_trace,
        confidence=report.overall_confidence,
        supporting_features=supporting,
        benchmark_context=benchmark_ctx,
        processing_time_ms=elapsed,
        engine_version=report.analysis_metadata.engine_version,
    )


def _build_executive_summary(report: Any) -> ExecutiveSummary:
    """Build executive summary from the report."""
    decision = report.investment_decision
    synthesis = report.decision_synthesis

    findings: list[str] = []
    if synthesis and synthesis.executive_summary:
        findings.append(synthesis.executive_summary)
    if synthesis and synthesis.executive_summary_key_points:
        findings.extend(synthesis.executive_summary_key_points[:3])

    headline = (
        f"{report.startup.name}: {decision.category.value.replace('_', ' ').title()}"
        if decision
        else f"{report.startup.name} Analysis"
    )

    return ExecutiveSummary(
        headline=headline,
        overview=synthesis.executive_summary if synthesis else "",
        key_findings=findings,
        confidence_level=report.overall_confidence,
    )


def _extract_strengths(report: Any) -> list[StrengthItem]:
    """Extract strengths from the report."""
    items: list[StrengthItem] = []

    for assessment in report.dimension_assessments:
        if assessment.score is not None and assessment.score >= 60:
            items.append(
                StrengthItem(
                    title=f"{assessment.dimension}: Strong Performance",
                    description=assessment.summary,
                    dimension=assessment.dimension,
                    severity="high" if assessment.score >= 80 else "moderate",
                    evidence=[e.statement for e in assessment.supporting_evidence[:3]],
                )
            )

    if report.investment_readiness and report.investment_readiness.key_strengths:
        for s in report.investment_readiness.key_strengths[:3]:
            items.append(
                StrengthItem(
                    title=s,
                    description="",
                    dimension="readiness",
                    severity="moderate",
                )
            )

    return items


def _extract_weaknesses(report: Any) -> list[WeaknessItem]:
    """Extract weaknesses from the report."""
    items: list[WeaknessItem] = []

    for assessment in report.dimension_assessments:
        if assessment.score is not None and assessment.score < 40:
            items.append(
                WeaknessItem(
                    title=f"{assessment.dimension}: Below Threshold",
                    description=assessment.summary,
                    dimension=assessment.dimension,
                    severity="high" if assessment.score < 20 else "moderate",
                    evidence=[e.statement for e in assessment.supporting_evidence[:3]],
                )
            )

    if report.investment_readiness and report.investment_readiness.key_concerns:
        for c in report.investment_readiness.key_concerns[:3]:
            items.append(
                WeaknessItem(
                    title=c,
                    description="",
                    dimension="readiness",
                    severity="moderate",
                )
            )

    return items


def _extract_opportunities(report: Any) -> list[OpportunityItemResponse]:
    """Extract opportunities from the report."""
    items: list[OpportunityItemResponse] = []

    synthesis = report.decision_synthesis
    if synthesis and synthesis.opportunities:
        for opp in synthesis.opportunities:
            items.append(
                OpportunityItemResponse(
                    title=opp.label,
                    description="; ".join(opp.statements[:2]) if opp.statements else "",
                    dimension=opp.dimension,
                    impact=opp.impact.value,
                )
            )

    for score in report.scores:
        if score.score >= 70:
            items.append(
                OpportunityItemResponse(
                    title=f"Strong {score.dimension}",
                    description=score.rationale[:200] if score.rationale else "",
                    dimension=score.dimension,
                    impact="high",
                )
            )

    return items


def _extract_risks(report: Any) -> list[RiskItemResponse]:
    """Extract risks from the report."""
    items: list[RiskItemResponse] = []

    synthesis = report.decision_synthesis
    if synthesis and synthesis.risks:
        for risk in synthesis.risks:
            items.append(
                RiskItemResponse(
                    title=risk.label,
                    description="; ".join(risk.statements[:2]) if risk.statements else "",
                    dimension=risk.dimension,
                    severity=risk.severity.value,
                )
            )

    if report.investment_readiness and report.investment_readiness.gaps:
        for g in report.investment_readiness.gaps[:3]:
            items.append(
                RiskItemResponse(
                    title=f"Data Gap: {g}",
                    description=g,
                    dimension="data_quality",
                    severity="low",
                )
            )

    return items


def _extract_evidence(report: Any) -> list[EvidenceEntry]:
    """Extract evidence entries from the report."""
    entries: list[EvidenceEntry] = []
    for ev in report.evidence[:20]:
        entries.append(
            EvidenceEntry(
                claim=ev.statement,
                domain=ev.domain,
                category=ev.category,
                source=ev.source,
                trust_score=ev.relevance_score,
                relevance_score=ev.relevance_score,
            )
        )
    return entries


def _build_decision_trace(report: Any) -> DecisionTraceEntry:
    """Build decision trace entry from the report."""
    decision = report.investment_decision
    if not decision:
        return DecisionTraceEntry()

    rationale = decision.rationale
    return DecisionTraceEntry(
        category=decision.category.value,
        conviction=decision.conviction.value,
        composite_score=decision.composite_score,
        margin_to_next_category=decision.margin_to_next_category,
        rationale_for=rationale.primary_reasons_for,
        rationale_against=rationale.primary_reasons_against,
        missing_information=rationale.missing_information,
    )


def _build_benchmark_context(
    report: Any, store: Any = None
) -> BenchmarkContext:
    """Build benchmark context from historical data."""
    import statistics

    from predictron_engine.dataset.store import DatasetStore

    scores: list[float] = []
    category_dist: dict[str, int] = {}

    if store is not None and isinstance(store, DatasetStore):
        for rid in store.list_records():
            rec = store.load_record(rid)
            if rec:
                scores.append(rec.prediction.composite_score)
                cat = rec.prediction.decision.value
                category_dist[cat] = category_dist.get(cat, 0) + 1

    mean = statistics.mean(scores) if scores else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    percentile = 0.0
    if scores:
        below = sum(1 for s in scores if s < report.overall_score)
        percentile = (below / len(scores)) * 100.0

    return BenchmarkContext(
        composite_score=report.overall_score,
        benchmark_mean=mean,
        benchmark_std_dev=std,
        percentile_rank=percentile,
        sample_size=len(scores),
        category_distribution=category_dist,
    )


def _build_supporting_features(report: Any) -> dict[str, object]:
    """Build supporting features dictionary."""
    return {
        "dimension_scores": {
            s.dimension: s.score for s in report.scores
        },
        "overall_score": report.overall_score,
        "overall_confidence": report.overall_confidence,
        "evidence_count": len(report.evidence),
        "observation_count": len(report.observations),
        "recommendation_count": len(report.recommendations),
    }
