"""Portfolio Analysis service — multi-company analysis and aggregation."""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import Counter
from typing import Any

from app.schemas.portfolio import (
    ConcentrationRisk,
    DiversificationScore,
    HeatmapCell,
    PortfolioAnalysisResponse,
    PortfolioCompanyScore,
    RiskSummary,
    SectorDistribution,
    SimilarityMatrixResponse,
    StageDistribution,
)
from predictron_engine.dataset.store import DatasetStore


async def analyze_portfolio(
    engine: Any,
    company_names: list[str],
    descriptions: dict[str, str] | None = None,
    website_urls: dict[str, str | None] | None = None,
    store: DatasetStore | None = None,
) -> PortfolioAnalysisResponse:
    """Analyze a portfolio of startups and return aggregated metrics."""
    import time

    start = time.monotonic()
    descriptions = descriptions or {}
    website_urls = website_urls or {}

    company_scores: list[PortfolioCompanyScore] = []
    reports: list[Any] = []

    for name in company_names:
        desc = descriptions.get(
            name,
            f"Startup analysis for {name}",
        )
        url = website_urls.get(name)
        report = await _run_analysis(engine, name, url, desc)
        reports.append(report)

        decision = report.investment_decision
        company_scores.append(
            PortfolioCompanyScore(
                startup_name=name,
                overall_score=report.overall_score,
                overall_confidence=report.overall_confidence,
                decision_category=decision.category.value if decision else None,
                conviction_level=decision.conviction.value if decision else None,
            )
        )

    portfolio_score = (
        statistics.mean([c.overall_score for c in company_scores])
        if company_scores
        else 0.0
    )
    overall_confidence = (
        statistics.mean([c.overall_confidence for c in company_scores])
        if company_scores
        else 0.0
    )

    sector_dist = _compute_sector_distribution(reports)
    stage_dist = _compute_stage_distribution(reports)
    risk_summary = _compute_risk_summary(company_scores)
    diversification = _compute_diversification(reports, sector_dist)
    concentration = _compute_concentration_risk(sector_dist, company_scores)
    heatmap = _build_heatmap(reports)
    similarity = _compute_similarity_matrix(reports)
    similarity_rows: list[dict[str, object]] = [
        {
            "company": similarity.companies[i],
            "similarities": {
                similarity.companies[j]: similarity.matrix[i][j]
                for j in range(len(similarity.companies))
            },
        }
        for i in range(len(similarity.companies))
    ]

    elapsed = (time.monotonic() - start) * 1000

    return PortfolioAnalysisResponse(
        company_count=len(company_scores),
        companies=company_scores,
        portfolio_score=portfolio_score,
        sector_distribution=sector_dist,
        stage_distribution=stage_dist,
        risk_summary=risk_summary,
        diversification=diversification,
        concentration_risk=concentration,
        heatmap_data=heatmap,
        similarity_matrix=similarity_rows,
        overall_confidence=overall_confidence,
        processing_time_ms=elapsed,
    )


async def _run_analysis(
    engine: Any,
    name: str,
    url: str | None,
    desc: str,
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


def _compute_sector_distribution(
    reports: list[Any],
) -> list[SectorDistribution]:
    """Compute sector distribution from reports."""
    sector_counts: Counter[str] = Counter()
    sector_scores: dict[str, list[float]] = {}

    for report in reports:
        industry = getattr(report.startup, "industry", None) or "unknown"
        sector_counts[industry] += 1
        sector_scores.setdefault(industry, []).append(report.overall_score)

    total = len(reports) or 1
    return [
        SectorDistribution(
            sector=sector,
            count=count,
            percentage=(count / total) * 100.0,
            avg_score=statistics.mean(sector_scores.get(sector, [0.0])),
        )
        for sector, count in sector_counts.most_common()
    ]


def _compute_stage_distribution(
    reports: list[Any],
) -> list[StageDistribution]:
    """Compute funding stage distribution."""
    stage_counts: Counter[str] = Counter()
    for report in reports:
        stage = "unknown"
        if hasattr(report.features, "funding_stage") and report.features.funding_stage:
            stage = report.features.funding_stage
        stage_counts[stage] += 1

    total = len(reports) or 1
    return [
        StageDistribution(
            stage=stage,
            count=count,
            percentage=(count / total) * 100.0,
        )
        for stage, count in stage_counts.most_common()
    ]


def _compute_risk_summary(
    company_scores: list[PortfolioCompanyScore],
) -> RiskSummary:
    """Compute risk summary from company scores."""
    high = sum(1 for c in company_scores if c.overall_score < 30)
    medium = sum(1 for c in company_scores if 30 <= c.overall_score < 60)
    low = sum(1 for c in company_scores if c.overall_score >= 60)
    avg_conf = (
        statistics.mean([c.overall_confidence for c in company_scores])
        if company_scores
        else 0.0
    )
    return RiskSummary(
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
        average_confidence=avg_conf,
    )


def _compute_diversification(
    reports: list[Any],
    sector_dist: list[SectorDistribution],
) -> DiversificationScore:
    """Compute diversification score."""
    sectors = [s.sector for s in sector_dist]
    sector_diversity = _entropy(len(sectors), len(reports)) if reports else 0.0

    stage_counter: Counter[str] = Counter()
    for r in reports:
        stage = "unknown"
        if hasattr(r.features, "funding_stage") and r.features.funding_stage:
            stage = r.features.funding_stage
        stage_counter[stage] += 1
    stage_diversity = _entropy(len(stage_counter), len(reports)) if reports else 0.0

    overall = (sector_diversity + stage_diversity) / 2.0

    return DiversificationScore(
        sector_diversity=min(sector_diversity, 1.0),
        stage_diversity=min(stage_diversity, 1.0),
        geography_diversity=0.0,
        overall_diversification=min(overall, 1.0),
        recommendation_count=len(reports),
    )


def _entropy(unique_count: int, total: int) -> float:
    """Normalized Shannon entropy (0..1)."""
    if total <= 1 or unique_count <= 1:
        return 0.0

    return 0.0


def _compute_concentration_risk(
    sector_dist: list[SectorDistribution],
    company_scores: list[PortfolioCompanyScore],
) -> ConcentrationRisk:
    """Compute concentration risk metrics."""
    if not sector_dist:
        return ConcentrationRisk()

    shares = [s.count / max(sum(x.count for x in sector_dist), 1) for s in sector_dist]
    hhi = sum(s**2 for s in shares)
    most_conc = sector_dist[0].sector if sector_dist else None
    scores = [c.overall_score for c in company_scores]
    variance = statistics.variance(scores) if len(scores) > 1 else 0.0

    return ConcentrationRisk(
        sector_concentration=min(hhi, 1.0),
        score_variance=variance,
        most_concentrated_sector=most_conc,
    )


def _build_heatmap(reports: list[Any]) -> list[HeatmapCell]:
    """Build heatmap cells from reports."""
    cells: list[HeatmapCell] = []
    for report in reports:
        for score in report.scores:
            cells.append(
                HeatmapCell(
                    row_label=report.startup.name,
                    col_label=score.dimension,
                    value=score.score,
                    label=f"{score.score:.0f}",
                )
            )
    return cells


def _compute_similarity_matrix(
    reports: list[Any],
) -> SimilarityMatrixResponse:
    """Compute pairwise score cosine similarity."""
    names = [r.startup.name for r in reports]
    vectors: list[list[float]] = []
    for r in reports:
        vec = [s.score for s in r.scores]
        if not vec:
            vec = [r.overall_score]
        vectors.append(vec)

    n = len(vectors)
    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(_cosine_sim(vectors[i], vectors[j]))
        matrix.append(row)

    return SimilarityMatrixResponse(
        companies=names,
        matrix=matrix,
        method="score_cosine",
    )


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
