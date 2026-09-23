"""Official Predictron API client.

``PredictronClient`` wraps every public API endpoint behind typed, sync
resource objects:

* ``client.venture`` — venture analysis and explainability.
* ``client.portfolio`` — portfolio and similarity analysis.
* ``client.compare`` — cross-company comparison.
* ``client.due_diligence`` — due diligence report generation.
* ``client.search(...)`` / ``client.search_resource`` — unified search.
* ``client.batch`` — async batch job management and streaming.
* ``client.analyze`` — legacy single-startup analysis.
* ``client.health`` — health and readiness checks.
* ``client.companies`` — Company Intelligence Hub profile endpoints.
* ``client.monitor`` — continuous intelligence & drift detection.

The client only talks to the API. No business logic is duplicated.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from predictron_sdk.auth import (
    AuthProvider,
    BearerTokenAuth,
    NullAuth,
)
from predictron_sdk.batch import (
    DEFAULT_POLL_INTERVAL,
    batch_items_from_names,
    build_batch_items,
    items_from_requests,
    wait_for_completion,
)
from predictron_sdk.config import SDKConfig, build_config
from predictron_sdk.models import (
    AnalysisDetail,
    AnalysisList,
    AnalysisSummary,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchEvent,
    BatchItem,
    BatchJobCancelled,
    BatchJobDetail,
    BatchJobList,
    BatchJobProgress,
    BatchJobResultItem,
    BatchJobSummary,
    BatchSubmission,
    BenchmarkComparison,
    CompanyProfile,
    ComparisonRequest,
    DecisionExplain,
    DecisionTrace,
    DueDiligenceReport,
    DueDiligenceRequest,
    FeatureList,
    FullComparison,
    Health,
    KnowledgeGraphSummary,
    LearningBias,
    LearningConfidence,
    LearningKnowledge,
    LearningObservations,
    LearningPatterns,
    LearningRecommendations,
    LearningReportDetail,
    LearningReports,
    LearningSummary,
    MonitorDrift,
    MonitorHealthList,
    MonitorReanalysis,
    MonitorSummary,
    MonitorTrends,
    PortfolioAnalysis,
    PortfolioRequest,
    Readiness,
    SearchByType,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SignalAggregation,
    SignalTimeline,
    SignalTrends,
    SimilarityMatrix,
    VentureAnalysis,
    VentureRequest,
    to_payload,
)
from predictron_sdk.pagination import Page, Paginator, page_from_offset
from predictron_sdk.retry import RetryPolicy
from predictron_sdk.stream import BatchJobStream
from predictron_sdk.transport import APIResponse, Transport, parse_model

__all__ = [
    "PredictronClient",
    "VentureResource",
    "PortfolioResource",
    "CompareResource",
    "DueDiligenceResource",
    "SearchResource",
    "BatchResource",
    "AnalyzeResource",
    "HealthResource",
    "CompaniesResource",
    "MonitorResource",
    "LearningResource",
    "make_bearer_client",
]

M = TypeVar("M", bound=BaseModel)
SleepFn = Callable[[float], None]


class _Resource:
    """Base class for typed API resource wrappers."""

    def __init__(self, client: PredictronClient) -> None:
        self._client = client

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> APIResponse:
        return self._client._transport.request(
            method,
            path,
            params=params,
            json=json,
            headers=headers,
        )

    def _post_model(
        self, path: str, model: BaseModel, response_model: type[M]
    ) -> M:
        response = self._request("POST", path, json=to_payload(model))
        return parse_model(response_model, response.json)

    def _get_model(
        self,
        path: str,
        response_model: type[M],
        *,
        params: Mapping[str, str] | None = None,
    ) -> M:
        response = self._request("GET", path, params=params)
        return parse_model(response_model, response.json)

    def _resolve_venture(
        self,
        request: VentureRequest | None,
        *,
        startup_name: str | None,
        description: str | None,
        website_url: str | None,
        pitch_deck_url: str | None,
        founder_linkedin_urls: Sequence[str] | None,
    ) -> VentureRequest:
        if request is not None:
            return request
        if startup_name is None or description is None:
            raise ValueError("startup_name and description are required")
        return VentureRequest(
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=list(founder_linkedin_urls or []),
        )

    def _resolve_analyze(
        self,
        request: AnalyzeRequest | None,
        *,
        startup_name: str | None,
        description: str | None,
        website_url: str | None,
        pitch_deck_url: str | None,
        founder_linkedin_urls: Sequence[str] | None,
    ) -> AnalyzeRequest:
        if request is not None:
            return request
        if startup_name is None or description is None:
            raise ValueError("startup_name and description are required")
        return AnalyzeRequest(
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=list(founder_linkedin_urls or []),
        )

    def _resolve_portfolio(
        self,
        request: PortfolioRequest | None,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None,
        website_urls: Mapping[str, str | None] | None,
    ) -> PortfolioRequest:
        if request is not None:
            return request
        return PortfolioRequest(
            company_names=list(company_names),
            descriptions=dict(descriptions or {}),
            website_urls={k: v for k, v in (website_urls or {}).items()},
        )

    def _resolve_comparison(
        self,
        request: ComparisonRequest | None,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None,
        website_urls: Mapping[str, str | None] | None,
    ) -> ComparisonRequest:
        if request is not None:
            return request
        return ComparisonRequest(
            company_names=list(company_names),
            descriptions=dict(descriptions or {}),
            website_urls={k: v for k, v in (website_urls or {}).items()},
        )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResource(_Resource):
    """Health and readiness endpoints."""

    def check(self) -> Health:
        return self._get_model("/health", Health)

    def health(self) -> Health:
        return self.check()

    def readiness(self) -> Readiness:
        return self._get_model("/health/readiness", Readiness)


# ---------------------------------------------------------------------------
# Analyze (legacy single-startup analysis)
# ---------------------------------------------------------------------------


class AnalyzeResource(_Resource):
    """Legacy single-startup analysis endpoints."""

    def analyze(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: AnalyzeRequest | None = None,
    ) -> AnalyzeResponse:
        request = self._resolve_analyze(
            request,
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
        )
        return self._post_model("/analyze", request, AnalyzeResponse)

    def list_analyses(
        self,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> AnalysisList:
        return self._get_model(
            "/analyze",
            AnalysisList,
            params={"offset": str(offset), "limit": str(limit)},
        )

    def get(self, analysis_id: str) -> AnalysisDetail:
        return self._get_model(f"/analyze/{analysis_id}", AnalysisDetail)

    def paginate(
        self,
        *,
        limit: int = 20,
    ) -> Paginator[AnalysisSummary]:
        def fetch(token: str | int | None, page_limit: int) -> Page[AnalysisSummary]:
            offset = 0 if token is None else int(token)
            page = self.list_analyses(offset=offset, limit=page_limit)
            return page_from_offset(page.analyses, page.total, offset, page_limit)

        return Paginator(fetch, limit=limit)


# ---------------------------------------------------------------------------
# Venture analysis
# ---------------------------------------------------------------------------


class VentureResource(_Resource):
    """Venture analysis, explainability and signal endpoints."""

    def evaluate(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: VentureRequest | None = None,
    ) -> VentureAnalysis:
        request = self._resolve_venture(
            request,
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
        )
        return self._post_model("/venture", request, VentureAnalysis)

    def explain(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: VentureRequest | None = None,
    ) -> DecisionExplain:
        request = self._resolve_venture(
            request,
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
        )
        return self._post_model("/venture/explain", request, DecisionExplain)

    def trace(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: VentureRequest | None = None,
    ) -> DecisionTrace:
        request = self._resolve_venture(
            request,
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
        )
        return self._post_model("/venture/trace", request, DecisionTrace)

    def features(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: VentureRequest | None = None,
    ) -> FeatureList:
        request = self._resolve_venture(
            request,
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
        )
        return self._post_model("/venture/features", request, FeatureList)

    def benchmark(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: VentureRequest | None = None,
    ) -> BenchmarkComparison:
        request = self._resolve_venture(
            request,
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
        )
        return self._post_model("/venture/benchmark", request, BenchmarkComparison)

    def knowledge_graph(self) -> KnowledgeGraphSummary:
        return self._get_model("/venture/knowledge-graph", KnowledgeGraphSummary)

    def signals(self, company_id: str) -> SignalTimeline:
        return self._get_model(
            f"/venture/signals/{company_id}", SignalTimeline
        )

    def signal_trends(self, company_id: str) -> SignalTrends:
        return self._get_model(
            f"/venture/signals/{company_id}/trends", SignalTrends
        )

    def signal_aggregation(self, company_id: str) -> SignalAggregation:
        return self._get_model(
            f"/venture/signals/{company_id}/aggregation", SignalAggregation
        )


# ---------------------------------------------------------------------------
# Portfolio analysis
# ---------------------------------------------------------------------------


class PortfolioResource(_Resource):
    """Portfolio analysis endpoints."""

    def analyze(
        self,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None = None,
        website_urls: Mapping[str, str | None] | None = None,
        request: PortfolioRequest | None = None,
    ) -> PortfolioAnalysis:
        request = self._resolve_portfolio(
            request,
            company_names=company_names,
            descriptions=descriptions,
            website_urls=website_urls,
        )
        return self._post_model("/portfolio", request, PortfolioAnalysis)

    def similarity(
        self,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None = None,
        website_urls: Mapping[str, str | None] | None = None,
        request: PortfolioRequest | None = None,
    ) -> SimilarityMatrix:
        request = self._resolve_portfolio(
            request,
            company_names=company_names,
            descriptions=descriptions,
            website_urls=website_urls,
        )
        return self._post_model("/portfolio/similarity", request, SimilarityMatrix)


# ---------------------------------------------------------------------------
# Company comparison
# ---------------------------------------------------------------------------


class CompareResource(_Resource):
    """Cross-company comparison endpoints."""

    def compare(
        self,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None = None,
        website_urls: Mapping[str, str | None] | None = None,
        request: ComparisonRequest | None = None,
    ) -> FullComparison:
        request = self._resolve_comparison(
            request,
            company_names=company_names,
            descriptions=descriptions,
            website_urls=website_urls,
        )
        return self._post_model("/compare", request, FullComparison)


# ---------------------------------------------------------------------------
# Due diligence
# ---------------------------------------------------------------------------


class DueDiligenceResource(_Resource):
    """Due diligence report generation."""

    def generate(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: DueDiligenceRequest | None = None,
    ) -> DueDiligenceReport:
        if request is None:
            if startup_name is None or description is None:
                raise ValueError("startup_name and description are required")
            request = DueDiligenceRequest(
                startup_name=startup_name,
                description=description,
                website_url=website_url,
                pitch_deck_url=pitch_deck_url,
                founder_linkedin_urls=list(founder_linkedin_urls or []),
            )
        return self._post_model("/due-diligence", request, DueDiligenceReport)

    def report(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: DueDiligenceRequest | None = None,
    ) -> DueDiligenceReport:
        return self.generate(
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
            request=request,
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchResource(_Resource):
    """Unified search endpoints."""

    def search(
        self,
        *,
        query: str,
        search_type: str = "all",
        offset: int = 0,
        limit: int = 20,
        request: SearchRequest | None = None,
    ) -> SearchResponse:
        request = (
            request
            if request is not None
            else SearchRequest(
                query=query,
                search_type=search_type,
                offset=offset,
                limit=limit,
            )
        )
        return self._post_model("/search", request, SearchResponse)

    def get(
        self,
        *,
        query: str,
        search_type: str = "all",
        offset: int = 0,
        limit: int = 20,
    ) -> SearchResponse:
        return self._get_model(
            "/search",
            SearchResponse,
            params={
                "q": query,
                "search_type": search_type,
                "offset": str(offset),
                "limit": str(limit),
            },
        )

    def by_type(
        self,
        *,
        search_type: str,
        query: str,
        offset: int = 0,
        limit: int = 20,
    ) -> SearchByType:
        return self._get_model(
            f"/search/by-type/{search_type}",
            SearchByType,
            params={
                "q": query,
                "offset": str(offset),
                "limit": str(limit),
            },
        )

    def paginate(
        self,
        *,
        query: str,
        search_type: str = "all",
        limit: int = 20,
    ) -> Paginator[SearchResultItem]:
        def fetch(
            token: str | int | None, page_limit: int
        ) -> Page[SearchResultItem]:
            offset = 0 if token is None else int(token)
            page = self.search(
                query=query,
                search_type=search_type,
                offset=offset,
                limit=page_limit,
            )
            return page_from_offset(page.results, page.total, offset, page_limit)

        return Paginator(fetch, limit=limit)


#: Type alias so ``client.search`` stays a callable while the resource type
#: remains fully typed.
SearchResourceType = SearchResource


# ---------------------------------------------------------------------------
# Company Intelligence Hub
# ---------------------------------------------------------------------------


class CompaniesResource(_Resource):
    """Company Intelligence Hub profile endpoints.

    Read-only: ``profile`` returns the unified intelligence profile of a
    stored company with guided coverage (live registry data merged with
    grounded offline dataset intelligence).
    """

    def profile(self, company_id: str) -> CompanyProfile:
        """Fetch the unified intelligence profile for a company."""
        return self._get_model(f"/companies/{company_id}/profile", CompanyProfile)


# ---------------------------------------------------------------------------
# Continuous intelligence & drift detection (Phase 6)
# ---------------------------------------------------------------------------


class MonitorResource(_Resource):
    """Continuous intelligence & drift-detection endpoints.

    Read-only monitoring views over the live prediction ledger (summary,
    health, stale, overdue, re-analysis) plus time-series trends and drift
    over the recorded snapshot history.
    """

    def summary(self) -> MonitorSummary:
        """Live monitoring summary over your forecasts and evaluations."""
        return self._get_model("/monitor/summary", MonitorSummary)

    def rollup(self) -> MonitorSummary:
        """Rolling-window monitoring rollup (default window 30)."""
        return self._get_model("/monitor/rollup", MonitorSummary)

    def health(self) -> MonitorHealthList:
        """Derived forecast-health rows (active due overdue stale resolved)."""
        return self._get_model("/monitor/health", MonitorHealthList)

    def stale(self) -> MonitorHealthList:
        """Stale forecasts (frozen snapshot older than the named threshold)."""
        return self._get_model("/monitor/stale", MonitorHealthList)

    def overdue(self) -> MonitorHealthList:
        """Overdue forecasts (due time passed with no outcome attached)."""
        return self._get_model("/monitor/overdue", MonitorHealthList)

    def reanalysis(self) -> MonitorReanalysis:
        """Deterministic re-analysis recommendations for your scope."""
        return self._get_model("/monitor/reanalysis", MonitorReanalysis)

    def trends(self, *, period: str = "daily") -> MonitorTrends:
        """Dashboard time-series over the recorded snapshot history."""
        return self._get_model(
            "/monitor/trends",
            MonitorTrends,
            params={"period": period},
        )

    def drift(
        self,
        *,
        before_id: str | None = None,
        after_id: str | None = None,
        period: str = "daily",
    ) -> MonitorDrift:
        """Deterministic drift report between two recorded snapshots.

        Without ids the latest two ``period`` snapshots are compared.
        """
        params = {"period": period}
        if before_id is not None:
            params["before_id"] = before_id
        if after_id is not None:
            params["after_id"] = after_id
        return self._get_model("/monitor/drift", MonitorDrift, params=params)


# ---------------------------------------------------------------------------
# Continuous learning intelligence (Phase 7)
# ---------------------------------------------------------------------------


class LearningResource(_Resource):
    """Continuous learning intelligence endpoints.

    Read-only learning views over the live evaluated prediction ledger
    (summary, patterns, observations, confidence, bias, per-dimension
    knowledge, recommendations) plus recorded learning-report history.
    """

    def summary(self) -> LearningSummary:
        """Live learning summary over your evaluated population."""
        return self._get_model("/learning/summary", LearningSummary)

    def patterns(self) -> LearningPatterns:
        """Per-dimension cohort patterns over your evaluated population."""
        return self._get_model("/learning/patterns", LearningPatterns)

    def observations(self) -> LearningObservations:
        """Canonical observations about your evaluated population."""
        return self._get_model("/learning/observations", LearningObservations)

    def confidence(self) -> LearningConfidence:
        """Population confidence statistics and calibration digest."""
        return self._get_model("/learning/confidence", LearningConfidence)

    def bias(self) -> LearningBias:
        """Confidence-bias view plus the population metric surface."""
        return self._get_model("/learning/bias", LearningBias)

    def recommendations(self) -> LearningRecommendations:
        """Deterministic rule-based recommendations for your population."""
        return self._get_model("/learning/recommendations", LearningRecommendations)

    def knowledge(self, dimension: str) -> LearningKnowledge:
        """Per-dimension knowledge (sector/stage/country/technology/...)."""
        return self._get_model(
            f"/learning/knowledge/{dimension}", LearningKnowledge
        )

    def sector(self) -> LearningKnowledge:
        """Sector knowledge (shortcut for ``knowledge("sector")``)."""
        return self._get_model("/learning/sector", LearningKnowledge)

    def technology(self) -> LearningKnowledge:
        """Technology knowledge."""
        return self._get_model("/learning/technology", LearningKnowledge)

    def country(self) -> LearningKnowledge:
        """Country knowledge."""
        return self._get_model("/learning/country", LearningKnowledge)

    def stage(self) -> LearningKnowledge:
        """Stage knowledge."""
        return self._get_model("/learning/stage", LearningKnowledge)

    def founders(self) -> LearningKnowledge:
        """Founder knowledge."""
        return self._get_model("/learning/founders", LearningKnowledge)

    def business_model(self) -> LearningKnowledge:
        """Business-model knowledge."""
        return self._get_model("/learning/business-model", LearningKnowledge)

    def reports(self, *, period: str = "daily") -> LearningReports:
        """Recorded learning snapshot headers."""
        return self._get_model(
            "/learning/reports",
            LearningReports,
            params={"period": period},
        )

    def report(self, snapshot_id: str) -> LearningReportDetail:
        """Full recorded learning report for a snapshot id."""
        return self._get_model(
            f"/learning/reports/{snapshot_id}", LearningReportDetail
        )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class BatchResource(_Resource):
    """Async batch job management endpoints."""

    def submit(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        job_name: str = "",
    ) -> BatchJobSummary:
        submission = BatchSubmission(
            job_name=job_name, items=[dict(item) for item in items]
        )
        return self._post_model("/batch", submission, BatchJobSummary)

    def submit_typed(
        self,
        items: Sequence[BatchItem],
        *,
        job_name: str = "",
    ) -> BatchJobSummary:
        return self.submit(build_batch_items(items), job_name=job_name)

    def evaluate(
        self,
        startup_names: Sequence[str],
        *,
        descriptions: Mapping[str, str] | None = None,
        website_urls: Mapping[str, str | None] | None = None,
        job_name: str = "",
    ) -> BatchJobSummary:
        items = batch_items_from_names(
            startup_names,
            descriptions=descriptions,
            website_urls=website_urls,
        )
        return self.submit(items, job_name=job_name)

    def from_requests(
        self,
        requests: Sequence[AnalyzeRequest | VentureRequest],
        *,
        job_name: str = "",
    ) -> BatchJobSummary:
        payload_items = items_from_requests(requests)
        return self.submit(payload_items, job_name=job_name)

    def list_jobs(
        self,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> BatchJobList:
        return self._get_model(
            "/batch",
            BatchJobList,
            params={"offset": str(offset), "limit": str(limit)},
        )

    def status(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> BatchJobDetail:
        return self._get_model(
            f"/batch/{job_id}",
            BatchJobDetail,
            params={"offset": str(offset), "limit": str(limit)},
        )

    def progress(self, job_id: str) -> BatchJobProgress:
        return self._get_model(
            f"/batch/{job_id}/progress", BatchJobProgress
        )

    def cancel(self, job_id: str) -> BatchJobCancelled:
        return parse_model(
            BatchJobCancelled,
            self._request("POST", f"/batch/{job_id}/cancel").json,
        )

    def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        timeout: float | None = None,
    ) -> BatchJobDetail:
        wait_for_completion(
            poll=lambda: self.progress(job_id),
            poll_interval=poll_interval,
            timeout=timeout,
            sleep_fn=self._client._sleep_fn,
        )
        return self.status(job_id)

    def stream(
        self,
        job_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> Iterator[BatchEvent]:
        stream = BatchJobStream(
            poll=lambda: self.progress(job_id),
            poll_interval=poll_interval,
            sleep_fn=self._client._sleep_fn,
        )
        return stream.events()

    def result_pages(
        self,
        job_id: str,
        *,
        limit: int = 100,
    ) -> Paginator[BatchJobResultItem]:
        def fetch(
            token: str | int | None, page_limit: int
        ) -> Page[BatchJobResultItem]:
            offset = 0 if token is None else int(token)
            detail = self.status(job_id, offset=offset, limit=page_limit)
            has_more = detail.cursor is not None
            return Page(
                items=detail.results,
                total=detail.total_items,
                limit=page_limit,
                has_more=has_more,
                next_token=detail.cursor,
            )

        return Paginator(fetch, limit=limit)

    def results(
        self,
        job_id: str,
        *,
        limit: int = 100,
    ) -> list[BatchJobResultItem]:
        return self.result_pages(job_id, limit=limit).all()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PredictronClient:
    """Synchronous client for the Predictron API.

    Arguments may also be supplied via environment variables
    (``PREDICTRON_API_KEY``, ``PREDICTRON_BASE_URL``, ...).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_policy: RetryPolicy | None = None,
        auth: AuthProvider | None = None,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
        config: SDKConfig | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config if config is not None else build_config(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            additional_headers=dict(headers or {}) if headers is not None else None,
        )
        if auth is None and self._config.api_key:
            auth = BearerTokenAuth(self._config.api_key)
        self._auth = auth if auth is not None else NullAuth()
        self._sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        self._retry_policy = (
            retry_policy
            if retry_policy is not None
            else self._config.retry_policy
        )
        self._transport = Transport(
            config=self._config,
            auth=self._auth,
            retry_policy=self._retry_policy,
            http_client=http_client,
            transport=transport,
            sleep_fn=self._sleep_fn,
        )
        self._venture = VentureResource(self)
        self._portfolio = PortfolioResource(self)
        self._compare = CompareResource(self)
        self._due_diligence = DueDiligenceResource(self)
        self._search = SearchResource(self)
        self._batch = BatchResource(self)
        self._analyze = AnalyzeResource(self)
        self._health = HealthResource(self)
        self._companies = CompaniesResource(self)
        self._monitor = MonitorResource(self)
        self._learning = LearningResource(self)

    # --- resources -------------------------------------------------------

    @property
    def venture(self) -> VentureResource:
        return self._venture

    @property
    def portfolio(self) -> PortfolioResource:
        return self._portfolio

    @property
    def comparison(self) -> CompareResource:
        return self._compare

    @property
    def due_diligence(self) -> DueDiligenceResource:
        return self._due_diligence

    @property
    def batch(self) -> BatchResource:
        return self._batch

    @property
    def analyze(self) -> AnalyzeResource:
        return self._analyze

    @property
    def health(self) -> HealthResource:
        return self._health

    @property
    def companies(self) -> CompaniesResource:
        """Company Intelligence Hub profile endpoints."""
        return self._companies

    @property
    def monitor(self) -> MonitorResource:
        """Continuous intelligence & drift detection endpoints."""
        return self._monitor

    @property
    def learning(self) -> LearningResource:
        """Continuous learning intelligence endpoints."""
        return self._learning

    @property
    def search_resource(self) -> SearchResource:
        """Resource-style search interface (for pagination helpers)."""
        return self._search

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def api_prefix(self) -> str:
        return self._config.api_prefix

    @property
    def auth(self) -> AuthProvider:
        return self._auth

    # --- top-level convenience methods ----------------------------------

    def evaluate(
        self,
        *,
        startup_name: str | None = None,
        description: str | None = None,
        website_url: str | None = None,
        pitch_deck_url: str | None = None,
        founder_linkedin_urls: Sequence[str] | None = None,
        request: VentureRequest | None = None,
    ) -> VentureAnalysis:
        """Analyze a single startup (alias of ``venture.evaluate``)."""
        return self._venture.evaluate(
            startup_name=startup_name,
            description=description,
            website_url=website_url,
            pitch_deck_url=pitch_deck_url,
            founder_linkedin_urls=founder_linkedin_urls,
            request=request,
        )

    def search(
        self,
        *,
        query: str,
        search_type: str = "all",
        offset: int = 0,
        limit: int = 20,
        request: SearchRequest | None = None,
    ) -> SearchResponse:
        """Unified search (alias of ``search.search``)."""
        return self._search.search(
            query=query,
            search_type=search_type,
            offset=offset,
            limit=limit,
            request=request,
        )

    def search_get(
        self,
        *,
        query: str,
        search_type: str = "all",
        offset: int = 0,
        limit: int = 20,
    ) -> SearchResponse:
        return self._search.get(
            query=query,
            search_type=search_type,
            offset=offset,
            limit=limit,
        )

    def search_by_type(
        self,
        *,
        search_type: str,
        query: str,
        offset: int = 0,
        limit: int = 20,
    ) -> SearchByType:
        return self._search.by_type(
            search_type=search_type,
            query=query,
            offset=offset,
            limit=limit,
        )

    def compare(
        self,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None = None,
        website_urls: Mapping[str, str | None] | None = None,
        request: ComparisonRequest | None = None,
    ) -> FullComparison:
        return self._compare.compare(
            company_names=company_names,
            descriptions=descriptions,
            website_urls=website_urls,
            request=request,
        )

    def portfolio_analyze(
        self,
        *,
        company_names: Sequence[str],
        descriptions: Mapping[str, str] | None = None,
        website_urls: Mapping[str, str | None] | None = None,
        request: PortfolioRequest | None = None,
    ) -> PortfolioAnalysis:
        """Analyze a portfolio (alias of ``portfolio.analyze``)."""
        return self._portfolio.analyze(
            company_names=company_names,
            descriptions=descriptions,
            website_urls=website_urls,
            request=request,
        )

    def health_status(self) -> Health:
        """Check the API health (alias of ``health.check``)."""
        return self._health.check()

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._transport.close()

    def __enter__(self) -> PredictronClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def from_env(cls, **kwargs: Any) -> PredictronClient:
        """Build a client using environment-based configuration."""
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"<PredictronClient base_url={self._config.base_url!r} "
            f"auth={type(self._auth).__name__}>"
        )


def make_bearer_client(
    api_key: str,
    *,
    base_url: str | None = None,
    **kwargs: Any,
) -> PredictronClient:
    """Create a client authenticated with a bearer token API key."""
    return PredictronClient(api_key=api_key, base_url=base_url, **kwargs)
