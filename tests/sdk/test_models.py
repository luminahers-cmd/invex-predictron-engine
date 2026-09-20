"""Tests for SDK models — construction, validation and defaults."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from predictron_sdk.models import (
    AnalysisDetail,
    AnalysisList,
    AnalysisScores,
    AnalysisSummary,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchEvent,
    BatchItem,
    BatchJobCancelled,
    BatchJobDetail,
    BatchJobProgress,
    BatchJobResultItem,
    BatchJobSummary,
    BatchSubmission,
    BenchmarkComparison,
    BenchmarkComparisonDetail,
    BenchmarkContext,
    CompanySearchResult,
    ComparisonRequest,
    Contribution,
    ContributionDiffItem,
    DecisionExplain,
    DecisionTrace,
    DecisionTraceEntry,
    DimensionScore,
    DueDiligenceReport,
    DueDiligenceRequest,
    EvidenceEntry,
    ExecutiveSummary,
    Explanation,
    FeatureDiff,
    FeatureList,
    FeatureSnapshot,
    FullComparison,
    GraphNodeSearchResult,
    GroundTruthEntry,
    Health,
    HeatmapCell,
    JobStatus,
    KnowledgeGraphDiffs,
    KnowledgeGraphSummary,
    PortfolioAnalysis,
    PortfolioCompanyScore,
    PortfolioComparisonItem,
    PortfolioRequest,
    Readiness,
    RiskItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SectorDistribution,
    Signal,
    SignalAggregation,
    SignalSearchResult,
    SignalTimeline,
    SignalTrends,
    SimilarityMatrix,
    Strength,
    TrendResult,
    VentureAnalysis,
    VentureRequest,
)


def test_analyze_request_payload_shape() -> None:
    request = AnalyzeRequest(
        startup_name="A",
        description="A long enough description.",
    )
    assert request.startup_name == "A"
    assert request.founder_linkedin_urls == []


def test_analyze_request_requires_description_length() -> None:
    with pytest.raises(PydanticValidationError):
        AnalyzeRequest(startup_name="A", description="short")


def test_analyze_response_full() -> None:
    payload = {
        "id": "x",
        "startup_name": "A",
        "venture_score": 10,
        "market_score": 20,
        "founder_score": 30,
        "traction_score": 40,
        "recommendations": ["r"],
        "confidence": 0.5,
    }
    model = AnalyzeResponse.model_validate(payload)
    assert model.venture_score == 10
    assert model.confidence == 0.5


def test_analyze_scores_constraints() -> None:
    with pytest.raises(PydanticValidationError):
        AnalysisScores(
            venture_score=101,
            market_score=1,
            founder_score=1,
            traction_score=1,
        )


def test_analysis_summary_created_at_datetime() -> None:
    model = AnalysisSummary(
        id="1",
        startup_name="A",
        venture_score=10,
        market_score=10,
        founder_score=10,
        traction_score=10,
        confidence=0.5,
        created_at="2026-01-01T00:00:00Z",
    )
    assert isinstance(model.created_at, datetime)


def test_analysis_list_defaults() -> None:
    assert AnalysisList().total == 0
    assert AnalysisList().analyses == []


def test_analysis_detail_full_report_default() -> None:
    assert AnalysisDetail.model_validate(
        {
            "id": "1",
            "startup_name": "A",
            "website": "https://a.example.com",
            "description": "desc",
            "venture_score": 1,
            "market_score": 1,
            "founder_score": 1,
            "traction_score": 1,
            "confidence": 0.5,
            "created_at": "2026-01-01T00:00:00Z",
        }
    ).full_report == {}


def test_venture_request_fields() -> None:
    request = VentureRequest(
        startup_name="N",
        description="A sufficiently long description.",
        website_url="https://w.example.com",
        founder_linkedin_urls=["https://li.com/a"],
    )
    assert request.website_url == "https://w.example.com"
    assert request.founder_linkedin_urls == ["https://li.com/a"]


def test_venture_analysis_dimension_scores() -> None:
    model = VentureAnalysis.model_validate(
        {
            "startup_name": "A",
            "overall_score": 70.0,
            "overall_confidence": 0.7,
            "dimension_scores": [
                {"dimension": "market", "score": 80.0, "rationale": "big"}
            ],
        }
    )
    assert model.dimension_scores[0].dimension == "market"
    assert model.dimension_scores[0].score == 80.0


def test_dimension_score_bounds() -> None:
    with pytest.raises(PydanticValidationError):
        DimensionScore(dimension="d", score=120.0)


def test_venture_analysis_created_at_parsing() -> None:
    model = VentureAnalysis.model_validate(
        {
            "startup_name": "A",
            "overall_score": 1,
            "overall_confidence": 0.1,
            "created_at": "2026-03-01T10:00:00+00:00",
        }
    )
    assert model.created_at is not None


def test_explanation_defaults() -> None:
    model = Explanation(company_id="c", headline="h")
    assert model.strengths == []
    assert model.weaknesses == []
    assert model.neutral_factors == []


def test_contribution_any_value() -> None:
    model = Contribution.model_validate(
        {
            "feature_id": "f",
            "feature_name": "n",
            "category": "c",
            "contribution_type": "t",
            "raw_value": {"nested": [1, 2]},
        }
    )
    assert model.raw_value == {"nested": [1, 2]}


def test_decision_explain_defaults() -> None:
    assert DecisionExplain(company_id="c").positive_contributions == []


def test_trace_node_edges() -> None:
    trace = DecisionTrace.model_validate(
        {
            "company_id": "c",
            "trace_id": "t",
            "nodes": [],
            "edges": [["a", "b"]],
        }
    )
    assert trace.edges == [["a", "b"]]
    assert trace.node_count == 0


def test_feature_list_count() -> None:
    model = FeatureList(
        company_id="c",
        features=[FeatureSnapshot(feature_id="f", feature_name="n", category="c")],
        feature_count=1,
    )
    assert model.feature_count == 1


def test_knowledge_graph_defaults() -> None:
    model = KnowledgeGraphSummary()
    assert model.node_count == 0
    assert model.nodes == []


def test_signal_timeline_counts() -> None:
    model = SignalTimeline(
        company_id="c",
        signals=[Signal(signal_id="s1", company_id="c", signal_type="hiring")],
        signal_count=1,
    )
    assert model.signal_count == 1


def test_signal_confidence_default() -> None:
    assert Signal().confidence == 1.0


def test_trend_result_required_fields() -> None:
    with pytest.raises(PydanticValidationError):
        TrendResult(name="t", value=1.0, direction="up")


def test_signal_trends_mapping() -> None:
    model = SignalTrends.model_validate(
        {
            "company_id": "c",
            "trends": {
                "hiring": {
                    "name": "hiring",
                    "value": 2.0,
                    "direction": "up",
                    "window_days": 90,
                    "explanation": "x",
                }
            },
        }
    )
    assert model.trends["hiring"].value == 2.0


def test_signal_aggregation_defaults() -> None:
    model = SignalAggregation(company_id="c")
    assert model.momentum_score == 0.0


def test_benchmark_comparison() -> None:
    model = BenchmarkComparison(startup_name="A", sample_size=10)
    assert model.sample_size == 10
    assert model.composite_score == 0.0


def test_ground_truth_entry_optional() -> None:
    model = GroundTruthEntry(startup_name="A")
    assert model.decision_match is None


def test_portfolio_request_min_length() -> None:
    with pytest.raises(PydanticValidationError):
        PortfolioRequest(company_names=["only-one"])


def test_portfolio_analysis_nested_defaults() -> None:
    model = PortfolioAnalysis(companies=[])
    assert model.risk_summary.high_risk_count == 0
    assert model.diversification.overall_diversification == 0.0


def test_portfolio_company_score_bounds() -> None:
    with pytest.raises(PydanticValidationError):
        PortfolioCompanyScore(startup_name="A", overall_score=200, overall_confidence=0.5)


def test_sector_distribution_requires_fields() -> None:
    with pytest.raises(PydanticValidationError):
        SectorDistribution(sector="s", count=1)


def test_heatmap_cell() -> None:
    cell = HeatmapCell(row_label="A", col_label="B", value=0.7)
    assert cell.label == ""


def test_similarity_matrix() -> None:
    model = SimilarityMatrix(companies=["A", "B"], matrix=[[1.0, 0.5], [0.5, 1.0]])
    assert model.method == "score_cosine"
    assert len(model.matrix) == 2


def test_comparison_request_max_length() -> None:
    with pytest.raises(PydanticValidationError):
        ComparisonRequest(company_names=[f"c{i}" for i in range(11)])


def test_full_comparison_nested_defaults() -> None:
    model = FullComparison()
    assert model.feature_diffs.companies == []
    assert model.overall_summary == {}


def test_feature_diff_any_values() -> None:
    model = FeatureDiff(feature="brand")
    assert model.company_a is None
    assert model.direction == "equal"


def test_contribution_diff_item() -> None:
    model = ContributionDiffItem(feature="f", contributions={"A": 1.0})
    assert model.max_contribution == 0.0


def test_knowledge_graph_diffs() -> None:
    model = KnowledgeGraphDiffs(companies=["A"])
    assert model.graph_summary == {}


def test_benchmark_comparison_detail() -> None:
    model = BenchmarkComparisonDetail(companies=["A", "B"])
    assert model.benchmark_metrics == []


def test_due_diligence_request() -> None:
    request = DueDiligenceRequest(startup_name="A", description="A sufficiently long one.")
    assert request.founder_linkedin_urls == []


def test_executive_summary_defaults() -> None:
    model = ExecutiveSummary()
    assert model.headline == ""
    assert model.confidence_level == 0.0


def test_strength_defaults() -> None:
    model = Strength(title="t")
    assert model.severity == "moderate"
    assert model.evidence == []


def test_risk_item_mitigation_default() -> None:
    model = RiskItem(title="r", severity="high")
    assert model.mitigation == ""


def test_evidence_entry_scores() -> None:
    model = EvidenceEntry(claim="c")
    assert model.trust_score == 0.0
    assert model.relevance_score == 1.0


def test_decision_trace_entry_defaults() -> None:
    model = DecisionTraceEntry()
    assert model.rationale_for == []


def test_benchmark_context_defaults() -> None:
    model = BenchmarkContext()
    assert model.sample_size == 0


def test_due_diligence_report_nested() -> None:
    model = DueDiligenceReport(startup_name="A")
    assert model.confidence == 0.0
    assert model.benchmark_context.composite_score == 0.0


def test_search_request_bounds() -> None:
    with pytest.raises(PydanticValidationError):
        SearchRequest(query="")
    with pytest.raises(PydanticValidationError):
        SearchRequest(query="q", limit=500)


def test_search_result_item_required_fields() -> None:
    model = SearchResultItem(result_type="company", id="1")
    assert model.score == 0.0


def test_search_response_defaults() -> None:
    model = SearchResponse()
    assert model.total == 0
    assert model.results == []


def test_company_search_result_defaults() -> None:
    model = CompanySearchResult()
    assert model.industries == []
    assert model.founded_year is None


def test_signal_search_result_confidence() -> None:
    assert SignalSearchResult().confidence == 1.0


def test_graph_node_search_result() -> None:
    model = GraphNodeSearchResult(node_id="n")
    assert model.connected_companies == []


def test_job_status_values() -> None:
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.CANCELLED.value == "cancelled"


def test_batch_item_min_desc() -> None:
    with pytest.raises(PydanticValidationError):
        BatchItem(startup_name="A", description="short")


def test_batch_submission_items_required() -> None:
    with pytest.raises(PydanticValidationError):
        BatchSubmission(job_name="j", items=[])


def test_batch_summary_status_enum() -> None:
    model = BatchJobSummary(job_id="1", status="running")
    assert model.status is JobStatus.RUNNING


def test_batch_job_result_item_enum() -> None:
    model = BatchJobResultItem(index=0, status="failed")
    assert model.status is JobStatus.FAILED


def test_batch_job_detail_cursor() -> None:
    model = BatchJobDetail(job_id="1", cursor="next-page")
    assert model.cursor == "next-page"


def test_batch_job_progress_defaults() -> None:
    model = BatchJobProgress(job_id="1")
    assert model.progress_pct == 0.0
    assert model.current_item is None


def test_batch_job_cancelled() -> None:
    model = BatchJobCancelled(job_id="1")
    assert model.status == "cancelled"


def test_batch_event_defaults() -> None:
    assert BatchEvent().event_type == "progress"
    assert BatchEvent().data == {}


def test_health_model() -> None:
    model = Health(status="degraded")
    assert model.startup_state == "unknown"
    assert model.engine_reachable is False


def test_readiness_model() -> None:
    model = Readiness(status="ready")
    assert model.db_healthy is False


def test_to_payload_round_trip() -> None:
    from predictron_sdk.models import to_payload

    request = VentureRequest(
        startup_name="A",
        description="A sufficiently long description.",
        website_url=None,
    )
    payload = to_payload(request)
    assert payload["startup_name"] == "A"
    assert "website_url" not in payload  # excluded when None


def test_models_ignore_extra_fields() -> None:
    model = AnalyzeResponse.model_validate(
        {
            "startup_name": "A",
            "venture_score": 1,
            "market_score": 1,
            "founder_score": 1,
            "traction_score": 1,
            "confidence": 0.5,
            "some_new_field": "ignored",
        }
    )
    assert model.startup_name == "A"


def test_portfolio_comparison_item() -> None:
    model = PortfolioComparisonItem(startup_name="A")
    assert model.dimension_scores == {}


@pytest.mark.parametrize(
    "cls,field",
    [
        (SignalTimeline, "company_id"),
        (SignalAggregation, "company_id"),
        (SignalTrends, "company_id"),
        (BatchJobSummary, "job_id"),
        (BatchJobDetail, "job_id"),
        (BatchJobProgress, "job_id"),
        (Explanation, "company_id"),
        (DecisionExplain, "company_id"),
        (DecisionTrace, "company_id"),
        (FeatureList, "company_id"),
        (BenchmarkComparison, "startup_name"),
    ],
)
def test_required_string_field(cls, field) -> None:
    with pytest.raises(PydanticValidationError):
        cls.model_validate({})


def test_optional_defaults_never_none_where_set() -> None:
    model = Health()
    assert model.status == "ok"
    assert model.startup_state == "unknown"
