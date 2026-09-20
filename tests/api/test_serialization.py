"""Schema serialization and validation tests."""

from __future__ import annotations

from app.schemas.batch import (
    BatchJobListResponse,
    BatchJobProgressResponse,
    BatchJobRequest,
    BatchJobResultItem,
    BatchJobSummary,
    JobStatus,
)
from app.schemas.comparison import (
    ComparisonRequest,
    ContributionDiffItem,
    FeatureDiff,
    FullComparisonResponse,
)
from app.schemas.due_diligence import (
    BenchmarkContext,
    DecisionTraceEntry,
    DueDiligenceReportResponse,
    DueDiligenceRequest,
    EvidenceEntry,
    ExecutiveSummary,
    OpportunityItemResponse,
    RiskItemResponse,
    StrengthItem,
    WeaknessItem,
)
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
from app.schemas.search import (
    CompanySearchResult,
    KnowledgeGraphNodeSearchResult,
    SearchByTypeResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SignalSearchResult,
)
from app.schemas.venture import (
    BenchmarkComparisonResponse,
    DimensionScore,
    ExplanationResponse,
    FeatureSnapshotResponse,
    KnowledgeGraphSummaryResponse,
    SignalTimelineResponse,
    TraceNodeResponse,
    VentureAnalysisRequest,
    VentureAnalysisResponse,
)


class TestVentureSchemaSerialization:
    """Verify Venture API schema serialization and deserialization."""

    def test_venture_request_roundtrip(self):
        req = VentureAnalysisRequest(
            startup_name="TestCo",
            description="A test company description for schema tests.",
            website_url="https://test.example.com",
        )
        d = req.model_dump()
        req2 = VentureAnalysisRequest.model_validate(d)
        assert req2.startup_name == "TestCo"

    def test_venture_response_json_roundtrip(self):
        resp = VentureAnalysisResponse(
            startup_name="X",
            overall_score=75.0,
            overall_confidence=0.85,
            dimension_scores=[
                DimensionScore(dimension="market", score=80.0, rationale="Strong"),
            ],
            decision_category="invest",
            conviction_level="high",
        )
        j = resp.model_dump_json()
        resp2 = VentureAnalysisResponse.model_validate_json(j)
        assert resp2.overall_score == 75.0

    def test_explanation_response_defaults(self):
        exp = ExplanationResponse(company_id="co_1", headline="Strong")
        assert exp.strengths == []
        assert exp.weaknesses == []
        assert exp.confidence_factors == []

    def test_feature_snapshot_response(self):
        fs = FeatureSnapshotResponse(
            feature_id="f1",
            feature_name="revenue",
            category="company",
            value=1000000,
            value_type="float",
            status="computed",
        )
        assert fs.feature_id == "f1"

    def test_trace_node_response(self):
        tn = TraceNodeResponse(
            node_id="n1",
            node_type="RAW_FEATURE",
            label="revenue feature",
        )
        assert tn.node_id == "n1"
        assert tn.input_node_ids == []

    def test_signal_timeline_response_defaults(self):
        st = SignalTimelineResponse(company_id="co1")
        assert st.signal_count == 0
        assert st.signals == []

    def test_knowledge_graph_summary_response_defaults(self):
        kg = KnowledgeGraphSummaryResponse()
        assert kg.node_count == 0
        assert kg.node_type_counts == {}

    def test_benchmark_comparison_response(self):
        bc = BenchmarkComparisonResponse(
            startup_name="X",
            composite_score=70.0,
            benchmark_mean=50.0,
            benchmark_std_dev=10.0,
            percentile_rank=90.0,
            score_z_score=2.0,
        )
        assert bc.percentile_rank == 90.0


class TestPortfolioSchemaSerialization:
    """Verify Portfolio API schema serialization."""

    def test_portfolio_company_score(self):
        cs = PortfolioCompanyScore(
            startup_name="A",
            overall_score=75.0,
            overall_confidence=0.8,
        )
        d = cs.model_dump()
        assert d["startup_name"] == "A"

    def test_sector_distribution(self):
        sd = SectorDistribution(
            sector="fintech",
            count=5,
            percentage=50.0,
            avg_score=72.0,
        )
        assert sd.count == 5

    def test_stage_distribution(self):
        sd = StageDistribution(stage="seed", count=3, percentage=60.0)
        assert sd.stage == "seed"

    def test_risk_summary(self):
        rs = RiskSummary(high_risk_count=1, medium_risk_count=2, low_risk_count=3)
        assert rs.high_risk_count == 1

    def test_concentration_risk(self):
        cr = ConcentrationRisk(sector_concentration=0.8)
        assert cr.sector_concentration == 0.8

    def test_diversification_score(self):
        ds = DiversificationScore(overall_diversification=0.7)
        assert ds.overall_diversification == 0.7

    def test_heatmap_cell(self):
        hc = HeatmapCell(
            row_label="A",
            col_label="market",
            value=75.0,
            label="75",
        )
        assert hc.value == 75.0

    def test_similarity_matrix(self):
        sm = SimilarityMatrixResponse(
            companies=["A", "B"],
            matrix=[[1.0, 0.5], [0.5, 1.0]],
        )
        assert sm.matrix[0][1] == 0.5

    def test_portfolio_analysis_response_roundtrip(self):
        par = PortfolioAnalysisResponse(
            company_count=2,
            portfolio_score=70.0,
            overall_confidence=0.75,
        )
        d = par.model_dump()
        par2 = PortfolioAnalysisResponse.model_validate(d)
        assert par2.portfolio_score == 70.0


class TestComparisonSchemaSerialization:
    """Verify Comparison API schema serialization."""

    def test_comparison_request(self):
        req = ComparisonRequest(
            company_names=["A", "B"],
            descriptions={"A": "desc A", "B": "desc B"},
        )
        assert len(req.company_names) == 2

    def test_feature_diff(self):
        fd = FeatureDiff(
            feature="market_score",
            company_a=70.0,
            company_b=65.0,
            difference=5.0,
            direction="positive",
        )
        assert fd.direction == "positive"

    def test_contribution_diff_item(self):
        cd = ContributionDiffItem(
            feature="revenue_growth",
            contributions={"A": 0.8, "B": 0.6},
            max_contribution=0.8,
            min_contribution=0.6,
        )
        assert cd.max_contribution == 0.8

    def test_full_comparison_response_defaults(self):
        fc = FullComparisonResponse()
        assert fc.companies == []


class TestDueDiligenceSchemaSerialization:
    """Verify Due Diligence API schema serialization."""

    def test_due_diligence_request(self):
        req = DueDiligenceRequest(
            startup_name="DDCo",
            description="Due diligence schema test company for validation.",
        )
        assert req.startup_name == "DDCo"

    def test_executive_summary(self):
        es = ExecutiveSummary(
            headline="Test",
            overview="Overview text",
            confidence_level=0.9,
        )
        assert es.confidence_level == 0.9

    def test_strength_item(self):
        si = StrengthItem(
            title="Strong Team",
            description="Experienced founding team",
            dimension="founder_quality",
            severity="high",
        )
        assert si.severity == "high"

    def test_weakness_item(self):
        wi = WeaknessItem(
            title="Early Stage",
            dimension="traction",
            severity="moderate",
        )
        assert wi.title == "Early Stage"

    def test_opportunity_item_response(self):
        oi = OpportunityItemResponse(
            title="Market Gap",
            impact="high",
        )
        assert oi.impact == "high"

    def test_risk_item_response(self):
        ri = RiskItemResponse(
            title="Competition",
            severity="high",
            mitigation="Focus on niche",
        )
        assert ri.mitigation == "Focus on niche"

    def test_evidence_entry(self):
        ee = EvidenceEntry(
            claim="Market growing at 20% CAGR",
            domain="market",
            trust_score=0.85,
        )
        assert ee.trust_score == 0.85

    def test_decision_trace_entry(self):
        dte = DecisionTraceEntry(
            category="invest",
            conviction="high",
            composite_score=72.5,
        )
        assert dte.conviction == "high"

    def test_benchmark_context(self):
        bc = BenchmarkContext(
            composite_score=70.0,
            percentile_rank=85.0,
        )
        assert bc.percentile_rank == 85.0

    def test_due_diligence_report_response(self):
        resp = DueDiligenceReportResponse(
            startup_name="DDCo",
            confidence=0.8,
        )
        assert resp.startup_name == "DDCo"
        assert resp.strengths == []


class TestSearchSchemaSerialization:
    """Verify Search API schema serialization."""

    def test_search_request(self):
        req = SearchRequest(query="fintech", search_type="industry")
        assert req.query == "fintech"

    def test_search_result_item(self):
        item = SearchResultItem(
            result_type="company",
            id="rec1",
            name="TestCo",
        )
        assert item.result_type == "company"

    def test_search_response(self):
        resp = SearchResponse(
            query="test",
            total=5,
            results=[SearchResultItem(result_type="company", id="1", name="X")],
        )
        assert resp.total == 5

    def test_company_search_result(self):
        csr = CompanySearchResult(
            startup_name="Co",
            industries=["fintech"],
        )
        assert csr.startup_name == "Co"

    def test_signal_search_result(self):
        ssr = SignalSearchResult(
            signal_id="sig1",
            signal_type="funding_round",
        )
        assert ssr.signal_type == "funding_round"

    def test_knowledge_graph_node_search_result(self):
        kgsr = KnowledgeGraphNodeSearchResult(
            node_id="node1",
            node_type="company",
            label="TestCo",
        )
        assert kgsr.node_type == "company"

    def test_search_by_type_response(self):
        resp = SearchByTypeResponse(search_type="company", total=0)
        assert resp.search_type == "company"


class TestBatchSchemaSerialization:
    """Verify Batch API schema serialization."""

    def test_job_status_enum(self):
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_batch_job_request(self):
        req = BatchJobRequest(
            job_name="test",
            items=[{"startup_name": "X", "description": "Desc X"}],
        )
        assert len(req.items) == 1

    def test_batch_job_summary(self):
        summary = BatchJobSummary(
            job_id="abc-123",
            job_name="test",
            status=JobStatus.PENDING,
            total_items=5,
        )
        assert summary.total_items == 5

    def test_batch_job_result_item(self):
        item = BatchJobResultItem(
            index=0,
            status=JobStatus.COMPLETED,
            startup_name="TestCo",
            result_id="r1",
            processing_time_ms=123.4,
        )
        assert item.processing_time_ms == 123.4

    def test_batch_job_list_response(self):
        resp = BatchJobListResponse(jobs=[], total=0)
        assert resp.total == 0

    def test_batch_job_progress_response(self):
        p = BatchJobProgressResponse(
            job_id="j1",
            status=JobStatus.RUNNING,
            total_items=10,
            completed_items=5,
            progress_pct=50.0,
        )
        assert p.progress_pct == 50.0
