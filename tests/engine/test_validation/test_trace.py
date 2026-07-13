"""Tests for the traceability system."""

from predictron_engine.models.report import (
    ConfidenceAssessment,
    DimensionAssessment,
    EvidenceItem,
    Observation,
    Recommendation,
    ScoreResult,
)
from predictron_engine.validation.trace import (
    TraceGraph,
    TraceGraphBuilder,
    TraceNode,
    _content_id,
    _make_id,
)


class TestMakeId:
    """Tests for the _make_id utility."""

    def test_deterministic(self):
        assert _make_id("reason", "observation", 0) == "reason:observation:0"

    def test_different_indices(self):
        id1 = _make_id("reason", "observation", 0)
        id2 = _make_id("reason", "observation", 1)
        assert id1 != id2

    def test_different_stages(self):
        id1 = _make_id("extract", "features", 0)
        id2 = _make_id("reason", "observation", 0)
        assert id1 != id2


class TestContentId:
    """Tests for the _content_id utility."""

    def test_deterministic(self):
        assert _content_id("hello") == _content_id("hello")

    def test_different_content(self):
        assert _content_id("hello") != _content_id("world")

    def test_hash_prefix(self):
        cid = _content_id("test")
        assert cid.startswith("hash:")


class TestTraceNode:
    """Tests for TraceNode model."""

    def test_required_fields(self):
        node = TraceNode(
            artifact_id="test:1",
            stage="extract",
            artifact_type="ExtractedFeatures",
        )
        assert node.artifact_id == "test:1"
        assert node.stage == "extract"
        assert node.input_ids == []
        assert node.metadata == {}

    def test_with_inputs(self):
        node = TraceNode(
            artifact_id="reason:obs:0",
            stage="reason",
            artifact_type="Observation",
            input_ids=["extract:features:0", "evidence:item:0"],
        )
        assert len(node.input_ids) == 2


class TestTraceGraphBuilder:
    """Tests for the TraceGraphBuilder."""

    def _make_obs(self, dim: str, evidence_refs: list[str] | None = None) -> Observation:
        return Observation(
            dimension=dim,
            category="test",
            statement=f"Obs for {dim}",
            evidence=evidence_refs or [],
            confidence=0.7,
            importance=0.5,
            source_rule="TestRule",
        )

    def _make_assess(self, dim: str) -> DimensionAssessment:
        return DimensionAssessment(
            dimension=dim,
            summary=f"Summary for {dim}",
            rationale="Test rationale",
            confidence=0.6,
        )

    def _make_evidence(self, domain: str, statement: str) -> EvidenceItem:
        return EvidenceItem(
            domain=domain,
            category="test",
            statement=statement,
            source="test_source",
        )

    def _make_rec(self, category: str, title: str) -> Recommendation:
        return Recommendation(
            category=category,
            action=f"Action for {category}",
            priority="medium",
            title=title,
            description="Desc",
            metadata={"strategy": "test"},
        )

    def test_build_empty(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(data_completeness=0.0)
        graph = TraceGraphBuilder().build(
            startup_name="Test",
            features=features,
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert isinstance(graph, TraceGraph)
        assert len(graph.nodes) >= 2
        assert "input:startup:0" in graph.nodes
        assert "extract:features:0" in graph.nodes

    def test_build_with_evidence(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(
            industry="fintech", data_completeness=0.5
        )
        evidence = [
            self._make_evidence("industry", "Fintech market is large"),
        ]
        graph = TraceGraphBuilder().build(
            startup_name="Test",
            features=features,
            evidence=evidence,
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert "evidence:item:0" in graph.nodes
        assert graph.edges["evidence:item:0"] == ["extract:features:0"]

    def test_build_with_observations(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(data_completeness=0.5)
        obs = [self._make_obs("market_opportunity")]
        graph = TraceGraphBuilder().build(
            startup_name="Test",
            features=features,
            evidence=[],
            observations=obs,
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert "reason:observation:0" in graph.nodes
        node = graph.nodes["reason:observation:0"]
        assert node.metadata["dimension"] == "market_opportunity"

    def test_build_full_pipeline(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(
            industry="saas", data_completeness=0.6
        )
        evidence = [
            self._make_evidence("industry", "SaaS market is growing"),
        ]
        obs = [self._make_obs("market_opportunity", ["SaaS market is growing"])]
        assessments = [self._make_assess("market_opportunity")]
        scores = [ScoreResult(dimension="market_opportunity", score=65.0)]
        recs = [self._make_rec("opportunity", "Market Analysis")]

        conf = [
            ConfidenceAssessment(
                dimension="market_opportunity",
                confidence=0.6,
                data_completeness=0.5,
            )
        ]

        graph = TraceGraphBuilder().build(
            startup_name="TestCo",
            features=features,
            evidence=evidence,
            observations=obs,
            assessments=assessments,
            scores=scores,
            recommendations=recs,
            confidence=conf,
        )

        assert "recommend:rec:0" in graph.nodes
        assert "confidence:assess:0" in graph.nodes
        assert "score:result:0" in graph.nodes
        assert "evaluate:assessment:0" in graph.nodes

    def test_reverse_edges_populated(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(data_completeness=0.5)
        graph = TraceGraphBuilder().build(
            startup_name="T",
            features=features,
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert "extract:features:0" in graph.reverse_edges.get(
            "input:startup:0", []
        )

    def test_get_trace_paths(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(data_completeness=0.5)
        graph = TraceGraphBuilder().build(
            startup_name="T",
            features=features,
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        paths = TraceGraphBuilder().get_trace_paths(
            graph, "extract:features:0"
        )
        assert len(paths) >= 1
        assert paths[0].target_id == "extract:features:0"
        assert paths[0].source_id == "input:startup:0"

    def test_get_all_downstream(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures(data_completeness=0.5)
        graph = TraceGraphBuilder().build(
            startup_name="T",
            features=features,
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        downstream = TraceGraphBuilder().get_all_downstream(
            graph, "input:startup:0"
        )
        assert "extract:features:0" in downstream

    def test_get_trace_paths_nonexistent(self):
        graph = TraceGraph()
        paths = TraceGraphBuilder().get_trace_paths(graph, "nonexistent")
        assert paths == []
