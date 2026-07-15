"""Tests for individual dimension evaluators."""


from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.evaluators.business_model import BusinessModelEvaluator
from predictron_engine.evaluation.evaluators.data_quality import DataQualityEvaluator
from predictron_engine.evaluation.evaluators.market import MarketEvaluator
from predictron_engine.evaluation.evaluators.risk import RiskEvaluator
from predictron_engine.evaluation.evaluators.team import TeamEvaluator
from predictron_engine.evaluation.evaluators.technology import TechnologyEvaluator
from predictron_engine.evaluation.evaluators.traction import TractionEvaluator


class TestMarketEvaluator:
    """Tests for the market dimension evaluator."""

    def test_dimension_property(self):
        evaluator = MarketEvaluator()
        assert evaluator.dimension == "market_opportunity"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = MarketEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "market_opportunity"
        assert result.summary
        assert result.rationale
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_with_no_observations(self, sample_features):
        evaluator = MarketEvaluator()
        result = evaluator.evaluate(sample_features, [], [])

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "market_opportunity"
        assert result.confidence == 0.0
        assert len(result.supporting_observations) == 0

    def test_evaluate_includes_metadata(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = MarketEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert "industry" in result.metadata
        assert "geography" in result.metadata


class TestTeamEvaluator:
    """Tests for the team dimension evaluator."""

    def test_dimension_property(self):
        evaluator = TeamEvaluator()
        assert evaluator.dimension == "founder_quality"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = TeamEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "founder_quality"
        assert result.summary
        assert result.rationale
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_with_no_observations(self, sample_features):
        evaluator = TeamEvaluator()
        result = evaluator.evaluate(sample_features, [], [])

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "founder_quality"
        assert result.confidence == 0.0

    def test_evaluate_includes_metadata(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = TeamEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert "founder_count" in result.metadata
        assert "team_size" in result.metadata


class TestTechnologyEvaluator:
    """Tests for the technology dimension evaluator."""

    def test_dimension_property(self):
        evaluator = TechnologyEvaluator()
        assert evaluator.dimension == "product_strength"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = TechnologyEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "product_strength"
        assert result.summary
        assert result.rationale
        assert 0.0 <= result.confidence <= 1.0

    def test_evaluate_with_no_observations(self, sample_features):
        evaluator = TechnologyEvaluator()
        result = evaluator.evaluate(sample_features, [], [])

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "product_strength"
        assert result.confidence == 0.0


class TestTractionEvaluator:
    """Tests for the traction dimension evaluator."""

    def test_dimension_property(self):
        evaluator = TractionEvaluator()
        assert evaluator.dimension == "traction_signals"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = TractionEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "traction_signals"
        assert result.summary
        assert result.rationale
        assert 0.0 <= result.confidence <= 1.0


class TestBusinessModelEvaluator:
    """Tests for the business model dimension evaluator."""

    def test_dimension_property(self):
        evaluator = BusinessModelEvaluator()
        assert evaluator.dimension == "business_model_viability"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = BusinessModelEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "business_model_viability"
        assert result.summary
        assert result.rationale


class TestRiskEvaluator:
    """Tests for the risk dimension evaluator."""

    def test_dimension_property(self):
        evaluator = RiskEvaluator()
        assert evaluator.dimension == "competitive_position"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = RiskEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "competitive_position"
        assert result.summary
        assert result.rationale

    def test_evaluate_with_no_observations(self, sample_features):
        evaluator = RiskEvaluator()
        result = evaluator.evaluate(sample_features, [], [])

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "competitive_position"
        assert result.confidence == 0.0
        assert len(result.supporting_observations) == 0

    def test_metadata_includes_competition_fields(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        evaluator = RiskEvaluator()
        features = ExtractedFeatures(
            market_concentration="concentrated",
            competitive_density="dense",
            network_effect_competition="strong_network_effects",
            competitive_moat_indicators=["proprietary_data"],
            switching_cost_signals=["integration_lock_in"],
            barriers_to_entry=["regulatory"],
            open_source_competition=["tool"],
        )
        result = evaluator.evaluate(features, [], [])
        assert result.metadata["market_concentration"] == "concentrated"
        assert result.metadata["competitive_density"] == "dense"
        assert result.metadata["network_effect_competition"] == "strong_network_effects"
        assert result.metadata["moat_count"] == 1
        assert result.metadata["switching_cost_count"] == 1
        assert result.metadata["barrier_count"] == 1
        assert result.metadata["open_source_competition_count"] == 1

    def test_metadata_defaults_to_unknown(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures

        evaluator = RiskEvaluator()
        features = ExtractedFeatures()
        result = evaluator.evaluate(features, [], [])
        assert result.metadata["market_concentration"] == "unknown"
        assert result.metadata["competitive_density"] == "unknown"
        assert result.metadata["network_effect_competition"] == "unknown"
        assert result.metadata["moat_count"] == 0


class TestDataQualityEvaluator:
    """Tests for the data quality dimension evaluator."""

    def test_dimension_property(self):
        evaluator = DataQualityEvaluator()
        assert evaluator.dimension == "data_quality"

    def test_evaluate_with_observations(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = DataQualityEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert isinstance(result, DimensionAssessment)
        assert result.dimension == "data_quality"
        assert result.summary
        assert result.rationale

    def test_evaluate_includes_metadata(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = DataQualityEvaluator()
        result = evaluator.evaluate(sample_features, sample_observations, sample_evidence)

        assert "data_completeness" in result.metadata
        assert "description_length" in result.metadata
