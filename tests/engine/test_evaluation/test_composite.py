"""Tests for the CompositeEvaluator orchestrator."""


from predictron_engine.evaluation.composite import CompositeEvaluator
from predictron_engine.evaluation.evaluation_models import (
    DimensionAssessment,
    EvaluationResult,
)


class TestCompositeEvaluator:
    """Tests for the CompositeEvaluator orchestrator."""

    def test_default_evaluators(self):
        evaluator = CompositeEvaluator()
        assert len(evaluator._evaluators) == 7

    def test_custom_evaluators(self):
        class MockEvaluator:
            @property
            def dimension(self) -> str:
                return "test_dimension"

            def evaluate(self, features, observations, evidence):
                return DimensionAssessment(
                    dimension="test_dimension",
                    summary="Test summary",
                    rationale="Test rationale",
                    confidence=0.5,
                )

        custom = [MockEvaluator()]
        evaluator = CompositeEvaluator(evaluators=custom)
        assert len(evaluator._evaluators) == 1

    def test_evaluate_returns_evaluation_result(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = CompositeEvaluator()
        result = evaluator.evaluate(
            sample_features, sample_observations, sample_evidence
        )

        assert isinstance(result, EvaluationResult)
        assert isinstance(result.assessments, list)
        assert result.dimensions_assessed >= 0
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_evaluate_with_empty_inputs(self, sample_features):
        evaluator = CompositeEvaluator()
        result = evaluator.evaluate(sample_features, [], [])

        assert isinstance(result, EvaluationResult)
        assert len(result.assessments) == 7
        assert result.dimensions_assessed == 7
        assert result.overall_confidence == 0.0

    def test_evaluate_with_rich_inputs(
        self, sample_features, sample_observations, sample_evidence
    ):
        evaluator = CompositeEvaluator()
        result = evaluator.evaluate(
            sample_features, sample_observations, sample_evidence
        )

        assert result.dimensions_assessed == 7
        assert result.overall_summary
        assert "7 dimensions" in result.overall_summary

    def test_evaluator_failure_continues(self, sample_features):
        class FailingEvaluator:
            @property
            def dimension(self) -> str:
                return "failing"

            def evaluate(self, features, observations, evidence):
                raise ValueError("Intentional failure")

        evaluator = CompositeEvaluator(evaluators=[FailingEvaluator()])
        result = evaluator.evaluate(sample_features, [], [])

        assert isinstance(result, EvaluationResult)
        assert len(result.assessments) == 0

    def test_mixed_evaluators(self, sample_features, sample_observations, sample_evidence):
        class FailingEvaluator:
            @property
            def dimension(self) -> str:
                return "failing"

            def evaluate(self, features, observations, evidence):
                raise ValueError("Intentional failure")

        evaluator = CompositeEvaluator(
            evaluators=[FailingEvaluator(), *CompositeEvaluator()._evaluators[:1]]
        )
        result = evaluator.evaluate(
            sample_features, sample_observations, sample_evidence
        )

        assert isinstance(result, EvaluationResult)
        assert len(result.assessments) >= 1
