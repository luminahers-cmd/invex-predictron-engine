"""Tests for evaluation utility functions."""


from predictron_engine.evaluation.evaluators.base import (
    calculate_average_confidence,
    filter_evidence_by_domain,
    filter_observations,
    generate_rationale,
    generate_summary,
)
from predictron_engine.models.report import EvidenceItem, Observation


class TestFilterObservations:
    """Tests for the filter_observations utility."""

    def test_filters_by_dimension(self):
        obs = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Test",
            ),
            Observation(
                dimension="founder_quality",
                category="team_assessment",
                statement="Test",
            ),
        ]
        result = filter_observations(obs, "market_opportunity")
        assert len(result) == 1
        assert result[0].dimension == "market_opportunity"

    def test_returns_empty_for_no_match(self):
        obs = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Test",
            ),
        ]
        result = filter_observations(obs, "nonexistent")
        assert len(result) == 0

    def test_returns_empty_for_empty_input(self):
        result = filter_observations([], "market_opportunity")
        assert len(result) == 0


class TestFilterEvidenceByDomain:
    """Tests for the filter_evidence_by_domain utility."""

    def test_filters_by_domain(self):
        evidence = [
            EvidenceItem(
                domain="industry",
                category="market_size",
                statement="Test",
                source="test",
            ),
            EvidenceItem(
                domain="team",
                category="founder",
                statement="Test",
                source="test",
            ),
        ]
        result = filter_evidence_by_domain(evidence, "industry")
        assert len(result) == 1
        assert result[0].domain == "industry"

    def test_returns_empty_for_no_match(self):
        evidence = [
            EvidenceItem(
                domain="industry",
                category="market_size",
                statement="Test",
                source="test",
            ),
        ]
        result = filter_evidence_by_domain(evidence, "nonexistent")
        assert len(result) == 0


class TestCalculateAverageConfidence:
    """Tests for the calculate_average_confidence utility."""

    def test_average_with_observations(self):
        obs = [
            Observation(
                dimension="test",
                category="test",
                statement="Test",
                confidence=0.8,
            ),
            Observation(
                dimension="test",
                category="test",
                statement="Test",
                confidence=0.6,
            ),
        ]
        result = calculate_average_confidence(obs)
        assert abs(result - 0.7) < 1e-10

    def test_average_empty(self):
        result = calculate_average_confidence([])
        assert result == 0.0


class TestGenerateSummary:
    """Tests for the generate_summary utility."""

    def test_summary_with_observations(self):
        obs = [
            Observation(
                dimension="test",
                category="test",
                statement="Test",
                confidence=0.5,
            ),
        ]
        result = generate_summary("test", obs, [])
        assert "test" in result
        assert "1 observations" in result

    def test_summary_without_observations(self):
        result = generate_summary("test", [], [])
        assert "No observations" in result


class TestGenerateRationale:
    """Tests for the generate_rationale utility."""

    def test_rationale_with_observations(self):
        obs = [
            Observation(
                dimension="test",
                category="test",
                statement="Key finding",
                confidence=0.5,
            ),
        ]
        result = generate_rationale("test", obs, [])
        assert "Key finding" in result

    def test_rationale_without_observations(self):
        result = generate_rationale("test", [], [])
        assert "Insufficient data" in result
