"""Tests for recommendation strategy helper utilities."""

import pytest

from predictron_engine.models.report import DimensionAssessment, Observation
from predictron_engine.recommendations.strategies.base import (
    assessment_confidence,
    filter_assessments,
    filter_observations,
    observation_confidence,
)


class TestFilterObservations:
    """Tests for the filter_observations helper."""

    def test_filters_by_dimension(self):
        obs = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="s1",
                confidence=0.5,
                importance=0.5,
                source_rule="test",
            ),
            Observation(
                dimension="founder_quality",
                category="test",
                statement="s2",
                confidence=0.6,
                importance=0.5,
                source_rule="test",
            ),
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="s3",
                confidence=0.7,
                importance=0.5,
                source_rule="test",
            ),
        ]
        result = filter_observations(obs, "market_opportunity")
        assert len(result) == 2
        assert all(o.dimension == "market_opportunity" for o in result)

    def test_empty_observations(self):
        assert filter_observations([], "market_opportunity") == []

    def test_no_match(self):
        obs = [
            Observation(
                dimension="founder_quality",
                category="test",
                statement="s",
                confidence=0.5,
                importance=0.5,
                source_rule="test",
            ),
        ]
        assert filter_observations(obs, "market_opportunity") == []


class TestFilterAssessments:
    """Tests for the filter_assessments helper."""

    def test_filters_by_dimension(self):
        assess = [
            DimensionAssessment(
                dimension="product_strength",
                summary="s1",
                rationale="r1",
                confidence=0.6,
            ),
            DimensionAssessment(
                dimension="market_opportunity",
                summary="s2",
                rationale="r2",
                confidence=0.7,
            ),
        ]
        result = filter_assessments(assess, "product_strength")
        assert len(result) == 1
        assert result[0].dimension == "product_strength"

    def test_empty_assessments(self):
        assert filter_assessments([], "product_strength") == []

    def test_no_match(self):
        assess = [
            DimensionAssessment(
                dimension="market_opportunity",
                summary="s",
                rationale="r",
                confidence=0.5,
            ),
        ]
        assert filter_assessments(assess, "product_strength") == []


class TestObservationConfidence:
    """Tests for the observation_confidence helper."""

    def test_average_confidence(self):
        obs = [
            Observation(
                dimension="d",
                category="c",
                statement="s",
                confidence=0.6,
                importance=0.5,
                source_rule="r",
            ),
            Observation(
                dimension="d",
                category="c",
                statement="s",
                confidence=0.8,
                importance=0.5,
                source_rule="r",
            ),
        ]
        assert observation_confidence(obs) == pytest.approx(0.7)

    def test_empty_observations(self):
        assert observation_confidence([]) == 0.0

    def test_single_observation(self):
        obs = [
            Observation(
                dimension="d",
                category="c",
                statement="s",
                confidence=0.45,
                importance=0.5,
                source_rule="r",
            ),
        ]
        assert observation_confidence(obs) == 0.45


class TestAssessmentConfidence:
    """Tests for the assessment_confidence helper."""

    def test_average_confidence(self):
        assess = [
            DimensionAssessment(
                dimension="d", summary="s", rationale="r", confidence=0.4
            ),
            DimensionAssessment(
                dimension="d", summary="s", rationale="r", confidence=0.8
            ),
        ]
        assert assessment_confidence(assess) == pytest.approx(0.6)

    def test_empty_assessments(self):
        assert assessment_confidence([]) == 0.0

    def test_single_assessment(self):
        assess = [
            DimensionAssessment(
                dimension="d", summary="s", rationale="r", confidence=0.55
            ),
        ]
        assert assessment_confidence(assess) == 0.55
