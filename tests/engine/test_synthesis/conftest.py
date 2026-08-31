"""Shared fixtures and builders for Sprint 6C synthesis tests."""

from __future__ import annotations

import pytest

from predictron_engine.decision.models import (
    ConfidenceBreakdown,
    ConfidenceFactor,
    ConfidenceLevel,
    DecisionConfidence,
    UncertaintyBreakdown,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConvictionLevel,
    DecisionCategory,
    DecisionRationale,
    DimensionAssessment,
    EvidenceCitation,
    InvestmentDecision,
    InvestmentReadiness,
    Observation,
    Recommendation,
    ScoreResult,
    SignalRelationship,
)


def make_citation(
    claim: str = "Supporting claim",
    domain: str = "market",
    category: str = "market_context",
) -> EvidenceCitation:
    return EvidenceCitation(claim=claim, domain=domain, category=category)


def make_observation(
    dimension: str = "market_opportunity",
    *,
    statement: str = "Observation statement",
    confidence: float = 0.8,
    importance: float = 0.7,
    trust_score: float = 0.0,
    category: str = "context",
    citations: list[EvidenceCitation] | None = None,
    provenance_document_ids: list[str] | None = None,
) -> Observation:
    return Observation(
        dimension=dimension,
        category=category,
        statement=statement,
        confidence=confidence,
        importance=importance,
        trust_score=trust_score,
        citations=citations or [],
        provenance_document_ids=provenance_document_ids or [],
    )


def make_score(dimension: str, score: float) -> ScoreResult:
    return ScoreResult(
        dimension=dimension,
        score=score,
        rationale=f"{dimension} rationale",
        evidence=[],
    )


def make_assessment(
    dimension: str,
    *,
    confidence: float = 0.9,
    supporting_observations: list[Observation] | None = None,
) -> DimensionAssessment:
    return DimensionAssessment(
        dimension=dimension,
        summary=f"{dimension} summary",
        rationale=f"{dimension} rationale",
        confidence=confidence,
        supporting_observations=supporting_observations or [],
    )


def make_recommendation(
    action: str = "Request the pitch deck.",
    *,
    category: str = "due_diligence",
    priority: str = "medium",
    confidence: float = 0.5,
    expected_confidence: float | None = None,
    citations: list[EvidenceCitation] | None = None,
    supporting_observations: list[Observation] | None = None,
) -> Recommendation:
    return Recommendation(
        category=category,
        action=action,
        priority=priority,
        rationale=f"Because of {category}",
        confidence=confidence,
        expected_confidence=expected_confidence,
        citations=citations or [],
        supporting_observations=supporting_observations or [],
    )


def make_features(**overrides) -> ExtractedFeatures:
    defaults: dict = {
        "data_completeness": 0.7,
        "founder_profile_count": 2,
        "technology_stack": ["python"],
        "risk_confidence": 0.6,
    }
    defaults.update(overrides)
    return ExtractedFeatures(**defaults)


def make_relationship(
    source: str,
    target: str,
    relationship_type: str = "reinforcing",
    *,
    description: str = "Signals move together",
    confidence: float = 0.8,
) -> SignalRelationship:
    return SignalRelationship(
        source_dimension=source,
        target_dimension=target,
        relationship_type=relationship_type,
        description=description,
        confidence=confidence,
        source_observations=[],
    )


def make_readiness(**overrides) -> InvestmentReadiness:
    defaults: dict = {
        "readiness_score": 70.0,
        "readiness_level": "strong",
        "key_strengths": [],
        "key_concerns": [],
        "gaps": [],
    }
    defaults.update(overrides)
    return InvestmentReadiness(**defaults)


def make_decision(
    composite: float = 63.0,
    *,
    margin: float = 7.0,
    next_threshold: float = 70.0,
    data_quality_modifier: float = 1.0,
    risk_modifier: float = 1.0,
    category: DecisionCategory = DecisionCategory.INVEST,
) -> InvestmentDecision:
    return InvestmentDecision(
        category=category,
        conviction=ConvictionLevel.HIGH,
        composite_score=composite,
        rationale=DecisionRationale(),
        next_category_threshold=next_threshold,
        margin_to_next_category=margin,
        data_quality_modifier=data_quality_modifier,
        risk_modifier=risk_modifier,
    )


def make_factor(name: str, value: float, weight: float) -> ConfidenceFactor:
    return ConfidenceFactor(
        name=name,
        value=value,
        weight=weight,
        contribution=value * weight,
    )


def make_confidence(
    value: float = 0.55,
    *,
    uncertainty: float = 0.4,
    level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    drivers: dict[str, float] | None = None,
) -> DecisionConfidence:
    effective_drivers = drivers if drivers else {"missing_evidence": 0.5}
    uncertainty_factors = [
        make_factor(name, driver_value, 1.0 / len(effective_drivers))
        for name, driver_value in effective_drivers.items()
    ]
    return DecisionConfidence(
        confidence=value,
        level=level,
        uncertainty_score=uncertainty,
        breakdown=ConfidenceBreakdown(composite=value),
        uncertainty_breakdown=UncertaintyBreakdown(
            factors=uncertainty_factors,
            composite=uncertainty,
        ),
        supporting_factors=["Evidence quality is adequate."],
        weakening_factors=["Some evidence is missing."],
    )


@pytest.fixture
def sample_scores() -> list[ScoreResult]:
    return [
        make_score("market_opportunity", 75.0),
        make_score("product_strength", 40.0),
        make_score("team_execution", 52.0),
    ]


@pytest.fixture
def sample_assessments() -> list[DimensionAssessment]:
    return [
        make_assessment("market_opportunity", confidence=0.85),
        make_assessment("product_strength", confidence=0.3),
        make_assessment("team_execution", confidence=0.8),
    ]
