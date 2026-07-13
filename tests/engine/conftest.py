import pytest
from pydantic import HttpUrl

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.engine import PredictronEngine
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    EvidenceItem,
    Observation,
    Recommendation,
    ScoreResult,
)
from predictron_engine.models.startup import Startup

"""Shared test fixtures for the Predictron Engine test suite."""


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def minimal_request() -> StartupAnalysisRequest:
    """A valid request with only required fields."""
    return StartupAnalysisRequest(
        startup_name="TestCo",
        website=HttpUrl("https://testco.example.com"),
        description="A test startup company that builds innovative software solutions.",
    )


@pytest.fixture
def full_request() -> StartupAnalysisRequest:
    """A valid request with all fields populated."""
    return StartupAnalysisRequest(
        startup_name="FullCo",
        website=HttpUrl("https://fullco.example.com"),
        description=(
            "FullCo is a B2B SaaS platform that provides AI-powered analytics "
            "for the fintech industry. The company has raised a Series A round "
            "and serves enterprise clients across North America."
        ),
        pitch_deck_url=HttpUrl("https://fullco.example.com/pitch.pdf"),
        founder_linkedin_urls=[
            HttpUrl("https://linkedin.com/in/founder1"),
            HttpUrl("https://linkedin.com/in/founder2"),
        ],
    )


@pytest.fixture
def saas_request() -> StartupAnalysisRequest:
    """A request for a SaaS startup with rich description."""
    return StartupAnalysisRequest(
        startup_name="SaaSCo",
        website=HttpUrl("https://saasco.example.com"),
        description=(
            "SaaSCo is an enterprise SaaS platform providing subscription-based "
            "customer relationship management tools for mid-market B2B companies. "
            "Founded in 2020 with a seed round, the platform serves over 200 "
            "enterprise clients with recurring revenue."
        ),
        pitch_deck_url=HttpUrl("https://saasco.example.com/deck.pdf"),
        founder_linkedin_urls=[
            HttpUrl("https://linkedin.com/in/ceo"),
        ],
    )


@pytest.fixture
def sample_startup() -> Startup:
    """A pre-built Startup model for unit testing."""
    return Startup(
        name="SampleCo",
        website="https://sampleco.example.com",
        description="A sample startup for testing purposes.",
        pitch_deck_url=None,
        founder_linkedin_urls=[],
        raw_data={},
    )


@pytest.fixture
def sample_collected_data() -> CollectedData:
    """A pre-built CollectedData model for unit testing."""
    return CollectedData(
        startup_name="SampleCo",
        website_domain="sampleco.example.com",
        description_tokens=["sample", "startup", "testing", "purposes"],
        description_word_count=4,
        has_website=True,
        has_pitch_deck=False,
        founder_count=0,
        url_metadata={"website_url": "https://sampleco.example.com"},
        enrichment_signals={},
    )


@pytest.fixture
def sample_features() -> ExtractedFeatures:
    """A pre-built ExtractedFeatures model for unit testing."""
    return ExtractedFeatures(
        industry="enterprise_saas",
        sub_industry=None,
        business_model="saas",
        funding_stage="seed",
        geography=None,
        headquarters_region=None,
        technology_stack=[],
        customer_type="b2b",
        team_size_indicator=None,
        founded_year=2020,
        has_revenue=True,
        key_keywords=["enterprise", "saas", "platform", "subscription"],
        description_length=250,
        has_pitch_deck=True,
        founder_profile_count=2,
        data_completeness=0.5,
    )


@pytest.fixture
def minimal_features() -> ExtractedFeatures:
    """Features with most fields missing (low data completeness)."""
    return ExtractedFeatures(
        description_length=50,
        data_completeness=0.1,
    )


@pytest.fixture
def sample_evidence() -> list[EvidenceItem]:
    """A pre-built list of evidence items for unit testing."""
    return [
        EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Enterprise SaaS market exceeds $500 billion.",
            source="knowledge/taxonomies.py:enterprise_saas",
            relevance_score=1.0,
        ),
        EvidenceItem(
            domain="business_model",
            category="unit_economics",
            statement="SaaS businesses typically achieve 70-90% gross margins.",
            source="knowledge/taxonomies.py:saas",
            relevance_score=1.0,
        ),
    ]


@pytest.fixture
def sample_observations() -> list[Observation]:
    """A pre-built list of observations for unit testing."""
    return [
        Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Startup operates in the enterprise_saas industry.",
            evidence=["industry=enterprise_saas"],
            confidence=0.7,
            importance=0.8,
            source_rule="MarketContextRule",
        ),
        Observation(
            dimension="business_model_viability",
            category="business_model_assessment",
            statement="Startup follows a saas business model.",
            evidence=["business_model=saas"],
            confidence=0.6,
            importance=0.7,
            source_rule="BusinessModelContextRule",
        ),
    ]


@pytest.fixture
def sample_scores() -> list[ScoreResult]:
    """A pre-built list of scores for unit testing."""
    return [
        ScoreResult(
            dimension="market_opportunity",
            score=65.0,
            rationale="Placeholder scoring.",
            evidence=["Startup operates in the enterprise_saas industry."],
        ),
        ScoreResult(
            dimension="product_strength",
            score=50.0,
            rationale="Placeholder scoring.",
            evidence=[],
        ),
        ScoreResult(
            dimension="founder_quality",
            score=53.0,
            rationale="Placeholder scoring.",
            evidence=[],
        ),
    ]


@pytest.fixture
def sample_recommendations() -> list[Recommendation]:
    """A pre-built list of recommendations for unit testing."""
    return [
        Recommendation(
            category="due_diligence",
            action="Request the pitch deck for deeper analysis.",
            priority="high",
            rationale="No pitch deck was provided.",
        )
    ]


@pytest.fixture
def sample_confidence() -> list[ConfidenceAssessment]:
    """A pre-built list of confidence assessments for unit testing."""
    return [
        ConfidenceAssessment(
            dimension="market_opportunity",
            confidence=0.55,
            factors=["data_completeness=0.50", "observation_count=1"],
            data_completeness=0.5,
        ),
        ConfidenceAssessment(
            dimension="product_strength",
            confidence=0.40,
            factors=["data_completeness=0.50", "observation_count=0"],
            data_completeness=0.5,
        ),
    ]


@pytest.fixture
def engine() -> PredictronEngine:
    """A default PredictronEngine with all standard implementations."""
    return PredictronEngine()
