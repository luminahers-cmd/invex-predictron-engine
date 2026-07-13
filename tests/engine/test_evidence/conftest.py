import pytest

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sample_startup() -> Startup:
    return Startup(
        name="SampleCo",
        website="https://sampleco.example.com",
        description="A sample startup for testing purposes.",
        pitch_deck_url=None,
        founder_linkedin_urls=[],
        raw_data={},
    )


@pytest.fixture
def rich_features() -> ExtractedFeatures:
    """Features with multiple domains populated for comprehensive testing."""
    return ExtractedFeatures(
        industry="fintech",
        business_model="saas",
        funding_stage="seed",
        geography="north_america",
        headquarters_region="north_america",
        technology_stack=["python", "aws", "react"],
        customer_type="b2b",
        has_revenue=True,
        key_keywords=["fintech", "saas", "payments"],
        description_length=300,
        has_pitch_deck=True,
        founder_profile_count=2,
        data_completeness=0.7,
    )


@pytest.fixture
def minimal_features() -> ExtractedFeatures:
    """Features with almost nothing populated."""
    return ExtractedFeatures(
        description_length=50,
        data_completeness=0.1,
    )


@pytest.fixture
def industry_only_features() -> ExtractedFeatures:
    """Features with only industry populated."""
    return ExtractedFeatures(
        industry="healthtech",
        description_length=100,
        data_completeness=0.2,
    )


@pytest.fixture
def saas_features() -> ExtractedFeatures:
    """Features for a SaaS enterprise startup."""
    return ExtractedFeatures(
        industry="enterprise_saas",
        business_model="saas",
        funding_stage="series_a",
        geography="north_america",
        technology_stack=["python", "react", "postgresql"],
        customer_type="b2b",
        has_revenue=True,
        description_length=400,
        has_pitch_deck=True,
        founder_profile_count=3,
        data_completeness=0.8,
    )
