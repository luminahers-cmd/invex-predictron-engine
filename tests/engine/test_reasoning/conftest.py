import pytest

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.extracted_features import ExtractedFeatures


def make_trust(overall: float):
    """Build a deterministic TrustScore with a single overall value."""
    from predictron_engine.evidence.provenance import (
        TrustFactor,
        TrustScore,
    )

    return TrustScore(
        overall=overall,
        factors=[
            TrustFactor(name="authority", weight=1.0, value=overall),
        ],
    )


def make_document(
    doc_id: str,
    url: str = "https://example.com/about",
    trust: float | None = None,
    provider: str = "website",
    title: str = "Example About",
):
    """Build a successful EvidenceDocument with optional trust metadata."""
    from datetime import UTC, datetime

    from predictron_engine.evidence.models import (
        DocumentMetadata,
        DocumentStatus,
        EvidenceDocument,
        PageType,
    )

    metadata = DocumentMetadata(source_provider=provider)
    if trust is not None:
        metadata = DocumentMetadata(
            source_provider=provider,
            trust_score=make_trust(trust),
        )

    return EvidenceDocument(
        id=doc_id,
        original_url=url,
        url=url,
        page_type=PageType.ABOUT,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        response_time_ms=10,
        http_status=200,
        title=title,
        metadata=metadata,
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
def rich_evidence() -> list[EvidenceItem]:
    """A rich set of evidence items across multiple domains."""
    return [
        EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Global fintech market exceeds $300 billion.",
            source="knowledge/taxonomies.py:fintech",
            relevance_score=1.0,
        ),
        EvidenceItem(
            domain="industry",
            category="regulatory",
            statement="Financial services require regulatory compliance.",
            source="knowledge/taxonomies.py:fintech",
            relevance_score=1.0,
        ),
        EvidenceItem(
            domain="business_model",
            category="unit_economics",
            statement="SaaS businesses achieve 70-90% gross margins.",
            source="knowledge/taxonomies.py:saas",
            relevance_score=1.0,
        ),
        EvidenceItem(
            domain="funding_stage",
            category="expectations",
            statement="Seed companies are building MVP and initial customers.",
            source="knowledge/stages.py:seed",
            relevance_score=1.0,
        ),
        EvidenceItem(
            domain="technology",
            category="ecosystem",
            statement="Python has a large, mature ecosystem.",
            source="knowledge/tech:python",
            relevance_score=1.0,
        ),
        EvidenceItem(
            domain="geography",
            category="market_size",
            statement="North America has the largest VC market globally.",
            source="knowledge/geography:north_america",
            relevance_score=1.0,
        ),
    ]
