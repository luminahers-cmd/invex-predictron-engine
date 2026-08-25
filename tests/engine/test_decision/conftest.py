"""Shared fixtures and builders for decision calibration tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictron_engine.decision.calibration import compute_decision_confidence
from predictron_engine.evidence.models import (
    DocumentMetadata,
    DocumentStatus,
    EvidenceBundle,
    EvidenceDocument,
    EvidenceSource,
    IntelligenceSummary,
    PageType,
)
from predictron_engine.evidence.provenance import TrustScore
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import DimensionAssessment, Observation

FETCHED_AT = datetime(2024, 1, 1, tzinfo=UTC)


def make_document(
    doc_id: str = "doc-1",
    *,
    provider: str = "website",
    trust: float | None = 0.8,
) -> EvidenceDocument:
    """Build a minimal successful evidence document."""
    metadata = None
    if trust is not None:
        metadata = DocumentMetadata(
            source_provider=provider,
            trust_score=TrustScore(overall=trust),
        )
    elif provider:
        metadata = DocumentMetadata(source_provider=provider)
    return EvidenceDocument(
        id=doc_id,
        original_url=f"https://example.com/{doc_id}",
        url=f"https://example.com/{doc_id}",
        page_type=PageType.HOMEPAGE,
        status=DocumentStatus.SUCCESS,
        fetched_at=FETCHED_AT,
        response_time_ms=100,
        http_status=200,
        metadata=metadata,
    )


def make_source(success: bool = True) -> EvidenceSource:
    return EvidenceSource(
        original_url="https://example.com",
        page_type=PageType.HOMEPAGE,
        fetched_at=FETCHED_AT,
        success=success,
    )


def make_observation(
    dimension: str = "market",
    *,
    confidence: float = 0.9,
    agreement: float = 1.0,
    conflict_count: int = 0,
    category: str = "market_context",
    trust_score: float = 0.0,
) -> Observation:
    return Observation(
        dimension=dimension,
        category=category,
        statement="Observation statement",
        confidence=confidence,
        evidence_agreement_ratio=agreement,
        evidence_conflict_count=conflict_count,
        trust_score=trust_score,
    )


def make_assessment(
    dimension: str = "market",
    *,
    confidence: float = 0.9,
) -> DimensionAssessment:
    return DimensionAssessment(
        dimension=dimension,
        summary="Assessment summary",
        rationale="Assessment rationale",
        confidence=confidence,
    )


def make_bundle(
    *documents: EvidenceDocument,
    sources: list[EvidenceSource] | None = None,
    intelligence: IntelligenceSummary | None = None,
    trust_summary: object = None,
) -> EvidenceBundle:
    kwargs: dict = {}
    if sources is not None:
        kwargs["sources"] = sources
    if intelligence is not None:
        kwargs["intelligence"] = intelligence
    if trust_summary is not None:
        kwargs["trust_summary"] = trust_summary
    return EvidenceBundle(
        startup_name="TestCo",
        documents=list(documents),
        **kwargs,
    )


@pytest.fixture
def strong_inputs():
    """A high-quality pipeline scenario: trusted, diverse, consistent."""
    bundle = make_bundle(
        make_document("doc-1", provider="website", trust=0.9),
        make_document("doc-2", provider="search", trust=0.85),
        make_document("doc-3", provider="crunchbase", trust=0.9),
        sources=[make_source(True), make_source(True)],
        intelligence=IntelligenceSummary(
            average_quality=0.9,
            classification_confidence=1.0,
        ),
    )
    observations = [
        make_observation("market", trust_score=0.9),
        make_observation("team", trust_score=0.85),
        make_observation("technology"),
    ]
    assessments = [
        make_assessment("market", confidence=0.9),
        make_assessment("team", confidence=0.9),
        make_assessment("technology", confidence=0.9),
    ]
    features = ExtractedFeatures(
        data_completeness=0.95,
        founder_profile_count=2,
        technology_stack=["python", "aws"],
    )
    return {
        "bundle": bundle,
        "observations": observations,
        "assessments": assessments,
        "features": features,
    }


@pytest.fixture
def weak_inputs():
    """A low-signal scenario: no evidence, no reasoning, sparse data."""
    features = ExtractedFeatures(data_completeness=0.05)
    return {
        "bundle": None,
        "observations": [],
        "assessments": [],
        "features": features,
    }


def compute(**kwargs):
    """Compute calibration with keyword defaults for brevity."""
    return compute_decision_confidence(**kwargs)
