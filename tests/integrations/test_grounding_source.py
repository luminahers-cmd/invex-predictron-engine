"""WI-8 tests: live grounding provider validation.

These tests exercise the *live* grounding adapter
(:class:`~predictron_engine.integrations.grounding.SyncEvidenceSource`)
against the real, provider-backed evidence boundary — the async
:class:`EvidenceOrchestrator` — with **no network access**.  They validate
the architecture the adapter depends on:

* the adapter satisfies the synchronous ``DocumentSource`` protocol and
  returns exactly the documents the evidence subsystem produced (fully
  enriched, deterministically ordered);
* a **provider** failure is isolated by :class:`EvidenceOrchestrator`
  (recorded as a failed ``ProviderRun``, never raised), so the adapter
  yields no documents and the configured
  :class:`GroundingPolicy` ``on_missing_document`` behaviour decides the
  outcome; and
* grounding, the collector registry, and the collection engine compose
  end to end.

The adapter introduces **no failure-handling boundary of its own**: the
production best-effort guarantee that a failed collection never aborts an
analysis lives one layer up, at
:meth:`ResearchAnalysisIntegration.augment
<predictron_engine.integrations.research_analysis.ResearchAnalysisIntegration.augment>`.
These tests cover the provider-isolation behaviour that guarantee relies
on.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from predictron_engine.evidence.models import EvidenceDocument
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    EvidenceProvider,
    ProviderResult,
)
from predictron_engine.evidence.replay.provider import ReplayEvidenceProvider
from predictron_engine.integrations.grounding import SyncEvidenceSource
from predictron_engine.research import (
    CollectionStatus,
    EvidenceCollectionEngine,
    ExtractionClaimProjector,
    GroundedCollector,
    GroundingPolicy,
    MissingDocumentPolicy,
    ProductCollector,
    ReplayDocumentSource,
    ResearchPriority,
    ResearchTask,
    all_topic_ids,
    grounded_collector_registry,
)
from predictron_engine.research.exceptions import InvalidEvidence
from predictron_engine.research.grounding import DocumentSource

DATASET = "sample_evidence"
COMPANY = "ExampleCorp"
WEBSITE = "https://example.com"


def _task(topic_id: str = "product") -> ResearchTask:
    return ResearchTask(
        task_id=f"research_{topic_id}",
        topic_id=topic_id,
        title=f"Research {topic_id}",
        description=f"Research {topic_id} for the company.",
        priority=ResearchPriority.HIGH,
        priority_score=0.8,
        importance=0.8,
        prediction_impact=0.8,
        freshness_requirement_days=365,
        dependency_ids=(),
        source_categories=(),
        priority_rank=1,
        execution_order=1,
    )


def _orchestrator(
    providers: Sequence[EvidenceProvider],
) -> EvidenceOrchestrator:
    return EvidenceOrchestrator(providers=list(providers))


def _provider_orchestrator() -> EvidenceOrchestrator:
    """A real orchestrator over the offline replay provider (no network)."""
    return _orchestrator([ReplayEvidenceProvider(DATASET)])


def _live_source() -> SyncEvidenceSource:
    return SyncEvidenceSource(_provider_orchestrator())


def _failing_source() -> SyncEvidenceSource:
    return SyncEvidenceSource(_orchestrator([_FailingProvider()]))


def _grounded(
    source: DocumentSource,
    policy: GroundingPolicy | None = None,
    *,
    fallback: object | None = None,
) -> GroundedCollector:
    return GroundedCollector(
        policy or GroundingPolicy(),
        source=source,
        projector=ExtractionClaimProjector(),
        fallback=fallback,  # type: ignore[arg-type]
    )


class _FailingProvider:
    """A provider whose collection always raises (never network)."""

    name = "failing"

    def can_collect(self, context: CollectContext) -> bool:
        return True

    async def collect(self, context: CollectContext) -> ProviderResult:
        raise RuntimeError("provider exploded")


# ----------------------------------------------------------------------
# Adapter contract
# ----------------------------------------------------------------------


def test_live_source_satisfies_document_source() -> None:
    assert isinstance(_live_source(), DocumentSource)


def test_live_source_supports_factory_collaborator() -> None:
    source = SyncEvidenceSource(_provider_orchestrator)
    documents = source.documents(company_name=COMPANY, website=WEBSITE)
    assert documents
    assert all(
        isinstance(document, EvidenceDocument) for document in documents
    )


def test_live_source_rejects_invalid_collaborator() -> None:
    with pytest.raises(TypeError):
        SyncEvidenceSource(object())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Provider-backed collection (offline replay through the real orchestrator)
# ----------------------------------------------------------------------


def test_live_source_returns_enriched_documents() -> None:
    documents = _live_source().documents(company_name=COMPANY, website=WEBSITE)
    assert documents
    assert all(
        isinstance(document, EvidenceDocument) for document in documents
    )
    assert all(document.metadata is not None for document in documents)
    assert any(
        document.metadata is not None
        and document.metadata.trust_score is not None
        for document in documents
    )


def test_live_source_is_deterministic() -> None:
    first = _live_source().documents(company_name=COMPANY, website=WEBSITE)
    second = _live_source().documents(company_name=COMPANY, website=WEBSITE)
    assert tuple(document.id for document in first) == tuple(
        document.id for document in second
    )


def test_live_source_empty_website_yields_no_documents() -> None:
    assert _live_source().documents(company_name=COMPANY, website="") == ()


# ----------------------------------------------------------------------
# Provider isolation (owned by EvidenceOrchestrator, not the adapter)
# ----------------------------------------------------------------------


async def test_orchestrator_isolates_provider_failure() -> None:
    orchestrator = _orchestrator([_FailingProvider()])
    bundle = await orchestrator.collect(COMPANY, WEBSITE)
    assert bundle.documents == []
    assert [run.success for run in bundle.providers] == [False]
    assert bundle.providers[0].failure_reason


def test_live_source_provider_failure_yields_no_documents() -> None:
    assert (
        _failing_source().documents(company_name=COMPANY, website=WEBSITE)
        == ()
    )


# ----------------------------------------------------------------------
# Provider-backed grounding end to end
# ----------------------------------------------------------------------


def test_provider_backed_evidence_is_grounded_version_two() -> None:
    evidence = _grounded(_live_source()).collect(
        _task(), company_name=COMPANY, website=WEBSITE
    )
    assert evidence
    for item in evidence:
        assert item.evidence_id_version == "2"
        assert item.document_ids
        assert item.citation_ids
        assert item.trust_score is not None


def test_provider_backed_evidence_is_deterministic() -> None:
    first = _grounded(_live_source()).collect(
        _task(), company_name=COMPANY, website=WEBSITE
    )
    second = _grounded(_live_source()).collect(
        _task(), company_name=COMPANY, website=WEBSITE
    )
    assert [item.evidence_id for item in first] == [
        item.evidence_id for item in second
    ]


def test_provider_backed_evidence_imports_supporting_trust() -> None:
    evidence = _grounded(_live_source()).collect(
        _task(), company_name=COMPANY, website=WEBSITE
    )
    documents = _live_source().documents(company_name=COMPANY, website=WEBSITE)
    trust_objects = [
        document.metadata.trust_score
        for document in documents
        if document.metadata and document.metadata.trust_score
    ]
    assert evidence
    assert all(
        any(item.trust_score == trust for trust in trust_objects)
        for item in evidence
    )


def test_provider_backed_registry_drives_collection_engine() -> None:
    registry = grounded_collector_registry(
        GroundingPolicy(),
        source=_live_source(),
        projector=ExtractionClaimProjector(),
    )
    engine = EvidenceCollectionEngine(registry=registry)
    result = engine.execute_for_company(COMPANY, website=WEBSITE)
    assert result.collection.evidence
    assert all(
        item.evidence_id_version == "2" for item in result.collection.evidence
    )


# ----------------------------------------------------------------------
# Missing documents degrade through the existing GroundingPolicy
# ----------------------------------------------------------------------


def test_provider_failure_skip_policy_returns_no_evidence() -> None:
    collector = _grounded(
        _failing_source(),
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.SKIP),
    )
    assert (
        collector.collect(_task(), company_name=COMPANY, website=WEBSITE) == ()
    )


def test_provider_failure_error_policy_raises_invalid_evidence() -> None:
    collector = _grounded(
        _failing_source(),
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.ERROR),
    )
    with pytest.raises(InvalidEvidence):
        collector.collect(_task(), company_name=COMPANY, website=WEBSITE)


def test_provider_failure_fallback_policy_defers_to_collector() -> None:
    fallback = ProductCollector()
    collector = _grounded(
        _failing_source(),
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.FALLBACK),
        fallback=fallback,
    )
    produced = collector.collect(
        _task(), company_name=COMPANY, website=WEBSITE
    )
    expected = fallback.collect(_task(), company_name=COMPANY, website=WEBSITE)
    assert [item.evidence_id for item in produced] == [
        item.evidence_id for item in expected
    ]


def test_provider_failure_does_not_abort_collection_engine() -> None:
    registry = grounded_collector_registry(
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.SKIP),
        source=_failing_source(),
        projector=ExtractionClaimProjector(),
    )
    engine = EvidenceCollectionEngine(registry=registry)
    result = engine.execute_for_company(COMPANY, website=WEBSITE)
    assert result.collection.evidence == ()
    assert result.status is CollectionStatus.SUCCESS
    assert not result.failed_tasks


def test_provider_failure_error_policy_records_failed_tasks() -> None:
    registry = grounded_collector_registry(
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.ERROR),
        source=_failing_source(),
        projector=ExtractionClaimProjector(),
    )
    engine = EvidenceCollectionEngine(registry=registry)
    result = engine.execute_for_company(COMPANY, website=WEBSITE)
    assert result.collection.evidence == ()
    assert result.status is CollectionStatus.FAILED


# ----------------------------------------------------------------------
# Opt-in invariant: default grounding is untouched
# ----------------------------------------------------------------------


def test_grounded_collector_supports_every_topic() -> None:
    collector = _grounded(ReplayDocumentSource(DATASET), GroundingPolicy())
    assert set(collector.supported_topics) == set(all_topic_ids())
