"""Tests for the WI-3 grounding stage.

Covers the public interfaces, policy validation, replay behavior, the
delegating projector, grounded evidence construction, determinism, the
registry factory, and the invariants that keep grounding opt-in and the
research package free of trust computation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import predictron_engine.research as research
from predictron_engine.evidence.replay.dataset import (
    EvidenceCorpusError,
    load_corpus,
    rebuild_bundle,
    resolve_dataset_path,
)
from predictron_engine.research import (
    Evidence,
    EvidenceCollectionEngine,
    EvidenceCollector,
    ExtractionClaimProjector,
    GroundedCollector,
    GroundingPolicy,
    MissingDocumentPolicy,
    PlannerInput,
    ProductCollector,
    ProjectedClaim,
    ReplayDocumentSource,
    ResearchPlanner,
    ResearchPriority,
    ResearchTask,
    all_topic_ids,
    grounded_collector_registry,
)
from predictron_engine.research.exceptions import (
    InvalidCollectionInputError,
    InvalidEvidence,
)
from predictron_engine.research.grounding import (
    ClaimProjector,
    DocumentSource,
)
from predictron_engine.research.registry import COLLECTOR_REGISTRY

SAMPLE = "sample_evidence"
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


class _EmptySource:
    """A DocumentSource that always returns no documents."""

    def documents(
        self, *, company_name: str, website: str = ""
    ) -> tuple[object, ...]:
        return ()


class _StaticProjector:
    """A ClaimProjector that returns fixed claims over all documents."""

    def __init__(self, *claims: ProjectedClaim) -> None:
        self._claims = claims

    def project(
        self, task: ResearchTask, documents: tuple[object, ...], *, company_name: str
    ) -> tuple[ProjectedClaim, ...]:
        return self._claims


def _selected_documents(source: ReplayDocumentSource, policy: GroundingPolicy) -> list:
    documents = source.documents(company_name="ExampleCorp", website=WEBSITE)
    eligible = []
    for document in documents:
        trust = 0.0
        if document.metadata and document.metadata.trust_score:
            trust = document.metadata.trust_score.overall
        if trust >= policy.min_document_trust:
            eligible.append(document)
    eligible.sort(key=lambda document: document.id)
    return eligible[: policy.max_documents]


# ----------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------


def test_replay_source_satisfies_document_source() -> None:
    assert isinstance(ReplayDocumentSource(SAMPLE), DocumentSource)


def test_projector_satisfies_claim_projector() -> None:
    assert isinstance(ExtractionClaimProjector(), ClaimProjector)


def test_grounded_collector_is_evidence_collector() -> None:
    collector = GroundedCollector(
        GroundingPolicy(),
        source=ReplayDocumentSource(SAMPLE),
        projector=ExtractionClaimProjector(),
    )
    assert isinstance(collector, EvidenceCollector)
    assert collector.supports("product")
    assert set(collector.metadata().supported_topics) == set(all_topic_ids())


def test_sync_source_satisfies_document_source() -> None:
    from predictron_engine.integrations.grounding import SyncEvidenceSource

    class _Orchestrator:
        async def collect(self, startup_name: str, website: str) -> object:
            raise AssertionError("should not be called")

    assert isinstance(SyncEvidenceSource(_Orchestrator()), DocumentSource)


# ----------------------------------------------------------------------
# GroundingPolicy
# ----------------------------------------------------------------------


def test_policy_defaults() -> None:
    policy = GroundingPolicy()
    assert policy.max_documents == 5
    assert policy.min_document_trust == 0.0
    assert policy.min_claim_confidence == 0.0
    assert policy.max_claims_per_task == 3
    assert policy.on_missing_document is MissingDocumentPolicy.FALLBACK


def test_policy_rejects_invalid_documents_cap() -> None:
    with pytest.raises(InvalidCollectionInputError):
        GroundingPolicy(max_documents=0)


def test_policy_rejects_invalid_claims_cap() -> None:
    with pytest.raises(InvalidCollectionInputError):
        GroundingPolicy(max_claims_per_task=0)


def test_policy_rejects_out_of_range_trust() -> None:
    with pytest.raises(InvalidCollectionInputError):
        GroundingPolicy(min_document_trust=1.5)


def test_policy_rejects_out_of_range_confidence() -> None:
    with pytest.raises(InvalidCollectionInputError):
        GroundingPolicy(min_claim_confidence=-0.1)


def test_policy_rejects_non_enum_missing_policy() -> None:
    with pytest.raises(InvalidCollectionInputError):
        GroundingPolicy(on_missing_document="skip")  # type: ignore[arg-type]


def test_policy_rejects_bool_caps() -> None:
    with pytest.raises(InvalidCollectionInputError):
        GroundingPolicy(max_documents=True)


# ----------------------------------------------------------------------
# ProjectedClaim
# ----------------------------------------------------------------------


def test_projected_claim_normalises_document_ids() -> None:
    claim = ProjectedClaim(
        category="industry",
        claim="Software",
        confidence=1.0,
        document_ids=("b", "a", "a"),
    )
    assert claim.document_ids == ("a", "b")


def test_projected_claim_rejects_empty_text() -> None:
    with pytest.raises(InvalidEvidence):
        ProjectedClaim(category="x", claim="   ", confidence=1.0)


def test_projected_claim_rejects_bad_confidence() -> None:
    with pytest.raises(InvalidEvidence):
        ProjectedClaim(category="x", claim="y", confidence=1.5)


# ----------------------------------------------------------------------
# ReplayDocumentSource
# ----------------------------------------------------------------------


def test_replay_source_returns_enriched_documents() -> None:
    documents = ReplayDocumentSource(SAMPLE).documents(
        company_name="ExampleCorp", website=WEBSITE
    )
    assert documents
    assert all(document.metadata is not None for document in documents)
    assert any(
        document.metadata is not None and document.metadata.trust_score is not None
        for document in documents
    )


def test_replay_source_matches_committed_corpus() -> None:
    bundle = rebuild_bundle(load_corpus(resolve_dataset_path(SAMPLE)))
    documents = ReplayDocumentSource(SAMPLE).documents(company_name="x")
    assert tuple(bundle.documents) == documents


def test_replay_source_is_deterministic() -> None:
    source = ReplayDocumentSource(SAMPLE)
    first = source.documents(company_name="ExampleCorp")
    second = ReplayDocumentSource(SAMPLE).documents(company_name="ExampleCorp")
    assert tuple(document.id for document in first) == tuple(
        document.id for document in second
    )


def test_replay_source_missing_corpus_raises() -> None:
    with pytest.raises(EvidenceCorpusError):
        ReplayDocumentSource("does_not_exist").documents(company_name="x")


# ----------------------------------------------------------------------
# ExtractionClaimProjector (delegating, no invented mapping)
# ----------------------------------------------------------------------


def test_projector_projects_only_extracted_feature_fields() -> None:
    from predictron_engine.models.extracted_features import ExtractedFeatures

    documents = tuple(
        ReplayDocumentSource(SAMPLE).documents(company_name="ExampleCorp")
    )
    claims = ExtractionClaimProjector().project(
        _task(), documents, company_name="ExampleCorp"
    )
    assert claims
    field_names = set(type(ExtractedFeatures()).model_fields)
    for claim in claims:
        assert claim.category in field_names
        assert claim.confidence == 1.0
        assert claim.document_ids == tuple(sorted(d.id for d in documents))


def test_projector_returns_empty_without_documents() -> None:
    projector = ExtractionClaimProjector()
    assert projector.project(_task(), (), company_name="ExampleCorp") == ()


def test_projector_rejects_non_task() -> None:
    with pytest.raises(InvalidCollectionInputError):
        ExtractionClaimProjector().project(
            object(), (), company_name="ExampleCorp"  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# GroundedCollector
# ----------------------------------------------------------------------


def _grounded_collector(
    policy: GroundingPolicy | None = None,
    *,
    source: object | None = None,
    projector: object | None = None,
    fallback: EvidenceCollector | None = None,
) -> GroundedCollector:
    return GroundedCollector(
        policy or GroundingPolicy(),
        source=source or ReplayDocumentSource(SAMPLE),  # type: ignore[arg-type]
        projector=projector or ExtractionClaimProjector(),  # type: ignore[arg-type]
        fallback=fallback,
    )


def test_grounded_evidence_is_version_two_and_linked() -> None:
    collector = _grounded_collector()
    evidence = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    assert evidence
    for item in evidence:
        assert item.evidence_id_version == "2"
        assert item.document_ids
        assert item.citation_ids
        assert item.trust_score is not None


def test_grounded_evidence_is_deterministic() -> None:
    collector = _grounded_collector()
    first = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    second = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    assert [item.evidence_id for item in first] == [
        item.evidence_id for item in second
    ]


def test_grounded_evidence_imports_supporting_trust() -> None:
    policy = GroundingPolicy()
    collector = _grounded_collector(policy)
    evidence = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    selected = _selected_documents(ReplayDocumentSource(SAMPLE), policy)
    trust_objects = [
        document.metadata.trust_score
        for document in selected
        if document.metadata and document.metadata.trust_score
    ]
    assert all(
        any(item.trust_score == trust for trust in trust_objects)
        for item in evidence
    )


def test_grounded_collector_respects_max_claims() -> None:
    collector = _grounded_collector(GroundingPolicy(max_claims_per_task=1))
    evidence = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    assert len(evidence) == 1


def test_grounded_collector_caps_documents() -> None:
    policy = GroundingPolicy(max_documents=2)
    collector = _grounded_collector(policy)
    evidence = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    selected_ids = {
        document.id
        for document in _selected_documents(ReplayDocumentSource(SAMPLE), policy)
    }
    for item in evidence:
        assert set(item.document_ids) <= selected_ids
        assert len(item.document_ids) <= 2


def test_grounded_collector_drops_unknown_document_claims() -> None:
    policy = GroundingPolicy()
    selected = _selected_documents(ReplayDocumentSource(SAMPLE), policy)
    selected_ids = tuple(document.id for document in selected)
    projector = _StaticProjector(
        ProjectedClaim("a", "kept", 1.0, (selected_ids[0],)),
        ProjectedClaim("b", "dropped", 1.0, ("does-not-exist",)),
    )
    collector = _grounded_collector(policy, projector=projector)
    evidence = collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE)
    assert [item.category for item in evidence] == ["a"]
    assert set(evidence[0].document_ids) <= set(selected_ids)


def test_grounded_collector_missing_skip_returns_empty() -> None:
    collector = _grounded_collector(
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.SKIP),
        source=_EmptySource(),
    )
    assert collector.collect(_task(), company_name="ExampleCorp") == ()


def test_grounded_collector_missing_error_raises() -> None:
    collector = _grounded_collector(
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.ERROR),
        source=_EmptySource(),
    )
    with pytest.raises(InvalidEvidence):
        collector.collect(_task(), company_name="ExampleCorp")


def test_grounded_collector_missing_fallback_without_collector_skips() -> None:
    collector = _grounded_collector(
        source=_EmptySource(),
    )
    assert collector.collect(_task(), company_name="ExampleCorp") == ()


def test_grounded_collector_falls_back_to_collector() -> None:
    fallback = ProductCollector()
    collector = _grounded_collector(source=_EmptySource(), fallback=fallback)
    expected = fallback.collect(_task(), company_name="ExampleCorp")
    produced = collector.collect(_task(), company_name="ExampleCorp")
    assert [item.evidence_id for item in produced] == [
        item.evidence_id for item in expected
    ]


def test_grounded_collector_drops_unknown_cited_claim() -> None:
    projector = _StaticProjector(
        ProjectedClaim(category="x", claim="y", confidence=1.0, document_ids=("nope",))
    )
    collector = _grounded_collector(
        GroundingPolicy(on_missing_document=MissingDocumentPolicy.SKIP),
        projector=projector,
    )
    assert collector.collect(_task(), company_name="ExampleCorp", website=WEBSITE) == ()


def test_grounded_collector_rejects_blank_company() -> None:
    with pytest.raises(InvalidCollectionInputError):
        _grounded_collector().collect(_task(), company_name="   ")


def test_grounded_collector_rejects_non_task() -> None:
    with pytest.raises(InvalidCollectionInputError):
        _grounded_collector().collect(object(), company_name="x")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_grounded_registry_does_not_mutate_shared_registry() -> None:
    before = COLLECTOR_REGISTRY.list_collectors()
    registry = grounded_collector_registry(
        GroundingPolicy(),
        source=ReplayDocumentSource(SAMPLE),
        projector=ExtractionClaimProjector(),
    )
    assert registry is not COLLECTOR_REGISTRY
    assert COLLECTOR_REGISTRY.list_collectors() == before
    assert not COLLECTOR_REGISTRY.has_collector("grounded_collector")
    assert registry.has_collector("grounded_collector")


def test_grounded_registry_covers_every_topic() -> None:
    registry = grounded_collector_registry(
        GroundingPolicy(),
        source=ReplayDocumentSource(SAMPLE),
        projector=ExtractionClaimProjector(),
    )
    assert set(registry.topics_covered()) == set(all_topic_ids())
    for topic_id in all_topic_ids():
        assert isinstance(registry.resolve(topic_id), GroundedCollector)


def test_grounded_registry_drives_collection_engine() -> None:
    registry = grounded_collector_registry(
        GroundingPolicy(),
        source=ReplayDocumentSource(SAMPLE),
        projector=ExtractionClaimProjector(),
    )
    engine = EvidenceCollectionEngine(registry=registry)
    result = engine.execute_for_company("ExampleCorp", website=WEBSITE)
    assert result.collection.evidence
    assert all(item.evidence_id_version == "2" for item in result.collection.evidence)


def test_default_registry_stays_placeholder_version_one() -> None:
    engine = EvidenceCollectionEngine()
    result = engine.execute_for_company("ExampleCorp")
    assert result.collection.evidence
    assert all(item.evidence_id_version == "1" for item in result.collection.evidence)


def test_default_plan_evidence_ids_unchanged_by_grounding_import() -> None:
    plan = ResearchPlanner().create_plan(PlannerInput(company_name="Acme"))
    engine = EvidenceCollectionEngine()
    result = engine.execute(plan)
    for item in result.collection.evidence:
        assert item.evidence_id_version == "1"


# ----------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------


def test_research_package_never_references_trust_computation() -> None:
    package_root = Path(research.__file__).parent  # type: ignore[arg-type]
    for path in package_root.glob("*.py"):
        assert "compute_document_trust" not in path.read_text(encoding="utf-8")


def test_grounding_does_not_change_placeholder_evidence_shape() -> None:
    task = _task("product")
    placeholder = ProductCollector().collect(task, company_name="Acme")
    assert all(isinstance(item, Evidence) for item in placeholder)
    assert all(item.evidence_id_version == "1" for item in placeholder)
