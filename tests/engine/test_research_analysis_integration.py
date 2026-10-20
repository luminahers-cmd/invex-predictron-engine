"""Tests for the WI-6 research/analysis integration seam.

Covers the public boundary surface, the projection (identity, provenance,
and imported trust preserved), the opt-in engine wiring (byte-identical
when disabled), grounded execution through the existing review pieces,
and the invariants that keep the seam additive and trust-importing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from predictron_engine.engine import PredictronEngine
from predictron_engine.evidence.evidence_models import EvidenceSet
from predictron_engine.integrations import research_analysis
from predictron_engine.integrations.research_analysis import (
    RESEARCH_PROVENANCE_SCHEMA,
    OrchestratedResearchSource,
    ResearchAnalysisIntegration,
    ResearchEvidenceProjector,
    ResearchEvidenceSource,
    ResearchResultSource,
    grounded_research_source,
)
from predictron_engine.models.report import EvidenceItem, Report
from predictron_engine.research import (
    Evidence,
    PlannerInput,
    ResearchExecutionResult,
    ResearchOrchestrator,
)
from predictron_engine.research.exceptions import InvalidEvidence
from predictron_engine.research.grounding import (
    ExtractionClaimProjector,
    ReplayDocumentSource,
)
from predictron_engine.research.provenance_bridge import trust_score_payload

SAMPLE = "sample_evidence"
COMPANY = "ExampleCorp"
WEBSITE = "https://example.com"


# ----------------------------------------------------------------------
# Helpers / doubles
# ----------------------------------------------------------------------


class _EmptySource:
    """A ResearchEvidenceSource that always yields nothing."""

    def evidence(
        self, *, company_name: str, website: str = ""
    ) -> tuple[Evidence, ...]:
        return ()


class _FailingSource:
    """A ResearchEvidenceSource that raises while collecting."""

    def evidence(
        self, *, company_name: str, website: str = ""
    ) -> tuple[Evidence, ...]:
        raise RuntimeError("research backend unavailable")


def _grounded_source() -> OrchestratedResearchSource:
    return grounded_research_source(
        ReplayDocumentSource(SAMPLE), ExtractionClaimProjector()
    )


def _grounded_evidence() -> tuple[Evidence, ...]:
    return _grounded_source().evidence(company_name=COMPANY, website=WEBSITE)


def _execution_result() -> ResearchExecutionResult:
    return ResearchOrchestrator().execute(
        PlannerInput(company_name=COMPANY, website=WEBSITE)
    )


def _empty_set() -> EvidenceSet:
    return EvidenceSet(items=[], provider_count=0, feature_coverage=0.0)


def _record(item: EvidenceItem) -> dict[str, Any]:
    """Parse a projected item's provenance record, asserting it exists."""
    assert item.provenance_record is not None
    parsed = json.loads(item.provenance_record)
    assert isinstance(parsed, dict)
    return parsed


def _stable(report: Report) -> dict[str, Any]:
    """A report dump with volatile run metadata normalised away."""
    data = report.model_dump()
    data.get("startup", {}).pop("normalized_at", None)
    metadata = data.get("analysis_metadata", {})
    metadata.pop("timestamp", None)
    metadata.pop("processing_time_ms", None)
    return data


# ----------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------


def test_orchestrated_source_satisfies_protocol() -> None:
    assert isinstance(_grounded_source(), ResearchEvidenceSource)


def test_result_source_satisfies_protocol() -> None:
    assert isinstance(
        ResearchResultSource(_execution_result()), ResearchEvidenceSource
    )


def test_grounded_factory_returns_orchestrated_source() -> None:
    assert isinstance(_grounded_source(), OrchestratedResearchSource)


def test_integration_rejects_non_source() -> None:
    with pytest.raises(TypeError):
        ResearchAnalysisIntegration(object())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Projection: field mapping, identity, provenance, imported trust
# ----------------------------------------------------------------------


def test_projection_maps_engine_visible_fields() -> None:
    source = _grounded_evidence()[0]
    projected = ResearchEvidenceProjector().project(source)
    assert projected.domain == source.topic_id
    assert projected.category == source.category
    assert projected.statement == source.claim
    assert projected.source == source.collector_id
    assert projected.relevance_score == source.confidence
    assert projected.citations == []


def test_projection_is_one_to_one() -> None:
    evidence = _grounded_evidence()
    items = ResearchEvidenceProjector().project_all(evidence)
    assert len(items) == len(evidence)
    assert len({item.provenance_record for item in items}) == len(items)


def test_projection_preserves_evidence_identity() -> None:
    evidence = _grounded_evidence()
    items = ResearchEvidenceProjector().project_all(evidence)
    by_id = {e.evidence_id: e for e in evidence}
    for item in items:
        record = _record(item)
        source = by_id[record["research_evidence_id"]]
        assert record["research_evidence_id"] == source.evidence_id
        assert (
            record["research_evidence_id_version"]
            == source.evidence_id_version
        )
        assert record["research_document_ids"] == list(source.document_ids)
        assert record["research_citation_ids"] == list(source.citation_ids)


def test_projection_preserves_provenance_reference() -> None:
    source = _grounded_evidence()[0]
    record = _record(ResearchEvidenceProjector().project(source))
    reference = source.reference
    assert record["research_source_url"] == reference.url
    assert record["research_source_category"] == reference.source_category
    assert record["research_source_identifier"] == reference.source_identifier
    assert record["url"] == reference.url
    assert record["document_type"] == reference.source_category


def test_projection_imports_trust_without_recomputing() -> None:
    for source in _grounded_evidence():
        record = _record(ResearchEvidenceProjector().project(source))
        # The serialised trust is exactly the payload of the imported object.
        assert record["research_trust_score"] == trust_score_payload(
            source.trust_score
        )
        if source.trust_score is not None:
            assert record["trust_score"] == source.trust_score.overall
            assert (
                record["trust_level"]
                == source.trust_score.source_trust_level.value
            )


def test_projection_is_deterministic_and_order_independent() -> None:
    evidence = _grounded_evidence()
    forward = ResearchEvidenceProjector().project_all(evidence)
    backward = ResearchEvidenceProjector().project_all(
        tuple(reversed(evidence))
    )
    assert [item.provenance_record for item in forward] == [
        item.provenance_record for item in backward
    ]


def test_projection_emits_provenance_schema_tag() -> None:
    record = _record(
        ResearchEvidenceProjector().project(_grounded_evidence()[0])
    )
    assert record["record_schema"] == RESEARCH_PROVENANCE_SCHEMA


def test_projection_rejects_non_evidence() -> None:
    with pytest.raises(InvalidEvidence):
        ResearchEvidenceProjector().project(object())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Boundary: augment semantics
# ----------------------------------------------------------------------


def test_augment_empty_source_returns_input_unchanged() -> None:
    integration = ResearchAnalysisIntegration(_EmptySource())
    base = _empty_set()
    assert integration.augment(base, company_name=COMPANY) is base


def test_augment_failure_returns_input_unchanged() -> None:
    integration = ResearchAnalysisIntegration(_FailingSource())
    base = _empty_set()
    assert integration.augment(base, company_name=COMPANY) is base


def test_augment_rejects_non_evidence_set() -> None:
    integration = ResearchAnalysisIntegration(_EmptySource())
    with pytest.raises(TypeError):
        integration.augment(object(), company_name=COMPANY)  # type: ignore[arg-type]


def test_augment_appends_and_preserves_set_metadata() -> None:
    integration = ResearchAnalysisIntegration(_grounded_source())
    base = EvidenceSet(items=[], provider_count=3, feature_coverage=0.75)
    augmented = integration.augment(
        base, company_name=COMPANY, website=WEBSITE
    )
    assert augmented is not base
    assert base.items == []
    assert len(augmented.items) == len(_grounded_evidence())
    assert augmented.provider_count == 3
    assert augmented.feature_coverage == 0.75


def test_augment_keeps_original_items_as_prefix() -> None:
    existing = EvidenceItem(
        domain="industry",
        category="market_size",
        statement="Existing engine evidence.",
        source="knowledge/taxonomies.py",
        relevance_score=1.0,
    )
    base = EvidenceSet(
        items=[existing], provider_count=1, feature_coverage=1.0
    )
    augmented = ResearchAnalysisIntegration(_grounded_source()).augment(
        base, company_name=COMPANY, website=WEBSITE
    )
    assert augmented.items[0] == existing
    assert augmented.items[1:]


# ----------------------------------------------------------------------
# Engine wiring: opt-in and byte-identical when disabled
# ----------------------------------------------------------------------


def test_engine_defaults_without_integration() -> None:
    assert PredictronEngine()._research_integration is None


def test_engine_disabled_is_byte_identical(minimal_request) -> None:
    default = PredictronEngine().analyze(minimal_request, request_id="fixed")
    explicit = PredictronEngine(research_integration=None).analyze(
        minimal_request, request_id="fixed"
    )
    assert _stable(default) == _stable(explicit)
    assert default.evidence == explicit.evidence


def test_engine_disabled_does_not_add_research_evidence(
    minimal_request,
) -> None:
    report = PredictronEngine().analyze(minimal_request, request_id="fixed")
    assert all(item.provenance_record is None for item in report.evidence)


def test_engine_enabled_appends_projected_research_evidence(
    minimal_request,
) -> None:
    baseline = PredictronEngine().analyze(minimal_request, request_id="fixed")
    integration = ResearchAnalysisIntegration(_grounded_source())
    augmented = PredictronEngine(research_integration=integration).analyze(
        minimal_request, request_id="fixed"
    )
    assert len(augmented.evidence) > len(baseline.evidence)
    # The original engine evidence is preserved as the untouched prefix.
    assert augmented.evidence[: len(baseline.evidence)] == baseline.evidence

    source_ids = {e.evidence_id for e in _grounded_evidence()}
    projected_ids = {
        _record(item).get("research_evidence_id")
        for item in augmented.evidence
        if item.provenance_record
    }
    assert projected_ids == source_ids


def test_engine_enabled_with_existing_result_source(minimal_request) -> None:
    integration = ResearchAnalysisIntegration(
        ResearchResultSource(_execution_result())
    )
    report = PredictronEngine(research_integration=integration).analyze(
        minimal_request, request_id="fixed"
    )
    projected = [i for i in report.evidence if i.provenance_record]
    assert projected
    assert all(
        _record(i).get("research_evidence_id_version") == "1"
        for i in projected
    )


# ----------------------------------------------------------------------
# Grounded execution through the existing components
# ----------------------------------------------------------------------


def test_grounded_source_executes_version_two_evidence() -> None:
    evidence = _grounded_evidence()
    assert evidence
    assert all(item.evidence_id_version == "2" for item in evidence)
    assert all(item.trust_score is not None for item in evidence)


def test_grounded_source_is_deterministic() -> None:
    first = _grounded_source().evidence(company_name=COMPANY, website=WEBSITE)
    second = _grounded_source().evidence(company_name=COMPANY, website=WEBSITE)
    assert [item.evidence_id for item in first] == [
        item.evidence_id for item in second
    ]


# ----------------------------------------------------------------------
# Invariants: additive, trust-importing, purity of the research package
# ----------------------------------------------------------------------


def test_module_never_recomputes_trust_or_provenance() -> None:
    text = Path(research_analysis.__file__).read_text(encoding="utf-8")
    assert "compute_document_trust" not in text
    assert "build_provenance_record" not in text


def test_module_does_not_import_the_app_layer() -> None:
    text = Path(research_analysis.__file__).read_text(encoding="utf-8")
    assert "from app" not in text
    assert "import app" not in text
