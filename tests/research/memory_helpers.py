"""Builders for the Phase 8 Sprint 6 research memory tests.

The helpers here are deliberately plain functions rather than fixtures: the
graph layer tests want to control every field of every value, and a factory
that hides a field is a factory that cannot be used to prove that field is
optional.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from predictron_engine.research.execution_report import (
    REPORT_SCHEMA_VERSION,
    ExecutionSummary,
    ResearchReport,
)
from predictron_engine.research.execution_state import ExecutionState
from predictron_engine.research.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    Provenance,
    RelationshipType,
    ValidationMarker,
)
from predictron_engine.research.linking import entity_id, normalize_name
from predictron_engine.research.models import Evidence, EvidenceReference


def make_evidence(
    *,
    task_id: str = "task-1",
    claim: str = "Acme builds widgets",
    url: str = "https://example.test/a",
    source: str = "sec-1",
    title: str = "Acme filing",
    topic_id: str = "topic-1",
    category: str = "company",
    confidence: float = 0.8,
    evidence_id_value: str = "",
) -> Evidence:
    """Build a deterministic :class:`Evidence` item."""
    return Evidence(
        task_id=task_id,
        topic_id=topic_id,
        collector_id="sec",
        category=category,
        claim=claim,
        confidence=confidence,
        reference=EvidenceReference(
            source_category="filing",
            source_identifier=source,
            url=url,
            description=title,
        ),
        evidence_id=evidence_id_value,
    )


def make_report(
    report_id: str = "rep-1",
    company_name: str = "Acme",
    state: ExecutionState = ExecutionState.COMPLETED,
) -> ResearchReport:
    """Build a deterministic :class:`ResearchReport`."""
    return ResearchReport(
        schema_version=REPORT_SCHEMA_VERSION,
        report_id=report_id,
        execution_id=f"exec-{report_id}",
        company_name=company_name,
        plan_id="plan-1",
        state=state,
        summary=ExecutionSummary(),
    )


def make_entity(
    name: str,
    entity_type: EntityType | str = EntityType.TECHNOLOGY,
    *,
    identifier: str | None = None,
    canonical_name: str = "",
    aliases: tuple[str, ...] = (),
    attributes: Mapping[str, Any] | None = None,
    evidence_ids: tuple[str, ...] = (),
    source_ids: tuple[str, ...] = (),
    urls: tuple[str, ...] = (),
    report_ids: tuple[str, ...] = ("rep-1",),
    confidence: float = 0.9,
    last_updated: str = "rep-1",
    validation: ValidationMarker | None = None,
    topic_id: str = "",
    record_id: str = "rep-1",
) -> GraphEntity:
    """Build a fully-provenanced node for the graph-level tests."""
    resolved_type = EntityType(entity_type)
    return GraphEntity(
        entity_id=(
            identifier
            if identifier is not None
            else entity_id(resolved_type.value, name)
        ),
        entity_type=resolved_type,
        name=name,
        canonical_name=canonical_name or normalize_name(name),
        aliases=aliases,
        attributes=tuple(sorted((attributes or {}).items())),
        provenance=Provenance(
            evidence_ids=evidence_ids,
            source_ids=source_ids,
            urls=urls,
            report_ids=report_ids,
        ),
        confidence=confidence,
        last_updated=last_updated,
        validation=validation or ValidationMarker(),
        topic_id=topic_id,
        record_id=record_id,
    )


def make_relationship(
    source_id: str,
    target_id: str,
    relationship_type: RelationshipType | str,
    *,
    confidence: float = 0.9,
    evidence_ids: tuple[str, ...] = (),
    source_ids: tuple[str, ...] = (),
    report_ids: tuple[str, ...] = ("rep-1",),
    timestamp: str = "rep-1",
    attributes: Mapping[str, Any] | None = None,
) -> GraphRelationship:
    """Build a fully-provenanced edge for the graph-level tests."""
    return GraphRelationship(
        source_id=source_id,
        target_id=target_id,
        relationship_type=RelationshipType(relationship_type),
        confidence=confidence,
        provenance=Provenance(
            evidence_ids=evidence_ids,
            source_ids=source_ids,
            report_ids=report_ids,
        ),
        timestamp=timestamp,
        evidence_ids=evidence_ids,
        attributes=tuple(sorted((attributes or {}).items())),
    )
