"""Ports for research — the persistence seams and the execution seam.

This module is the *seam* later sprints will implement.  It declares the
behavioural contracts the research core depends on — a sequence index, a
graph container, a record ledger, and a pipeline executor — without naming
any concrete implementation.

What a port is here
-------------------
A port describes a **capability**, not a class.  Nothing in this module
inherits from, instantiates, or imports at runtime any of the concrete
implementations; the in-memory stores and the pipeline orchestrator simply
satisfy the contracts structurally, so they remain the defaults and no
behaviour changes.

Design rules
------------
* Structural typing only — the protocols are ``Protocol`` subclasses, and the
  concrete implementations never inherit from them.
* ``@runtime_checkable`` — construction sites validate collaborators with
  ``isinstance``, the same way the concrete classes were validated before.
* Minimal surface — every declared member is long-lived API that a future
  backend must honour, so only the members the research code paths actually
  exercise are declared.  Reconstruction helpers such as ``from_dict``
  deliberately stay on the concrete classes, because rebuilding state from a
  document is the implementation's own concern.

The ports
---------
``MemoryIndexPort``
    The append-order index.  Records, graph facts, and history all sequence
    through one of these, which is what makes snapshot order reproducible.
``GraphStorePort``
    The append-only container of nodes and edges that the knowledge graph
    reads through and :class:`~predictron_engine.research.memory.ResearchMemory`
    writes through.
``MemoryStorePort``
    The ledger of reports and evidence, plus the duplicate-aware write path
    research memory calls directly.
``ResearchExecutorPort``
    The execution boundary of a full pipeline pass — exactly
    ``execute(planner_input) -> ResearchExecutionResult``.  The research loop
    runs through this contract instead of duck-typing an ``execute``
    attribute, so any executor that satisfies it structurally is a valid
    orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from predictron_engine.research.execution_report import (
        ResearchExecutionResult,
        ResearchReport,
    )
    from predictron_engine.research.graph_models import (
        EntityType,
        GraphEntity,
        GraphRelationship,
        GraphStats,
        GraphValidation,
        RelationshipType,
    )
    from predictron_engine.research.graph_store import (
        EntityWrite,
        GraphDirection,
        MemoryRecordKey,
        RelationshipWrite,
    )
    from predictron_engine.research.memory_store import (
        EvidenceWrite,
        ReportWrite,
        TopicRetrieval,
    )
    from predictron_engine.research.models import Evidence, PlannerInput


@runtime_checkable
class MemoryIndexPort(Protocol):
    """The shared append-order index every research store sequences through.

    One index serves the record store, the graph store, and the history log,
    so "the newest report" and "the newest graph fact" come from a single
    deterministic order rather than from a clock.
    """

    def allocate(
        self,
        record_type: str,
        record_id: str,
        *,
        topic_id: str = "",
        company_name: str = "",
        stored_at: str = "",
    ) -> MemoryRecordKey:
        """Return the key for a record, allocating a sequence if it is new.

        Re-allocating an already indexed ``(record_type, record_id)`` pair
        must be a no-op that returns the existing key, because replaying a
        report may not reorder anything stored after it.
        """
        ...

    def next_sequence(self) -> int:
        """Return the sequence the next allocation will receive."""
        ...

    def key_for(self, record_type: str, record_id: str) -> MemoryRecordKey | None:
        """Return the key already allocated for a record, or ``None``."""
        ...

    def keys(self, record_type: str | None = None) -> tuple[MemoryRecordKey, ...]:
        """Return every key in sequence order, optionally narrowed to one type."""
        ...

    def to_dict(self) -> dict[str, object]:
        """Serialize the index into its JSON-ready snapshot document."""
        ...

    def __len__(self) -> int:
        """Return how many ``(record_type, record_id)`` pairs are indexed."""
        ...


@runtime_checkable
class GraphStorePort(Protocol):
    """The append-only container of nodes and edges behind the knowledge graph.

    The knowledge graph reads through this contract and research memory
    writes through it, so a backend only has to reproduce append, merge, and
    ordered reads to be a drop-in replacement.
    """

    index: MemoryIndexPort

    def add_entity(
        self,
        entity: GraphEntity,
        *,
        stored_at: str = "",
        record_id: str = "",
    ) -> EntityWrite:
        """Store a node, merging it with any node already present.

        The write must return the pre-write value so the caller can snapshot
        it into history; replacing a stored node outright is not permitted.
        """
        ...

    def add_relationship(
        self,
        relationship: GraphRelationship,
        *,
        stored_at: str = "",
    ) -> RelationshipWrite:
        """Store an edge, strengthening it when it already exists."""
        ...

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Return the node for ``entity_id`` or ``None``."""
        ...

    def entity_type(self, entity_id: str) -> EntityType | None:
        """Return the type of a node, or ``None`` when it is absent."""
        ...

    def entity_ids(self) -> tuple[str, ...]:
        """Return every node identifier, sorted."""
        ...

    def entities(self) -> tuple[GraphEntity, ...]:
        """Return every node, identifier-sorted."""
        ...

    def entities_of_type(self, entity_type: EntityType) -> tuple[GraphEntity, ...]:
        """Return every node of ``entity_type``, identifier-sorted."""
        ...

    def relationships(self) -> tuple[GraphRelationship, ...]:
        """Return every edge, key-sorted."""
        ...

    def relationships_by_key(
        self,
        key: tuple[str, str, str],
    ) -> GraphRelationship | None:
        """Return the edge stored under a canonical key or ``None``."""
        ...

    def relationship_count(self) -> int:
        """Return how many edges are stored."""
        ...

    def get_relationship(
        self,
        source_id: str,
        relationship_type: RelationshipType,
        target_id: str,
    ) -> GraphRelationship | None:
        """Return the edge between two endpoints or ``None``."""
        ...

    def incident_keys(
        self,
        entity_id: str,
        direction: GraphDirection,
    ) -> tuple[tuple[str, str, str], ...]:
        """Return the edge keys touching ``entity_id``, key-sorted.

        ``direction`` is required rather than defaulted: a default value
        would need the concrete direction enum at import time, and this
        module must stay free of runtime imports from the stores it ports.
        """
        ...

    def other_endpoint(self, key: tuple[str, str, str], entity_id: str) -> str:
        """Return the endpoint of ``key`` that is not ``entity_id``."""
        ...

    def stats(self) -> GraphStats:
        """Return the deterministic structural counters for the graph."""
        ...

    def validate(self) -> GraphValidation:
        """Return the groundedness report for the stored graph."""
        ...

    def to_dict(self) -> dict[str, object]:
        """Serialize the store into its JSON-ready snapshot document."""
        ...

    def __len__(self) -> int:
        """Return how many nodes are stored."""
        ...


@runtime_checkable
class MemoryStorePort(Protocol):
    """The ledger of reports and evidence research memory writes through.

    The contract covers what research memory itself exercises: the two write
    paths, the read paths behind its public getters and retrieval calls, the
    deterministic counters, and the snapshot document.  Narrower helpers such
    as ``reports_for_company`` or ``covers_topic`` stay concrete because
    research memory reaches them through ``retrieve_latest`` and
    ``topic_records`` instead.
    """

    index: MemoryIndexPort

    def add_evidence(
        self,
        evidence: Evidence,
        *,
        stored_at: str = "",
        report_id: str = "",
    ) -> EvidenceWrite:
        """Store an evidence item, recognising a repeat observation as a duplicate."""
        ...

    def add_report(
        self,
        report: ResearchReport,
        *,
        evidence: Iterable[Evidence] = (),
        stored_at: str = "",
        topic_id: str = "",
    ) -> tuple[ReportWrite, tuple[EvidenceWrite, ...]]:
        """Store a report and the evidence it carries, reporting what changed.

        Replaying an already stored report must be a no-op that returns the
        original receipt, so a re-run of the pipeline cannot reorder or
        inflate anything.
        """
        ...

    def is_empty(self) -> bool:
        """Return whether nothing has been recorded."""
        ...

    def counts(self) -> dict[str, int]:
        """Return the deterministic record counters for the store."""
        ...

    def has_evidence(self, evidence_id: str) -> bool:
        """Return whether ``evidence_id`` is stored."""
        ...

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Return the stored evidence item or ``None``."""
        ...

    def get_report(self, report_id: str) -> ResearchReport | None:
        """Return the stored report or ``None``."""
        ...

    def reports(self) -> tuple[ResearchReport, ...]:
        """Return every stored report, sequence-ordered."""
        ...

    def evidence(self) -> tuple[Evidence, ...]:
        """Return every stored evidence item, sequence-ordered."""
        ...

    def retrieve_latest(
        self,
        company_name: str = "",
        topic_id: str = "",
        limit: int = 1,
    ) -> tuple[ResearchReport, ...]:
        """Return the ``limit`` newest matching reports, newest first."""
        ...

    def topic_records(self, topic_id: str) -> TopicRetrieval:
        """Return the reports and evidence recorded for ``topic_id``."""
        ...

    def to_dict(self) -> dict[str, object]:
        """Serialize the store into its JSON-ready snapshot document."""
        ...


@runtime_checkable
class ResearchExecutorPort(Protocol):
    """Execute one full deterministic research pipeline pass.

    This is the execution boundary of the research loop: everything it needs
    from an orchestrator is a single ``PlannerInput`` in, one
    :class:`~predictron_engine.research.execution_report.ResearchExecutionResult`
    out.  The contract is deliberately minimal — no concurrency, lifecycle,
    or factory members — so a backend or a test double only has to reproduce
    the deterministic pass to be a drop-in executor.
    """

    def execute(
        self,
        planner_input: PlannerInput,
    ) -> ResearchExecutionResult:
        """Run one pass for the company in ``planner_input``."""
        ...
