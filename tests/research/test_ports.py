"""Tests for the research ports — Phase 8 Sprint 8, work items 4 and 5.

The ports are a seam, not a behaviour change.  These tests pin that the
in-memory classes satisfy the protocols structurally, that
:class:`~predictron_engine.research.memory.ResearchMemory` accepts
protocol-typed collaborators (including doubles that are not the concrete
classes), that the shared-sequence invariant is enforced for the new
parameters, that the research loop accepts any struct-typed executor, that
the ports themselves never import a concrete implementation, and that the
snapshot documents did not move.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from predictron_engine import research
from predictron_engine.canonical import canonical_json
from predictron_engine.research.exceptions import (
    InvalidGraphModelError,
    InvalidMemoryInputError,
)
from predictron_engine.research.graph_models import (
    GRAPH_SCHEMA_VERSION,
    EntityType,
    GraphRelationship,
    GraphStats,
    GraphValidation,
    RelationshipType,
)
from predictron_engine.research.graph_store import (
    STORE_SCHEMA_VERSION,
    EntityWrite,
    GraphStore,
    MemoryIndex,
    MemoryRecordKey,
    RelationshipWrite,
)
from predictron_engine.research.knowledge_graph import KnowledgeGraph
from predictron_engine.research.memory import (
    MEMORY_SNAPSHOT_SCHEMA_VERSION,
    EntityObservation,
    RelationshipObservation,
    ResearchMemory,
)
from predictron_engine.research.memory_store import (
    MEMORY_SCHEMA_VERSION,
    EvidenceWrite,
    MemoryStore,
    ReportWrite,
    TopicRetrieval,
)
from predictron_engine.research.orchestrator import ResearchOrchestrator
from predictron_engine.research.ports import (
    GraphStorePort,
    MemoryIndexPort,
    MemoryStorePort,
    ResearchExecutorPort,
)

from .memory_helpers import make_evidence, make_report

PORTS_PATH = Path(__file__).resolve().parents[2] / "predictron_engine" / "research" / "ports.py"

PORTS: tuple[type, ...] = (
    MemoryIndexPort,
    GraphStorePort,
    MemoryStorePort,
    ResearchExecutorPort,
)


# -- doubles -----------------------------------------------------------


class SequenceIndexDouble:
    """A protocol-only index: the contract, without ``MemoryIndex``."""

    def __init__(self) -> None:
        self._next = 1
        self._keys: dict[tuple[str, str], MemoryRecordKey] = {}

    def allocate(
        self,
        record_type: str,
        record_id: str,
        *,
        topic_id: str = "",
        company_name: str = "",
        stored_at: str = "",
    ) -> MemoryRecordKey:
        composite = (record_type.strip(), record_id.strip())
        existing = self._keys.get(composite)
        if existing is not None:
            return existing
        key = MemoryRecordKey(
            record_type=composite[0],
            record_id=composite[1],
            sequence=self._next,
            topic_id=topic_id,
            company_name=company_name,
            stored_at=stored_at,
        )
        self._next += 1
        self._keys[composite] = key
        return key

    def next_sequence(self) -> int:
        return self._next

    def key_for(self, record_type: str, record_id: str) -> MemoryRecordKey | None:
        return self._keys.get((record_type.strip(), record_id.strip()))

    def keys(self, record_type: str | None = None) -> tuple[MemoryRecordKey, ...]:
        selected = [
            key
            for key in self._keys.values()
            if record_type is None or key.record_type == record_type.strip()
        ]
        return tuple(sorted(selected, key=lambda key: (key.sequence, key.record_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "next_sequence": self._next,
            "keys": [key.to_dict() for key in self.keys()],
        }

    def __len__(self) -> int:
        return len(self._keys)


class GraphStoreDouble:
    """A protocol-only graph store: writes are recorded, reads return nothing."""

    def __init__(self, index: MemoryIndexPort) -> None:
        self.index = index
        self.entity_writes: list[str] = []
        self.relationship_writes: list[str] = []

    def add_entity(
        self,
        entity: Any,
        *,
        stored_at: str = "",
        record_id: str = "",
    ) -> EntityWrite:
        self.entity_writes.append(entity.entity_id)
        return EntityWrite(
            entity=entity, previous=None, sequence=self.index.next_sequence()
        )

    def add_relationship(
        self,
        relationship: GraphRelationship,
        *,
        stored_at: str = "",
    ) -> RelationshipWrite:
        self.relationship_writes.append(relationship.relationship_id())
        return RelationshipWrite(
            relationship=relationship,
            previous=None,
            sequence=self.index.next_sequence(),
        )

    def get_entity(self, entity_id: str) -> None:
        return None

    def entity_type(self, entity_id: str) -> None:
        return None

    def entity_ids(self) -> tuple[str, ...]:
        return ()

    def entities(self) -> tuple[Any, ...]:
        return ()

    def entities_of_type(self, entity_type: EntityType) -> tuple[Any, ...]:
        return ()

    def relationships(self) -> tuple[GraphRelationship, ...]:
        return ()

    def relationships_by_key(
        self, key: tuple[str, str, str]
    ) -> GraphRelationship | None:
        return None

    def relationship_count(self) -> int:
        return 0

    def get_relationship(
        self,
        source_id: str,
        relationship_type: RelationshipType,
        target_id: str,
    ) -> GraphRelationship | None:
        return None

    def incident_keys(
        self, entity_id: str, direction: Any
    ) -> tuple[tuple[str, str, str], ...]:
        return ()

    def other_endpoint(self, key: tuple[str, str, str], entity_id: str) -> str:
        return ""

    def stats(self) -> GraphStats:
        return GraphStats(0, 0, 0, 0, 0, 0)

    def validate(self) -> GraphValidation:
        return GraphValidation()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "index": self.index.to_dict(),
            "entities": [],
            "relationships": [],
        }

    def __len__(self) -> int:
        return 0


class RecordsDouble:
    """A protocol-only record store: writes are recorded, reads return nothing."""

    def __init__(self, index: MemoryIndexPort) -> None:
        self.index = index
        self.reports_written: list[str] = []
        self.evidence_written: list[str] = []

    def add_evidence(
        self,
        evidence: Any,
        *,
        stored_at: str = "",
        report_id: str = "",
    ) -> EvidenceWrite:
        self.evidence_written.append(evidence.evidence_id)
        return EvidenceWrite(
            evidence=evidence,
            duplicate_of=None,
            sequence=self.index.next_sequence(),
        )

    def add_report(
        self,
        report: Any,
        *,
        evidence: Any = (),
        stored_at: str = "",
        topic_id: str = "",
    ) -> tuple[ReportWrite, tuple[EvidenceWrite, ...]]:
        self.reports_written.append(report.report_id)
        self.evidence_written.extend(item.evidence_id for item in evidence)
        return (
            ReportWrite(
                report=report, previous=None, sequence=self.index.next_sequence()
            ),
            (),
        )

    def is_empty(self) -> bool:
        return True

    def counts(self) -> dict[str, int]:
        return {
            "reports": 0,
            "evidence": 0,
            "duplicates": 0,
            "indexed": len(self.index),
        }

    def has_evidence(self, evidence_id: str) -> bool:
        return False

    def get_evidence(self, evidence_id: str) -> None:
        return None

    def get_report(self, report_id: str) -> None:
        return None

    def reports(self) -> tuple[Any, ...]:
        return ()

    def evidence(self) -> tuple[Any, ...]:
        return ()

    def retrieve_latest(
        self,
        company_name: str = "",
        topic_id: str = "",
        limit: int = 1,
    ) -> tuple[Any, ...]:
        return ()

    def topic_records(self, topic_id: str) -> TopicRetrieval:
        return TopicRetrieval(topic_id=topic_id.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "index": self.index.to_dict(),
            "reports": [],
            "evidence": [],
            "report_evidence": {},
            "evidence_reports": {},
        }


# -- helpers -----------------------------------------------------------


def gpt_observation(evidence_id: str) -> EntityObservation:
    return EntityObservation(
        "GPT",
        EntityType.TECHNOLOGY,
        attributes={"generation": 4},
        evidence_ids=(evidence_id,),
        confidence=0.8,
    )


def has_technology(evidence_id: str) -> RelationshipObservation:
    return RelationshipObservation(
        relationship_type=RelationshipType.HAS_TECHNOLOGY,
        source_name="Acme",
        source_type=EntityType.COMPANY,
        target_name="GPT",
        target_type=EntityType.TECHNOLOGY,
        evidence_ids=(evidence_id,),
        confidence=0.7,
    )


def populate(memory: ResearchMemory) -> None:
    evidence = make_evidence()
    memory.add_report(make_report("rep-1"), evidence=[evidence])
    memory.add_report(
        make_report("rep-2"),
        entities=[gpt_observation(evidence.evidence_id)],
        relationships=[has_technology(evidence.evidence_id)],
    )


def _runtime_imports() -> list[str]:
    tree = ast.parse(PORTS_PATH.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def _guarded_imports() -> set[str]:
    tree = ast.parse(PORTS_PATH.read_text(encoding="utf-8"))
    guarded: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        if not isinstance(node.test, ast.Name) or node.test.id != "TYPE_CHECKING":
            continue
        for child in node.body:
            if isinstance(child, ast.ImportFrom) and child.module:
                guarded.add(child.module)
    return guarded


# -- protocol shape ----------------------------------------------------


@pytest.mark.parametrize("port", PORTS)
def test_every_port_is_runtime_checkable(port: type) -> None:
    assert isinstance(object(), port) is False


@pytest.mark.parametrize("port", PORTS)
def test_a_port_cannot_be_instantiated(port: type) -> None:
    with pytest.raises(TypeError):
        port()


@pytest.mark.parametrize(
    ("port", "implementation"),
    [
        (MemoryIndexPort, MemoryIndex),
        (GraphStorePort, GraphStore),
        (MemoryStorePort, MemoryStore),
        (ResearchExecutorPort, ResearchOrchestrator),
    ],
)
def test_concrete_stores_do_not_inherit_from_their_port(
    port: type, implementation: type
) -> None:
    assert port not in implementation.__mro__


def test_the_ports_make_only_typing_imports_at_runtime() -> None:
    assert set(_runtime_imports()) == {"__future__", "typing"}


def test_the_ports_type_check_against_the_stores_they_port() -> None:
    assert _guarded_imports() == {
        "collections.abc",
        "predictron_engine.research.execution_report",
        "predictron_engine.research.graph_models",
        "predictron_engine.research.graph_store",
        "predictron_engine.research.memory_store",
        "predictron_engine.research.models",
    }


def test_the_ports_are_exported_from_the_research_package() -> None:
    for name in (
        "MemoryIndexPort",
        "GraphStorePort",
        "MemoryStorePort",
        "ResearchExecutorPort",
    ):
        assert name in research.__all__
        assert getattr(research, name) is not None


# -- the in-memory defaults satisfy the ports --------------------------


def test_the_in_memory_implementations_satisfy_their_ports() -> None:
    assert isinstance(MemoryIndex(), MemoryIndexPort)
    assert isinstance(GraphStore(), GraphStorePort)
    assert isinstance(KnowledgeGraph().store, GraphStorePort)
    assert isinstance(MemoryStore(), MemoryStorePort)
    assert isinstance(ResearchOrchestrator(), ResearchExecutorPort)


def test_the_executor_port_describes_only_the_execution_boundary() -> None:
    members = {
        name
        for name, member in vars(ResearchExecutorPort).items()
        if not name.startswith("_")
    }
    assert members == {"execute"}


def test_research_memory_defaults_to_the_in_memory_implementations() -> None:
    memory = ResearchMemory()
    assert isinstance(memory.index, MemoryIndex)
    assert isinstance(memory.store, GraphStore)
    assert isinstance(memory.records, MemoryStore)
    assert memory.graph.store is memory.store
    assert memory.records.index is memory.index
    assert memory.store.index is memory.index
    assert memory.deduplication is memory.records.deduplication


def test_research_memory_validates_collaborators_against_the_ports() -> None:
    with pytest.raises(InvalidMemoryInputError, match="MemoryIndexPort"):
        ResearchMemory(index="index")
    with pytest.raises(InvalidMemoryInputError, match="GraphStorePort"):
        ResearchMemory(store="store")
    with pytest.raises(InvalidMemoryInputError, match="MemoryStorePort"):
        ResearchMemory(records="records")
    with pytest.raises(InvalidMemoryInputError, match="KnowledgeGraph"):
        ResearchMemory(graph="graph")
    with pytest.raises(InvalidMemoryInputError, match="HistoryLog"):
        ResearchMemory(history="history")


def test_the_knowledge_graph_accepts_a_graph_store_port() -> None:
    double = GraphStoreDouble(MemoryIndex())
    graph = KnowledgeGraph(store=double)
    assert graph.store is double
    with pytest.raises(InvalidGraphModelError, match="GraphStorePort"):
        KnowledgeGraph(store="nope")


def test_the_memory_store_accepts_a_memory_index_port() -> None:
    double = SequenceIndexDouble()
    store = MemoryStore(index=double)
    assert store.index is double
    with pytest.raises(InvalidMemoryInputError, match="MemoryIndexPort"):
        MemoryStore(index="nope")


def test_the_graph_store_accepts_a_memory_index_port() -> None:
    double = SequenceIndexDouble()
    store = GraphStore(index=double)
    assert store.index is double


# -- the shared-sequence invariant -------------------------------------


def test_a_store_must_allocate_through_the_memory_index() -> None:
    with pytest.raises(InvalidMemoryInputError, match="own index"):
        ResearchMemory(index=MemoryIndex(), store=GraphStore())


def test_records_must_allocate_through_the_memory_index() -> None:
    with pytest.raises(InvalidMemoryInputError, match="own index"):
        ResearchMemory(index=MemoryIndex(), records=MemoryStore())


def test_a_store_supplied_with_a_graph_must_be_that_graphs_store() -> None:
    graph = KnowledgeGraph()
    with pytest.raises(InvalidMemoryInputError, match="graph's own"):
        ResearchMemory(graph=graph, store=GraphStore())


def test_a_graph_may_be_supplied_together_with_its_own_store() -> None:
    graph = KnowledgeGraph()
    memory = ResearchMemory(
        graph=graph,
        store=graph.store,
        index=graph.store.index,
    )
    assert memory.graph is graph
    assert memory.store is graph.store
    assert memory.index is graph.store.index
    assert memory.records.index is memory.index


def test_a_store_supplied_on_its_own_supplies_the_index() -> None:
    double = GraphStoreDouble(MemoryIndex())
    memory = ResearchMemory(store=double)
    assert memory.index is double.index
    assert memory.records.index is double.index
    assert memory.graph.store is double


# -- protocol-only doubles are injectable ------------------------------


def test_a_protocol_only_index_is_injectable() -> None:
    double = SequenceIndexDouble()
    assert isinstance(double, MemoryIndexPort)
    memory = ResearchMemory(index=double)
    assert memory.index is double
    assert memory.records.index is double
    assert memory.store.index is double


def test_a_protocol_only_index_produces_an_identical_snapshot() -> None:
    plain = ResearchMemory()
    populate(plain)

    wired = ResearchMemory(index=SequenceIndexDouble())
    populate(wired)

    assert canonical_json(wired.to_dict()) == canonical_json(plain.to_dict())
    assert wired.memory_key() == plain.memory_key()
    assert wired.counts() == plain.counts()
    assert canonical_json(wired.report()) == canonical_json(plain.report())


def test_a_protocol_only_graph_store_is_injectable() -> None:
    index = MemoryIndex()
    double = GraphStoreDouble(index)
    assert isinstance(double, GraphStorePort)

    memory = ResearchMemory(index=index, store=double)
    assert memory.store is double
    assert memory.graph.store is double
    assert memory.is_empty() is True
    assert memory.get_company("Acme") is None

    memory.add_report(make_report("rep-1"), evidence=[make_evidence()])

    assert double.entity_writes
    assert memory.get_company("Acme") is None
    assert memory.records.reports()


def test_a_protocol_only_record_store_is_injectable() -> None:
    index = MemoryIndex()
    double = RecordsDouble(index)
    assert isinstance(double, MemoryStorePort)

    memory = ResearchMemory(index=index, records=double)
    assert memory.records is double
    assert memory.is_empty() is True

    memory.add_report(make_report("rep-1"), evidence=[make_evidence()])

    assert double.reports_written == ["rep-1"]
    assert len(double.evidence_written) == 1
    assert memory.reports() == ()
    assert memory.get_report("rep-1") is None
    assert memory.retrieve_latest() == ()
    assert memory.retrieve_by_topic("topic-1").is_empty()
    assert memory.to_dict()["records"] == double.to_dict()


# -- nothing about the documents moved ---------------------------------


def test_the_snapshot_schema_versions_are_unchanged() -> None:
    assert MEMORY_SNAPSHOT_SCHEMA_VERSION == "1.0.0"
    assert MEMORY_SCHEMA_VERSION == "1.0.0"
    assert STORE_SCHEMA_VERSION == "1.0.0"
    assert GRAPH_SCHEMA_VERSION == "1.0.0"


def test_a_default_snapshot_still_has_exactly_its_four_sections() -> None:
    memory = ResearchMemory()
    populate(memory)
    assert set(memory.to_dict()) == {
        "schema_version",
        "graph",
        "records",
        "history",
    }
    assert memory.to_dict()["schema_version"] == MEMORY_SNAPSHOT_SCHEMA_VERSION
