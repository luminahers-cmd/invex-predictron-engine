"""Research memory — Phase 8 Sprint 6.

:class:`ResearchMemory` is the single entry point the rest of the engine uses
to remember what research found.  It is the public API this sprint is
delivering::

    memory = ResearchMemory()
    memory.add_report(report)
    company = memory.get_company("OpenAI")
    memory.graph

What it is
----------
Memory is **append-only**.  Nothing it stores is ever edited or removed:

* a fact observed twice *merges* into the graph and the pre-merge value is
  snapshotted into the history log, so the earlier belief survives;
* an evidence item re-observed is recognised as a duplicate and the original
  value is kept, because the first observation is the one with the original
  context;
* a report replayed by a re-run is a no-op that keeps its original position.

What it is not
--------------
It is not a database, and it does not pretend to be one.  There is no
transaction, no index tuning, and no external service.  What it does have is a
seam for one: everything stateful lives behind
:class:`~predictron_engine.research.memory_store.MemoryStore` and
:class:`~predictron_engine.research.graph_store.GraphStore`, so a persistent
backend only has to supply *sequence allocation* and *document round-tripping*.

Determinism
-----------
Nothing here reads a clock, calls a model, or iterates an unordered
collection.  Ordering comes from the shared
:class:`~predictron_engine.research.graph_store.MemoryIndex`; the caller
supplies every ``stored_at`` label; and a caller who wants reproducible
snapshots can take one and keep it forever, because the values are frozen.

Groundedness
------------
A fact can only enter the graph through an
:class:`EntityObservation` or a :class:`RelationshipObservation` that names
the evidence supporting it, and every relationship must reference evidence
that memory actually holds.  A report with no evidence therefore records the
company it is about and nothing else — the graph never invents a competitor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from predictron_engine.research.deduplication import DeduplicationIndex
from predictron_engine.research.exceptions import (
    InvalidMemoryInputError,
    InvalidRelationshipError,
)
from predictron_engine.research.execution_report import ResearchReport
from predictron_engine.research.graph_models import (
    AttributeValue,
    EntitySnapshot,
    EntityType,
    GraphEntity,
    GraphRelationship,
    GraphStats,
    GraphValidation,
    Provenance,
    RelationshipType,
    ValidationMarker,
)
from predictron_engine.research.graph_store import (
    GraphDirection,
    GraphStore,
    MemoryIndex,
)
from predictron_engine.research.history import (
    HistoryChange,
    HistoryEntry,
    HistoryLog,
)
from predictron_engine.research.knowledge_graph import (
    KnowledgeGraph,
    SearchScope,
)
from predictron_engine.research.linking import EntityLinker, normalize_name
from predictron_engine.research.memory_store import (
    MemoryStore,
    ReportWrite,
    TopicRetrieval,
)
from predictron_engine.research.models import Evidence

#: Serialization schema version for a whole research memory snapshot.
MEMORY_SNAPSHOT_SCHEMA_VERSION = "1.0.0"

#: The record type a relationship-support history change is filed under.
RECORD_HISTORY = "history"

#: The field-name prefix a relationship-support change is recorded under.
RELATIONSHIP_CHANGE_PREFIX = "relationship"


@dataclass(frozen=True)
class EntityObservation:
    """One observed fact about one entity, stated by name.

    Observations carry *names*, not identifiers.  The memory facade resolves
    the name through the entity linker at write time, so ``"Open AI"`` and
    ``"OpenAI Inc."`` land on the same node without the caller having to know
    that they should.

    Parameters
    ----------
    name:
        The name as the research stated it.
    entity_type:
        Which kind of entity this is.
    attributes:
        Scalar attributes to merge, e.g. ``{"employees": 1400}``.
    aliases:
        Additional observed name variants for this entity.
    confidence:
        Confidence in the observation as a whole, within ``[0, 1]``.
    validation:
        The validation marker the supporting evidence produced.
    evidence_ids:
        The evidence identifiers backing the observation.  At least one is
        required for any entity that is not itself an evidence node.
    topic_id:
        The research topic this was observed for, when known.
    source_ids / urls / titles:
        Provenance carried over from the source that produced the fact.
    """

    name: str
    entity_type: EntityType
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    confidence: float = 0.0
    validation: ValidationMarker = field(default_factory=ValidationMarker)
    evidence_ids: tuple[str, ...] = ()
    topic_id: str = ""
    source_ids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidMemoryInputError(
                "an observation needs a non-empty name"
            )
        if not isinstance(self.entity_type, EntityType):
            raise InvalidMemoryInputError(
                "entity_type must be an EntityType for EntityObservation"
            )
        if not isinstance(self.attributes, Mapping):
            raise InvalidMemoryInputError(
                "attributes must be a mapping for EntityObservation"
            )
        for key, value in self.attributes.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidMemoryInputError(
                    "attribute keys must be non-empty strings"
                )
            if not isinstance(value, str | int | float | bool):
                raise InvalidMemoryInputError(
                    f"attribute {key!r} must be a JSON scalar"
                )
        if isinstance(self.aliases, str) or not isinstance(self.aliases, tuple):
            raise InvalidMemoryInputError(
                "aliases must be a tuple of strings for EntityObservation"
            )
        for alias in self.aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise InvalidMemoryInputError(
                    "alias entries must be non-empty strings"
                )
        if isinstance(self.evidence_ids, str) or not isinstance(
            self.evidence_ids, tuple
        ):
            raise InvalidMemoryInputError(
                "evidence_ids must be a tuple of strings for EntityObservation"
            )
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise InvalidMemoryInputError(
                    "evidence_ids entries must be non-empty strings"
                )
        if not isinstance(self.validation, ValidationMarker):
            raise InvalidMemoryInputError(
                "validation must be a ValidationMarker for EntityObservation"
            )
        for name in ("topic_id",):
            if not isinstance(getattr(self, name), str):
                raise InvalidMemoryInputError(
                    f"{name} must be a string for EntityObservation"
                )
        for name in ("source_ids", "urls", "titles"):
            if not isinstance(getattr(self, name), tuple):
                raise InvalidMemoryInputError(
                    f"{name} must be a tuple for EntityObservation"
                )

    @property
    def canonical_name(self) -> str:
        """Return the normalized form of the observed name."""
        return normalize_name(self.name)

    @property
    def is_evidence_node(self) -> bool:
        """Return whether this observation anchors a piece of evidence."""
        return self.entity_type is EntityType.EVIDENCE

    def requires_evidence(self) -> bool:
        """Return whether the observation must cite supporting evidence."""
        return bool(self.evidence_ids) or self.is_evidence_node

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "attributes": dict(self.attributes),
            "aliases": list(self.aliases),
            "confidence": self.confidence,
            "validation": self.validation.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "topic_id": self.topic_id,
            "source_ids": list(self.source_ids),
            "urls": list(self.urls),
            "titles": list(self.titles),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EntityObservation:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        attributes = data.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise InvalidMemoryInputError("expected mapping for 'attributes'")
        return cls(
            name=_require_str(data, "name"),
            entity_type=EntityType(_require_str(data, "entity_type")),
            attributes={
                key: value
                for key, value in attributes.items()
                if isinstance(key, str)
            },
            aliases=tuple(_require_str_list(data, "aliases")),
            confidence=_require_float(data, "confidence"),
            validation=ValidationMarker.from_dict(
                _require_mapping(data, "validation")
            ),
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
            topic_id=_require_str(data, "topic_id", default=""),
            source_ids=tuple(_require_str_list(data, "source_ids")),
            urls=tuple(_require_str_list(data, "urls")),
            titles=tuple(_require_str_list(data, "titles")),
        )


@dataclass(frozen=True)
class RelationshipObservation:
    """One observed connection between two entities, stated by name.

    Endpoints are named rather than identified for the same reason entity
    observations are: the facade resolves them, so the caller states the fact
    and memory decides which nodes it joins.
    """

    relationship_type: RelationshipType
    source_name: str
    source_type: EntityType
    target_name: str
    target_type: EntityType
    confidence: float = 0.0
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    timestamp: str = ""
    topic_id: str = ""
    source_ids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.relationship_type, RelationshipType):
            raise InvalidMemoryInputError(
                "relationship_type must be a RelationshipType for "
                "RelationshipObservation"
            )
        for name in ("source_name", "target_name"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise InvalidMemoryInputError(
                    f"{name} must not be empty for RelationshipObservation"
                )
        for name in ("source_type", "target_type"):
            if not isinstance(getattr(self, name), EntityType):
                raise InvalidMemoryInputError(
                    f"{name} must be an EntityType for RelationshipObservation"
                )
        if not isinstance(self.attributes, Mapping):
            raise InvalidMemoryInputError(
                "attributes must be a mapping for RelationshipObservation"
            )
        for key, value in self.attributes.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidMemoryInputError(
                    "attribute keys must be non-empty strings"
                )
            if not isinstance(value, str | int | float | bool):
                raise InvalidMemoryInputError(
                    f"attribute {key!r} must be a JSON scalar"
                )
        for name in ("evidence_ids", "source_ids", "urls", "titles"):
            if isinstance(getattr(self, name), str) or not isinstance(
                getattr(self, name), tuple
            ):
                raise InvalidMemoryInputError(
                    f"{name} must be a tuple for RelationshipObservation"
                )
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise InvalidMemoryInputError(
                    "evidence_ids entries must be non-empty strings"
                )
        for name in ("timestamp", "topic_id"):
            if not isinstance(getattr(self, name), str):
                raise InvalidMemoryInputError(
                    f"{name} must be a string for RelationshipObservation"
                )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "relationship_type": self.relationship_type.value,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "target_name": self.target_name,
            "target_type": self.target_type.value,
            "confidence": self.confidence,
            "attributes": dict(self.attributes),
            "evidence_ids": list(self.evidence_ids),
            "timestamp": self.timestamp,
            "topic_id": self.topic_id,
            "source_ids": list(self.source_ids),
            "urls": list(self.urls),
            "titles": list(self.titles),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
    ) -> RelationshipObservation:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        attributes = data.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise InvalidMemoryInputError("expected mapping for 'attributes'")
        return cls(
            relationship_type=RelationshipType(
                _require_str(data, "relationship_type")
            ),
            source_name=_require_str(data, "source_name"),
            source_type=EntityType(_require_str(data, "source_type")),
            target_name=_require_str(data, "target_name"),
            target_type=EntityType(_require_str(data, "target_type")),
            confidence=_require_float(data, "confidence"),
            attributes={
                key: value
                for key, value in attributes.items()
                if isinstance(key, str)
            },
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
            timestamp=_require_str(data, "timestamp", default=""),
            topic_id=_require_str(data, "topic_id", default=""),
            source_ids=tuple(_require_str_list(data, "source_ids")),
            urls=tuple(_require_str_list(data, "urls")),
            titles=tuple(_require_str_list(data, "titles")),
        )


@dataclass(frozen=True)
class MemoryWrite:
    """Everything one :meth:`ResearchMemory.add_report` call appended.

    The value is a receipt, not a log: it names what was created, what was
    merged, what was recognised as a duplicate, and what was appended to
    history, so a caller can report on a write without re-deriving any of it.
    """

    report: ReportWrite
    evidence_ids: tuple[str, ...] = ()
    duplicate_evidence_ids: tuple[str, ...] = ()
    created_entity_ids: tuple[str, ...] = ()
    updated_entity_ids: tuple[str, ...] = ()
    created_relationship_ids: tuple[str, ...] = ()
    strengthened_relationship_ids: tuple[str, ...] = ()
    history_sequences: tuple[int, ...] = ()

    @property
    def report_id(self) -> str:
        """Return the report this write recorded."""
        return self.report.report_id

    @property
    def created(self) -> bool:
        """Return whether the report was newly recorded."""
        return self.report.created

    @property
    def change_count(self) -> int:
        """Return how many graph facts this write created or changed."""
        return (
            len(self.created_entity_ids)
            + len(self.updated_entity_ids)
            + len(self.created_relationship_ids)
            + len(self.strengthened_relationship_ids)
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "report": self.report.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "duplicate_evidence_ids": list(self.duplicate_evidence_ids),
            "created_entity_ids": list(self.created_entity_ids),
            "updated_entity_ids": list(self.updated_entity_ids),
            "created_relationship_ids": list(self.created_relationship_ids),
            "strengthened_relationship_ids": list(
                self.strengthened_relationship_ids
            ),
            "history_sequences": list(self.history_sequences),
        }


class ResearchMemory:
    """Append-only, deterministic memory for everything research found.

    Parameters
    ----------
    index:
        The shared sequence index.  When omitted a fresh one is created.
    graph:
        An existing graph to append to.  When omitted a new one is built on
        ``index``.
    linker:
        The entity linker.  When omitted the graph's own linker is used.
    history:
        An existing history log.  When omitted an empty one is created.

    Attributes
    ----------
    graph:
        The :class:`~predictron_engine.research.knowledge_graph.KnowledgeGraph`
        holding every fact, exposed as ``memory.graph``.
    records:
        The :class:`~predictron_engine.research.memory_store.MemoryStore`
        holding every report and evidence item.
    index:
        The shared :class:`~predictron_engine.research.graph_store.MemoryIndex`
        that orders records, facts, and history with one sequence.
    history:
        The :class:`~predictron_engine.research.history.HistoryLog`.
    """

    def __init__(
        self,
        index: MemoryIndex | None = None,
        graph: KnowledgeGraph | None = None,
        history: HistoryLog | None = None,
    ) -> None:
        if index is None:
            index = MemoryIndex()
        elif not isinstance(index, MemoryIndex):
            raise InvalidMemoryInputError(
                "index must be a MemoryIndex instance"
            )
        if graph is None:
            graph = KnowledgeGraph(store=GraphStore(index=index))
        elif not isinstance(graph, KnowledgeGraph):
            raise InvalidMemoryInputError(
                "graph must be a KnowledgeGraph instance"
            )
        if history is not None and not isinstance(history, HistoryLog):
            raise InvalidMemoryInputError(
                "history must be a HistoryLog instance"
            )
        self.index = index
        self.graph = graph
        self.deduplication = DeduplicationIndex()
        self.records = MemoryStore(index=index, deduplication=self.deduplication)
        self._history = history if history is not None else HistoryLog()

    # -- Views -------------------------------------------------------

    @property
    def history(self) -> HistoryLog:
        """Return the append-only history log."""
        return self._history

    @property
    def linker(self) -> EntityLinker:
        """Return the entity linker the graph resolves names through."""
        return self.graph.linker

    def is_empty(self) -> bool:
        """Return whether memory holds nothing at all."""
        return self.records.is_empty() and self.graph.is_empty()

    def counts(self) -> dict[str, int]:
        """Return the deterministic record and history counters."""
        return {
            **self.records.counts(),
            "entities": len(self.graph),
            "relationships": self.graph.relationship_count(),
            "history": len(self._history),
        }

    def stats(self) -> GraphStats:
        """Return the graph's structural counters."""
        return self.graph.stats()

    def validate(self) -> GraphValidation:
        """Return the graph's groundedness report."""
        return self.graph.validate()

    # -- Writing -----------------------------------------------------

    def add_evidence(
        self,
        evidence: Evidence,
        *,
        stored_at: str = "",
    ) -> str:
        """Store one evidence item and return its identifier.

        The item is only ever stored: a re-observation is recognised as a
        duplicate and the original value is kept untouched.
        """
        write = self.records.add_evidence(evidence, stored_at=stored_at)
        return write.evidence_id

    def add_report(
        self,
        report: ResearchReport,
        *,
        evidence: Iterable[Evidence] = (),
        entities: Sequence[EntityObservation] = (),
        relationships: Sequence[RelationshipObservation] = (),
        stored_at: str = "",
        topic_id: str = "",
    ) -> MemoryWrite:
        """Record a finished report and everything it established.

        The call is one deterministic transaction over four append-only
        stores, applied in a fixed order:

        1. the report and its evidence are written to the record store,
        2. each :class:`EntityObservation` is resolved to a node, merged into
           the graph, and — when the merge changed anything — snapshotted
           into history with the derived changes,
        3. each :class:`RelationshipObservation` is resolved to an edge and
           strengthened if it already existed, and
        4. relationship support changes are appended to the source entity's
           history.

        Replaying the same report is a no-op that returns the original
        receipt, so a re-run of the pipeline cannot duplicate history.
        """
        if not isinstance(report, ResearchReport):
            raise InvalidMemoryInputError(
                "add_report expects a ResearchReport instance"
            )
        report_write, evidence_writes = self.records.add_report(
            report,
            evidence=evidence,
            stored_at=stored_at,
            topic_id=topic_id,
        )
        if report_write.replayed:
            return MemoryWrite(
                report=report_write,
                evidence_ids=tuple(
                    write.evidence_id for write in evidence_writes
                ),
            )
        evidence_ids = tuple(write.evidence_id for write in evidence_writes)
        duplicated = tuple(
            write.evidence_id
            for write in evidence_writes
            if write.is_duplicate
        )
        created_entities: list[str] = []
        updated_entities: list[str] = []
        created_relationships: list[str] = []
        strengthened_relationships: list[str] = []
        history_sequences: list[int] = []
        company_id = self.linker.resolve(
            EntityType.COMPANY.value, report.company_name
        )
        self._observe_company(
            report,
            company_id=company_id,
            stored_at=stored_at,
            history_sequences=history_sequences,
            created=created_entities,
            updated=updated_entities,
        )
        for write in evidence_writes:
            self._observe_evidence(
                write.evidence,
                report=report,
                stored_at=stored_at,
                created=created_entities,
                history_sequences=history_sequences,
            )
        for observation in entities:
            self._write_entity(
                observation,
                report=report,
                stored_at=stored_at,
                created=created_entities,
                updated=updated_entities,
                history_sequences=history_sequences,
            )
        for relation in relationships:
            self._write_relationship(
                relation,
                report=report,
                stored_at=stored_at,
                created=created_relationships,
                strengthened=strengthened_relationships,
                history_sequences=history_sequences,
            )
        return MemoryWrite(
            report=report_write,
            evidence_ids=evidence_ids,
            duplicate_evidence_ids=duplicated,
            created_entity_ids=tuple(created_entities),
            updated_entity_ids=tuple(updated_entities),
            created_relationship_ids=tuple(created_relationships),
            strengthened_relationship_ids=tuple(strengthened_relationships),
            history_sequences=tuple(history_sequences),
        )

    def _observe_company(
        self,
        report: ResearchReport,
        *,
        company_id: str,
        stored_at: str,
        history_sequences: list[int],
        created: list[str],
        updated: list[str],
    ) -> None:
        """Record the company a report is about, grounded in the report itself.

        A finished report is itself evidence that research happened, so the
        company node is always created — provenance is the report identifier
        and the node records the outcome the report reached.  A later report
        about the same company merges into the existing node and, when the
        outcome changed, is snapshotted into history exactly like any other
        merge.
        """
        entity = GraphEntity(
            entity_id=company_id,
            entity_type=EntityType.COMPANY,
            name=report.company_name,
            canonical_name=normalize_name(report.company_name),
            aliases=(),
            attributes=(
                ("last_report_state", report.state.value),
                ("last_report_successful", report.is_successful),
            ),
            provenance=Provenance(report_ids=(report.report_id,)),
            confidence=1.0,
            last_updated=report.report_id,
            validation=ValidationMarker(),
            topic_id="",
            record_id=report.report_id,
        )
        write = self.graph.store.add_entity(
            entity,
            stored_at=stored_at,
            record_id=report.report_id,
        )
        self._record_entity_write(
            entity=write.entity,
            previous=write.previous,
            report=report,
            stored_at=stored_at,
            created=created,
            updated=updated,
            history_sequences=history_sequences,
        )

    def _observe_evidence(
        self,
        evidence: Evidence,
        *,
        report: ResearchReport,
        stored_at: str,
        created: list[str],
        history_sequences: list[int],
    ) -> None:
        """Create the first-class evidence node for a stored item.

        This is what makes ``EVIDENCE_ENTITY`` — "Evidence -> Entity" — a
        real edge rather than a bookkeeping column: the claim an item makes
        becomes a node in its own right, and every fact it supports can be
        walked back to it.  The node is named after the claim, so two items
        asserting the same thing land on one node and their support unions,
        while the *items* themselves stay separate in the record store.

        The node is created for every item the report carried, not only for
        the ones the deduplication index called new.  Those are different
        questions: the record store decides whether an item has been seen
        before, while the graph must still represent every distinct claim
        that was observed.  Skipping a repeat would erase a genuine
        observation whenever two documents shared a title and nothing else.
        """
        entity_id = self.linker.resolve(
            EntityType.EVIDENCE.value, evidence.claim
        )
        entity = GraphEntity(
            entity_id=entity_id,
            entity_type=EntityType.EVIDENCE,
            name=evidence.claim,
            canonical_name=normalize_name(evidence.claim) or evidence.evidence_id,
            aliases=(),
            attributes=(
                ("category", evidence.category),
                ("confidence", evidence.confidence),
            ),
            provenance=Provenance(
                evidence_ids=(evidence.evidence_id,),
                source_ids=(evidence.reference.source_identifier,),
                urls=(evidence.reference.url,) if evidence.reference.url else (),
                titles=(
                    (evidence.reference.description,)
                    if evidence.reference.description
                    else ()
                ),
                report_ids=(report.report_id,),
            ),
            confidence=evidence.confidence,
            last_updated=report.report_id,
            validation=ValidationMarker(),
            topic_id=evidence.topic_id,
            record_id=report.report_id,
        )
        write = self.graph.store.add_entity(
            entity,
            stored_at=stored_at,
            record_id=report.report_id,
        )
        self._record_entity_write(
            entity=write.entity,
            previous=write.previous,
            report=report,
            stored_at=stored_at,
            created=created,
            updated=[],
            history_sequences=history_sequences,
        )

    def _write_entity(
        self,
        observation: EntityObservation,
        *,
        report: ResearchReport,
        stored_at: str,
        created: list[str],
        updated: list[str],
        history_sequences: list[int],
    ) -> None:
        """Resolve, ground, and merge one entity observation."""
        self._require_grounding(observation)
        entity_id = self.linker.resolve(
            observation.entity_type.value, observation.name
        )
        for alias in observation.aliases:
            self.linker.resolve(observation.entity_type.value, alias)
        entity = GraphEntity(
            entity_id=entity_id,
            entity_type=observation.entity_type,
            name=observation.name,
            canonical_name=observation.canonical_name,
            aliases=observation.aliases,
            attributes=tuple(sorted(observation.attributes.items())),
            provenance=Provenance(
                evidence_ids=observation.evidence_ids,
                source_ids=observation.source_ids,
                urls=observation.urls,
                titles=observation.titles,
                report_ids=(report.report_id,),
            ),
            confidence=observation.confidence,
            last_updated=report.report_id,
            validation=observation.validation,
            topic_id=observation.topic_id,
            record_id=report.report_id,
        )
        write = self.graph.store.add_entity(
            entity,
            stored_at=stored_at,
            record_id=report.report_id,
        )
        self._record_entity_write(
            entity=write.entity,
            previous=write.previous,
            report=report,
            stored_at=stored_at,
            created=created,
            updated=updated,
            history_sequences=history_sequences,
        )

    def _record_entity_write(
        self,
        *,
        entity: GraphEntity,
        previous: GraphEntity | None,
        report: ResearchReport,
        stored_at: str,
        created: list[str],
        updated: list[str],
        history_sequences: list[int],
    ) -> None:
        """Snapshot one merged node into history and note what it changed.

        Only a change in what memory *believes* counts: the name, the
        aliases, the attributes, the confidence, the validation verdict, or
        the topic.  Bookkeeping — which report last touched the node, and the
        union of reports that have grounded it — is deliberately excluded, so
        a re-run that restates the same finding appends no history.  Without
        that exclusion the log would record the research pipeline's activity
        rather than what was actually learned.
        """
        if previous is None:
            created.append(entity.entity_id)
        elif _belief_of(previous) == _belief_of(entity):
            return
        else:
            updated.append(entity.entity_id)
        version = len(self._history.snapshots_for(entity.entity_id)) + 1
        history_sequences.append(
            self._append_history(
                entity=entity,
                previous=(
                    previous.snapshot(version - 1, stored_at)
                    if previous is not None
                    else None
                ),
                current=entity.snapshot(version, stored_at),
                recorded_at=stored_at or report.report_id,
                source=report.report_id,
            )
        )

    def _write_relationship(
        self,
        observation: RelationshipObservation,
        *,
        report: ResearchReport,
        stored_at: str,
        created: list[str],
        strengthened: list[str],
        history_sequences: list[int],
    ) -> None:
        """Resolve, ground, and strengthen one relationship observation."""
        if not observation.evidence_ids:
            raise InvalidRelationshipError(
                "a relationship observation must cite the evidence "
                "supporting it"
            )
        unknown = [
            evidence_id
            for evidence_id in observation.evidence_ids
            if not self.records.has_evidence(evidence_id)
        ]
        if unknown:
            raise InvalidMemoryInputError(
                f"relationship evidence is not recorded: {sorted(unknown)}"
            )
        source_id = self._require_stored(
            observation.source_type, observation.source_name
        )
        target_id = self._require_stored(
            observation.target_type, observation.target_name
        )
        relationship = GraphRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=observation.relationship_type,
            confidence=observation.confidence,
            provenance=Provenance(
                evidence_ids=observation.evidence_ids,
                source_ids=observation.source_ids,
                urls=observation.urls,
                titles=observation.titles,
                report_ids=(report.report_id,),
            ),
            timestamp=observation.timestamp,
            evidence_ids=observation.evidence_ids,
            attributes=tuple(sorted(observation.attributes.items())),
        )
        write = self.graph.store.add_relationship(
            relationship, stored_at=stored_at
        )
        relationship_id = write.relationship.relationship_id()
        if write.created:
            created.append(relationship_id)
            return
        if not write.strengthened:
            return
        strengthened.append(relationship_id)
        source = self.graph.get_entity(source_id)
        if source is None:
            return
        previous_support = (
            write.previous.evidence_ids if write.previous is not None else ()
        )
        version = len(self._history.snapshots_for(source_id)) + 1
        snapshot = source.snapshot(version, stored_at)
        recorded_at = stored_at or report.report_id
        self._history = self._history.append(
            HistoryEntry(
                entity_id=source_id,
                entity_type=source.entity_type,
                sequence=self._reserve_history_sequence(),
                recorded_at=recorded_at,
                source=report.report_id,
                previous=snapshot,
                current=snapshot,
                changes=(
                    HistoryChange(
                        field_name=(
                            f"{RELATIONSHIP_CHANGE_PREFIX}."
                            f"{observation.relationship_type.value}"
                        ),
                        previous_value=",".join(sorted(previous_support)),
                        new_value=",".join(
                            sorted(write.relationship.evidence_ids)
                        ),
                        timestamp=recorded_at,
                        source=report.report_id,
                        evidence_ids=tuple(
                            sorted(write.relationship.evidence_ids)
                        ),
                        validation_status=source.validation.status,
                    ),
                ),
                evidence_ids=write.relationship.evidence_ids,
                validation_status=source.validation.status,
            )
        )
        history_sequences.append(self._history.ordered()[-1].sequence)

    def _require_grounding(self, observation: EntityObservation) -> None:
        """Reject an observation that cites evidence memory does not hold."""
        if observation.is_evidence_node:
            return
        if not observation.evidence_ids:
            raise InvalidMemoryInputError(
                f"entity observation for {observation.name!r} cites no evidence"
            )
        unknown = [
            evidence_id
            for evidence_id in observation.evidence_ids
            if not self.records.has_evidence(evidence_id)
        ]
        if unknown:
            raise InvalidMemoryInputError(
                f"entity evidence is not recorded: {sorted(unknown)}"
            )

    def _require_stored(
        self,
        entity_type: EntityType,
        name: str,
    ) -> str:
        """Return the identifier of a node memory already holds.

        A relationship may only join nodes that are already grounded, so a
        caller cannot create an edge out of thin air by naming an endpoint
        that was never observed.
        """
        entity = self.graph.get_entity_by_name(name, entity_type)
        if entity is not None:
            return entity.entity_id
        identifier = self.linker.peek(entity_type.value, name)
        if identifier is not None and self.graph.get_entity(identifier) is not None:
            return identifier
        raise InvalidMemoryInputError(
            f"relationship endpoint is not in memory: "
            f"{entity_type.value} {name!r}"
        )

    def _append_history(
        self,
        *,
        entity: GraphEntity,
        previous: EntitySnapshot | None,
        current: EntitySnapshot,
        recorded_at: str,
        source: str,
    ) -> int:
        """Append a derived history entry and return its sequence."""
        self._history = self._history.record(
            entity_id=entity.entity_id,
            entity_type=entity.entity_type,
            sequence=self._reserve_history_sequence(),
            recorded_at=recorded_at,
            source=source,
            previous=previous,
            current=current,
        )
        return self._history.ordered()[-1].sequence

    def _reserve_history_sequence(self) -> int:
        """Reserve and return the sequence the next history entry will use.

        History shares the memory index's namespace, so an entry can never
        collide with a record or a graph fact, and a snapshot that includes
        all three stays internally ordered.  The reservation identifier
        embeds the current sequence, which makes it unique by construction.
        """
        key = self.index.allocate(
            RECORD_HISTORY, f"{self.index.next_sequence():012d}"
        )
        return key.sequence

    # -- Reading -----------------------------------------------------

    def get_company(self, name: str) -> GraphEntity | None:
        """Return the company node for ``name``, or ``None``.

        Name resolution is alias-aware, so ``get_company("Open AI")`` finds
        the node that was created from ``"OpenAI Inc."``.
        """
        return self.graph.get_company(name)

    def get_entity(
        self,
        name_or_id: str,
        entity_type: EntityType | None = None,
    ) -> GraphEntity | None:
        """Return a node by identifier or by any of its known names."""
        if not isinstance(name_or_id, str) or not name_or_id.strip():
            return None
        candidate = name_or_id.strip()
        direct = self.graph.get_entity(candidate)
        if direct is not None:
            return direct
        return self.graph.get_entity_by_name(candidate, entity_type)

    def get_history(
        self,
        name_or_id: str,
    ) -> tuple[HistoryEntry, ...]:
        """Return every history entry for a node, in append order."""
        entity = self.get_entity(name_or_id)
        if entity is None:
            return ()
        return self._history.for_entity(entity.entity_id)

    def get_changes(
        self,
        name_or_id: str,
        field_name: str = "",
    ) -> tuple[HistoryEntry, ...]:
        """Return the history entries in which something was learned.

        Every entry is a change of some kind, so with no ``field_name`` this
        is the entity's whole history.  Passing ``field_name`` narrows the
        result to the entries that moved that one field, which is how a
        caller asks "when did headcount last move?".  Structural learnings
        that are not attribute moves -- learning a new name for a known
        entity, or the first time an edge was supported -- do not appear in
        the narrowed result even though their entry exists.
        """
        if not field_name.strip():
            return self.get_history(name_or_id)
        return tuple(
            entry
            for entry in self.get_history(name_or_id)
            if entry.change_for(field_name.strip()) is not None
        )

    def get_relationships(
        self,
        name_or_id: str = "",
        relationship_type: RelationshipType | None = None,
        direction: GraphDirection = GraphDirection.BOTH,
    ) -> tuple[GraphRelationship, ...]:
        """Return the edges touching a node, or every edge when unnamed.

        With no ``name_or_id`` this is the whole relationship set, which is
        what makes it the general form of the same call a caller makes for a
        single company.  A *named* entity that memory does not hold has no
        edges, so a typo cannot silently widen the answer from one node to
        the entire graph.
        """
        if not name_or_id.strip():
            return self.graph.get_relationships(
                None, relationship_type, direction
            )
        entity = self.get_entity(name_or_id)
        if entity is None:
            return ()
        return self.graph.get_relationships(
            entity.entity_id, relationship_type, direction
        )

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Return a stored evidence item or ``None``."""
        return self.records.get_evidence(evidence_id)

    def get_report(self, report_id: str) -> ResearchReport | None:
        """Return a stored report or ``None``."""
        return self.records.get_report(report_id)

    def reports(self) -> tuple[ResearchReport, ...]:
        """Return every stored report, in append order."""
        return self.records.reports()

    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = 10,
    ) -> tuple[GraphEntity, ...]:
        """Return the nodes matching ``query``, best match first.

        Matching is deterministic substring matching over normalized text, so
        the same query always returns the same nodes in the same order.
        """
        return self.graph.search(query, entity_type, scope, limit)

    def retrieve_latest(
        self,
        company_name: str = "",
        topic_id: str = "",
        limit: int = 1,
    ) -> tuple[ResearchReport, ...]:
        """Return the ``limit`` newest matching reports, newest first.

        Newest means the highest shared-index sequence — a deterministic
        append order, not a comparison of ``stored_at`` labels, so the answer
        never depends on how a date happens to be formatted.
        """
        return self.records.retrieve_latest(company_name, topic_id, limit)

    def retrieve_by_topic(self, topic_id: str) -> TopicRetrieval:
        """Return everything memory holds for ``topic_id``.

        The answer spans all four stores: the reports that covered the topic,
        the evidence collected for it, the graph nodes extracted from it, and
        the edges between those nodes.
        """
        records = self.records.topic_records(topic_id)
        entity_ids = tuple(
            entity.entity_id
            for entity in self.graph.entities()
            if entity.topic_id == topic_id.strip()
        )
        selected = set(entity_ids)
        relationship_keys = tuple(
            relationship.key
            for relationship in self.graph.relationships()
            if relationship.source_id in selected
            and relationship.target_id in selected
        )
        return TopicRetrieval(
            topic_id=records.topic_id,
            reports=records.reports,
            evidence=records.evidence,
            entity_ids=entity_ids,
            relationship_keys=relationship_keys,
        )

    def entities(self) -> tuple[GraphEntity, ...]:
        """Return every node in the graph, identifier-sorted."""
        return self.graph.entities()

    def entity_ids(self) -> tuple[str, ...]:
        """Return every node identifier, sorted."""
        return self.graph.entity_ids()

    def evidence_ids(self) -> tuple[str, ...]:
        """Return every stored evidence identifier, sorted."""
        return tuple(
            item.evidence_id for item in self.records.evidence()
        )

    def __len__(self) -> int:
        """Return how many nodes the graph holds."""
        return len(self.graph)

    def __contains__(self, name_or_id: object) -> bool:
        """Return whether memory holds a node for the given name or id."""
        return isinstance(name_or_id, str) and self.get_entity(name_or_id) is not None

    # -- Serialization ------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize the whole of memory into a JSON-ready dictionary.

        The document is byte-stable for a given memory: the graph, the
        records, and the history are each emitted in sorted or sequence
        order, so two identical memories serialize to identical text.
        """
        return {
            "schema_version": MEMORY_SNAPSHOT_SCHEMA_VERSION,
            "graph": self.graph.to_dict(),
            "records": self.records.to_dict(),
            "history": self._history.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ResearchMemory:
        """Rebuild memory from :meth:`to_dict` output.

        The graph, the records, and the history all come back sharing the
        single sequence index that was serialized with them, so append
        ordering survives the round trip exactly.
        """
        if not isinstance(data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a mapping")
        graph_data = data.get("graph")
        if not isinstance(graph_data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a 'graph' mapping")
        graph = KnowledgeGraph.from_dict(graph_data)
        memory = cls(index=graph.store.index, graph=graph)
        records_data = data.get("records")
        if records_data is not None:
            if not isinstance(records_data, Mapping):
                raise InvalidMemoryInputError(
                    "expected mapping for 'records'"
                )
            memory.records = MemoryStore.from_dict(
                records_data,
                index=graph.store.index,
                deduplication=memory.deduplication,
            )
        history_data = data.get("history")
        if history_data is not None:
            if not isinstance(history_data, Mapping):
                raise InvalidMemoryInputError(
                    "expected mapping for 'history'"
                )
            memory._history = HistoryLog.from_dict(history_data)
        return memory

    def memory_key(self) -> str:
        """Return the stable content key identifying this memory snapshot."""
        return self.graph.graph_key()

    def report(self) -> dict[str, object]:
        """Return a compact, JSON-ready overview of the whole memory."""
        return {
            "schema_version": MEMORY_SNAPSHOT_SCHEMA_VERSION,
            "memory_key": self.memory_key(),
            "counts": self.counts(),
            "stats": self.stats().to_dict(),
            "validation": self.validate().to_dict(),
        }


def _belief_of(entity: GraphEntity) -> tuple[object, ...]:
    """Return the part of a node that constitutes what memory believes.

    Comparing this instead of the whole node is what separates "we learned
    something" from "we ran the pipeline again": ``provenance``,
    ``last_updated`` and ``record_id`` all move on every re-observation, but
    a name, alias, attribute, confidence or validation verdict that has not
    moved is not a change in belief.
    """
    return (
        entity.name,
        entity.canonical_name,
        entity.aliases,
        entity.attributes,
        entity.confidence,
        entity.validation,
        entity.topic_id,
    )


def _require_str(
    data: Mapping[str, object],
    key: str,
    *,
    default: str = "",
) -> str:
    """Read and validate a string field."""
    value = data.get(key, default)
    if not isinstance(value, str):
        raise InvalidMemoryInputError(f"expected string for '{key}'")
    return value


def _require_float(data: Mapping[str, object], key: str) -> float:
    """Read and validate a numeric field."""
    value = data.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidMemoryInputError(f"expected number for '{key}'")
    return float(value)


def _require_mapping(
    data: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    """Read and validate a nested mapping field."""
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise InvalidMemoryInputError(f"expected mapping for '{key}'")
    return value


def _require_str_list(data: Mapping[str, object], key: str) -> list[str]:
    """Read and validate a string-list field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidMemoryInputError(f"expected list for '{key}'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidMemoryInputError(
                f"expected list of strings for '{key}'"
            )
        result.append(item)
    return result


__all__ = [
    "MEMORY_SNAPSHOT_SCHEMA_VERSION",
    "EntityObservation",
    "MemoryWrite",
    "RelationshipObservation",
    "ResearchMemory",
]
