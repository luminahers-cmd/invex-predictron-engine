"""Append-only storage for the research knowledge graph — Phase 8 Sprint 6.

This module is the *write path* of the graph.  It owns:

* :class:`MemoryIndex` — the deterministic append-order counter every
  mutable part of research memory shares, so "latest" and "history" have a
  single, caller-controlled notion of order.  The index never reads a clock:
  the caller supplies the ``stored_at`` label, which is what makes retrieval
  reproducible.
* :class:`EntityWrite` / :class:`RelationshipWrite` — the return values that
  report *what changed*, which is precisely what the history log needs to
  append a :class:`~predictron_engine.research.history.HistoryEntry`.
* :class:`GraphStore` — the append-only container for
  :class:`~predictron_engine.research.graph_models.GraphEntity` and
  :class:`~predictron_engine.research.graph_models.GraphRelationship`
  values, with the adjacency indexes traversal needs.

Why a store and not a plain graph object
----------------------------------------
Splitting the write path from the read path is what lets future sprints add a
persistent backend without redesign.  A backend only has to implement
*allocate a sequence number* and *round-trip a document*; traversal, search,
and validation stay in :mod:`predictron_engine.research.knowledge_graph` and
keep working unchanged.

Append-only guarantees
----------------------
* :meth:`GraphStore.add_entity` never replaces a node's facts.  It merges
  the new observation into a *new* node value and reports the previous one
  through :class:`EntityWrite`, so the caller can snapshot it into history.
* :meth:`GraphStore.add_relationship` never replaces an edge.  A repeat
  observation only ever *strengthens* the existing edge's support set and
  confidence, and again reports the previous value.
* Every edge must reference two nodes that already exist and whose types
  satisfy the relationship's spec, so the graph cannot hold a dangling or
  mistyped edge even transiently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from predictron_engine.research.exceptions import (
    ImmutableRecordError,
    InvalidGraphModelError,
    InvalidMemoryInputError,
    InvalidRelationshipError,
)
from predictron_engine.research.graph_models import (
    GRAPH_SCHEMA_VERSION,
    RELATIONSHIP_SPECS,
    AttributeValue,
    EntityType,
    GraphEntity,
    GraphRelationship,
    GraphStats,
    GraphValidation,
    Provenance,
    RelationshipType,
    ValidationMarker,
)
from predictron_engine.research.ports import MemoryIndexPort

#: Serialization schema version for the graph store and the memory index.
STORE_SCHEMA_VERSION = "1.0.0"

#: Prefix applied to a canonical graph key.
GRAPH_KEY_PREFIX = "graph_"

#: The direction an adjacency lookup walks.


class GraphDirection(str, Enum):
    """Which way an adjacency lookup walks.

    * ``OUTGOING`` — follow edges from the entity to its targets.
    * ``INCOMING`` — follow edges from the entity to its sources.
    * ``BOTH`` — follow edges either way.
    """

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"

    def includes_outgoing(self) -> bool:
        """Return whether this direction follows edges forward."""
        return self in (GraphDirection.OUTGOING, GraphDirection.BOTH)

    def includes_incoming(self) -> bool:
        """Return whether this direction follows edges backward."""
        return self in (GraphDirection.INCOMING, GraphDirection.BOTH)


@dataclass(frozen=True)
class MemoryRecordKey:
    """The deterministic position of one stored memory record.

    Parameters
    ----------
    record_type:
        The kind of record, e.g. ``"report"``, ``"evidence"``,
        ``"validation"``, ``"execution"``, ``"entity"``, or
        ``"relationship"``.
    record_id:
        The record's own stable identifier.
    sequence:
        The append order.  Strictly increasing per allocation and the only
        ordering research memory relies on.
    topic_id:
        The research topic the record belongs to, when known.
    company_name:
        The company the record is about, when known.
    stored_at:
        The caller-supplied, caller-deterministic label for the append.
    """

    record_type: str
    record_id: str
    sequence: int
    topic_id: str = ""
    company_name: str = ""
    stored_at: str = ""

    def __post_init__(self) -> None:
        for name in ("record_type", "record_id"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise InvalidMemoryInputError(f"{name} must not be empty")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise InvalidMemoryInputError("sequence must be an int")
        if self.sequence < 1:
            raise InvalidMemoryInputError("sequence must be at least 1")
        for name in ("topic_id", "company_name", "stored_at"):
            if not isinstance(getattr(self, name), str):
                raise InvalidMemoryInputError(f"{name} must be a string")
        object.__setattr__(self, "record_type", self.record_type.strip())
        object.__setattr__(self, "record_id", self.record_id.strip())
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "company_name", self.company_name.strip())
        object.__setattr__(self, "stored_at", self.stored_at.strip())

    @property
    def composite(self) -> tuple[str, str]:
        """Return the ``(record_type, record_id)`` identity of the record."""
        return (self.record_type, self.record_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "record_type": self.record_type,
            "record_id": self.record_id,
            "sequence": self.sequence,
            "topic_id": self.topic_id,
            "company_name": self.company_name,
            "stored_at": self.stored_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryRecordKey:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            record_type=_require_str(data, "record_type"),
            record_id=_require_str(data, "record_id"),
            sequence=_require_int(data, "sequence"),
            topic_id=_require_str(data, "topic_id", default=""),
            company_name=_require_str(data, "company_name", default=""),
            stored_at=_require_str(data, "stored_at", default=""),
        )


class MemoryIndex:
    """The deterministic append-order index shared by every memory store.

    Allocating a sequence is the only mutation.  Allocating a
    ``(record_type, record_id)`` pair that is already indexed is a no-op
    that returns the existing key, so replaying a report cannot shift the
    order of anything stored after it.
    """

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
        """Return the key for a record, allocating a sequence if needed.

        An existing record keeps its original sequence, which is what makes
        "the latest report for this company" stable no matter how many times
        the same report is added.
        """
        if not isinstance(record_type, str) or not record_type.strip():
            raise InvalidMemoryInputError("record_type must not be empty")
        if not isinstance(record_id, str) or not record_id.strip():
            raise InvalidMemoryInputError("record_id must not be empty")
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
        """Return the sequence the next allocation will receive."""
        return self._next

    def key_for(self, record_type: str, record_id: str) -> MemoryRecordKey | None:
        """Return the key for a record, or ``None`` when absent."""
        return self._keys.get((record_type.strip(), record_id.strip()))

    def keys(
        self,
        record_type: str | None = None,
    ) -> tuple[MemoryRecordKey, ...]:
        """Return every key, ordered by sequence.

        Passing ``record_type`` narrows the result to one kind of record.
        """
        selected = [
            key
            for key in self._keys.values()
            if record_type is None or key.record_type == record_type.strip()
        ]
        return tuple(sorted(selected, key=lambda key: (key.sequence, key.record_id)))

    def record_ids(self, record_type: str) -> tuple[str, ...]:
        """Return the identifiers stored for ``record_type``, sequence-ordered."""
        return tuple(
            key.record_id
            for key in self.keys(record_type)
        )

    def latest(self, record_type: str) -> MemoryRecordKey | None:
        """Return the highest-sequence key for ``record_type`` or ``None``."""
        keys = self.keys(record_type)
        return keys[-1] if keys else None

    def contains(self, record_type: str, record_id: str) -> bool:
        """Return whether a record has been indexed."""
        return (record_type.strip(), record_id.strip()) in self._keys

    def __len__(self) -> int:
        """Return how many records are indexed."""
        return len(self._keys)

    def __contains__(self, composite: object) -> bool:
        """Return whether a ``(record_type, record_id)`` pair is indexed."""
        if not isinstance(composite, tuple) or len(composite) != 2:
            return False
        return composite in self._keys

    def to_dict(self) -> dict[str, object]:
        """Serialize the index into a JSON-ready dictionary."""
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "next_sequence": self._next,
            "keys": [key.to_dict() for key in self.keys()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MemoryIndex:
        """Rebuild an index from :meth:`to_dict` output."""
        if not isinstance(data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a mapping")
        entries = data.get("keys", [])
        if not isinstance(entries, list):
            raise InvalidMemoryInputError("expected list for 'keys'")
        index = cls()
        restored: list[MemoryRecordKey] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise InvalidMemoryInputError(
                    "expected mappings for 'keys' entries"
                )
            restored.append(MemoryRecordKey.from_dict(entry))
        sequences = [key.sequence for key in restored]
        if len(set(sequences)) != len(sequences):
            raise ImmutableRecordError(
                "memory index sequences must be unique"
            )
        for key in sorted(restored, key=lambda item: item.sequence):
            if index.contains(key.record_type, key.record_id):
                raise ImmutableRecordError(
                    "memory index records must be unique"
                )
            index.allocate(
                key.record_type,
                key.record_id,
                topic_id=key.topic_id,
                company_name=key.company_name,
                stored_at=key.stored_at,
            )
        declared = data.get("next_sequence", index._next)
        if isinstance(declared, int) and not isinstance(declared, bool):
            index._next = max(declared, index._next)
        return index


@dataclass(frozen=True)
class EntityWrite:
    """The outcome of writing one node.

    ``previous`` is the node value as it stood *before* this write, or
    ``None`` when the node is new.  Passing it straight to
    :meth:`~predictron_engine.research.history.HistoryLog.record` is all the
    history layer needs; the caller never has to reconstruct the past.
    """

    entity: GraphEntity
    previous: GraphEntity | None
    sequence: int

    @property
    def created(self) -> bool:
        """Return whether this write introduced the node."""
        return self.previous is None

    @property
    def changed(self) -> bool:
        """Return whether this write altered the node's stored value."""
        return self.previous is not None and self.previous != self.entity

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "entity": self.entity.to_dict(),
            "created": self.created,
            "changed": self.changed,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class RelationshipWrite:
    """The outcome of writing one edge.

    Like :class:`EntityWrite`, ``previous`` is the pre-write value so the
    caller can record that an already-known relationship was re-observed —
    which is a fact worth keeping, not an update worth applying.
    """

    relationship: GraphRelationship
    previous: GraphRelationship | None
    sequence: int

    @property
    def created(self) -> bool:
        """Return whether this write introduced the edge."""
        return self.previous is None

    @property
    def strengthened(self) -> bool:
        """Return whether this write added support to an existing edge."""
        return self.previous is not None and self.previous != self.relationship

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "relationship": self.relationship.to_dict(),
            "created": self.created,
            "strengthened": self.strengthened,
            "sequence": self.sequence,
        }


class GraphStore:
    """The append-only container behind the research knowledge graph.

    The store holds the current value of every node and edge, the adjacency
    indexes traversal needs, and nothing else.  History lives in
    :class:`~predictron_engine.research.history.HistoryLog`; the store only
    hands out the previous values needed to build it.
    """

    def __init__(self, index: MemoryIndexPort | None = None) -> None:
        self.index: MemoryIndexPort = index if index is not None else MemoryIndex()
        self._entities: dict[str, GraphEntity] = {}
        self._relationships: dict[tuple[str, str, str], GraphRelationship] = {}
        self._outgoing: dict[str, set[tuple[str, str, str]]] = {}
        self._incoming: dict[str, set[tuple[str, str, str]]] = {}

    # -- Entities ----------------------------------------------------

    def has_entity(self, entity_id: str) -> bool:
        """Return whether ``entity_id`` is stored."""
        return entity_id in self._entities

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Return the node for ``entity_id`` or ``None``."""
        return self._entities.get(entity_id)

    def entity_type(self, entity_id: str) -> EntityType | None:
        """Return the type of ``entity_id`` or ``None`` when absent."""
        entity = self._entities.get(entity_id)
        return entity.entity_type if entity is not None else None

    def add_entity(
        self,
        entity: GraphEntity,
        *,
        stored_at: str = "",
        record_id: str = "",
    ) -> EntityWrite:
        """Store ``entity``, merging it into any node already present.

        A brand-new node is stored verbatim.  An existing node is *never*
        replaced: the new observation is merged into a new value via
        :meth:`GraphEntity.with_updates` and the pre-write value comes back
        in the result, ready to be snapshotted into history.
        """
        if not isinstance(entity, GraphEntity):
            raise InvalidGraphModelError("add_entity expects a GraphEntity")
        existing = self._entities.get(entity.entity_id)
        if existing is None:
            stored = entity
        elif existing.entity_type is not entity.entity_type:
            raise InvalidGraphModelError(
                f"entity {entity.entity_id} is already stored as "
                f"{existing.entity_type.value}"
            )
        else:
            stored = existing.with_updates(
                provenance=entity.provenance,
                attributes=dict(entity.attributes),
                confidence=entity.confidence,
                last_updated=entity.last_updated,
                validation=entity.validation,
                aliases=(entity.name, *entity.aliases),
                record_id=record_id,
            )
        key = self.index.allocate(
            "entity",
            entity.entity_id,
            topic_id=entity.topic_id,
            stored_at=stored_at,
        )
        self._entities[entity.entity_id] = stored
        return EntityWrite(
            entity=stored,
            previous=existing,
            sequence=key.sequence,
        )

    def update_entity(
        self,
        entity_id: str,
        *,
        provenance: Provenance | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        confidence: float | None = None,
        last_updated: str | None = None,
        validation: ValidationMarker | None = None,
        aliases: tuple[str, ...] = (),
        record_id: str = "",
        stored_at: str = "",
    ) -> EntityWrite:
        """Apply an incremental update to a stored node.

        Returns the merged node together with the pre-update value.  Raises
        when the node is unknown, because an update to something that was
        never observed would silently invent a fact.
        """
        existing = self._entities.get(entity_id)
        if existing is None:
            raise InvalidGraphModelError(
                f"cannot update unknown entity {entity_id!r}"
            )
        merged = existing.with_updates(
            provenance=provenance,
            attributes=attributes,
            confidence=confidence,
            last_updated=last_updated,
            validation=validation,
            aliases=aliases,
            record_id=record_id,
        )
        key = self.index.allocate("entity", entity_id, stored_at=stored_at)
        self._entities[entity_id] = merged
        return EntityWrite(
            entity=merged,
            previous=existing,
            sequence=key.sequence,
        )

    def entity_ids(self) -> tuple[str, ...]:
        """Return every stored node identifier, sorted."""
        return tuple(sorted(self._entities))

    def entities(self) -> tuple[GraphEntity, ...]:
        """Return every node, identifier-sorted."""
        return tuple(self._entities[entity_id] for entity_id in sorted(self._entities))

    def entities_of_type(self, entity_type: EntityType) -> tuple[GraphEntity, ...]:
        """Return every node of ``entity_type``, identifier-sorted."""
        return tuple(
            entity
            for entity in self.entities()
            if entity.entity_type is entity_type
        )

    def __len__(self) -> int:
        """Return how many nodes are stored."""
        return len(self._entities)

    # -- Relationships -----------------------------------------------

    def add_relationship(
        self,
        relationship: GraphRelationship,
        *,
        stored_at: str = "",
    ) -> RelationshipWrite:
        """Store ``relationship``, strengthening it if it already exists.

        A repeat observation unions the supporting evidence and takes the
        higher confidence; it never rewrites the edge into something else.
        Both endpoints must already be stored, and their types must satisfy
        the relationship's spec, or the write is rejected.
        """
        if not isinstance(relationship, GraphRelationship):
            raise InvalidRelationshipError(
                "add_relationship expects a GraphRelationship"
            )
        self._require_endpoints(relationship)
        existing = self._relationships.get(relationship.key)
        if existing is None:
            stored = relationship
        else:
            stored = existing.with_support(
                evidence_ids=relationship.evidence_ids,
                provenance=relationship.provenance,
                confidence=max(existing.confidence, relationship.confidence),
                timestamp=relationship.timestamp or existing.timestamp,
                attributes=dict(relationship.attributes),
            )
        key = self.index.allocate(
            "relationship",
            stored.relationship_id(),
            stored_at=stored_at,
        )
        self._relationships[stored.key] = stored
        self._link(stored)
        return RelationshipWrite(
            relationship=stored,
            previous=existing,
            sequence=key.sequence,
        )

    def _require_endpoints(self, relationship: GraphRelationship) -> None:
        """Reject an edge whose endpoints are missing or mistyped."""
        source = self._entities.get(relationship.source_id)
        target = self._entities.get(relationship.target_id)
        if source is None or target is None:
            missing = [
                entity_id
                for entity_id, entity in (
                    (relationship.source_id, source),
                    (relationship.target_id, target),
                )
                if entity is None
            ]
            raise InvalidRelationshipError(
                f"relationship endpoints are not stored: {sorted(missing)}"
            )
        relationship.check_endpoints(
            source.entity_type, target.entity_type
        )

    def _link(self, relationship: GraphRelationship) -> None:
        """Index an edge in the adjacency maps."""
        key = relationship.key
        self._outgoing.setdefault(relationship.source_id, set()).add(key)
        self._incoming.setdefault(relationship.target_id, set()).add(key)

    def has_relationship(
        self,
        source_id: str,
        relationship_type: RelationshipType,
        target_id: str,
    ) -> bool:
        """Return whether an edge exists between the two endpoints."""
        return (source_id, relationship_type.value, target_id) in self._relationships

    def get_relationship(
        self,
        source_id: str,
        relationship_type: RelationshipType,
        target_id: str,
    ) -> GraphRelationship | None:
        """Return the stored edge between the two endpoints or ``None``."""
        if not isinstance(relationship_type, RelationshipType):
            raise InvalidRelationshipError(
                "get_relationship expects a RelationshipType"
            )
        return self._relationships.get(
            (source_id, relationship_type.value, target_id)
        )

    def relationships(self) -> tuple[GraphRelationship, ...]:
        """Return every edge, key-sorted."""
        return tuple(
            self._relationships[key] for key in sorted(self._relationships)
        )

    def relationships_by_key(
        self,
        key: tuple[str, str, str],
    ) -> GraphRelationship | None:
        """Return the edge stored under a canonical ``key`` or ``None``.

        This is the read counterpart to :meth:`incident_keys`: a caller that
        walks the adjacency index gets the key first and the edge second,
        which is the order traversal needs and which avoids building a
        dictionary of every edge on the way.
        """
        return self._relationships.get(key)

    def relationship_count(self) -> int:
        """Return how many edges are stored."""
        return len(self._relationships)

    def incident_keys(
        self,
        entity_id: str,
        direction: GraphDirection = GraphDirection.OUTGOING,
    ) -> tuple[tuple[str, str, str], ...]:
        """Return the edge keys touching ``entity_id``, key-sorted.

        For a symmetric relationship the canonical key orders its endpoints,
        so :meth:`other_endpoint` must be used to walk them.
        """
        if not isinstance(direction, GraphDirection):
            raise InvalidGraphModelError("direction must be a GraphDirection")
        keys: set[tuple[str, str, str]] = set()
        if direction.includes_outgoing():
            keys |= self._outgoing.get(entity_id, set())
        if direction.includes_incoming():
            keys |= self._incoming.get(entity_id, set())
        return tuple(sorted(keys))

    def other_endpoint(
        self,
        key: tuple[str, str, str],
        entity_id: str,
    ) -> str:
        """Return the endpoint of ``key`` that is not ``entity_id``."""
        source_id, _type, target_id = key
        if entity_id == source_id:
            return target_id
        if entity_id == target_id:
            return source_id
        raise InvalidGraphModelError(
            f"edge {key} does not touch {entity_id!r}"
        )

    def endpoints(self, key: tuple[str, str, str]) -> tuple[str, str]:
        """Return both endpoints of ``key`` in canonical order."""
        return (key[0], key[2])

    # -- Reporting ---------------------------------------------------

    def stats(self) -> GraphStats:
        """Return the deterministic structural counters for the graph."""
        entity_counts: dict[str, int] = {}
        validated = 0
        referenced: set[str] = set()
        topics: set[str] = set()
        evidence_nodes = 0
        for entity in self._entities.values():
            key = entity.entity_type.value
            entity_counts[key] = entity_counts.get(key, 0) + 1
            if entity.validation.is_validated:
                validated += 1
            referenced.update(entity.evidence_ids)
            if entity.topic_id:
                topics.add(entity.topic_id)
            if entity.is_evidence:
                evidence_nodes += 1
        relationship_counts: dict[str, int] = {}
        supporting: set[str] = set()
        for relationship in self._relationships.values():
            key = relationship.relationship_type.value
            relationship_counts[key] = relationship_counts.get(key, 0) + 1
            supporting.update(relationship.evidence_ids)
        referenced |= supporting
        return GraphStats(
            entity_count=len(self._entities),
            relationship_count=len(self._relationships),
            evidence_node_count=evidence_nodes,
            business_entity_count=len(self._entities) - evidence_nodes,
            validated_entity_count=validated,
            referenced_evidence_count=len(referenced),
            entity_counts=tuple(entity_counts.items()),
            relationship_counts=tuple(relationship_counts.items()),
            topics=tuple(sorted(topics)),
        )

    def validate(self) -> GraphValidation:
        """Check the graph for dangling, ungrounded, or mistyped elements.

        A clean store always validates: dangling edges are impossible
        through the write path, and every relationship is required to carry
        evidence.  The check is kept because a store can also be rebuilt from
        a serialized document, and a future persistent backend will load
        through exactly that path.
        """
        dangling: list[str] = []
        missing_provenance: list[str] = []
        invalid_relationships: list[str] = []
        for key, relationship in sorted(self._relationships.items()):
            source = self._entities.get(relationship.source_id)
            target = self._entities.get(relationship.target_id)
            if source is None or target is None:
                dangling.append(
                    f"{relationship.source_id} -[{relationship.relationship_type.value}]-> "
                    f"{relationship.target_id}"
                )
                continue
            if not relationship.evidence_ids:
                missing_provenance.append(
                    relationship.relationship_id()
                )
                continue
            spec = RELATIONSHIP_SPECS[relationship.relationship_type]
            if not spec.supports(source.entity_type, target.entity_type):
                invalid_relationships.append(
                    f"{relationship.relationship_id()}: "
                    f"{source.entity_type.value} -> {target.entity_type.value}"
                )
        for entity_id, entity in sorted(self._entities.items()):
            if entity.provenance.is_empty and not entity.is_evidence:
                missing_provenance.append(entity_id)
        return GraphValidation(
            dangling=tuple(sorted(dangling)),
            missing_provenance=tuple(sorted(missing_provenance)),
            invalid_relationships=tuple(sorted(invalid_relationships)),
        )

    # -- Serialization ------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize the store into a JSON-ready dictionary.

        Nodes and edges are emitted in sorted order and the index comes last,
        so the document is byte-stable for a given store and can be hashed or
        diffed directly.
        """
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "index": self.index.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities()],
            "relationships": [
                relationship.to_dict() for relationship in self.relationships()
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GraphStore:
        """Rebuild a store from :meth:`to_dict` output.

        Nodes are restored before edges so the endpoint checks in
        :meth:`add_relationship` succeed, and the restored index keeps the
        original sequence numbering.
        """
        if not isinstance(data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a mapping")
        index_data = data.get("index")
        store = cls(
            MemoryIndex.from_dict(index_data)
            if isinstance(index_data, Mapping)
            else MemoryIndex()
        )
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            raise InvalidMemoryInputError("expected list for 'entities'")
        for entry in entities:
            if not isinstance(entry, Mapping):
                raise InvalidMemoryInputError(
                    "expected mappings for 'entities' entries"
                )
            store.add_entity(GraphEntity.from_dict(entry))
        relationships = data.get("relationships", [])
        if not isinstance(relationships, list):
            raise InvalidMemoryInputError("expected list for 'relationships'")
        for entry in relationships:
            if not isinstance(entry, Mapping):
                raise InvalidMemoryInputError(
                    "expected mappings for 'relationships' entries"
                )
            store.add_relationship(GraphRelationship.from_dict(entry))
        return store


def _require_str(
    data: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read and validate a string field from a serialized dictionary."""
    value = data.get(key, default)
    if not isinstance(value, str):
        raise InvalidMemoryInputError(f"expected string for '{key}'")
    return value


def _require_int(data: Mapping[str, object], key: str) -> int:
    """Read and validate an int field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMemoryInputError(f"expected int for '{key}'")
    return value


__all__ = [
    "GRAPH_KEY_PREFIX",
    "STORE_SCHEMA_VERSION",
    "EntityWrite",
    "GraphDirection",
    "GraphStore",
    "MemoryIndex",
    "MemoryRecordKey",
    "RelationshipWrite",
]
