"""The research knowledge graph — Phase 8 Sprint 6.

:class:`KnowledgeGraph` is the *read path* over the append-only
:class:`~predictron_engine.research.graph_store.GraphStore`.  It answers the
questions future sprints will keep asking:

* what is known about a company, an entity, or a topic?
* what is connected to what, and how strongly?
* which evidence supports this fact, and which report observed it?
* how do two entities relate, if at all?
* what changed since the last time we looked?

Everything is deterministic.  Traversal visits neighbours in a fixed sort
order, breadth-first searches break ties by node identifier, search results
are rank-ordered then identifier-ordered, and :meth:`KnowledgeGraph.graph_key`
is a content hash over a canonical document — so two graphs built from the
same observations hash identically on any machine.

The graph also owns a :class:`~predictron_engine.research.linking.EntityLinker`,
which is what makes ``memory.get_company("Open AI")`` find the node created
from ``"OpenAI Inc."``.

Isolation without a database
----------------------------
:meth:`KnowledgeGraph.subgraph` returns a new graph over a selected set of
nodes, built by copying the frozen value objects.  Because nothing is shared
and nothing is mutated, the original graph cannot be changed through the
copy — which is the same isolation guarantee a real backend would give, and
the reason a future graph database is an additive change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from predictron_engine.canonical import (
    RESEARCH_DIGEST_LENGTH,
    canonical_json,
    text_digest,
)
from predictron_engine.research.exceptions import (
    InvalidGraphModelError,
    InvalidMemoryInputError,
)
from predictron_engine.research.graph_models import (
    GRAPH_SCHEMA_VERSION,
    EntityType,
    GraphEntity,
    GraphRelationship,
    GraphStats,
    GraphValidation,
    RelationshipType,
)
from predictron_engine.research.graph_store import (
    GRAPH_KEY_PREFIX,
    GraphDirection,
    GraphStore,
)
from predictron_engine.research.linking import (
    LINKING_VERSION,
    EntityLinker,
    normalize_name,
)
from predictron_engine.research.ports import GraphStorePort

#: Serialization schema version for the canonical graph document.
KNOWLEDGE_GRAPH_SCHEMA_VERSION = GRAPH_SCHEMA_VERSION

#: Default how many search results :meth:`KnowledgeGraph.search` returns.
DEFAULT_SEARCH_LIMIT = 10

#: The default traversal depth when a caller does not supply one.
DEFAULT_TRAVERSAL_DEPTH = 2

#: A walk that never leaves the start node, used for a depth of zero.
_TRAVERSAL_DIRECTION = GraphDirection.BOTH


class SearchScope(str, Enum):
    """How much of a node a search is allowed to match.

    * ``ALL`` — name, aliases, and attribute values.
    * ``NAME`` — the name and its aliases only.
    * ``ATTRIBUTES`` — attribute keys and values only.
    """

    ALL = "all"
    NAME = "name"
    ATTRIBUTES = "attributes"

    def includes_name(self) -> bool:
        """Return whether this scope matches names."""
        return self in (SearchScope.ALL, SearchScope.NAME)

    def includes_attributes(self) -> bool:
        """Return whether this scope matches attributes."""
        return self in (SearchScope.ALL, SearchScope.ATTRIBUTES)


@dataclass(frozen=True)
class GraphNeighbor:
    """One entity reached from another, with the edge that reached it.

    ``depth`` is the number of edges walked from the traversal's start, so
    a direct neighbour has depth ``1``.  Results are sorted by
    ``(entity_type, entity_id, relationship_type)``, which is stable because
    every component is derived from content.
    """

    entity_id: str
    entity_type: EntityType
    name: str
    relationship_type: RelationshipType
    confidence: float
    depth: int
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise InvalidGraphModelError("entity_id must not be empty")
        if not isinstance(self.entity_type, EntityType):
            raise InvalidGraphModelError(
                "entity_type must be an EntityType for GraphNeighbor"
            )
        if not isinstance(self.relationship_type, RelationshipType):
            raise InvalidGraphModelError(
                "relationship_type must be a RelationshipType for GraphNeighbor"
            )
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise InvalidGraphModelError(
                "depth must be an int for GraphNeighbor"
            )
        if self.depth < 1:
            raise InvalidGraphModelError(
                "depth must be at least 1 for GraphNeighbor"
            )
        object.__setattr__(self, "entity_id", self.entity_id.strip())
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(self.evidence_ids)),
        )

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Return the deterministic ordering key for this neighbour."""
        return (
            self.entity_type.value,
            self.entity_id,
            self.relationship_type.value,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "depth": self.depth,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GraphNeighbor:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            entity_id=_require_str(data, "entity_id"),
            entity_type=EntityType(_require_str(data, "entity_type")),
            name=_require_str(data, "name", default=""),
            relationship_type=RelationshipType(
                _require_str(data, "relationship_type")
            ),
            confidence=_require_float(data, "confidence"),
            depth=_require_int(data, "depth"),
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
        )


@dataclass(frozen=True)
class TraversalResult:
    """The deterministic outcome of a bounded graph walk.

    The result records *what was reached* and *how*, so a caller can answer
    "show me this company's two-hop neighbourhood" and then ask why any
    particular node is in it.

    Parameters
    ----------
    start_id:
        Where the walk began.
    max_depth:
        How many edges the walk was allowed to follow.
    entities:
        Every node reached, excluding the start node, identifier-sorted.
    relationships:
        Every edge walked, key-sorted.
    neighbors:
        Every reach event, sorted by :attr:`GraphNeighbor.sort_key`.
    """

    start_id: str
    max_depth: int
    entities: tuple[GraphEntity, ...] = ()
    relationships: tuple[GraphRelationship, ...] = ()
    neighbors: tuple[GraphNeighbor, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.start_id, str) or not self.start_id.strip():
            raise InvalidGraphModelError("start_id must not be empty")
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            raise InvalidGraphModelError("max_depth must be an int")
        if self.max_depth < 0:
            raise InvalidGraphModelError("max_depth must not be negative")
        object.__setattr__(self, "start_id", self.start_id.strip())

    def __len__(self) -> int:
        """Return how many nodes the walk reached, excluding the start."""
        return len(self.entities)

    def entity_ids(self) -> tuple[str, ...]:
        """Return the reached node identifiers, sorted."""
        return tuple(entity.entity_id for entity in self.entities)

    def has_entity(self, entity_id: str) -> bool:
        """Return whether ``entity_id`` was reached."""
        return any(
            entity.entity_id == entity_id for entity in self.entities
        )

    def depth_of(self, entity_id: str) -> int | None:
        """Return the depth ``entity_id`` was reached at, or ``None``."""
        for neighbor in self.neighbors:
            if neighbor.entity_id == entity_id:
                return neighbor.depth
        return None

    def of_type(self, entity_type: EntityType) -> tuple[GraphEntity, ...]:
        """Return the reached nodes of ``entity_type``, identifier-sorted."""
        return tuple(
            entity
            for entity in self.entities
            if entity.entity_type is entity_type
        )

    def evidence_ids(self) -> tuple[str, ...]:
        """Return every evidence identifier reached, sorted."""
        collected: set[str] = set()
        for entity in self.entities:
            collected.update(entity.evidence_ids)
        for relationship in self.relationships:
            collected.update(relationship.evidence_ids)
        return tuple(sorted(collected))

    def relationship_count(self) -> int:
        """Return how many edges were walked."""
        return len(self.relationships)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "start_id": self.start_id,
            "max_depth": self.max_depth,
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [
                relationship.to_dict() for relationship in self.relationships
            ],
            "neighbors": [neighbor.to_dict() for neighbor in self.neighbors],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TraversalResult:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            start_id=_require_str(data, "start_id"),
            max_depth=_require_int(data, "max_depth"),
            entities=tuple(
                GraphEntity.from_dict(item)
                for item in _require_list_of_dicts(data, "entities")
            ),
            relationships=tuple(
                GraphRelationship.from_dict(item)
                for item in _require_list_of_dicts(data, "relationships")
            ),
            neighbors=tuple(
                GraphNeighbor.from_dict(item)
                for item in _require_list_of_dicts(data, "neighbors")
            ),
        )


class KnowledgeGraph:
    """A deterministic, evidence-backed graph of everything research found.

    Parameters
    ----------
    store:
        The append-only container.  A fresh one is created when omitted.
    linker:
        The entity linker used to resolve name variants.  A fresh linker is
        created when omitted.

    The graph is read-only by design: every mutation goes through
    :class:`~predictron_engine.research.graph_store.GraphStore`, so the graph
    can hand out frozen views without a caller being able to corrupt them.
    """

    def __init__(
        self,
        store: GraphStorePort | None = None,
        linker: EntityLinker | None = None,
    ) -> None:
        if store is not None and not isinstance(store, GraphStorePort):
            raise InvalidGraphModelError("store must implement GraphStorePort")
        if linker is not None and not isinstance(linker, EntityLinker):
            raise InvalidGraphModelError("linker must be an EntityLinker instance")
        self.store: GraphStorePort = store if store is not None else GraphStore()
        self.linker = linker if linker is not None else EntityLinker()

    # -- Basic views -------------------------------------------------

    def __len__(self) -> int:
        """Return how many nodes the graph holds."""
        return len(self.store)

    def is_empty(self) -> bool:
        """Return whether the graph holds no nodes."""
        return not self.store.entity_ids()

    def entity_ids(self) -> tuple[str, ...]:
        """Return every node identifier, sorted."""
        return self.store.entity_ids()

    def entities(self) -> tuple[GraphEntity, ...]:
        """Return every node, identifier-sorted."""
        return self.store.entities()

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Return the node for ``entity_id`` or ``None``."""
        return self.store.get_entity(entity_id)

    def entities_of_type(self, entity_type: EntityType) -> tuple[GraphEntity, ...]:
        """Return every node of ``entity_type``, identifier-sorted."""
        return self.store.entities_of_type(entity_type)

    def entity_type(self, entity_id: str) -> EntityType | None:
        """Return the type of ``entity_id`` or ``None`` when absent."""
        return self.store.entity_type(entity_id)

    def get_entity_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> GraphEntity | None:
        """Return the node whose name resolves to ``name``.

        Resolution is deterministic and alias-aware: ``"Open AI"`` finds the
        node created from ``"OpenAI Inc."``.  When ``entity_type`` is given,
        only nodes of that type are considered, so a company and a product
        that share a name never shadow each other.
        """
        if not isinstance(name, str) or not name.strip():
            return None
        types = (
            (entity_type,) if entity_type is not None else tuple(EntityType)
        )
        for candidate in types:
            identifier = self.linker.peek(candidate.value, name)
            if identifier is not None:
                entity = self.store.get_entity(identifier)
                if entity is not None:
                    return entity
        normalized = normalize_name(name)
        if not normalized:
            return None
        for entity in self.entities():
            if entity_type is not None and entity.entity_type is not entity_type:
                continue
            # Compare normalized to normalized: an alias is stored as the
            # caller wrote it, so comparing it to a normalized query would
            # miss every alias that carries punctuation or different case.
            if entity.canonical_name == normalized or any(
                normalize_name(candidate) == normalized
                for candidate in entity.names
            ):
                return entity
        return None

    def get_company(self, name: str) -> GraphEntity | None:
        """Return the company node for ``name`` or ``None``.

        This is the entry point the public API advertises:
        ``memory.graph.get_company("OpenAI")``.
        """
        return self.get_entity_by_name(name, EntityType.COMPANY)

    def companies(self) -> tuple[GraphEntity, ...]:
        """Return every company node, identifier-sorted."""
        return self.entities_of_type(EntityType.COMPANY)

    def resolve(self, entity_type: EntityType, name: str) -> str | None:
        """Return the identifier ``name`` resolves to, without recording it."""
        return self.linker.peek(entity_type.value, name)

    # -- Relationships -----------------------------------------------

    def relationships(self) -> tuple[GraphRelationship, ...]:
        """Return every edge, key-sorted."""
        return self.store.relationships()

    def relationship_count(self) -> int:
        """Return how many edges the graph holds."""
        return self.store.relationship_count()

    def get_relationship(
        self,
        source_id: str,
        relationship_type: RelationshipType,
        target_id: str,
    ) -> GraphRelationship | None:
        """Return the edge between two endpoints or ``None``."""
        return self.store.get_relationship(
            source_id, relationship_type, target_id
        )

    def get_relationships(
        self,
        entity_id: str | None = None,
        relationship_type: RelationshipType | None = None,
        direction: GraphDirection = GraphDirection.BOTH,
    ) -> tuple[GraphRelationship, ...]:
        """Return the edges matching every supplied filter, key-sorted.

        ``entity_id=None`` returns every edge, which makes this the general
        form of ``get_relationships()`` in the public memory API.
        """
        if entity_id is None:
            candidates = list(self.store.relationships())
        else:
            candidates = []
            for key in self.store.incident_keys(entity_id, direction):
                relationship = self.store.relationships_by_key(key)
                if relationship is not None:
                    candidates.append(relationship)
        if relationship_type is not None:
            candidates = [
                relationship
                for relationship in candidates
                if relationship.relationship_type is relationship_type
            ]
        return tuple(
            sorted(candidates, key=lambda relationship: relationship.key)
        )

    def outgoing_relationships(
        self,
        entity_id: str,
        relationship_type: RelationshipType | None = None,
    ) -> tuple[GraphRelationship, ...]:
        """Return the edges leaving ``entity_id``, key-sorted."""
        return self.get_relationships(
            entity_id, relationship_type, GraphDirection.OUTGOING
        )

    def incoming_relationships(
        self,
        entity_id: str,
        relationship_type: RelationshipType | None = None,
    ) -> tuple[GraphRelationship, ...]:
        """Return the edges arriving at ``entity_id``, key-sorted."""
        return self.get_relationships(
            entity_id, relationship_type, GraphDirection.INCOMING
        )

    # -- Traversal ---------------------------------------------------

    def neighbors(
        self,
        entity_id: str,
        relationship_type: RelationshipType | None = None,
        direction: GraphDirection = GraphDirection.BOTH,
        depth: int = 1,
    ) -> tuple[GraphNeighbor, ...]:
        """Return the entities reachable from ``entity_id``, sorted.

        A ``depth`` of ``1`` returns the direct neighbours; larger values
        walk outward breadth-first.  Each entity appears at the *shallowest*
        depth it was reached at, so results never depend on traversal order.
        """
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise InvalidGraphModelError("depth must be an int")
        if depth < 1:
            raise InvalidGraphModelError("depth must be at least 1")
        if not isinstance(direction, GraphDirection):
            raise InvalidGraphModelError("direction must be a GraphDirection")
        if self.store.get_entity(entity_id) is None:
            return ()
        return self.traverse(
            entity_id,
            max_depth=depth,
            relationship_type=relationship_type,
            direction=direction,
        ).neighbors

    def traverse(
        self,
        start_id: str,
        max_depth: int = DEFAULT_TRAVERSAL_DEPTH,
        relationship_type: RelationshipType | None = None,
        direction: GraphDirection = _TRAVERSAL_DIRECTION,
    ) -> TraversalResult:
        """Walk outward from ``start_id`` breadth-first and return the result.

        The walk is deterministic: at every level, candidate keys are
        key-sorted before the frontier is built, so two runs over the same
        graph reach the same nodes at the same depths.
        """
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise InvalidGraphModelError("max_depth must be an int")
        if max_depth < 0:
            raise InvalidGraphModelError("max_depth must not be negative")
        if not isinstance(direction, GraphDirection):
            raise InvalidGraphModelError("direction must be a GraphDirection")
        if self.store.get_entity(start_id) is None:
            return TraversalResult(start_id=start_id, max_depth=max_depth)
        reached: dict[str, GraphNeighbor] = {}
        walked: dict[tuple[str, str, str], GraphRelationship] = {}
        visited = {start_id}
        frontier = [start_id]
        for depth in range(1, max_depth + 1):
            next_frontier: list[str] = []
            for current in sorted(frontier):
                for key in self.store.incident_keys(current, direction):
                    relationship = self.store.relationships_by_key(key)
                    if relationship is None:
                        continue
                    if (
                        relationship_type is not None
                        and relationship.relationship_type is not relationship_type
                    ):
                        continue
                    other_id = self.store.other_endpoint(key, current)
                    walked[key] = relationship
                    if other_id in visited:
                        continue
                    visited.add(other_id)
                    next_frontier.append(other_id)
                    other = self.store.get_entity(other_id)
                    if other is None:
                        continue
                    reached[other_id] = GraphNeighbor(
                        entity_id=other_id,
                        entity_type=other.entity_type,
                        name=other.name,
                        relationship_type=relationship.relationship_type,
                        confidence=relationship.confidence,
                        depth=depth,
                        evidence_ids=relationship.evidence_ids,
                    )
            if not next_frontier:
                break
            frontier = next_frontier
        return TraversalResult(
            start_id=start_id,
            max_depth=max_depth,
            entities=tuple(
                entity
                for entity in (
                    self.store.get_entity(entity_id) for entity_id in sorted(reached)
                )
                if entity is not None
            ),
            relationships=tuple(
                walked[key] for key in sorted(walked)
            ),
            neighbors=tuple(
                sorted(reached.values(), key=lambda neighbor: neighbor.sort_key)
            ),
        )

    def shortest_path(
        self,
        start_id: str,
        target_id: str,
        direction: GraphDirection = _TRAVERSAL_DIRECTION,
    ) -> tuple[str, ...] | None:
        """Return the shortest node path from ``start_id`` to ``target_id``.

        Returns ``None`` when the two are not connected.  Ties are broken by
        choosing the lexicographically smallest neighbour at each step, so
        the returned path is the same on every run.
        """
        if self.store.get_entity(start_id) is None:
            return None
        if self.store.get_entity(target_id) is None:
            return None
        if start_id == target_id:
            return (start_id,)
        visited = {start_id}
        frontier: list[tuple[str, ...]] = [(start_id,)]
        while frontier:
            following: list[tuple[str, ...]] = []
            for path in frontier:
                current = path[-1]
                for key in self.store.incident_keys(current, direction):
                    other_id = self.store.other_endpoint(key, current)
                    if other_id in visited:
                        continue
                    if other_id == target_id:
                        return (*path, other_id)
                    visited.add(other_id)
                    following.append((*path, other_id))
            if not following:
                return None
            following.sort()
            frontier = following
        return None

    def subgraph(
        self,
        entity_ids: Sequence[str],
        relationship_type: RelationshipType | None = None,
    ) -> KnowledgeGraph:
        """Return a new graph over ``entity_ids`` and the edges between them.

        The copy is deep and isolated: the nodes and edges are the same
        frozen values, but they live in a fresh store and a fresh linker, so
        nothing done to the copy can be observed through the original.
        """
        selected = tuple(dict.fromkeys(entity_ids))
        store = GraphStore()
        linker = EntityLinker()
        for entity_id in selected:
            entity = self.store.get_entity(entity_id)
            if entity is None:
                continue
            store.add_entity(entity)
            linker.resolve(entity.entity_type.value, entity.name)
            for alias in entity.aliases:
                linker.resolve(entity.entity_type.value, alias)
        for relationship in self.relationships():
            if (
                relationship_type is not None
                and relationship.relationship_type is not relationship_type
            ):
                continue
            if (
                store.has_entity(relationship.source_id)
                and store.has_entity(relationship.target_id)
            ):
                store.add_relationship(relationship)
        return KnowledgeGraph(store=store, linker=linker)

    def companies_for_topic(self, topic_id: str) -> tuple[GraphEntity, ...]:
        """Return the company nodes researched for ``topic_id``, sorted."""
        return tuple(
            entity
            for entity in self.companies()
            if entity.topic_id == topic_id.strip()
        )

    # -- Search ------------------------------------------------------

    def search(
        self,
        query: str,
        entity_type: EntityType | None = None,
        scope: SearchScope = SearchScope.ALL,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> tuple[GraphEntity, ...]:
        """Return the nodes matching ``query``, best match first.

        Matching is substring matching over normalized text — no fuzzy
        matching, no stemming, no ranking model.  Ties are broken by node
        identifier, so the result order is stable.

        Ranking is deterministic and explainable:

        1. an exact normalized-name match, then a canonical-name match,
        2. a name prefix match,
        3. an alias prefix match,
        4. any other substring match in scope,
        5. fewer attributes, then node identifier.
        """
        if not isinstance(query, str):
            raise InvalidGraphModelError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidGraphModelError("limit must be an int")
        if limit < 0:
            raise InvalidGraphModelError("limit must not be negative")
        if not isinstance(scope, SearchScope):
            raise InvalidGraphModelError("scope must be a SearchScope")
        normalized = normalize_name(query) if scope.includes_name() else query.strip().casefold()
        candidates: list[tuple[int, int, str, GraphEntity]] = []
        for entity in self.entities():
            if entity_type is not None and entity.entity_type is not entity_type:
                continue
            rank = _rank(entity, query, normalized, scope)
            if rank is None:
                continue
            candidates.append((rank, len(entity.attributes), entity.entity_id, entity))
        candidates.sort(key=lambda entry: entry[:3])
        return tuple(entry[3] for entry in candidates[:limit])

    def search_by_name(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> tuple[GraphEntity, ...]:
        """Return the nodes whose name matches ``query``, best match first."""
        return self.search(query, entity_type, SearchScope.NAME, limit)

    # -- Reporting ---------------------------------------------------

    def stats(self) -> GraphStats:
        """Return the deterministic structural counters for the graph."""
        return self.store.stats()

    def validate(self) -> GraphValidation:
        """Return the groundedness report for the graph."""
        return self.store.validate()

    def to_dict(self) -> dict[str, object]:
        """Serialize the graph into a JSON-ready dictionary."""
        return self.store.to_dict()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> KnowledgeGraph:
        """Rebuild a graph from :meth:`to_dict` output.

        The linker is rebuilt from the nodes' own names and aliases, so name
        resolution works identically on a deserialized graph.
        """
        if not isinstance(data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a mapping")
        graph = cls(store=GraphStore.from_dict(data))
        for entity in graph.entities():
            graph.linker.resolve(entity.entity_type.value, entity.name)
            for alias in entity.aliases:
                graph.linker.resolve(entity.entity_type.value, alias)
        return graph

    def canonical_json(self) -> str:
        """Return the canonical JSON document for this graph.

        Canonical means sorted keys, compact separators, and nodes and edges
        in sorted order — so the same graph always produces the same string,
        which is what :meth:`graph_key` hashes.  Delegates to the shared
        encoder in :mod:`predictron_engine.canonical`.
        """
        return canonical_json(self.to_dict())

    def graph_key(self) -> str:
        """Return the stable content hash identifying this graph.

        The digest covers the nodes, the edges, *and* the append ordering
        from the memory index, so two graphs with the same facts but observed
        in a different order are distinguishable — which is what a monitoring
        sprint needs in order to tell "nothing changed" from "something was
        re-observed".
        """
        canonical = self.canonical_json()
        digest = text_digest(canonical, length=RESEARCH_DIGEST_LENGTH)
        return f"{GRAPH_KEY_PREFIX}{digest}"

    def report(self) -> dict[str, object]:
        """Return a compact, JSON-ready overview of the graph."""
        return {
            "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
            "linking_version": LINKING_VERSION,
            "graph_key": self.graph_key(),
            "stats": self.stats().to_dict(),
            "validation": self.validate().to_dict(),
        }


def _rank(
    entity: GraphEntity,
    query: str,
    normalized: str,
    scope: SearchScope,
) -> int | None:
    """Return the deterministic match rank for ``entity``, or ``None``."""
    if not normalized:
        return None
    if scope.includes_name():
        if entity.canonical_name == normalized:
            return 0
        if entity.name.strip().casefold() == query.strip().casefold():
            return 0
        for alias in entity.aliases:
            if alias.strip().casefold() == query.strip().casefold():
                return 0
        if entity.canonical_name.startswith(normalized):
            return 1
        for alias in entity.aliases:
            if alias.strip().casefold().startswith(normalized):
                return 2
        if normalized in entity.canonical_name or any(
            normalized in alias.strip().casefold() for alias in entity.aliases
        ):
            return 3
    if scope.includes_attributes():
        for key, value in entity.attributes:
            haystack = f"{key} {value}".strip().casefold()
            if haystack.startswith(normalized):
                return 2
            if normalized in haystack:
                return 3
    return None


def _require_str(
    data: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    """Read and validate a string field from a serialized dictionary."""
    value = data.get(key, default)
    if not isinstance(value, str):
        raise InvalidGraphModelError(f"expected string for '{key}'")
    return value


def _require_int(data: Mapping[str, object], key: str) -> int:
    """Read and validate an int field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGraphModelError(f"expected int for '{key}'")
    return value


def _require_float(data: Mapping[str, object], key: str) -> float:
    """Read and validate a float field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidGraphModelError(f"expected number for '{key}'")
    return float(value)


def _require_str_list(
    data: Mapping[str, object],
    key: str,
) -> list[str]:
    """Read and validate a string-list field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidGraphModelError(f"expected list for '{key}'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise InvalidGraphModelError(
                f"expected list of strings for '{key}'"
            )
        result.append(item)
    return result


def _require_list_of_dicts(
    data: Mapping[str, object],
    key: str,
) -> list[dict[str, object]]:
    """Read and validate a list-of-dicts field."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise InvalidGraphModelError(f"expected list for '{key}'")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise InvalidGraphModelError(
                f"expected list of dicts for '{key}'"
            )
        result.append(item)
    return result


__all__ = [
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_TRAVERSAL_DEPTH",
    "KNOWLEDGE_GRAPH_SCHEMA_VERSION",
    "GraphNeighbor",
    "KnowledgeGraph",
    "SearchScope",
    "TraversalResult",
]
