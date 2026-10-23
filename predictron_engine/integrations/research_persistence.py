"""Durable filesystem backend for the research persistence ports.

This module is the **initial implementation** of the persistence seam declared
by :mod:`predictron_engine.research.ports`.  It provides persistent versions of
the four storage ports — :class:`~predictron_engine.research.ports.MemoryIndexPort`,
:class:`~predictron_engine.research.ports.GraphStorePort`,
:class:`~predictron_engine.research.ports.MemoryStorePort`, and
:class:`~predictron_engine.research.ports.HistoryPort` — backed by a single
canonical JSON document on the filesystem, and wires them into the existing
:class:`~predictron_engine.research.memory.ResearchMemory` facade.

Why this lives outside ``predictron_engine.research``
----------------------------------------------------
The research package is deliberately free of I/O: it declares the seam and the
in-memory defaults, but never touches a disk, a socket, or a clock.  A durable
backend is an *adapter* over that seam, so it belongs with the other adapters
in :mod:`predictron_engine.integrations` (which already host the live evidence
and engine bridges).  The recent research-package changes are type-level only —
history gained a port and the memory facade is typed against it — and no
behaviour changed there, as the existing suite proves.

How persistence works
---------------------
The in-memory stores already implement every behavioural guarantee research
memory relies on: append-only writes, merge-instead-of-replace, duplicate
recognition, and deterministic sequence allocation.  A persistent backend only
has to add two things — *sequence allocation* survives a restart, and
*documents* round-trip — which this module does by delegating every operation
to the in-memory implementation and re-serializing its snapshot document after
each write.

The persistent classes are **not** subclasses of the in-memory stores and do not
inherit from the ports.  They satisfy the protocols structurally, exactly like
the in-memory defaults do, so they are interchangeable collaborators.

Serialization contract
----------------------
* The persisted document is
  ``{"schema_version", "index", "graph", "records", "history"}``.
  Each section is the store's own ``to_dict()`` output; **no schema version and
  no field is invented here**.  Existing ``to_dict``/``from_dict`` behaviour is
  authoritative and honoured verbatim.
* Writes are canonical (``sort_keys``, compact, ASCII), so the same state always
  produces the same bytes.
* Writes are atomic (temporary file plus :func:`os.replace`), so a crash cannot
  leave a half-written document behind.
* Nothing reads a clock and nothing generates an identifier: every label and
  every sequence comes from the caller and the shared
  :class:`~predictron_engine.research.graph_store.MemoryIndex`, exactly as in the
  in-memory implementation.

Determinism and identity guarantees
-----------------------------------
* Entity, relationship, evidence, and report identifiers are computed by the
  unchanged research code, never by the backend, so they are identical whether
  or not persistence is enabled.
* Provenance is round-tripped through the stores' own ``from_dict``/``to_dict``,
  so it is byte-for-byte unchanged across a restart.
* ``KnowledgeGraph.graph_key`` (the canonical content hash) is computed over the
  restored graph document, which is identical to the original, so the hash is
  unchanged.

Scope
-----
This work item implements the durable backend only.  It deliberately does not
add an application API, observability, evidence redesign, a richer projection,
networking, live providers, a UI, benchmark changes, a migration framework, or a
schema redesign.

Nothing here changes the default system.  Persistence is opt-in: it is only used
when a caller explicitly opens a :class:`PersistentResearchMemory` or injects the
persistent collaborators into :class:`ResearchMemory`.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from predictron_engine.canonical import canonical_json
from predictron_engine.research.exceptions import ResearchMemoryError
from predictron_engine.research.execution_report import ResearchReport
from predictron_engine.research.graph_models import (
    EntitySnapshot,
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
    GraphStore,
    MemoryIndex,
    MemoryRecordKey,
    RelationshipWrite,
)
from predictron_engine.research.history import HistoryEntry, HistoryLog
from predictron_engine.research.memory import ResearchMemory
from predictron_engine.research.memory_store import (
    EvidenceWrite,
    MemoryStore,
    ReportWrite,
    TopicRetrieval,
)
from predictron_engine.research.models import Evidence
from predictron_engine.research.ports import MemoryIndexPort

#: Serialization schema version for the durable research-memory document.
PERSISTENCE_SCHEMA_VERSION = "1.1.0"

#: Document versions the loader accepts.  ``1.0.0`` documents predate the
#: persisted history section, so they load with an empty history log.
_SUPPORTED_SCHEMA_VERSIONS = ("1.0.0", "1.1.0")


class ResearchPersistenceError(ResearchMemoryError):
    """Raised when the durable research-memory backend cannot read or write.

    Persistence faults are memory faults, so the error derives from
    :class:`~predictron_engine.research.exceptions.ResearchMemoryError` and
    callers can handle them with the same ``except`` clause they already use
    for the rest of research memory.
    """


class FileDocumentStore:
    """A durable, canonical, atomic document store on the filesystem.

    The store holds exactly one JSON document at ``path``.  It never reads a
    clock, never generates an identifier, and never reorders content: it encodes
    through :func:`~predictron_engine.canonical.canonical_json`, so writing the
    same document twice produces byte-identical files.

    Parameters
    ----------
    path:
        The file the document is persisted to.  Parent directories are created
        on first write.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the path the document is stored at."""
        return self._path

    def exists(self) -> bool:
        """Return whether a document file is present."""
        return self._path.is_file()

    def read(self) -> dict[str, object] | None:
        """Return the stored document, or ``None`` when nothing is stored.

        An absent or empty file is a fresh store, not an error.  A file that is
        not a JSON object raises :class:`ResearchPersistenceError`.
        """
        if not self._path.is_file():
            return None
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as error:
            raise ResearchPersistenceError(
                f"cannot read research memory at {self._path}: {error}"
            ) from error
        if not text.strip():
            return None
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ResearchPersistenceError(
                f"invalid JSON in research memory at {self._path}: {error}"
            ) from error
        if not isinstance(document, dict):
            raise ResearchPersistenceError(
                f"research memory at {self._path} is not a JSON object"
            )
        return document

    def write(self, document: Mapping[str, object]) -> None:
        """Persist ``document`` atomically as canonical JSON.

        The document is written to a temporary file in the destination
        directory and then moved into place with :func:`os.replace`, so a
        reader never observes a partly written file.
        """
        text = canonical_json(dict(document))
        parent = self._path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            handle_fd, temp_name = tempfile.mkstemp(
                dir=str(parent),
                prefix=".research-memory-",
                suffix=".tmp",
            )
        except OSError as error:
            raise ResearchPersistenceError(
                f"cannot prepare research memory at {self._path}: {error}"
            ) from error
        try:
            with os.fdopen(
                handle_fd, "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(text)
            os.replace(temp_name, self._path)
        except OSError as error:
            _remove_quietly(temp_name)
            raise ResearchPersistenceError(
                f"cannot write research memory at {self._path}: {error}"
            ) from error


class PersistentMemoryBackend:
    """The shared durable state behind the four persistent ports.

    One backend owns one in-memory index, one in-memory graph store, one
    in-memory record store, and one in-memory history log, plus the file they
    are persisted to.  The four persistent ports are thin views over it, so
    records, facts, and history keep sharing a single append order exactly as
    they do in memory.

    Parameters
    ----------
    path:
        The document file.  When it exists it is loaded; when it does not, an
        empty memory is created and nothing is written until the first
        mutation.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._store = FileDocumentStore(path)
        document = self._store.read()
        version = _validated_schema_version(document)
        self.index: MemoryIndex = _restore_index(document)
        self.graph: GraphStore = _restore_graph(document, self.index)
        self.records: MemoryStore = _restore_records(document, self.index)
        self.history: HistoryLog = (
            _restore_history(document)
            if version == "1.1.0"
            else HistoryLog()
        )

    @property
    def path(self) -> Path:
        """Return the path the backend persists to."""
        return self._store.path

    def document(self) -> dict[str, object]:
        """Return the current durable document.

        The document is assembled from each store's own ``to_dict()`` output, so
        it carries the stores' existing schema versions and fields unchanged.
        """
        return {
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "index": self.index.to_dict(),
            "graph": self.graph.to_dict(),
            "records": self.records.to_dict(),
            "history": self.history.to_dict(),
        }

    def flush(self) -> None:
        """Persist the current state as one atomic canonical write."""
        self._store.write(self.document())


class PersistentMemoryIndex:
    """A :class:`~predictron_engine.research.ports.MemoryIndexPort` on disk.

    Allocation delegates to the shared in-memory index and then persists the
    whole document, so a sequence that has been allocated is durable before the
    call returns.
    """

    def __init__(self, backend: PersistentMemoryBackend) -> None:
        self._backend = backend

    def allocate(
        self,
        record_type: str,
        record_id: str,
        *,
        topic_id: str = "",
        company_name: str = "",
        stored_at: str = "",
    ) -> MemoryRecordKey:
        """Allocate a sequence for a record and persist the result."""
        key = self._backend.index.allocate(
            record_type,
            record_id,
            topic_id=topic_id,
            company_name=company_name,
            stored_at=stored_at,
        )
        self._backend.flush()
        return key

    def next_sequence(self) -> int:
        """Return the sequence the next allocation will receive."""
        return self._backend.index.next_sequence()

    def key_for(
        self, record_type: str, record_id: str
    ) -> MemoryRecordKey | None:
        """Return the key already allocated for a record, or ``None``."""
        return self._backend.index.key_for(record_type, record_id)

    def keys(
        self, record_type: str | None = None
    ) -> tuple[MemoryRecordKey, ...]:
        """Return every key in sequence order, optionally narrowed by type."""
        return self._backend.index.keys(record_type)

    def to_dict(self) -> dict[str, object]:
        """Serialize the index into its JSON-ready snapshot document."""
        return self._backend.index.to_dict()

    def __len__(self) -> int:
        """Return how many records are indexed."""
        return len(self._backend.index)


class PersistentGraphStore:
    """A :class:`~predictron_engine.research.ports.GraphStorePort` on disk.

    Reads delegate to the shared in-memory graph store.  Writes delegate, then
    persist the whole document, so an appended node or edge is durable before
    the call returns.
    """

    def __init__(
        self,
        backend: PersistentMemoryBackend,
        index: MemoryIndexPort,
    ) -> None:
        self._backend = backend
        self.index: MemoryIndexPort = index

    def add_entity(
        self,
        entity: GraphEntity,
        *,
        stored_at: str = "",
        record_id: str = "",
    ) -> EntityWrite:
        """Store a node, merging it with any node already present."""
        write = self._backend.graph.add_entity(
            entity,
            stored_at=stored_at,
            record_id=record_id,
        )
        self._backend.flush()
        return write

    def add_relationship(
        self,
        relationship: GraphRelationship,
        *,
        stored_at: str = "",
    ) -> RelationshipWrite:
        """Store an edge, strengthening it when it already exists."""
        write = self._backend.graph.add_relationship(
            relationship,
            stored_at=stored_at,
        )
        self._backend.flush()
        return write

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Return the node for ``entity_id`` or ``None``."""
        return self._backend.graph.get_entity(entity_id)

    def entity_type(self, entity_id: str) -> EntityType | None:
        """Return the type of a node, or ``None`` when it is absent."""
        return self._backend.graph.entity_type(entity_id)

    def entity_ids(self) -> tuple[str, ...]:
        """Return every node identifier, sorted."""
        return self._backend.graph.entity_ids()

    def entities(self) -> tuple[GraphEntity, ...]:
        """Return every node, identifier-sorted."""
        return self._backend.graph.entities()

    def entities_of_type(
        self, entity_type: EntityType
    ) -> tuple[GraphEntity, ...]:
        """Return every node of ``entity_type``, identifier-sorted."""
        return self._backend.graph.entities_of_type(entity_type)

    def relationships(self) -> tuple[GraphRelationship, ...]:
        """Return every edge, key-sorted."""
        return self._backend.graph.relationships()

    def relationships_by_key(
        self,
        key: tuple[str, str, str],
    ) -> GraphRelationship | None:
        """Return the edge stored under a canonical key or ``None``."""
        return self._backend.graph.relationships_by_key(key)

    def relationship_count(self) -> int:
        """Return how many edges are stored."""
        return self._backend.graph.relationship_count()

    def get_relationship(
        self,
        source_id: str,
        relationship_type: RelationshipType,
        target_id: str,
    ) -> GraphRelationship | None:
        """Return the edge between two endpoints or ``None``."""
        return self._backend.graph.get_relationship(
            source_id, relationship_type, target_id
        )

    def incident_keys(
        self,
        entity_id: str,
        direction: GraphDirection = GraphDirection.OUTGOING,
    ) -> tuple[tuple[str, str, str], ...]:
        """Return the edge keys touching ``entity_id``, key-sorted."""
        return self._backend.graph.incident_keys(entity_id, direction)

    def other_endpoint(self, key: tuple[str, str, str], entity_id: str) -> str:
        """Return the endpoint of ``key`` that is not ``entity_id``."""
        return self._backend.graph.other_endpoint(key, entity_id)

    def stats(self) -> GraphStats:
        """Return the deterministic structural counters for the graph."""
        return self._backend.graph.stats()

    def validate(self) -> GraphValidation:
        """Return the groundedness report for the stored graph."""
        return self._backend.graph.validate()

    def to_dict(self) -> dict[str, object]:
        """Serialize the store into its JSON-ready snapshot document."""
        return self._backend.graph.to_dict()

    def __len__(self) -> int:
        """Return how many nodes are stored."""
        return len(self._backend.graph)


class PersistentMemoryStore:
    """A :class:`~predictron_engine.research.ports.MemoryStorePort` on disk.

    Reads delegate to the shared in-memory record ledger.  Writes delegate, then
    persist the whole document, so a stored report or evidence item is durable
    before the call returns.
    """

    def __init__(
        self,
        backend: PersistentMemoryBackend,
        index: MemoryIndexPort,
    ) -> None:
        self._backend = backend
        self.index: MemoryIndexPort = index

    def add_evidence(
        self,
        evidence: Evidence,
        *,
        stored_at: str = "",
        report_id: str = "",
    ) -> EvidenceWrite:
        """Store an evidence item, recognising a repeat as a duplicate."""
        write = self._backend.records.add_evidence(
            evidence,
            stored_at=stored_at,
            report_id=report_id,
        )
        self._backend.flush()
        return write

    def add_report(
        self,
        report: ResearchReport,
        *,
        evidence: Iterable[Evidence] = (),
        stored_at: str = "",
        topic_id: str = "",
    ) -> tuple[ReportWrite, tuple[EvidenceWrite, ...]]:
        """Store a report and the evidence it carries, reporting what changed."""
        writes = self._backend.records.add_report(
            report,
            evidence=evidence,
            stored_at=stored_at,
            topic_id=topic_id,
        )
        self._backend.flush()
        return writes

    def is_empty(self) -> bool:
        """Return whether nothing has been recorded."""
        return self._backend.records.is_empty()

    def counts(self) -> dict[str, int]:
        """Return the deterministic record counters for the store."""
        return self._backend.records.counts()

    def has_evidence(self, evidence_id: str) -> bool:
        """Return whether ``evidence_id`` is stored."""
        return self._backend.records.has_evidence(evidence_id)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Return the stored evidence item or ``None``."""
        return self._backend.records.get_evidence(evidence_id)

    def get_report(self, report_id: str) -> ResearchReport | None:
        """Return the stored report or ``None``."""
        return self._backend.records.get_report(report_id)

    def reports(self) -> tuple[ResearchReport, ...]:
        """Return every stored report, sequence-ordered."""
        return self._backend.records.reports()

    def evidence(self) -> tuple[Evidence, ...]:
        """Return every stored evidence item, sequence-ordered."""
        return self._backend.records.evidence()

    def retrieve_latest(
        self,
        company_name: str = "",
        topic_id: str = "",
        limit: int = 1,
    ) -> tuple[ResearchReport, ...]:
        """Return the ``limit`` newest matching reports, newest first."""
        return self._backend.records.retrieve_latest(
            company_name, topic_id, limit
        )

    def topic_records(self, topic_id: str) -> TopicRetrieval:
        """Return the reports and evidence recorded for ``topic_id``."""
        return self._backend.records.topic_records(topic_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize the store into its JSON-ready snapshot document."""
        return self._backend.records.to_dict()


class PersistentHistoryStore:
    """A :class:`~predictron_engine.research.ports.HistoryPort` on disk.

    Reads delegate to the shared in-memory history log.  Writes rebind the
    backend's log to the value the in-memory implementation returns — the log
    has value semantics, so every write yields a new log — and then persist the
    whole document, so a recorded entry is durable before the call returns.
    """

    def __init__(self, backend: PersistentMemoryBackend) -> None:
        self._backend = backend

    def append(self, entry: HistoryEntry) -> PersistentHistoryStore:
        """Return this store after appending ``entry`` and persisting it."""
        self._backend.history = self._backend.history.append(entry)
        self._backend.flush()
        return self

    def record(
        self,
        *,
        entity_id: str,
        entity_type: EntityType,
        sequence: int,
        recorded_at: str,
        source: str,
        previous: EntitySnapshot | None,
        current: EntitySnapshot | None,
    ) -> PersistentHistoryStore:
        """Return this store after recording a derived entry and persisting it."""
        self._backend.history = self._backend.history.record(
            entity_id=entity_id,
            entity_type=entity_type,
            sequence=sequence,
            recorded_at=recorded_at,
            source=source,
            previous=previous,
            current=current,
        )
        self._backend.flush()
        return self

    def ordered(self) -> tuple[HistoryEntry, ...]:
        """Return every entry, ordered by sequence."""
        return self._backend.history.ordered()

    def for_entity(self, entity_id: str) -> tuple[HistoryEntry, ...]:
        """Return every entry for ``entity_id``, ordered by sequence."""
        return self._backend.history.for_entity(entity_id)

    def snapshots_for(self, entity_id: str) -> tuple[EntitySnapshot, ...]:
        """Return every snapshot recorded for ``entity_id``, in version order."""
        return self._backend.history.snapshots_for(entity_id)

    def to_dict(self) -> dict[str, object]:
        """Serialize the log into its JSON-ready snapshot document."""
        return self._backend.history.to_dict()

    def __len__(self) -> int:
        """Return how many entries are recorded."""
        return len(self._backend.history)

    def is_empty(self) -> bool:
        """Return whether the log holds no entries."""
        return self._backend.history.is_empty()


class PersistentResearchMemory:
    """A :class:`ResearchMemory` whose state is persisted to disk.

    The facade wires the four persistent ports over one shared backend and
    hands them to the unchanged :class:`ResearchMemory`, whose constructor
    enforces the shared-sequence invariant.  Reads and writes behave exactly as
    the in-memory implementation does; the only difference is that each write is
    flushed to the document before returning.

    Parameters
    ----------
    path:
        The document file.  When it exists it is loaded; when it does not, an
        empty memory is created.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.backend = PersistentMemoryBackend(path)
        self.index = PersistentMemoryIndex(self.backend)
        self.graph = PersistentGraphStore(self.backend, self.index)
        self.records = PersistentMemoryStore(self.backend, self.index)
        self.history = PersistentHistoryStore(self.backend)
        self.memory = ResearchMemory(
            store=self.graph,
            records=self.records,
            history=self.history,
        )

    @property
    def path(self) -> Path:
        """Return the path the memory persists to."""
        return self.backend.path

    def document(self) -> dict[str, object]:
        """Return the current durable document."""
        return self.backend.document()

    def flush(self) -> None:
        """Persist the current state without performing any other write."""
        self.backend.flush()


def open_persistent_memory(path: str | os.PathLike[str]) -> ResearchMemory:
    """Open (or create) a durable :class:`ResearchMemory` at ``path``.

    This is the convenience entry point: it builds the four persistent ports,
    wires them into the unchanged memory facade, and returns the facade.  The
    persistent collaborators remain reachable as ``memory.store``,
    ``memory.records``, ``memory.history``, and ``memory.index``.
    """
    return PersistentResearchMemory(path).memory


def _validated_schema_version(
    document: Mapping[str, object] | None,
) -> str | None:
    """Validate the persisted document version and return it.

    A fresh store has no document and therefore no version to check.  A stored
    document must name one of :data:`_SUPPORTED_SCHEMA_VERSIONS`; a missing,
    non-string, or unsupported version is rejected before any section is
    restored, so an unrecognized document is never partially interpreted.
    """
    if document is None:
        return None
    supported = ", ".join(_SUPPORTED_SCHEMA_VERSIONS)
    version = document.get("schema_version")
    if not isinstance(version, str):
        raise ResearchPersistenceError(
            "missing or invalid 'schema_version' in research memory; "
            f"supported versions: {supported}"
        )
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ResearchPersistenceError(
            f"unsupported persistence schema version {version!r} in research "
            f"memory; supported versions: {supported}"
        )
    return version


def _restore_index(
    document: Mapping[str, object] | None,
) -> MemoryIndex:
    """Rebuild the shared index from a persisted document."""
    if document is None:
        return MemoryIndex()
    section = document.get("index")
    if section is None:
        return MemoryIndex()
    return MemoryIndex.from_dict(_require_mapping(section, what="'index'"))


def _restore_graph(
    document: Mapping[str, object] | None,
    index: MemoryIndex,
) -> GraphStore:
    """Rebuild the graph store from a persisted document.

    Nodes are restored before edges so the endpoint checks succeed, and every
    allocation goes through ``index``, which already holds the original
    sequences, so the restored order is identical to the original.
    """
    store = GraphStore(index=index)
    if document is None:
        return store
    section = document.get("graph")
    if section is None:
        return store
    data = _require_mapping(section, what="'graph'")
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    if not isinstance(entities, list):
        raise ResearchPersistenceError("expected a list for 'graph.entities'")
    if not isinstance(relationships, list):
        raise ResearchPersistenceError(
            "expected a list for 'graph.relationships'"
        )
    for entry in entities:
        store.add_entity(
            GraphEntity.from_dict(
                _require_mapping(entry, what="a graph entity")
            )
        )
    for entry in relationships:
        store.add_relationship(
            GraphRelationship.from_dict(
                _require_mapping(entry, what="a graph relationship")
            )
        )
    return store


def _restore_records(
    document: Mapping[str, object] | None,
    index: MemoryIndex,
) -> MemoryStore:
    """Rebuild the record ledger from a persisted document.

    The shared ``index`` is injected so the records, the graph, and history keep
    one append order after a restart.
    """
    if document is None:
        return MemoryStore(index=index)
    section = document.get("records")
    if section is None:
        return MemoryStore(index=index)
    return MemoryStore.from_dict(
        _require_mapping(section, what="'records'"),
        index=index,
    )


def _restore_history(
    document: Mapping[str, object] | None,
) -> HistoryLog:
    """Rebuild the history log from a persisted document."""
    if document is None:
        return HistoryLog()
    section = document.get("history")
    if section is None:
        return HistoryLog()
    return HistoryLog.from_dict(_require_mapping(section, what="'history'"))


def _require_mapping(value: object, *, what: str) -> Mapping[str, object]:
    """Validate that a serialized section is a mapping."""
    if not isinstance(value, Mapping):
        raise ResearchPersistenceError(f"expected a mapping for {what}")
    return value


def _remove_quietly(path: str) -> None:
    """Remove a temporary file, ignoring any failure."""
    try:
        os.unlink(path)
    except OSError:
        pass


__all__ = [
    "PERSISTENCE_SCHEMA_VERSION",
    "FileDocumentStore",
    "PersistentGraphStore",
    "PersistentHistoryStore",
    "PersistentMemoryBackend",
    "PersistentMemoryIndex",
    "PersistentMemoryStore",
    "PersistentResearchMemory",
    "ResearchPersistenceError",
    "open_persistent_memory",
]
