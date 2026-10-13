"""Append-only historical tracking for research memory — Phase 8 Sprint 6.

Research memory must be able to answer "what did we believe about this
company last quarter, and why did that change?" without ever rewriting the
answer.  This module owns that guarantee.

Two frozen value objects carry it:

* :class:`HistoryChange` — one observed difference: the previous value, the
  new value, the timestamp, the source, the supporting evidence, and the
  validation status.  A change is a *statement about a difference*, so it is
  always attributable and never ambiguous.
* :class:`HistoryEntry` — one append: the entity, the before-snapshot, the
  after-snapshot, and the changes that were observed between them.

:class:`HistoryLog` is the append-only container.  It offers no ``set``,
``update``, or ``pop``: the only mutation is :meth:`HistoryLog.append`, and
:meth:`HistoryLog.restore` hands back an equal log with a *different*,
larger number of entries.  Entries are keyed by the append
:class:`~predictron_engine.research.memory_store.MemoryIndex.sequence`
number, which the caller supplies, so ordering is caller-deterministic and
the graph never reads a clock.

Two properties the rest of the layer depends on:

* **Nothing is overwritten** — appending a change for a field that already
  has one is allowed (a value can change more than once) and the earlier
  record is kept verbatim.
* **First state is a creation, not a change** — the entry that introduces
  an entity carries no :class:`HistoryChange` at all, so a reader can tell
  "we learned this" apart from "this stopped being true".
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from predictron_engine.canonical import (
    RESEARCH_DIGEST_LENGTH,
    text_digest,
)
from predictron_engine.research.exceptions import (
    ImmutableRecordError,
    InvalidGraphModelError,
    InvalidMemoryInputError,
)
from predictron_engine.research.graph_models import (
    AttributeValue,
    EntitySnapshot,
    EntityType,
    ValidationMarker,
)
from predictron_engine.research.validation_models import ValidationStatus

#: Serialization schema version for the history log.
HISTORY_SCHEMA_VERSION = "1.0.0"

#: Prefix applied to the deterministic identifier of a history entry.
HISTORY_ENTRY_PREFIX = "hist_"

#: Separator between the history prefix and the digest.
HISTORY_ENTRY_SEPARATOR = "_"

#: The :attr:`HistoryChange.kind` for a field observed for the first time.
CHANGE_KIND_ADDED = "added"

#: The :attr:`HistoryChange.kind` for a field that was removed.
CHANGE_KIND_REMOVED = "removed"

#: The :attr:`HistoryChange.kind` for a value that was replaced.
CHANGE_KIND_UPDATED = "updated"

#: The :attr:`HistoryChange.kind` for a value re-observed unchanged.
CHANGE_KIND_UNCHANGED = "unchanged"

#: The deterministic kinds a field-level change can report.
CHANGE_KINDS: tuple[str, ...] = (
    CHANGE_KIND_ADDED,
    CHANGE_KIND_REMOVED,
    CHANGE_KIND_UNCHANGED,
    CHANGE_KIND_UPDATED,
)

#: Sentinel recorded as the ``previous_value`` of a field that did not exist
#: and the ``new_value`` of a field that was removed.  It cannot collide
#: with a real value because no renderer emits a NUL character.
_ABSENT = "\x00absent"


def history_entry_id(entity_id: str, sequence: int) -> str:
    """Return the stable identifier of one history entry.

    The digest covers the entity and the append sequence, so a given
    entity's history is addressable and two entries can never collide.
    """
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ImmutableRecordError("entity_id must not be empty")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ImmutableRecordError("sequence must be an int")
    if sequence < 1:
        raise ImmutableRecordError("sequence must be at least 1")
    payload = f"{entity_id.strip()}\x1f{sequence}"
    digest = text_digest(payload, length=RESEARCH_DIGEST_LENGTH)
    return f"{HISTORY_ENTRY_PREFIX}{digest}"


@dataclass(frozen=True)
class HistoryChange:
    """One observed difference between a previous and a new value.

    Parameters
    ----------
    field_name:
        What changed: an attribute key, ``"name"``, ``"confidence"``, or
        ``"validation"``.
    previous_value:
        The value before the change, or ``None`` when the field was absent.
    new_value:
        The value after the change, or ``None`` when the field was removed.
    timestamp:
        Caller-supplied, caller-deterministic change marker.
    source:
        The report or record identifier that observed the change.
    evidence_ids:
        The evidence that supports the new value.
    validation_status:
        The validation status the new value carries.

    ``None`` is accepted for either value and is stored as the private
    absent marker, because a real attribute may legitimately hold the empty
    string and that must not be mistaken for "the field was not there".
    """

    field_name: str
    previous_value: str
    new_value: str
    timestamp: str
    source: str
    evidence_ids: tuple[str, ...] = ()
    validation_status: ValidationStatus = ValidationStatus.SKIPPED

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.strip():
            raise InvalidGraphModelError("field_name must not be empty")
        for name in ("previous_value", "new_value"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, _ABSENT)
            elif not isinstance(value, str):
                raise InvalidGraphModelError(
                    f"{name} must be a string or None for HistoryChange"
                )
        for name in ("timestamp", "source"):
            if not isinstance(getattr(self, name), str):
                raise InvalidGraphModelError(
                    f"{name} must be a string for HistoryChange"
                )
        if not isinstance(self.evidence_ids, tuple):
            raise InvalidGraphModelError(
                "evidence_ids must be a tuple for HistoryChange"
            )
        for evidence_id in self.evidence_ids:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise InvalidGraphModelError(
                    "evidence_ids entries must be non-empty strings"
                )
        if not isinstance(self.validation_status, ValidationStatus):
            raise InvalidGraphModelError(
                "validation_status must be a ValidationStatus"
            )
        object.__setattr__(self, "field_name", self.field_name.strip())
        object.__setattr__(self, "timestamp", self.timestamp.strip())
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(dict.fromkeys(self.evidence_ids)),
        )

    @property
    def was_absent(self) -> bool:
        """Return whether the field did not exist before this change."""
        return self.previous_value == _ABSENT

    @property
    def is_removed(self) -> bool:
        """Return whether the field no longer exists."""
        return self.new_value == _ABSENT

    @property
    def is_creation(self) -> bool:
        """Return whether the field appeared for the first time."""
        return self.was_absent

    @property
    def is_removal(self) -> bool:
        """Return whether the field was removed."""
        return self.is_removed

    @property
    def is_transition(self) -> bool:
        """Return whether an existing value was replaced."""
        return not self.was_absent and not self.is_removed

    @property
    def changed(self) -> bool:
        """Return whether the value actually differs."""
        return self.previous_value != self.new_value

    @property
    def kind(self) -> str:
        """Return the deterministic kind of change, from :data:`CHANGE_KINDS`."""
        if self.is_creation:
            return CHANGE_KIND_ADDED
        if self.is_removal:
            return CHANGE_KIND_REMOVED
        if not self.changed:
            return CHANGE_KIND_UNCHANGED
        return CHANGE_KIND_UPDATED

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "field_name": self.field_name,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
            "source": self.source,
            "evidence_ids": list(self.evidence_ids),
            "validation_status": self.validation_status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HistoryChange:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            field_name=_require_str(data, "field_name"),
            previous_value=_require_str(data, "previous_value", default=""),
            new_value=_require_str(data, "new_value", default=""),
            timestamp=_require_str(data, "timestamp", default=""),
            source=_require_str(data, "source", default=""),
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
            validation_status=ValidationStatus(
                _require_str(data, "validation_status")
            ),
        )


@dataclass(frozen=True)
class HistoryEntry:
    """One append to an entity's history.

    Parameters
    ----------
    entity_id:
        The entity this entry is about.
    entity_type:
        The entity's type, denormalized so a history reader never has to
        load the graph to interpret the entry.
    sequence:
        The caller-supplied append order.  Entries are always ordered by
        this value.
    recorded_at:
        The caller-supplied, caller-deterministic marker for the append.
    source:
        The report or record identifier that caused the append.
    previous:
        The value *before* the append, or ``None`` for the first one.
    current:
        The value *after* the append.
    changes:
        The observed differences, field-name sorted.
    evidence_ids:
        Every evidence identifier that contributed to the append.
    validation_status:
        The validation status recorded on the append.
    """

    entity_id: str
    entity_type: EntityType
    sequence: int
    recorded_at: str
    source: str
    previous: EntitySnapshot | None = None
    current: EntitySnapshot | None = None
    changes: tuple[HistoryChange, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    validation_status: ValidationStatus = ValidationStatus.SKIPPED

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise InvalidGraphModelError("entity_id must not be empty")
        if not isinstance(self.entity_type, EntityType):
            raise InvalidGraphModelError(
                "entity_type must be an EntityType for HistoryEntry"
            )
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise InvalidGraphModelError(
                "sequence must be an int for HistoryEntry"
            )
        if self.sequence < 1:
            raise InvalidGraphModelError(
                "sequence must be at least 1 for HistoryEntry"
            )
        for name in ("recorded_at", "source"):
            if not isinstance(getattr(self, name), str):
                raise InvalidGraphModelError(
                    f"{name} must be a string for HistoryEntry"
                )
        if self.previous is not None and not isinstance(
            self.previous, EntitySnapshot
        ):
            raise InvalidGraphModelError(
                "previous must be an EntitySnapshot for HistoryEntry"
            )
        if self.current is not None and not isinstance(
            self.current, EntitySnapshot
        ):
            raise InvalidGraphModelError(
                "current must be an EntitySnapshot for HistoryEntry"
            )
        if self.previous is None and self.changes:
            raise InvalidGraphModelError(
                "the first entry for an entity carries no changes"
            )
        if not isinstance(self.changes, tuple):
            raise InvalidGraphModelError(
                "changes must be a tuple for HistoryEntry"
            )
        names = [change.field_name for change in self.changes]
        if names != sorted(names):
            raise InvalidGraphModelError(
                "HistoryEntry changes must be ordered by field name"
            )
        if len(set(names)) != len(names):
            raise InvalidGraphModelError(
                "HistoryEntry changes must not repeat a field"
            )
        if not isinstance(self.validation_status, ValidationStatus):
            raise InvalidGraphModelError(
                "validation_status must be a ValidationStatus"
            )
        object.__setattr__(self, "entity_id", self.entity_id.strip())
        object.__setattr__(self, "recorded_at", self.recorded_at.strip())
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(
            self,
            "changes",
            tuple(
                change
                for change in self.changes
                if isinstance(change, HistoryChange)
            ),
        )
        object.__setattr__(
            self, "evidence_ids", tuple(dict.fromkeys(self.evidence_ids))
        )

    @property
    def entry_id(self) -> str:
        """Return the stable identifier of this entry."""
        return history_entry_id(self.entity_id, self.sequence)

    @property
    def is_creation(self) -> bool:
        """Return whether this entry is the entity's first record."""
        return self.previous is None

    @property
    def version(self) -> int:
        """Return the entity version this entry produced."""
        if self.current is None:
            raise InvalidGraphModelError(
                "a HistoryEntry without a current snapshot has no version"
            )
        return self.current.version

    def change_for(self, field_name: str) -> HistoryChange | None:
        """Return the change recorded for ``field_name`` or ``None``."""
        for change in self.changes:
            if change.field_name == field_name:
                return change
        return None

    def changed_fields(self) -> tuple[str, ...]:
        """Return the field names this entry observed changing."""
        return tuple(change.field_name for change in self.changes)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "entry_id": self.entry_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "sequence": self.sequence,
            "recorded_at": self.recorded_at,
            "source": self.source,
            "is_creation": self.is_creation,
            "previous": (
                self.previous.to_dict() if self.previous is not None else None
            ),
            "current": (
                self.current.to_dict() if self.current is not None else None
            ),
            "changes": [change.to_dict() for change in self.changes],
            "evidence_ids": list(self.evidence_ids),
            "validation_status": self.validation_status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HistoryEntry:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        previous = _optional_dict(data, "previous")
        current = _optional_dict(data, "current")
        return cls(
            entity_id=_require_str(data, "entity_id"),
            entity_type=EntityType(_require_str(data, "entity_type")),
            sequence=_require_int(data, "sequence"),
            recorded_at=_require_str(data, "recorded_at", default=""),
            source=_require_str(data, "source", default=""),
            previous=(
                EntitySnapshot.from_dict(previous) if previous else None
            ),
            current=(
                EntitySnapshot.from_dict(current) if current else None
            ),
            changes=tuple(
                HistoryChange.from_dict(item)
                for item in _require_list_of_dicts(data, "changes")
            ),
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
            validation_status=ValidationStatus(
                _require_str(data, "validation_status")
            ),
        )


class HistoryLog:
    """An append-only log of :class:`HistoryEntry` values.

    The log is deliberately hostile to mutation.  There is no way to replace
    an entry, and :meth:`append` returns the *new* log rather than mutating
    the old one, so an earlier snapshot of memory can be held and compared
    forever.
    """

    def __init__(
        self,
        entries: Iterable[HistoryEntry] = (),
    ) -> None:
        resolved: dict[int, HistoryEntry] = {}
        for entry in entries:
            if not isinstance(entry, HistoryEntry):
                raise InvalidMemoryInputError(
                    "history entries must be HistoryEntry instances"
                )
            if entry.sequence in resolved:
                raise ImmutableRecordError(
                    f"history sequence {entry.sequence} is already recorded"
                )
            resolved[entry.sequence] = entry
        self._entries: dict[int, HistoryEntry] = dict(sorted(resolved.items()))

    # -- Append ------------------------------------------------------

    def append(self, entry: HistoryEntry) -> HistoryLog:
        """Return a new log with ``entry`` appended.

        Re-appending an existing sequence raises rather than replacing, so
        history can never be rewritten — not even by a caller that made a
        mistake.
        """
        if not isinstance(entry, HistoryEntry):
            raise InvalidMemoryInputError(
                "append expects a HistoryEntry instance"
            )
        if entry.sequence in self._entries:
            raise ImmutableRecordError(
                f"history sequence {entry.sequence} is already recorded"
            )
        return HistoryLog((*self.ordered(), entry))

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
    ) -> HistoryLog:
        """Return a new log carrying a fully derived entry.

        The observed differences between ``previous`` and ``current`` are
        computed here, so callers cannot accidentally record a change they
        did not actually observe — or forget one they did.
        """
        changes = derive_changes(
            previous=previous,
            current=current,
            timestamp=recorded_at,
            source=source,
        )
        evidence_ids = tuple(
            sorted(
                {
                    *(
                        previous.provenance.evidence_ids
                        if previous is not None
                        else ()
                    ),
                    *(
                        current.provenance.evidence_ids
                        if current is not None
                        else ()
                    ),
                }
            )
        )
        status = (
            current.validation.status
            if current is not None
            else ValidationStatus.SKIPPED
        )
        return self.append(
            HistoryEntry(
                entity_id=entity_id,
                entity_type=entity_type,
                sequence=sequence,
                recorded_at=recorded_at,
                source=source,
                previous=previous,
                current=current,
                changes=changes,
                evidence_ids=evidence_ids,
                validation_status=status,
            )
        )

    # -- Retrieval ---------------------------------------------------

    def ordered(self) -> tuple[HistoryEntry, ...]:
        """Return every entry, ordered by sequence."""
        return tuple(self._entries[sequence] for sequence in sorted(self._entries))

    def entries(self) -> tuple[HistoryEntry, ...]:
        """Return every entry, ordered by sequence."""
        return self.ordered()

    def for_entity(self, entity_id: str) -> tuple[HistoryEntry, ...]:
        """Return every entry for ``entity_id``, ordered by sequence."""
        return tuple(
            entry
            for entry in self.ordered()
            if entry.entity_id == entity_id.strip()
        )

    def latest_for(self, entity_id: str) -> HistoryEntry | None:
        """Return the newest entry for ``entity_id`` or ``None``."""
        entries = self.for_entity(entity_id)
        return entries[-1] if entries else None

    def first_for(self, entity_id: str) -> HistoryEntry | None:
        """Return the oldest entry for ``entity_id`` or ``None``."""
        entries = self.for_entity(entity_id)
        return entries[0] if entries else None

    def changes_for(
        self,
        entity_id: str,
        field_name: str,
    ) -> tuple[HistoryChange, ...]:
        """Return every change recorded against ``field_name``, in order."""
        return tuple(
            change
            for entry in self.for_entity(entity_id)
            for change in entry.changes
            if change.field_name == field_name.strip()
        )

    def snapshots_for(self, entity_id: str) -> tuple[EntitySnapshot, ...]:
        """Return every snapshot recorded for ``entity_id``, in version order."""
        return tuple(
            entry.current
            for entry in self.for_entity(entity_id)
            if entry.current is not None
        )

    def last_value_for(self, entity_id: str, field_name: str) -> str | None:
        """Return the newest recorded value of ``field_name``.

        ``None`` means the field has never been recorded for the entity.
        A recorded *removal* returns the absent sentinel's public form
        (``""``) rather than ``None``, so absence and removal stay distinct.
        """
        changes = self.changes_for(entity_id, field_name)
        if not changes:
            return None
        value = changes[-1].new_value
        return "" if value == _ABSENT else value

    def entities(self) -> tuple[str, ...]:
        """Return every entity that has history, sorted."""
        return tuple(sorted({entry.entity_id for entry in self._entries.values()}))

    def next_sequence(self) -> int:
        """Return the sequence an appended entry would receive."""
        return max(self._entries, default=0) + 1

    def __len__(self) -> int:
        """Return how many entries are recorded."""
        return len(self._entries)

    def __contains__(self, sequence: object) -> bool:
        """Return whether ``sequence`` has been recorded."""
        return sequence in self._entries

    def __iter__(self) -> Iterator[HistoryEntry]:
        """Iterate the entries in sequence order."""
        return iter(self.ordered())

    def is_empty(self) -> bool:
        """Return whether the log holds no entries."""
        return not self._entries

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a log holding the same entries."""
        if not isinstance(other, HistoryLog):
            return NotImplemented
        return self.ordered() == other.ordered()

    def __hash__(self) -> int:
        """Return a hash of the ordered entries."""
        return hash(self.ordered())

    # -- Serialization ------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize the log into a JSON-ready dictionary."""
        return {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.ordered()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HistoryLog:
        """Rebuild a log from :meth:`to_dict` output."""
        if not isinstance(data, Mapping):
            raise InvalidMemoryInputError("from_dict expects a mapping")
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise InvalidMemoryInputError("expected list for 'entries'")
        return cls(
            HistoryEntry.from_dict(entry) for entry in entries
        )


def derive_changes(
    *,
    previous: EntitySnapshot | None,
    current: EntitySnapshot | None,
    timestamp: str,
    source: str,
) -> tuple[HistoryChange, ...]:
    """Return the observed differences between two snapshots.

    The comparison is exhaustive and deterministic.  It reports, in sorted
    field order:

    * ``name`` when the display name changed,
    * every attribute key that appeared, disappeared, or changed value,
    * ``confidence`` when the numeric confidence changed,
    * ``validation`` when the validation status changed.

    Passing ``previous=None`` yields *no* changes: the first observation of
    an entity is a creation, and is recorded as such by the absence of
    changes rather than by a synthetic "everything is new" entry.

    Comparison is on the string form of values, so an attribute that
    changed only in Python type (``1`` to ``True``) is reported as changed
    unless its string form is identical — deliberately conservative, since
    silently ignoring a type change would hide a real observation.
    """
    if current is None:
        return ()
    evidence_ids = current.provenance.evidence_ids
    status = current.validation.status
    if previous is None:
        return ()

    def _change(field_name: str, old: str, new: str) -> HistoryChange:
        return HistoryChange(
            field_name=field_name,
            previous_value=old,
            new_value=new,
            timestamp=timestamp,
            source=source,
            evidence_ids=evidence_ids,
            validation_status=status,
        )

    changes: list[HistoryChange] = []
    if previous.name != current.name:
        changes.append(_change("name", previous.name, current.name))
    old_attributes = previous.attribute_map()
    new_attributes = current.attribute_map()
    for key in sorted(set(old_attributes) | set(new_attributes)):
        old_value = old_attributes.get(key)
        new_value = new_attributes.get(key)
        if old_value is None and new_value is None:
            continue
        if old_value is None:
            changes.append(
                _change(key, _ABSENT, _render(new_value))
            )
            continue
        if new_value is None:
            changes.append(
                _change(key, _render(old_value), _ABSENT)
            )
            continue
        old_rendered = _render(old_value)
        new_rendered = _render(new_value)
        if old_rendered != new_rendered:
            changes.append(_change(key, old_rendered, new_rendered))
    if _render(previous.confidence) != _render(current.confidence):
        changes.append(
            _change(
                "confidence",
                _render(previous.confidence),
                _render(current.confidence),
            )
        )
    if previous.validation.status is not current.validation.status:
        changes.append(
            _change(
                "validation",
                previous.validation.status.value,
                current.validation.status.value,
            )
        )
    return tuple(sorted(changes, key=lambda change: change.field_name))


def _render(value: AttributeValue | None) -> str:
    """Render an attribute value deterministically for comparison."""
    if value is None:
        return _ABSENT
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def require_marker(validation: ValidationMarker) -> ValidationMarker:
    """Return ``validation`` after asserting it is a marker.

    Small public guard so callers that assemble a history entry by hand get
    the same error the dataclass would raise.
    """
    if not isinstance(validation, ValidationMarker):
        raise InvalidGraphModelError(
            "validation must be a ValidationMarker"
        )
    return validation


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


def _optional_dict(
    data: Mapping[str, object],
    key: str,
) -> dict[str, object] | None:
    """Read a nested dictionary field that may be absent or ``None``."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidGraphModelError(f"expected dict or null for '{key}'")
    return value


__all__ = [
    "CHANGE_KIND_ADDED",
    "CHANGE_KIND_REMOVED",
    "CHANGE_KIND_UNCHANGED",
    "CHANGE_KIND_UPDATED",
    "CHANGE_KINDS",
    "HISTORY_ENTRY_PREFIX",
    "HISTORY_ENTRY_SEPARATOR",
    "HISTORY_SCHEMA_VERSION",
    "HistoryChange",
    "HistoryEntry",
    "HistoryLog",
    "derive_changes",
    "history_entry_id",
    "require_marker",
]
