"""Research knowledge-graph value objects — Phase 8 Sprint 6.

This module owns the *vocabulary* of the research knowledge graph: the
entity types, the strongly typed relationship vocabulary, and the frozen
value objects that carry them.

* :class:`EntityType` — the kinds of node the graph holds.
* :class:`RelationshipType` — the kinds of edge the graph holds.
* :class:`RelationshipSpec` — the permitted endpoints for one relationship.
* :class:`Provenance` — where a fact came from.
* :class:`ValidationMarker` — how a fact fared against evidence validation.
* :class:`GraphEntity` — one node.
* :class:`GraphRelationship` — one edge.
* :class:`EntitySnapshot` — the immutable "previous value" of a node.
* :class:`GraphStats` / :class:`GraphValidation` — structural reporting.

Design rules
------------
**Strongly typed edges.**  Every :class:`RelationshipType` declares the
entity types permitted at each end through :data:`RELATIONSHIP_SPECS`, so
``FOUNDER_OF`` can only point from a :attr:`EntityType.FOUNDER` to a
:attr:`EntityType.COMPANY`, and a violation raises
:class:`~predictron_engine.research.exceptions.InvalidRelationshipError`
instead of quietly landing in the graph.

**Evidence or nothing.**  Every node and every edge must carry
:class:`Provenance` and at least one supporting evidence identifier.  A
relationship with no evidence cannot be created, which is how the graph
guarantees that nothing in it was invented.

**Append-only.**  Every object here is a ``frozen`` dataclass.  Updates go
through :meth:`GraphEntity.with_updates` and
:meth:`GraphRelationship.with_support`, which both return *new* objects
and never mutate their input.

**Deterministic.**  Attributes are stored as a key-sorted tuple, aliases and
evidence identifiers are deduplicated while preserving first-seen order, and
every derived ordering is sorted.  Identifiers are content hashes computed
by :mod:`predictron_engine.research.linking`, so they are stable across
processes and machines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeVar

from predictron_engine.canonical import RESEARCH_DIGEST_LENGTH, text_digest
from predictron_engine.research.exceptions import (
    InvalidGraphModelError,
    InvalidRelationshipError,
)
from predictron_engine.research.validation_models import (
    EvidenceConfidence,
    ValidationStatus,
)

#: Bound used by :func:`_require_enum` to read a closed vocabulary field.
EnumT = TypeVar("EnumT", bound=Enum)

#: Serialization schema version for every graph value object.
GRAPH_SCHEMA_VERSION = "1.0.0"

#: Prefix applied to content-derived relationship identifiers.
RELATIONSHIP_ID_PREFIX = "rel_"

#: Separator between the relationship prefix and the digest.
RELATIONSHIP_ID_SEPARATOR = "_"

#: A JSON-scalar graph attribute.  Attributes are deliberately scalar so a
#: node can never smuggle an unordered collection into a content hash.
AttributeValue = str | int | float | bool


class EntityType(str, Enum):
    """The kinds of node the research knowledge graph holds.

    The fifteen business types (``COMPANY`` through ``REVIEW``) are the
    researched subject matter.  :attr:`EVIDENCE` is the sixteenth: evidence
    items are first-class nodes so ``EVIDENCE_ENTITY`` — the
    "Evidence -> Entity" relationship — is a real edge in the graph rather
    than a bookkeeping column, and so any node can be traced back to the
    exact evidence that supports it.
    """

    COMPANY = "company"
    FOUNDER = "founder"
    TEAM = "team"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    CUSTOMER = "customer"
    MARKET = "market"
    COMPETITOR = "competitor"
    INVESTOR = "investor"
    FUNDING_ROUND = "funding_round"
    PARTNERSHIP = "partnership"
    PATENT = "patent"
    RISK = "risk"
    NEWS_EVENT = "news_event"
    REVIEW = "review"
    EVIDENCE = "evidence"


class RelationshipType(str, Enum):
    """The kinds of edge the research knowledge graph holds.

    Every member is strongly typed: the permitted endpoints are declared in
    :data:`RELATIONSHIP_SPECS` and enforced on construction.
    """

    FOUNDER_OF = "founder_of"
    PREVIOUS_COMPANY = "previous_company"
    HAS_PRODUCT = "has_product"
    HAS_TECHNOLOGY = "has_technology"
    OPERATES_IN = "operates_in"
    COMPETES_WITH = "competes_with"
    HAS_CUSTOMER = "has_customer"
    HAS_FUNDING = "has_funding"
    FUNDED_BY = "funded_by"
    HAS_PARTNERSHIP = "has_partnership"
    COMPETES_ON_TECHNOLOGY = "competes_on_technology"
    HAS_RISK = "has_risk"
    EVIDENCE_ENTITY = "evidence_entity"


#: The entity types that describe a researched business, i.e. everything
#: except the evidence nodes.
BUSINESS_ENTITY_TYPES: frozenset[EntityType] = frozenset(
    entity_type
    for entity_type in EntityType
    if entity_type is not EntityType.EVIDENCE
)


@dataclass(frozen=True)
class RelationshipSpec:
    """The endpoint contract for one :class:`RelationshipType`.

    Parameters
    ----------
    source_types:
        Entity types permitted at the source end.
    target_types:
        Entity types permitted at the target end.
    symmetric:
        Whether the relationship means the same thing in both directions.
        Symmetric relationships are stored with canonically ordered
        endpoints so ``(a, X, b)`` and ``(b, X, a)`` are the same edge.
    description:
        Human-readable statement of what the relationship asserts.
    """

    source_types: frozenset[EntityType]
    target_types: frozenset[EntityType]
    symmetric: bool
    description: str

    def supports(self, source: EntityType, target: EntityType) -> bool:
        """Return whether ``source`` -> ``target`` is permitted."""
        return source in self.source_types and target in self.target_types

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "source_types": sorted(item.value for item in self.source_types),
            "target_types": sorted(item.value for item in self.target_types),
            "symmetric": self.symmetric,
            "description": self.description,
        }


def _company_anchored(target: EntityType) -> RelationshipSpec:
    """Return a spec whose only permitted source is a company."""
    return RelationshipSpec(
        source_types=frozenset({EntityType.COMPANY}),
        target_types=frozenset({target}),
        symmetric=False,
        description=f"A company is connected to a {target.value}.",
    )


#: The complete, strongly typed relationship vocabulary.
RELATIONSHIP_SPECS: Mapping[RelationshipType, RelationshipSpec] = MappingProxyType(
    {
        RelationshipType.FOUNDER_OF: RelationshipSpec(
            source_types=frozenset({EntityType.FOUNDER}),
            target_types=frozenset({EntityType.COMPANY}),
            symmetric=False,
            description="A person founded a company.",
        ),
        RelationshipType.PREVIOUS_COMPANY: RelationshipSpec(
            source_types=frozenset({EntityType.FOUNDER}),
            target_types=frozenset({EntityType.COMPANY}),
            symmetric=False,
            description="A person was previously employed by a company.",
        ),
        RelationshipType.HAS_PRODUCT: _company_anchored(EntityType.PRODUCT),
        RelationshipType.HAS_TECHNOLOGY: _company_anchored(
            EntityType.TECHNOLOGY
        ),
        RelationshipType.OPERATES_IN: _company_anchored(EntityType.MARKET),
        RelationshipType.COMPETES_WITH: RelationshipSpec(
            source_types=frozenset({EntityType.COMPANY}),
            target_types=frozenset({EntityType.COMPETITOR}),
            symmetric=False,
            description="A company competes with a named competitor.",
        ),
        RelationshipType.HAS_CUSTOMER: _company_anchored(EntityType.CUSTOMER),
        RelationshipType.HAS_FUNDING: _company_anchored(
            EntityType.FUNDING_ROUND
        ),
        RelationshipType.FUNDED_BY: RelationshipSpec(
            source_types=frozenset({EntityType.FUNDING_ROUND}),
            target_types=frozenset({EntityType.INVESTOR}),
            symmetric=False,
            description="An investor funded a round.",
        ),
        RelationshipType.HAS_PARTNERSHIP: _company_anchored(
            EntityType.PARTNERSHIP
        ),
        RelationshipType.COMPETES_ON_TECHNOLOGY: RelationshipSpec(
            source_types=frozenset({EntityType.TECHNOLOGY}),
            target_types=frozenset({EntityType.COMPETITOR}),
            symmetric=False,
            description="A technology overlaps with a competitor's offering.",
        ),
        RelationshipType.HAS_RISK: RelationshipSpec(
            source_types=frozenset({EntityType.COMPANY}),
            target_types=frozenset({EntityType.RISK}),
            symmetric=False,
            description="A company is exposed to a risk.",
        ),
        RelationshipType.EVIDENCE_ENTITY: RelationshipSpec(
            source_types=frozenset({EntityType.EVIDENCE}),
            target_types=BUSINESS_ENTITY_TYPES,
            symmetric=False,
            description="An evidence item supports an entity.",
        ),
    }
)


@dataclass(frozen=True)
class ValidationMarker:
    """How a graph fact fared against evidence validation.

    ``status`` reuses the
    :class:`~predictron_engine.research.validation_models.ValidationStatus`
    vocabulary so memory never invents a parallel grading scale, and
    ``confidence`` reuses
    :class:`~predictron_engine.research.validation_models.EvidenceConfidence`.
    The default marker is ``SKIPPED`` at zero confidence, which is exactly
    what an unvalidated fact should look like.
    """

    status: ValidationStatus = ValidationStatus.SKIPPED
    confidence: EvidenceConfidence = EvidenceConfidence.UNKNOWN
    score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.status, ValidationStatus):
            raise InvalidGraphModelError(
                "status must be a ValidationStatus for ValidationMarker"
            )
        if not isinstance(self.confidence, EvidenceConfidence):
            raise InvalidGraphModelError(
                "confidence must be an EvidenceConfidence for ValidationMarker"
            )
        if not _in_unit_range(self.score):
            raise InvalidGraphModelError(
                "score must be within [0, 1] for ValidationMarker"
            )

    @classmethod
    def from_status(
        cls,
        status: ValidationStatus,
        score: float,
    ) -> ValidationMarker:
        """Build a marker whose band is derived from ``score``."""
        return cls(
            status=status,
            confidence=EvidenceConfidence.from_score(score),
            score=score,
        )

    @property
    def is_validated(self) -> bool:
        """Return whether the fact was actually validated."""
        return self.status is not ValidationStatus.SKIPPED

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence.value,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ValidationMarker:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            status=_require_enum(ValidationStatus, data, "status"),
            confidence=_require_enum(EvidenceConfidence, data, "confidence"),
            score=_require_float(data, "score"),
        )


@dataclass(frozen=True)
class Provenance:
    """Where a graph fact came from.

    Provenance is a union of every input that touched the fact, never a
    replacement: adding a second supporting source extends
    :attr:`evidence_ids` and never drops the first one.  The identifiers
    inside are reused verbatim from the earlier sprints — evidence
    identifiers, ``EvidenceReference`` source identifiers and URLs, and
    ``ResearchReport`` identifiers — so a memory reader can walk straight
    back to the artifact that produced a fact.
    """

    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()
    report_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "evidence_ids",
            "source_ids",
            "urls",
            "titles",
            "report_ids",
        ):
            object.__setattr__(
                self, name, _dedupe(getattr(self, name), f"Provenance.{name}")
            )

    @property
    def is_empty(self) -> bool:
        """Return whether no provenance at all was recorded."""
        return not (
            self.evidence_ids
            or self.source_ids
            or self.urls
            or self.titles
            or self.report_ids
        )

    def merged_with(self, other: Provenance) -> Provenance:
        """Return the union of ``self`` and ``other``, preserving order."""
        if not isinstance(other, Provenance):
            raise InvalidGraphModelError("merged_with expects a Provenance")
        return Provenance(
            evidence_ids=self.evidence_ids + other.evidence_ids,
            source_ids=self.source_ids + other.source_ids,
            urls=self.urls + other.urls,
            titles=self.titles + other.titles,
            report_ids=self.report_ids + other.report_ids,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "evidence_ids": list(self.evidence_ids),
            "source_ids": list(self.source_ids),
            "urls": list(self.urls),
            "titles": list(self.titles),
            "report_ids": list(self.report_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Provenance:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
            source_ids=tuple(_require_str_list(data, "source_ids")),
            urls=tuple(_require_str_list(data, "urls")),
            titles=tuple(_require_str_list(data, "titles")),
            report_ids=tuple(_require_str_list(data, "report_ids")),
        )


@dataclass(frozen=True)
class GraphEntity:
    """One node in the research knowledge graph.

    Parameters
    ----------
    entity_id:
        Stable, content-derived identifier from
        :func:`~predictron_engine.research.linking.entity_id`.
    entity_type:
        Which kind of node this is.
    name:
        The name exactly as the research stated it.
    canonical_name:
        The normalized name from
        :func:`~predictron_engine.research.linking.normalize_name`.
    aliases:
        Every observed name variant, in first-seen order.
    attributes:
        Key-sorted scalar attributes, as ``(key, value)`` pairs.
    provenance:
        Union of every input that produced or supported this node.
    confidence:
        Deterministic confidence in ``[0, 1]`` for the node as a whole.
    last_updated:
        Caller-supplied, caller-deterministic update marker (a report
        identifier, a date, or an explicit revision label).
    validation:
        The validation status this node's evidence produced.
    topic_id:
        The research topic this node was extracted for, when known.
    record_id:
        The identifier of the memory record that last touched this node.
    """

    entity_id: str
    entity_type: EntityType
    name: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    attributes: tuple[tuple[str, AttributeValue], ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.0
    last_updated: str = ""
    validation: ValidationMarker = field(default_factory=ValidationMarker)
    topic_id: str = ""
    record_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise InvalidGraphModelError("entity_id must not be empty")
        if not isinstance(self.entity_type, EntityType):
            raise InvalidGraphModelError(
                "entity_type must be an EntityType for GraphEntity"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidGraphModelError("name must not be empty")
        if not isinstance(self.canonical_name, str) or not self.canonical_name.strip():
            raise InvalidGraphModelError("canonical_name must not be empty")
        if not isinstance(self.provenance, Provenance):
            raise InvalidGraphModelError(
                "provenance must be a Provenance for GraphEntity"
            )
        if not isinstance(self.validation, ValidationMarker):
            raise InvalidGraphModelError(
                "validation must be a ValidationMarker for GraphEntity"
            )
        if not _in_unit_range(self.confidence):
            raise InvalidGraphModelError(
                "confidence must be within [0, 1] for GraphEntity"
            )
        for name in ("last_updated", "topic_id", "record_id"):
            if not isinstance(getattr(self, name), str):
                raise InvalidGraphModelError(
                    f"{name} must be a string for GraphEntity"
                )
        object.__setattr__(self, "entity_id", self.entity_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self, "canonical_name", self.canonical_name.strip()
        )
        object.__setattr__(self, "last_updated", self.last_updated.strip())
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "record_id", self.record_id.strip())
        object.__setattr__(
            self,
            "aliases",
            _dedupe(self.aliases, "GraphEntity.aliases", strip=True),
        )
        object.__setattr__(
            self,
            "attributes",
            sorted_attributes(self.attributes, "GraphEntity.attributes"),
        )

    # -- Views -------------------------------------------------------

    @property
    def is_evidence(self) -> bool:
        """Return whether this node anchors a piece of evidence."""
        return self.entity_type is EntityType.EVIDENCE

    @property
    def names(self) -> tuple[str, ...]:
        """Return the canonical name followed by every alias variant."""
        return (self.name, *self.aliases)

    def attribute(self, key: str, default: AttributeValue | None = None) -> AttributeValue | None:
        """Return the attribute stored under ``key`` or ``default``."""
        for stored_key, value in self.attributes:
            if stored_key == key:
                return value
        return default

    def has_alias(self, name: str) -> bool:
        """Return whether ``name`` is a recorded variant of this node."""
        return name.strip() in self.aliases

    def attribute_map(self) -> dict[str, AttributeValue]:
        """Return the attributes as a plain dictionary."""
        return dict(self.attributes)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return the evidence identifiers supporting this node."""
        return self.provenance.evidence_ids

    def supports_relationship(
        self, relationship_type: RelationshipType
    ) -> bool:
        """Return whether this node may be the source of ``relationship_type``."""
        return self.entity_type in RELATIONSHIP_SPECS[relationship_type].source_types

    def accepts_relationship(
        self, relationship_type: RelationshipType
    ) -> bool:
        """Return whether this node may be the target of ``relationship_type``."""
        return self.entity_type in RELATIONSHIP_SPECS[relationship_type].target_types

    # -- Append-only updates -----------------------------------------

    def with_updates(
        self,
        *,
        provenance: Provenance | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        confidence: float | None = None,
        last_updated: str | None = None,
        validation: ValidationMarker | None = None,
        aliases: tuple[str, ...] = (),
        record_id: str = "",
    ) -> GraphEntity:
        """Return a *new* node carrying the union of the supplied updates.

        Nothing is mutated: attributes merge with the new values winning,
        provenance unions, and aliases extend.  This is the only way a node
        changes, which is what keeps the graph's node history reconstructible
        from its snapshots.
        """
        merged_attributes = dict(self.attributes)
        if attributes:
            merged_attributes.update(attributes)
        marker = (
            validation
            if validation is not None and validation.is_validated
            else self.validation
        )
        return GraphEntity(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            name=self.name,
            canonical_name=self.canonical_name,
            aliases=_dedupe(
                self.aliases
                + _dedupe(
                    # A node's own name is not one of its aliases: re-observing
                    # the same name must not read as a newly learned one.
                    tuple(alias for alias in aliases if alias.strip() != self.name),
                    "aliases",
                    strip=True,
                ),
                "aliases",
                strip=True,
            ),
            attributes=tuple(sorted(merged_attributes.items())),
            provenance=(
                self.provenance
                if provenance is None
                else self.provenance.merged_with(provenance)
            ),
            confidence=self.confidence if confidence is None else confidence,
            last_updated=(
                self.last_updated if last_updated is None else last_updated
            ),
            validation=marker,
            topic_id=self.topic_id,
            record_id=record_id or self.record_id,
        )

    def snapshot(self, version: int, recorded_at: str) -> EntitySnapshot:
        """Return the immutable :class:`EntitySnapshot` of this node.

        The snapshot carries the node's aliases so that the set of names an
        entity was known by at a given version is recoverable from history
        alone -- learning a new name for a known fact is itself a change in
        what memory knows.
        """
        return EntitySnapshot(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            version=version,
            recorded_at=recorded_at,
            name=self.name,
            attributes=self.attributes,
            provenance=self.provenance,
            confidence=self.confidence,
            validation=self.validation,
            topic_id=self.topic_id,
            aliases=self.aliases,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "name": self.name,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "attributes": {key: value for key, value in self.attributes},
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "validation": self.validation.to_dict(),
            "topic_id": self.topic_id,
            "record_id": self.record_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GraphEntity:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            entity_id=_require_str(data, "entity_id"),
            entity_type=_require_enum(EntityType, data, "entity_type"),
            name=_require_str(data, "name"),
            canonical_name=_require_str(data, "canonical_name"),
            aliases=tuple(_require_str_list(data, "aliases")),
            attributes=_require_attribute_pairs(data, "attributes"),
            provenance=Provenance.from_dict(
                _require_dict(data, "provenance")
            ),
            confidence=_require_float(data, "confidence"),
            last_updated=_require_str(data, "last_updated", default=""),
            validation=ValidationMarker.from_dict(
                _require_dict(data, "validation")
            ),
            topic_id=_require_str(data, "topic_id", default=""),
            record_id=_require_str(data, "record_id", default=""),
        )


@dataclass(frozen=True)
class GraphRelationship:
    """One strongly typed, evidence-backed edge in the graph.

    Parameters
    ----------
    source_id:
        Entity identifier at the source end.
    target_id:
        Entity identifier at the target end.
    relationship_type:
        Which relationship this is.
    confidence:
        Deterministic confidence in ``[0, 1]`` that the relationship holds.
    provenance:
        Where the relationship came from.
    timestamp:
        Caller-supplied, caller-deterministic marker.  The graph never
        reads a clock; ordering comes from :attr:`MemoryIndex.sequence`.
    evidence_ids:
        Evidence identifiers supporting this edge; at least one is required.
    attributes:
        Key-sorted scalar attributes, e.g. a funding ``amount`` or a
        patent ``number``.
    """

    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float = 0.0
    provenance: Provenance = field(default_factory=Provenance)
    timestamp: str = ""
    evidence_ids: tuple[str, ...] = ()
    attributes: tuple[tuple[str, AttributeValue], ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_id", "target_id"):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise InvalidRelationshipError(f"{name} must not be empty")
        if self.source_id.strip() == self.target_id.strip():
            raise InvalidRelationshipError(
                "a relationship needs two distinct endpoints"
            )
        if not isinstance(self.relationship_type, RelationshipType):
            raise InvalidRelationshipError(
                "relationship_type must be a RelationshipType"
            )
        if not isinstance(self.provenance, Provenance):
            raise InvalidRelationshipError(
                "provenance must be a Provenance for GraphRelationship"
            )
        if not _in_unit_range(self.confidence):
            raise InvalidRelationshipError(
                "confidence must be within [0, 1] for GraphRelationship"
            )
        if not isinstance(self.timestamp, str):
            raise InvalidRelationshipError(
                "timestamp must be a string for GraphRelationship"
            )
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "target_id", self.target_id.strip())
        object.__setattr__(self, "timestamp", self.timestamp.strip())
        object.__setattr__(
            self,
            "evidence_ids",
            _dedupe(self.evidence_ids, "GraphRelationship.evidence_ids", strip=True),
        )
        if not self.evidence_ids:
            raise InvalidRelationshipError(
                "a relationship must be supported by at least one evidence id"
            )
        object.__setattr__(
            self,
            "attributes",
            sorted_attributes(self.attributes, "GraphRelationship.attributes"),
        )
        if not self.provenance.evidence_ids:
            object.__setattr__(
                self,
                "provenance",
                Provenance(
                    evidence_ids=self.evidence_ids,
                    source_ids=self.provenance.source_ids,
                    urls=self.provenance.urls,
                    titles=self.provenance.titles,
                    report_ids=self.provenance.report_ids,
                ),
            )

    # -- Views -------------------------------------------------------

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the canonical store key, with symmetric ends ordered."""
        spec = RELATIONSHIP_SPECS[self.relationship_type]
        if spec.symmetric and self.target_id < self.source_id:
            return (self.target_id, self.relationship_type.value, self.source_id)
        return (self.source_id, self.relationship_type.value, self.target_id)

    @property
    def is_symmetric(self) -> bool:
        """Return whether direction is semantically irrelevant."""
        return RELATIONSHIP_SPECS[self.relationship_type].symmetric

    def attribute(self, key: str, default: AttributeValue | None = None) -> AttributeValue | None:
        """Return the attribute stored under ``key`` or ``default``."""
        for stored_key, value in self.attributes:
            if stored_key == key:
                return value
        return default

    def attribute_map(self) -> dict[str, AttributeValue]:
        """Return the attributes as a plain dictionary."""
        return dict(self.attributes)

    def check_endpoints(
        self,
        source_type: EntityType,
        target_type: EntityType,
    ) -> None:
        """Raise unless ``source_type`` -> ``target_type`` is permitted.

        The graph calls this after it knows both endpoints; the check lives
        on the value object so it applies to hand-built relationships too.
        """
        if not isinstance(source_type, EntityType) or not isinstance(
            target_type, EntityType
        ):
            raise InvalidRelationshipError(
                "endpoint types must be EntityType instances"
            )
        spec = RELATIONSHIP_SPECS[self.relationship_type]
        if not spec.supports(source_type, target_type):
            raise InvalidRelationshipError(
                f"{self.relationship_type.value} does not allow "
                f"{source_type.value} -> {target_type.value}"
            )

    def with_support(
        self,
        *,
        evidence_ids: tuple[str, ...] = (),
        provenance: Provenance | None = None,
        confidence: float | None = None,
        timestamp: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> GraphRelationship:
        """Return a *new* edge strengthened by additional support.

        The union of the evidence identifiers becomes the new support set
        and the new confidence is the deterministic maximum of the two, so
        re-observing an existing relationship can only ever strengthen it.
        """
        merged_attributes = dict(self.attributes)
        if attributes:
            merged_attributes.update(attributes)
        support = _dedupe(
            self.evidence_ids + _dedupe(evidence_ids, "evidence_ids", strip=True),
            "evidence_ids",
            strip=True,
        )
        merged_provenance = (
            self.provenance
            if provenance is None
            else self.provenance.merged_with(provenance)
        )
        return GraphRelationship(
            source_id=self.source_id,
            target_id=self.target_id,
            relationship_type=self.relationship_type,
            confidence=(
                self.confidence
                if confidence is None
                else max(self.confidence, confidence)
            ),
            provenance=Provenance(
                evidence_ids=support,
                source_ids=merged_provenance.source_ids,
                urls=merged_provenance.urls,
                titles=merged_provenance.titles,
                report_ids=merged_provenance.report_ids,
            ),
            timestamp=(
                self.timestamp if timestamp is None else timestamp
            ),
            evidence_ids=support,
            attributes=tuple(sorted(merged_attributes.items())),
        )

    def relationship_id(self) -> str:
        """Return the stable, content-derived identifier of this edge."""
        return relationship_id(
            self.source_id, self.relationship_type, self.target_id
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
            "timestamp": self.timestamp,
            "evidence_ids": list(self.evidence_ids),
            "attributes": {key: value for key, value in self.attributes},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GraphRelationship:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            source_id=_require_str(data, "source_id"),
            target_id=_require_str(data, "target_id"),
            relationship_type=_require_enum(
                RelationshipType, data, "relationship_type"
            ),
            confidence=_require_float(data, "confidence"),
            provenance=Provenance.from_dict(
                _require_dict(data, "provenance")
            ),
            timestamp=_require_str(data, "timestamp", default=""),
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
            attributes=_require_attribute_pairs(data, "attributes"),
        )


@dataclass(frozen=True)
class EntitySnapshot:
    """The immutable "previous value" of a node at one version.

    Research memory never overwrites a node: it records a snapshot of the
    prior state, appends a :class:`~predictron_engine.research.history.HistoryChange`
    describing the delta, and stores the new node.  The snapshots are what
    make the full value history of a node reconstructible after the fact.
    """

    entity_id: str
    entity_type: EntityType
    version: int
    recorded_at: str
    name: str
    attributes: tuple[tuple[str, AttributeValue], ...]
    provenance: Provenance
    confidence: float
    validation: ValidationMarker
    topic_id: str = ""
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise InvalidGraphModelError("entity_id must not be empty")
        if not isinstance(self.entity_type, EntityType):
            raise InvalidGraphModelError(
                "entity_type must be an EntityType for EntitySnapshot"
            )
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise InvalidGraphModelError(
                "version must be an int for EntitySnapshot"
            )
        if self.version < 1:
            raise InvalidGraphModelError(
                "version must be at least 1 for EntitySnapshot"
            )
        if not isinstance(self.recorded_at, str):
            raise InvalidGraphModelError(
                "recorded_at must be a string for EntitySnapshot"
            )
        if not isinstance(self.provenance, Provenance):
            raise InvalidGraphModelError(
                "provenance must be a Provenance for EntitySnapshot"
            )
        if not isinstance(self.validation, ValidationMarker):
            raise InvalidGraphModelError(
                "validation must be a ValidationMarker for EntitySnapshot"
            )
        if not _in_unit_range(self.confidence):
            raise InvalidGraphModelError(
                "confidence must be within [0, 1] for EntitySnapshot"
            )
        if not isinstance(self.aliases, tuple) or not all(
            isinstance(alias, str) and alias.strip() for alias in self.aliases
        ):
            raise InvalidGraphModelError(
                "aliases must be non-empty strings for EntitySnapshot"
            )
        object.__setattr__(self, "entity_id", self.entity_id.strip())
        object.__setattr__(
            self,
            "attributes",
            sorted_attributes(self.attributes, "EntitySnapshot.attributes"),
        )

    def attribute_map(self) -> dict[str, AttributeValue]:
        """Return the snapshot attributes as a plain dictionary."""
        return dict(self.attributes)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "version": self.version,
            "recorded_at": self.recorded_at,
            "name": self.name,
            "attributes": {key: value for key, value in self.attributes},
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "validation": self.validation.to_dict(),
            "topic_id": self.topic_id,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EntitySnapshot:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            entity_id=_require_str(data, "entity_id"),
            entity_type=_require_enum(EntityType, data, "entity_type"),
            version=_require_int(data, "version"),
            recorded_at=_require_str(data, "recorded_at", default=""),
            name=_require_str(data, "name"),
            attributes=_require_attribute_pairs(data, "attributes"),
            provenance=Provenance.from_dict(
                _require_dict(data, "provenance")
            ),
            confidence=_require_float(data, "confidence"),
            validation=ValidationMarker.from_dict(
                _require_dict(data, "validation")
            ),
            topic_id=_require_str(data, "topic_id", default=""),
            aliases=tuple(_require_str_list(data, "aliases")),
        )


@dataclass(frozen=True)
class GraphStats:
    """Deterministic structural counters for a whole graph.

    Every per-type count is a key-sorted tuple rather than a dictionary so
    the value is hashable, order-stable, and safe to put straight into a
    content hash.
    """

    entity_count: int
    relationship_count: int
    evidence_node_count: int
    business_entity_count: int
    validated_entity_count: int
    referenced_evidence_count: int
    entity_counts: tuple[tuple[str, int], ...] = ()
    relationship_counts: tuple[tuple[str, int], ...] = ()
    topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "entity_count",
            "relationship_count",
            "evidence_node_count",
            "business_entity_count",
            "validated_entity_count",
            "referenced_evidence_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidGraphModelError(
                    f"{name} must be an int for GraphStats"
                )
            if value < 0:
                raise InvalidGraphModelError(
                    f"{name} must be non-negative for GraphStats"
                )
        object.__setattr__(
            self, "entity_counts", _sorted_counts(self.entity_counts, "entity_counts")
        )
        object.__setattr__(
            self,
            "relationship_counts",
            _sorted_counts(self.relationship_counts, "relationship_counts"),
        )
        topics = tuple(dict.fromkeys(self.topics))
        for topic in topics:
            if not isinstance(topic, str) or not topic.strip():
                raise InvalidGraphModelError(
                    "topics entries must be non-empty strings for GraphStats"
                )
        object.__setattr__(self, "topics", topics)

    def count_for_entity_type(self, entity_type: EntityType) -> int:
        """Return how many nodes have ``entity_type``."""
        for key, value in self.entity_counts:
            if key == entity_type.value:
                return value
        return 0

    def count_for_relationship(
        self, relationship_type: RelationshipType
    ) -> int:
        """Return how many edges have ``relationship_type``."""
        for key, value in self.relationship_counts:
            if key == relationship_type.value:
                return value
        return 0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "evidence_node_count": self.evidence_node_count,
            "business_entity_count": self.business_entity_count,
            "validated_entity_count": self.validated_entity_count,
            "referenced_evidence_count": self.referenced_evidence_count,
            "entity_counts": {key: value for key, value in self.entity_counts},
            "relationship_counts": {
                key: value for key, value in self.relationship_counts
            },
            "topics": list(self.topics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GraphStats:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            entity_count=_require_int(data, "entity_count"),
            relationship_count=_require_int(data, "relationship_count"),
            evidence_node_count=_require_int(data, "evidence_node_count"),
            business_entity_count=_require_int(data, "business_entity_count"),
            validated_entity_count=_require_int(data, "validated_entity_count"),
            referenced_evidence_count=_require_int(
                data, "referenced_evidence_count"
            ),
            entity_counts=_require_counts(data, "entity_counts"),
            relationship_counts=_require_counts(data, "relationship_counts"),
            topics=tuple(_require_str_list(data, "topics")),
        )


@dataclass(frozen=True)
class GraphValidation:
    """The result of checking a graph for dangling or ungrounded elements.

    ``valid`` is true only when every one of the three issue tuples is
    empty.  Issues are sorted so two runs over the same graph produce the
    same report.
    """

    dangling: tuple[str, ...] = ()
    missing_provenance: tuple[str, ...] = ()
    invalid_relationships: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the graph is fully grounded."""
        return not (self.dangling or self.missing_provenance or self.invalid_relationships)

    @property
    def issue_count(self) -> int:
        """Return how many issues were found in total."""
        return (
            len(self.dangling)
            + len(self.missing_provenance)
            + len(self.invalid_relationships)
        )

    def issues(self) -> tuple[str, ...]:
        """Return every issue, in deterministic category order."""
        return (
            *self.dangling,
            *self.missing_provenance,
            *self.invalid_relationships,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "valid": self.valid,
            "dangling": list(self.dangling),
            "missing_provenance": list(self.missing_provenance),
            "invalid_relationships": list(self.invalid_relationships),
            "issue_count": self.issue_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> GraphValidation:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            dangling=tuple(_require_str_list(data, "dangling")),
            missing_provenance=tuple(
                _require_str_list(data, "missing_provenance")
            ),
            invalid_relationships=tuple(
                _require_str_list(data, "invalid_relationships")
            ),
        )


def relationship_id(
    source_id: str,
    relationship_type: RelationshipType,
    target_id: str,
) -> str:
    """Return the stable content-derived identifier of one relationship.

    The digest covers the relationship type and both endpoints in canonical
    order, so re-observing the same edge always yields the same identifier
    regardless of how many times it is added.
    """
    if not isinstance(relationship_type, RelationshipType):
        raise InvalidRelationshipError(
            "relationship_id expects a RelationshipType"
        )
    if relationship_type in _SYMMETRIC_RELATIONSHIPS and target_id < source_id:
        source_id, target_id = target_id, source_id
    payload = f"{relationship_type.value}\x1f{source_id}\x1f{target_id}"
    return RELATIONSHIP_ID_PREFIX + text_digest(
        payload, length=RESEARCH_DIGEST_LENGTH
    )


_SYMMETRIC_RELATIONSHIPS: frozenset[RelationshipType] = frozenset(
    relationship_type
    for relationship_type, spec in RELATIONSHIP_SPECS.items()
    if spec.symmetric
)


def sorted_attributes(
    attributes: tuple[tuple[str, AttributeValue], ...],
    label: str = "attributes",
) -> tuple[tuple[str, AttributeValue], ...]:
    """Return ``attributes`` as a validated, key-sorted tuple.

    Sorting is what makes two nodes with the same facts but different
    attribute insertion order produce the same content hash.
    """
    pairs: dict[str, AttributeValue] = {}
    for entry in attributes:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise InvalidGraphModelError(
                f"{label} entries must be (key, value) pairs"
            )
        key, value = entry
        if not isinstance(key, str) or not key.strip():
            raise InvalidGraphModelError(
                f"{label} keys must be non-empty strings"
            )
        if key in pairs:
            raise InvalidGraphModelError(
                f"duplicate attribute key for {label}: {key!r}"
            )
        if not isinstance(value, str | int | float | bool):
            raise InvalidGraphModelError(
                f"attribute {key!r} must be a JSON scalar"
            )
        pairs[key] = value
    return tuple(sorted(pairs.items()))


def _sorted_counts(
    counts: tuple[tuple[str, int], ...],
    label: str,
) -> tuple[tuple[str, int], ...]:
    """Validate and key-sort a per-type counter tuple."""
    pairs: dict[str, int] = {}
    for entry in counts:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise InvalidGraphModelError(f"{label} entries must be pairs")
        key, value = entry
        if not isinstance(key, str) or not key.strip():
            raise InvalidGraphModelError(
                f"{label} keys must be non-empty strings"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidGraphModelError(
                f"{label} values must be non-negative ints"
            )
        if key in pairs:
            raise InvalidGraphModelError(f"duplicate {label} key: {key!r}")
        pairs[key] = value
    return tuple(sorted(pairs.items()))


def _require_attribute_pairs(
    data: Mapping[str, object],
    key: str,
) -> tuple[tuple[str, AttributeValue], ...]:
    """Read a serialized attribute mapping back into a pair tuple."""
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise InvalidGraphModelError(f"expected dict for '{key}'")
    pairs: list[tuple[str, AttributeValue]] = []
    for attribute_key, attribute_value in value.items():
        if not isinstance(attribute_key, str):
            raise InvalidGraphModelError(f"expected string keys for '{key}'")
        if not isinstance(attribute_value, str | int | float | bool):
            raise InvalidGraphModelError(
                f"attribute {attribute_key!r} must be a JSON scalar"
            )
        pairs.append((attribute_key, attribute_value))
    return tuple(pairs)


def _require_counts(
    data: Mapping[str, object],
    key: str,
) -> tuple[tuple[str, int], ...]:
    """Read a serialized counter mapping back into a pair tuple."""
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise InvalidGraphModelError(f"expected dict for '{key}'")
    pairs: list[tuple[str, int]] = []
    for count_key, count_value in value.items():
        if not isinstance(count_key, str):
            raise InvalidGraphModelError(f"expected string keys for '{key}'")
        if isinstance(count_value, bool) or not isinstance(count_value, int):
            raise InvalidGraphModelError(f"expected int values for '{key}'")
        pairs.append((count_key, count_value))
    return tuple(pairs)


def _dedupe(
    values: tuple[str, ...],
    label: str,
    *,
    strip: bool = False,
) -> tuple[str, ...]:
    """Validate a string tuple and drop repeats, keeping first-seen order."""
    if not isinstance(values, tuple):
        raise InvalidGraphModelError(f"{label} must be a tuple of strings")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise InvalidGraphModelError(f"{label} entries must be strings")
        candidate = value.strip() if strip else value
        if not candidate:
            raise InvalidGraphModelError(
                f"{label} entries must not be empty"
            )
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _in_unit_range(value: float) -> bool:
    """Whether a float lies in the closed unit interval."""
    return not isinstance(value, bool) and isinstance(value, int | float) and (
        0.0 <= float(value) <= 1.0
    )


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


def _require_enum(
    enum_type: type[EnumT],
    data: Mapping[str, object],
    key: str,
) -> EnumT:
    """Read and validate a closed vocabulary field.

    A payload that names a value outside the vocabulary is malformed input,
    not a programming error, so it is reported as
    :class:`InvalidGraphModelError` rather than leaking the ``ValueError``
    the ``Enum`` lookup would otherwise raise.
    """
    raw = _require_str(data, key)
    try:
        return enum_type(raw)
    except ValueError as error:
        raise InvalidGraphModelError(
            f"'{raw}' is not a valid {enum_type.__name__} for '{key}'"
        ) from error


def _require_float(data: Mapping[str, object], key: str) -> float:
    """Read and validate a float field from a serialized dictionary."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidGraphModelError(f"expected number for '{key}'")
    return float(value)


def _require_dict(
    data: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    """Read and validate a nested dictionary field."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise InvalidGraphModelError(f"expected dict for '{key}'")
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


__all__ = [
    "BUSINESS_ENTITY_TYPES",
    "GRAPH_SCHEMA_VERSION",
    "RELATIONSHIP_ID_PREFIX",
    "RELATIONSHIP_ID_SEPARATOR",
    "RELATIONSHIP_SPECS",
    "AttributeValue",
    "EntitySnapshot",
    "EntityType",
    "GraphEntity",
    "GraphRelationship",
    "GraphStats",
    "GraphValidation",
    "Provenance",
    "RelationshipSpec",
    "RelationshipType",
    "ValidationMarker",
    "relationship_id",
    "sorted_attributes",
]
