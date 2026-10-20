"""WI-6 — the grounded-research / analysis integration seam.

This module is the **only** place the autonomous research subsystem and
the deterministic analysis pipeline meet.  It exists so that research
output can flow into the engine's evidence plane through a clean, opt-in
boundary, without either package learning about the other:

* :mod:`predictron_engine.research` stays pure — it is never imported from
  and never imports the engine, ``app``, or this module's consumers.
* :class:`ResearchAnalysisIntegration` is injected into
  :class:`~predictron_engine.engine.PredictronEngine` **only** when
  research integration is explicitly requested.  With no boundary
  injected the pipeline is byte-identical: the same outputs, ids, hashes,
  reports, and benchmarks.

What the boundary does
----------------------
::

    ResearchEvidenceSource  ->  ResearchEvidenceProjector  ->  EvidenceSet

* A :class:`ResearchEvidenceSource` yields research :class:`Evidence`
  (either by executing grounded research through an injected executor, or
  by accepting an already-produced research result).
* :class:`ResearchEvidenceProjector` projects each :class:`Evidence` onto
  the engine's :class:`~predictron_engine.evidence.evidence_models.EvidenceItem`.
* :class:`ResearchAnalysisIntegration` appends the projected items to the
  gathered :class:`~predictron_engine.evidence.evidence_models.EvidenceSet`
  before the reasoning layer consumes it.

Invariants (never violated by the projection)
---------------------------------------------
* **Trust is imported, never recomputed.**  A research ``TrustScore``
  arrives as a value object and is serialised by
  :func:`~predictron_engine.research.provenance_bridge.trust_score_payload`,
  which dumps the object it was handed.  No factor is reweighted and no
  band is assigned here.
* **Evidence ids are never regenerated.**  The research ``evidence_id``,
  its ``evidence_id_version``, ``document_ids``, and ``citation_ids`` are
  copied verbatim into the projected provenance record — the same values
  the research subsystem minted.
* **Provenance is never rewritten.**  The imported
  :class:`~predictron_engine.research.models.EvidenceReference` fields are
  carried through unchanged alongside the identity above.

The projected item's ``provenance_record`` is a JSON document that begins
with the ``ProvenanceRecord``-compatible keys the existing reasoning
consumers already read (``document_id``, ``trust_score``) and then adds a
clearly namespaced ``research_*`` identity block.  Consuming code that
predates this module keeps working, because it reads the keys it always
read and ignores the rest.

Deliberately absent
-------------------
No persistence, no claim re-mapping, no API surface, no observability
sink, no evidence-model change, no live provider, and no networking.
Those are later work items; this module is the seam only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from predictron_engine.canonical import canonical_json
from predictron_engine.evidence.evidence_models import (
    EvidenceItem,
    EvidenceSet,
)
from predictron_engine.research.exceptions import InvalidEvidence
from predictron_engine.research.models import Evidence, PlannerInput
from predictron_engine.research.provenance_bridge import trust_score_payload

if TYPE_CHECKING:
    from predictron_engine.research.collector import EvidenceCollector
    from predictron_engine.research.execution_report import (
        ResearchExecutionResult,
    )
    from predictron_engine.research.grounding import (
        ClaimProjector,
        DocumentSource,
        GroundingPolicy,
    )
    from predictron_engine.research.ports import ResearchExecutorPort

logger = logging.getLogger(__name__)

#: Discriminator written into every projected provenance record.  It
#: marks the record as a research projection rather than a document
#: ``ProvenanceRecord``, so a future reader can tell the two apart
#: without having to guess from field presence.
RESEARCH_PROVENANCE_SCHEMA = "research_evidence_projection_v1"


@runtime_checkable
class ResearchEvidenceSource(Protocol):
    """A source of research :class:`Evidence` for one company.

    Implementations are side-effect free from the boundary's point of
    view: they return already-collected evidence and never score, rank,
    or mutate it.  The two shipped sources are
    :class:`ResearchResultSource` (accept an existing result) and
    :class:`OrchestratedResearchSource` (execute a grounded pass).
    """

    def evidence(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        """Return the research evidence available for ``company_name``."""
        ...


class ResearchResultSource:
    """A :class:`ResearchEvidenceSource` over an existing research result.

    This is the "accept existing research outputs" half of the seam: it
    performs no execution and no I/O, it simply exposes the evidence the
    supplied :class:`ResearchExecutionResult` already produced.
    """

    def __init__(self, result: ResearchExecutionResult) -> None:
        self._result = result

    @property
    def result(self) -> ResearchExecutionResult:
        """Return the research execution result this source wraps."""
        return self._result

    def evidence(
        self,
        *,
        company_name: str = "",
        website: str = "",
    ) -> tuple[Evidence, ...]:
        """Return the wrapped result's collected evidence, in order."""
        return _as_evidence(self._result.evidence())


class OrchestratedResearchSource:
    """A :class:`ResearchEvidenceSource` that executes grounded research.

    The executor is injected as a
    :class:`~predictron_engine.research.ports.ResearchExecutorPort` — most
    commonly a :class:`~predictron_engine.research.orchestrator.ResearchOrchestrator`
    already wired to a grounded collector registry (see
    :func:`grounded_research_source`).  This class owns no planning or
    collection logic; it builds a :class:`PlannerInput` and delegates.
    """

    def __init__(self, executor: ResearchExecutorPort) -> None:
        self._executor = executor

    @property
    def executor(self) -> ResearchExecutorPort:
        """Return the executor this source drives."""
        return self._executor

    def evidence(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        """Run one grounded research pass and return its evidence."""
        result = self._executor.execute(
            PlannerInput(company_name=company_name, website=website)
        )
        return _as_evidence(result.evidence())


class ResearchEvidenceProjector:
    """Projects research :class:`Evidence` onto engine :class:`EvidenceItem` s.

    The projection is total and lossless with respect to the research
    identity: every id, version, reference field, and imported trust
    value is carried into the projected provenance record.  Only the
    engine-visible surfaces (domain, category, statement, source,
    relevance) are lifted to first-class ``EvidenceItem`` fields.
    """

    def project(self, evidence: Evidence) -> EvidenceItem:
        """Return the :class:`EvidenceItem` projection of ``evidence``.

        Raises
        ------
        InvalidEvidence:
            When ``evidence`` is not a research :class:`Evidence`.
        """
        if not isinstance(evidence, Evidence):
            raise InvalidEvidence(
                "project expects a research Evidence instance"
            )
        return EvidenceItem(
            domain=evidence.topic_id,
            category=evidence.category,
            statement=evidence.claim,
            source=evidence.collector_id,
            relevance_score=evidence.confidence,
            provenance_record=canonical_json(_provenance_payload(evidence)),
        )

    def project_all(
        self,
        evidence: Sequence[Evidence],
    ) -> tuple[EvidenceItem, ...]:
        """Project ``evidence`` and order the result deterministically.

        Items are ordered by ``(topic_id, category, statement,
        evidence_id)`` so any insertion order yields the same projection —
        the same research output always produces the same engine evidence.
        """
        projected = [self.project(item) for item in evidence]
        projected.sort(
            key=lambda item: (
                item.domain,
                item.category,
                item.statement,
                _projected_evidence_id(item),
            )
        )
        return tuple(projected)


class ResearchAnalysisIntegration:
    """The opt-in boundary between research output and the engine.

    An instance is injected into
    :class:`~predictron_engine.engine.PredictronEngine`; when absent, the
    engine never calls into this module and every default output is
    unchanged.  The boundary is a pure function of its source and the
    gathered evidence set: it performs no I/O of its own and never
    mutates its inputs.
    """

    def __init__(
        self,
        source: ResearchEvidenceSource,
        *,
        projector: ResearchEvidenceProjector | None = None,
    ) -> None:
        if not isinstance(source, ResearchEvidenceSource):
            raise TypeError("source must implement ResearchEvidenceSource")
        self._source = source
        self._projector = (
            projector if projector is not None else ResearchEvidenceProjector()
        )

    @property
    def source(self) -> ResearchEvidenceSource:
        """Return the research evidence source this boundary reads from."""
        return self._source

    @property
    def projector(self) -> ResearchEvidenceProjector:
        """Return the projector this boundary applies."""
        return self._projector

    def augment(
        self,
        evidence_set: EvidenceSet,
        *,
        company_name: str,
        website: str = "",
    ) -> EvidenceSet:
        """Return ``evidence_set`` with projected research evidence appended.

        The original items, ``provider_count``, and ``feature_coverage``
        are preserved exactly; the projected research items are appended
        in deterministic order.  Research integration is best-effort: any
        failure while executing or projecting leaves the evidence set
        untouched, mirroring the pipeline's rule that evidence collection
        must never break an analysis.

        Returns
        -------
        EvidenceSet
            A new set when research evidence was added; the original set
            object, unchanged, when nothing was added or on failure.
        """
        if not isinstance(evidence_set, EvidenceSet):
            raise TypeError("augment expects an EvidenceSet")
        try:
            evidence = self._source.evidence(
                company_name=company_name, website=website
            )
            projected = self._projector.project_all(evidence)
        except Exception:  # noqa: BLE001 - integration must never break analysis
            logger.warning(
                "Research integration failed; continuing without it",
                exc_info=True,
            )
            return evidence_set
        if not projected:
            return evidence_set
        return EvidenceSet(
            items=[*evidence_set.items, *projected],
            provider_count=evidence_set.provider_count,
            feature_coverage=evidence_set.feature_coverage,
        )


def grounded_research_source(
    source: DocumentSource,
    projector: ClaimProjector,
    *,
    policy: GroundingPolicy | None = None,
    fallback: EvidenceCollector | None = None,
) -> OrchestratedResearchSource:
    """Build an executor-backed source over a grounded collector registry.

    Wires the existing, reviewed pieces — :func:`grounded_collector_registry`,
    :class:`~predictron_engine.research.execution.EvidenceCollectionEngine`,
    and :class:`~predictron_engine.research.orchestrator.ResearchOrchestrator`
    — so callers get "execute grounded research" without assembling the
    pipeline themselves.  The supplied ``source`` is any
    :class:`~predictron_engine.research.grounding.DocumentSource`, so this
    works offline with
    :class:`~predictron_engine.research.grounding.ReplayDocumentSource`
    and, unchanged, with the existing live adapter.

    This adds no provider, no networking, and no I/O of its own: it only
    chooses which already-implemented components to compose.
    """
    from predictron_engine.research.execution import EvidenceCollectionEngine
    from predictron_engine.research.grounding import GroundingPolicy
    from predictron_engine.research.orchestrator import ResearchOrchestrator
    from predictron_engine.research.registry import grounded_collector_registry

    resolved_policy = policy if policy is not None else GroundingPolicy()
    registry = grounded_collector_registry(
        resolved_policy,
        source=source,
        projector=projector,
        fallback=fallback,
    )
    engine = EvidenceCollectionEngine(registry=registry)
    return OrchestratedResearchSource(
        ResearchOrchestrator(collection_engine=engine)
    )


def _as_evidence(items: Sequence[object]) -> tuple[Evidence, ...]:
    """Keep only the research :class:`Evidence` members of ``items``.

    ``ResearchExecutionResult.evidence()`` is typed as ``tuple[object, ...]``
    because the executor port is deliberately generic; the members are the
    untouched :class:`Evidence` objects.  Filtering here restores the
    precise type without asserting blindly.
    """
    return tuple(item for item in items if isinstance(item, Evidence))


def _provenance_payload(evidence: Evidence) -> dict[str, object]:
    """Build the projected provenance record for one evidence item.

    The first block mirrors the ``ProvenanceRecord`` keys existing
    reasoning consumers read; every value is imported, never derived.  The
    ``research_*`` block preserves the claim's canonical identity and
    versioning verbatim.
    """
    trust = evidence.trust_score
    return {
        # ProvenanceRecord-compatible keys (read by existing consumers).
        "document_id": _primary_document_id(evidence),
        "url": evidence.reference.url,
        "final_url": "",
        "document_type": evidence.reference.source_category,
        "source_provider": evidence.collector_id,
        "trust_level": (
            trust.source_trust_level.value if trust is not None else "unknown"
        ),
        "trust_score": float(trust.overall) if trust is not None else 0.0,
        "fetched_at": None,
        "is_official": False,
        "is_duplicate": False,
        "duplicate_of": None,
        "freshness_days": int(trust.freshness_days)
        if trust is not None
        else 0,
        # Research identity (additive): preserves versioning + canonical ids.
        "record_schema": RESEARCH_PROVENANCE_SCHEMA,
        "research_evidence_id": evidence.evidence_id,
        "research_evidence_id_version": evidence.evidence_id_version,
        "research_task_id": evidence.task_id,
        "research_topic_id": evidence.topic_id,
        "research_collector_id": evidence.collector_id,
        "research_category": evidence.category,
        "research_confidence": evidence.confidence,
        "research_source_category": evidence.reference.source_category,
        "research_source_identifier": evidence.reference.source_identifier,
        "research_source_url": evidence.reference.url,
        "research_document_ids": list(evidence.document_ids),
        "research_citation_ids": list(evidence.citation_ids),
        "research_trust_score": trust_score_payload(trust),
    }


def _primary_document_id(evidence: Evidence) -> str:
    """Return the first document id as a ProvenanceRecord ``document_id``.

    ``document_ids`` is already sorted and unique, so the choice is
    deterministic.  The full id set is preserved under
    ``research_document_ids``; this single value exists only so the
    existing single-document provenance readers keep functioning.
    """
    return evidence.document_ids[0] if evidence.document_ids else ""


def _projected_evidence_id(item: EvidenceItem) -> str:
    """Return the research evidence id carried by a projected item.

    Read back out of the provenance record purely to feed the
    deterministic ordering in :meth:`ResearchEvidenceProjector.project_all`;
    the id itself is never recomputed.
    """
    import json

    encoded = item.provenance_record
    if not encoded:
        return ""
    try:
        parsed = json.loads(encoded)
    except (TypeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get("research_evidence_id")
    return value if isinstance(value, str) else ""


__all__ = [
    "RESEARCH_PROVENANCE_SCHEMA",
    "OrchestratedResearchSource",
    "ResearchAnalysisIntegration",
    "ResearchEvidenceProjector",
    "ResearchEvidenceSource",
    "ResearchResultSource",
    "grounded_research_source",
]
