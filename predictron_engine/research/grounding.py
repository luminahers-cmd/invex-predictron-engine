"""Grounding stage for the research subsystem.

Grounding replaces a topic's *placeholder* evidence with evidence linked
to real documents produced by the existing evidence subsystem.  It is an
**opt-in** substitution: the default registry, the placeholder collectors,
and every default identifier are untouched, so the research baseline is
byte-identical unless grounding is explicitly enabled.

The layer is deliberately thin and delegating:

* :class:`DocumentSource` / :class:`ClaimProjector` are the two seams the
  stage is built from.  A source returns the documents available for a
  company; a projector turns those documents into candidate claims.
* :class:`ReplayDocumentSource` is an offline, deterministic source over a
  committed evidence corpus.  The live adapter
  (:class:`~predictron_engine.integrations.grounding.SyncEvidenceSource`)
  lives outside this package so the research layer stays pure.
* :class:`ExtractionClaimProjector` runs the **existing** extraction
  pipeline and projects populated ``ExtractedFeatures`` fields.  It
  defines no topic-to-field mapping and no extraction logic of its own.
* :class:`GroundedCollector` is the :class:`EvidenceCollector` that
  applies a :class:`GroundingPolicy` and builds version ``"2"``
  :class:`Evidence`.

Trust is never computed here.  A ``TrustScore`` is imported from the
evidence subsystem as a value object and carried through untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable
from urllib.parse import urlsplit

from predictron_engine.evidence.provenance import TrustScore
from predictron_engine.research.collector import (
    EvidenceCollector,
    enforce_supported,
)
from predictron_engine.research.exceptions import (
    InvalidCollectionInputError,
    InvalidEvidence,
)
from predictron_engine.research.models import (
    Evidence,
    EvidenceReference,
    ResearchTask,
)
from predictron_engine.research.provenance_bridge import (
    GROUNDED_EVIDENCE_ID_VERSION,
    citation_ids,
    document_ids_for,
    normalise_link_ids,
)
from predictron_engine.research.topics import all_topic_ids

if TYPE_CHECKING:
    from pathlib import Path as _Path

    from predictron_engine.evidence.models import EvidenceDocument
    from predictron_engine.extraction.composite import CompositeExtractor
    from predictron_engine.models.extracted_features import ExtractedFeatures


@runtime_checkable
class DocumentSource(Protocol):
    """A source of evidence documents for one company.

    Implementations must be side-effect free from the research layer's
    point of view: they return documents, they do not call back into the
    collector, and they never score or rank.
    """

    def documents(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[EvidenceDocument, ...]:
        """Return the documents available for ``company_name``."""
        ...


@runtime_checkable
class ClaimProjector(Protocol):
    """Turns documents into candidate claims for one research task.

    A projector never computes trust and never mutates its inputs; it
    reads documents only as the values the source produced.
    """

    def project(
        self,
        task: ResearchTask,
        documents: tuple[EvidenceDocument, ...],
        *,
        company_name: str,
    ) -> tuple[ProjectedClaim, ...]:
        """Return the candidate claims supported by ``documents``."""
        ...


@dataclass(frozen=True)
class ProjectedClaim:
    """One candidate claim produced by a :class:`ClaimProjector`.

    ``document_ids`` cites the documents that support the claim.  The
    collector derives the final :class:`EvidenceReference` from those
    documents, so a projector does not need to build provenance itself.
    """

    category: str
    claim: str
    confidence: float
    document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        category = self.category.strip()
        claim = self.claim.strip()
        if not category:
            raise InvalidEvidence("projected claim category must not be empty")
        if not claim:
            raise InvalidEvidence("projected claim text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidEvidence("projected claim confidence must be within [0, 1]")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "claim", claim)
        object.__setattr__(
            self,
            "document_ids",
            normalise_link_ids(self.document_ids, "document_ids"),
        )


class MissingDocumentPolicy(str, Enum):
    """What a grounded collector does when a task yields no claims.

    * ``FALLBACK`` — defer to the configured fallback collector, or skip
      when none is configured.
    * ``SKIP`` — return no evidence (a successful, empty collection).
    * ``ERROR`` — raise :class:`InvalidEvidence` (a failed task).
    """

    FALLBACK = "fallback"
    SKIP = "skip"
    ERROR = "error"


@dataclass(frozen=True)
class GroundingPolicy:
    """Deterministic, serializable tuning for grounded collection.

    The policy holds only scalars and one enum — never callables — so it
    can be serialized without a schema version and passed across process
    boundaries.  Dependencies are injected into :class:`GroundedCollector`
    instead.
    """

    max_documents: int = 5
    min_document_trust: float = 0.0
    min_claim_confidence: float = 0.0
    max_claims_per_task: int = 3
    on_missing_document: MissingDocumentPolicy = MissingDocumentPolicy.FALLBACK

    def __post_init__(self) -> None:
        if isinstance(self.max_documents, bool) or not isinstance(
            self.max_documents, int
        ):
            raise InvalidCollectionInputError("max_documents must be an int")
        if self.max_documents < 1:
            raise InvalidCollectionInputError("max_documents must be at least 1")
        if isinstance(self.max_claims_per_task, bool) or not isinstance(
            self.max_claims_per_task, int
        ):
            raise InvalidCollectionInputError("max_claims_per_task must be an int")
        if self.max_claims_per_task < 1:
            raise InvalidCollectionInputError(
                "max_claims_per_task must be at least 1"
            )
        _check_unit_range("min_document_trust", self.min_document_trust)
        _check_unit_range("min_claim_confidence", self.min_claim_confidence)
        if not isinstance(self.on_missing_document, MissingDocumentPolicy):
            raise InvalidCollectionInputError(
                "on_missing_document must be a MissingDocumentPolicy"
            )


def _check_unit_range(name: str, value: float) -> None:
    """Raise when ``value`` is not a number in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidCollectionInputError(f"{name} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise InvalidCollectionInputError(f"{name} must be within [0, 1]")


class ReplayDocumentSource:
    """Offline :class:`DocumentSource` over a committed evidence corpus.

    The corpus stores the complete, post-enrichment bundle, so the
    documents it yields already carry trust metadata and no orchestrator
    (and therefore no I/O or async) is involved.
    """

    def __init__(self, dataset: str | _Path) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> str | _Path:
        """Return the dataset name/path this source was built from."""
        return self._dataset

    def documents(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[EvidenceDocument, ...]:
        """Return the corpus bundle's documents, in stored order."""
        from predictron_engine.evidence.replay.dataset import (
            load_corpus,
            rebuild_bundle,
            resolve_dataset_path,
        )

        if isinstance(self._dataset, str):
            path = resolve_dataset_path(self._dataset)
        else:
            path = Path(self._dataset)
        document = load_corpus(path)
        bundle = rebuild_bundle(document)
        return tuple(bundle.documents)


class ExtractionClaimProjector:
    """Projects populated ``ExtractedFeatures`` into candidate claims.

    The projector delegates to the existing extraction pipeline
    (:class:`~predictron_engine.extraction.composite.CompositeExtractor`)
    and emits one claim per **populated** feature field, in definition
    order.  It defines no topic-to-field mapping and no extraction logic:
    selecting which features matter for a topic would be an invented
    mapping, so every populated field is projected and
    :attr:`GroundingPolicy.max_claims_per_task` bounds the result.
    """

    def __init__(self, *, extractor: CompositeExtractor | None = None) -> None:
        self._extractor = extractor

    def project(
        self,
        task: ResearchTask,
        documents: tuple[EvidenceDocument, ...],
        *,
        company_name: str,
    ) -> tuple[ProjectedClaim, ...]:
        """Run extraction over ``documents`` and project populated fields."""
        if not isinstance(task, ResearchTask):
            raise InvalidCollectionInputError("project expects a ResearchTask")
        company = company_name.strip()
        if not company:
            raise InvalidCollectionInputError("company_name must not be empty")
        if not documents:
            return ()

        from predictron_engine.evidence.models import EvidenceBundle
        from predictron_engine.extraction.composite import CompositeExtractor
        from predictron_engine.models.collected_data import CollectedData
        from predictron_engine.models.startup import Startup

        combined_text = "\n\n".join(doc.text for doc in documents if doc.text)
        description = combined_text.strip() or company
        startup = Startup(name=company, website="", description=description)
        data = CollectedData(startup_name=company)
        bundle = EvidenceBundle(
            startup_name=company,
            website=None,
            documents=list(documents),
        )
        extractor = self._extractor or CompositeExtractor()
        features = extractor.extract(startup, data, bundle)

        document_ids = tuple(sorted(doc.id for doc in documents))
        claims: list[ProjectedClaim] = []
        for category, value in _populated_fields(features):
            claims.append(
                ProjectedClaim(
                    category=category,
                    claim=f"{category}: {_render_value(value)}",
                    confidence=1.0,
                    document_ids=document_ids,
                )
            )
        return tuple(claims)


class GroundedCollector(EvidenceCollector):
    """An :class:`EvidenceCollector` that emits document-grounded evidence.

    The collector applies a :class:`GroundingPolicy` to the documents a
    :class:`DocumentSource` returns and the claims a
    :class:`ClaimProjector` projects, then builds version ``"2"``
    :class:`Evidence`.  It performs no I/O of its own and never computes
    trust.
    """

    collector_id: ClassVar[str] = "grounded_collector"
    display_name: ClassVar[str] = "Grounded Evidence"
    description: ClassVar[str] = (
        "Evidence projected from documents retrieved by the evidence "
        "subsystem."
    )
    supported_topics: ClassVar[tuple[str, ...]] = tuple(all_topic_ids())

    def __init__(
        self,
        policy: GroundingPolicy,
        *,
        source: DocumentSource,
        projector: ClaimProjector,
        fallback: EvidenceCollector | None = None,
    ) -> None:
        if not isinstance(policy, GroundingPolicy):
            raise InvalidCollectionInputError("policy must be a GroundingPolicy")
        if not isinstance(source, DocumentSource):
            raise InvalidCollectionInputError("source must implement DocumentSource")
        if not isinstance(projector, ClaimProjector):
            raise InvalidCollectionInputError(
                "projector must implement ClaimProjector"
            )
        if fallback is not None and not isinstance(fallback, EvidenceCollector):
            raise InvalidCollectionInputError(
                "fallback must be an EvidenceCollector or None"
            )
        self._policy = policy
        self._source = source
        self._projector = projector
        self._fallback = fallback

    @property
    def policy(self) -> GroundingPolicy:
        """Return the policy this collector applies."""
        return self._policy

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        """Produce grounded evidence for ``task``.

        Raises
        ------
        InvalidCollectionInputError:
            When ``task`` is not a :class:`ResearchTask` or
            ``company_name`` is blank.
        UnsupportedTopic:
            When ``task.topic_id`` is not supported.
        InvalidEvidence:
            When the missing-document policy is ``ERROR`` and no claims
            survive.
        """
        if not isinstance(task, ResearchTask):
            raise InvalidCollectionInputError(
                "collect expects a ResearchTask instance"
            )
        enforce_supported(self, task.topic_id)
        company = company_name.strip()
        if not company:
            raise InvalidCollectionInputError("company_name must not be empty")

        documents = tuple(
            self._source.documents(company_name=company, website=website)
        )
        selected = self._select_documents(documents)
        claims = self._filter_claims(
            self._projector.project(task, selected, company_name=company),
            selected,
        )
        if not claims:
            return self._handle_missing(task, company_name=company, website=website)

        documents_by_id = {doc.id: doc for doc in selected}
        built = [
            self._build_evidence(task, claim, documents_by_id)
            for claim in claims
        ]
        return tuple(sorted(built, key=lambda item: item.evidence_id))

    # ------------------------------------------------------------------
    # Policy application
    # ------------------------------------------------------------------

    def _select_documents(
        self, documents: tuple[EvidenceDocument, ...]
    ) -> tuple[EvidenceDocument, ...]:
        """Filter by trust, then sort by id and cap to ``max_documents``."""
        minimum = self._policy.min_document_trust
        eligible = [
            document for document in documents if _document_trust(document) >= minimum
        ]
        eligible.sort(key=lambda document: document.id)
        return tuple(eligible[: self._policy.max_documents])

    def _filter_claims(
        self,
        claims: Sequence[ProjectedClaim],
        documents: tuple[EvidenceDocument, ...],
    ) -> tuple[ProjectedClaim, ...]:
        """Apply confidence/known-document filters, sort, and cap."""
        known = {document.id for document in documents}
        all_ids = tuple(document.id for document in documents)
        kept: list[ProjectedClaim] = []
        for claim in claims:
            if claim.confidence < self._policy.min_claim_confidence:
                continue
            cited = tuple(identifier for identifier in claim.document_ids if identifier in known)
            if claim.document_ids and not cited:
                continue
            if not claim.document_ids:
                cited = all_ids
            kept.append(replace(claim, document_ids=cited))
        kept.sort(
            key=lambda claim: (
                claim.category,
                claim.claim,
                claim.confidence,
                claim.document_ids,
            )
        )
        return tuple(kept[: self._policy.max_claims_per_task])

    def _handle_missing(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str,
    ) -> tuple[Evidence, ...]:
        """Apply :attr:`GroundingPolicy.on_missing_document`."""
        policy = self._policy.on_missing_document
        if policy is MissingDocumentPolicy.ERROR:
            raise InvalidEvidence(
                f"no grounded evidence for topic {task.topic_id!r} "
                f"from collector {self.collector_id!r}"
            )
        if policy is MissingDocumentPolicy.SKIP or self._fallback is None:
            return ()
        return tuple(
            self._fallback.collect(
                task, company_name=company_name, website=website
            )
        )

    # ------------------------------------------------------------------
    # Evidence construction
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        task: ResearchTask,
        claim: ProjectedClaim,
        documents_by_id: dict[str, EvidenceDocument],
    ) -> Evidence:
        """Build one version ``"2"`` :class:`Evidence` from a claim."""
        from predictron_engine.evidence.citation import format_citation
        from predictron_engine.models.report import EvidenceCitation

        supporting = [
            documents_by_id[identifier]
            for identifier in claim.document_ids
            if identifier in documents_by_id
        ]
        reference_document = sorted(
            supporting, key=lambda document: (-_document_trust(document), document.id)
        )[0]
        trust = _document_trust_score(reference_document)
        reference = EvidenceReference(
            source_category=_source_category(reference_document),
            source_identifier=_source_identifier(reference_document),
            url=str(reference_document.url),
            description=f"Grounded in document {reference_document.id}.",
        )
        citation = EvidenceCitation(
            claim=claim.claim,
            domain=task.topic_id,
            category=claim.category,
            source_document_ids=list(claim.document_ids),
            source_urls=[str(document.url) for document in supporting],
            best_trust_score=round(trust.overall, 4) if trust is not None else 0.0,
            provider=self.collector_id,
        )
        citation = citation.model_copy(
            update={"citation_text": format_citation(citation)}
        )
        citations = (citation,)
        return Evidence(
            task_id=task.task_id,
            topic_id=task.topic_id,
            collector_id=self.collector_id,
            category=claim.category,
            claim=claim.claim,
            confidence=claim.confidence,
            reference=reference,
            evidence_id_version=GROUNDED_EVIDENCE_ID_VERSION,
            document_ids=document_ids_for(citations),
            trust_score=trust,
            citation_ids=citation_ids(citations),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _document_trust(document: EvidenceDocument) -> float:
    """Return a document's overall trust, or ``0.0`` when it has none."""
    trust = _document_trust_score(document)
    return 0.0 if trust is None else float(trust.overall)


def _document_trust_score(document: EvidenceDocument) -> TrustScore | None:
    """Return the document's imported ``TrustScore``, or ``None``."""
    metadata = document.metadata
    if metadata is None:
        return None
    return metadata.trust_score


def _source_category(document: EvidenceDocument) -> str:
    """Return the document's category label (intelligence first)."""
    metadata = document.metadata
    if metadata is not None:
        return metadata.document_type.value
    return document.page_type.value


def _source_identifier(document: EvidenceDocument) -> str:
    """Return a stable identifier for the document's source host."""
    host = urlsplit(str(document.url)).netloc
    return host or document.id


def _populated_fields(features: ExtractedFeatures) -> list[tuple[str, Any]]:
    """Return the feature fields whose value differs from their default."""
    items: list[tuple[str, Any]] = []
    for name, field in type(features).model_fields.items():
        value = getattr(features, name)
        default = field.get_default(call_default_factory=True)
        if value != default:
            items.append((name, value))
    return items


def _render_value(value: Any) -> str:
    """Render a feature value deterministically as claim text."""
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={value[key]}" for key in sorted(value))
    return str(value)


__all__ = [
    "ClaimProjector",
    "DocumentSource",
    "ExtractionClaimProjector",
    "GroundedCollector",
    "GroundingPolicy",
    "MissingDocumentPolicy",
    "ProjectedClaim",
    "ReplayDocumentSource",
]
