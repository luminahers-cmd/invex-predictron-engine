"""Versioned provenance and trust links (Sprint 8, WI-2).

Three promises are asserted here, and they have to hold together:

* **v1 is frozen.**  Identity and serialization for an ungrounded claim
  are byte-for-byte what they were before linkage existed.  The golden
  identifiers below were captured before this work item landed; if one
  moves, stored repositories stop reproducing their own identifiers.
* **grounding is explicit.**  Linkage can only be recorded under
  evidence id version ``"2"``, and only when someone asks for it.
* **trust and citations are imported.**  Research carries what the
  evidence subsystem produced; it never recomputes, rebands, or
  re-derives either of them.

The final tests are source-level: they fail if a future change quietly
reintroduces trust scoring or a second digest implementation inside the
research layer, which is the failure mode that would make this work
item's guarantees worthless.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from predictron_engine.canonical import RESEARCH_DIGEST_LENGTH, content_digest
from predictron_engine.evidence.models import EvidenceDocument
from predictron_engine.evidence.provenance import (
    ProvenanceRecord,
    SourceTrustLevel,
    TrustFactor,
    TrustScore,
    compute_document_trust,
)
from predictron_engine.models.report import EvidenceCitation
from predictron_engine.research.exceptions import InvalidEvidence
from predictron_engine.research.graph_models import Provenance as GraphProvenance
from predictron_engine.research.models import Evidence, EvidenceReference
from predictron_engine.research.provenance_bridge import (
    CITATION_ID_PREFIX,
    CITATION_ID_VERSION,
    EVIDENCE_ID_VERSION,
    GROUNDED_EVIDENCE_ID_VERSION,
    SUPPORTED_EVIDENCE_ID_VERSIONS,
    V1_IDENTITY_FIELDS,
    citation_id,
    citation_ids,
    document_ids_for,
    evidence_identity_payload,
    normalise_link_ids,
    require_supported_version,
    trust_score_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SOURCE = (REPO_ROOT / "predictron_engine" / "research" / "provenance_bridge.py")
MODELS_SOURCE = (REPO_ROOT / "predictron_engine" / "research" / "models.py")

# Captured before WI-2.  These are the identifiers existing repositories
# already hold, so they must never move.
GOLDEN_V1_ID = "ev_eef0072081b6cdf6"
GOLDEN_V1_JSON = (
    '{"category":"team","claim":"Acme Corp has three founders.",'
    '"collector_id":"founder_collector","confidence":0.75,'
    '"evidence_id":"ev_eef0072081b6cdf6","reference":'
    '{"description":"About page","source_category":"primary_site",'
    '"source_identifier":"about_page","url":"https://example.test/about"},'
    '"task_id":"task_1","topic_id":"market"}'
)
GOLDEN_V1_UNICODE_ID = "ev_3d024233600629b5"
GOLDEN_V1_BOUNDARY_ID = "ev_6c646d1ab8e00d02"

V1_KEYS = {
    "evidence_id",
    "task_id",
    "topic_id",
    "collector_id",
    "category",
    "claim",
    "confidence",
    "reference",
}
V2_ONLY_KEYS = {"evidence_id_version", "document_ids", "trust_score", "citation_ids"}

# Deterministic inputs; the clock is pinned so the subsystem's freshness
# factor is reproducible across runs.
_FETCHED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _reference() -> EvidenceReference:
    return EvidenceReference(
        source_category="primary_site",
        source_identifier="about_page",
        url="https://example.test/about",
        description="About page",
    )


def _evidence(
    *,
    claim: str = "Acme Corp has three founders.",
    confidence: float = 0.75,
    task_id: str = "task_1",
    evidence_id_version: str = EVIDENCE_ID_VERSION,
    document_ids: tuple[str, ...] = (),
    trust_score: TrustScore | None = None,
    citation_ids: tuple[str, ...] = (),
) -> Evidence:
    return Evidence(
        task_id=task_id,
        topic_id="market",
        collector_id="founder_collector",
        category="team",
        claim=claim,
        confidence=confidence,
        reference=_reference(),
        evidence_id_version=evidence_id_version,
        document_ids=document_ids,
        trust_score=trust_score,
        citation_ids=citation_ids,
    )


def _subsystem_trust_score() -> TrustScore:
    """Produce trust the way the evidence subsystem produces it."""
    return compute_document_trust(
        authority_score=0.87,
        quality_score=0.91,
        fetched_at=_FETCHED_AT,
        is_official=True,
        now=_NOW,
    )


def _citation(
    claim: str = "Enterprise sales cycles are 12-18 months",
    document_ids: tuple[str, ...] = ("doc-b", "doc-a"),
) -> EvidenceCitation:
    return EvidenceCitation(
        claim=claim,
        domain="industry",
        category="sales_cycle",
        source_document_ids=list(document_ids),
        source_urls=["https://example.test/a", "https://example.test/b"],
        best_trust_score=0.81,
        citation_text=f"source: {claim}",
        provider="placeholder",
    )


# ────────────────────────────────────────────────────────────────────
# v1 identity and serialization are frozen
# ────────────────────────────────────────────────────────────────────


class TestV1IdentityIsFrozen:
    def test_golden_identifier_reproduced(self) -> None:
        assert _evidence().evidence_id == GOLDEN_V1_ID

    def test_golden_serialization_is_byte_identical(self) -> None:
        """Not merely equal — the stored bytes must match."""
        encoded = json.dumps(_evidence().to_dict(), sort_keys=True, separators=(",", ":"))
        assert encoded == GOLDEN_V1_JSON

    def test_non_ascii_claim_keeps_its_identifier(self) -> None:
        """Unicode is a real risk: an encoding change renumbers ids."""
        item = _evidence(
            task_id="task_2",
            claim="Acme \u00dcber Corp grows 12% \u2014 r\u00e9sum\u00e9s included.",
            confidence=1.0,
        )
        assert item.evidence_id == GOLDEN_V1_UNICODE_ID

    def test_boundary_confidence_keeps_its_identifier(self) -> None:
        item = _evidence(task_id="task_3", claim="x", confidence=0.0)
        assert item.evidence_id == GOLDEN_V1_BOUNDARY_ID

    def test_v1_serialization_exposes_no_linkage_keys(self) -> None:
        assert set(_evidence().to_dict()) == V1_KEYS

    def test_v1_coverage_is_exactly_the_original_fields(self) -> None:
        assert set(V1_IDENTITY_FIELDS) == {
            "task_id",
            "topic_id",
            "collector_id",
            "category",
            "claim",
            "confidence",
            "reference",
        }

    def test_v1_payload_ignores_grounded_fields(self) -> None:
        payload = evidence_identity_payload(
            EVIDENCE_ID_VERSION,
            {
                **{name: "value" for name in V1_IDENTITY_FIELDS},
                "evidence_id_version": EVIDENCE_ID_VERSION,
                "document_ids": ["doc-a"],
                "trust_score": {"overall": 0.5},
                "citation_ids": ["cit_abc"],
            },
        )
        assert set(payload) == set(V1_IDENTITY_FIELDS)


class TestLegacyPayloadsDeserializeUnchanged:
    def test_payload_without_version_keys_reproduces_its_id(self) -> None:
        legacy = json.loads(GOLDEN_V1_JSON)
        restored = Evidence.from_dict(legacy)
        assert restored.evidence_id == GOLDEN_V1_ID
        assert restored.evidence_id_version == EVIDENCE_ID_VERSION
        assert restored.document_ids == ()
        assert restored.trust_score is None
        assert restored.citation_ids == ()

    def test_legacy_payload_round_trips_to_itself(self) -> None:
        legacy = json.loads(GOLDEN_V1_JSON)
        assert Evidence.from_dict(legacy).to_dict() == legacy

    def test_json_round_trip_preserves_identity(self) -> None:
        item = _evidence()
        wire = json.loads(json.dumps(item.to_dict()))
        restored = Evidence.from_dict(wire)
        assert restored == item
        assert restored.evidence_id == item.evidence_id

    def test_deserialize_does_not_read_a_stale_identifier(self) -> None:
        """``evidence_id`` is recomputed, never trusted from the wire."""
        stale = json.loads(GOLDEN_V1_JSON)
        stale["evidence_id"] = "ev_0000000000000000"
        assert Evidence.from_dict(stale).evidence_id == GOLDEN_V1_ID


# ────────────────────────────────────────────────────────────────────
# Version 2 is explicit, opt-in, and complete
# ────────────────────────────────────────────────────────────────────


class TestGroundingRequiresVersionTwo:
    def test_default_version_is_one(self) -> None:
        assert _evidence().evidence_id_version == "1"
        assert EVIDENCE_ID_VERSION == "1"

    def test_document_ids_rejected_on_v1(self) -> None:
        with pytest.raises(InvalidEvidence, match="not covered by evidence_id_version"):
            _evidence(document_ids=("doc-a",))

    def test_trust_reference_rejected_on_v1(self) -> None:
        with pytest.raises(InvalidEvidence, match="not covered by evidence_id_version"):
            _evidence(trust_score=_subsystem_trust_score())

    def test_citation_ids_rejected_on_v1(self) -> None:
        with pytest.raises(InvalidEvidence, match="not covered by evidence_id_version"):
            _evidence(citation_ids=("cit_a",))

    def test_error_names_the_rehash_operation(self) -> None:
        """The message has to say how to proceed, not just what failed."""
        with pytest.raises(InvalidEvidence, match="rehash"):
            _evidence(document_ids=("doc-a",))

    def test_unsupported_version_rejected(self) -> None:
        with pytest.raises(InvalidEvidence, match="unsupported evidence_id_version"):
            _evidence(evidence_id_version="9")

    def test_supported_versions_are_one_and_two(self) -> None:
        assert SUPPORTED_EVIDENCE_ID_VERSIONS == frozenset({"1", "2"})
        assert require_supported_version(" 1 ") == "1"
        with pytest.raises(InvalidEvidence):
            require_supported_version("")


class TestRehash:
    def test_upgrades_and_changes_identity(self) -> None:
        original = _evidence()
        upgraded = original.rehash(to_version=GROUNDED_EVIDENCE_ID_VERSION)

        assert original.evidence_id_version == "1"
        assert upgraded.evidence_id_version == "2"
        assert upgraded.evidence_id != original.evidence_id

    def test_preserves_every_ungrounded_field(self) -> None:
        original = _evidence()
        upgraded = original.rehash(to_version="2")

        assert upgraded.task_id == original.task_id
        assert upgraded.topic_id == original.topic_id
        assert upgraded.collector_id == original.collector_id
        assert upgraded.category == original.category
        assert upgraded.claim == original.claim
        assert upgraded.confidence == original.confidence
        assert upgraded.reference == original.reference

    def test_is_idempotent(self) -> None:
        once = _evidence().rehash(to_version="2")
        twice = once.rehash(to_version="2")
        assert twice is once
        assert twice.evidence_id == once.evidence_id

    def test_v2_without_linkage_still_differs_from_v1(self) -> None:
        """A v2 id must not be mistakable for a v1 id even when empty —
        otherwise "which fields does this identifier cover?" is
        unanswerable from the identifier's own record."""
        item = _evidence().rehash(to_version="2")
        assert item.evidence_id != _evidence().evidence_id

    def test_downgrade_to_v1_restores_the_original_id(self) -> None:
        original = _evidence()
        restored = original.rehash(to_version="2").rehash(to_version="1")
        assert restored.evidence_id == original.evidence_id

    def test_downgrade_refused_while_grounded(self) -> None:
        """v1 identity cannot represent linkage, so downgrading would
        discard it silently."""
        grounded = _evidence(
            evidence_id_version="2",
            document_ids=("doc-a",),
            trust_score=_subsystem_trust_score(),
        )
        with pytest.raises(InvalidEvidence, match="not covered by evidence_id_version"):
            grounded.rehash(to_version="1")

    def test_unknown_target_version_rejected(self) -> None:
        with pytest.raises(InvalidEvidence, match="unsupported evidence_id_version"):
            _evidence().rehash(to_version="3")

    def test_never_migrates_implicitly(self) -> None:
        """Building an item never upgrades it: the default constructor
        must keep producing a v1 identifier."""
        assert _evidence(evidence_id_version="1").evidence_id == GOLDEN_V1_ID


class TestVersionTwoSerialization:
    def test_round_trips_through_json(self) -> None:
        item = _evidence(
            evidence_id_version="2",
            document_ids=("doc-a", "doc-b"),
            trust_score=_subsystem_trust_score(),
            citation_ids=("cit_a", "cit_b"),
        )
        wire = json.loads(json.dumps(item.to_dict()))
        restored = Evidence.from_dict(wire)

        assert restored == item
        assert restored.evidence_id == item.evidence_id
        assert restored.evidence_id_version == "2"

    def test_v2_payload_declares_its_version(self) -> None:
        payload = _evidence(evidence_id_version="2").to_dict()
        assert payload["evidence_id_version"] == "2"

    def test_v1_payload_omits_the_new_keys(self) -> None:
        assert set(_evidence().to_dict()).isdisjoint(V2_ONLY_KEYS)

    def test_v2_payload_carries_every_new_key(self) -> None:
        payload = _evidence(evidence_id_version="2").to_dict()
        assert V2_ONLY_KEYS <= set(payload)

    def test_identity_changes_when_any_link_changes(self) -> None:
        base = _evidence(evidence_id_version="2", document_ids=("doc-a",))
        assert (
            _evidence(evidence_id_version="2", document_ids=("doc-b",)).evidence_id
            != base.evidence_id
        )
        assert (
            _evidence(
                evidence_id_version="2", document_ids=("doc-a",), confidence=0.76
            ).evidence_id
            != base.evidence_id
        )


# ────────────────────────────────────────────────────────────────────
# References are sets: order must not leak into identity
# ────────────────────────────────────────────────────────────────────


class TestLinkNormalisation:
    def test_order_does_not_change_identity(self) -> None:
        forward = _evidence(
            evidence_id_version="2",
            document_ids=("doc-a", "doc-b", "doc-c"),
            citation_ids=("cit_1", "cit_2"),
        )
        reversed_ = _evidence(
            evidence_id_version="2",
            document_ids=("doc-c", "doc-b", "doc-a"),
            citation_ids=("cit_2", "cit_1"),
        )
        assert forward.evidence_id == reversed_.evidence_id

    def test_links_are_sorted_and_deduplicated(self) -> None:
        item = _evidence(
            evidence_id_version="2",
            document_ids=(" doc-b ", "doc-a", "doc-b"),
        )
        assert item.document_ids == ("doc-a", "doc-b")

    def test_blank_link_rejected(self) -> None:
        with pytest.raises(InvalidEvidence, match="must not be empty"):
            _evidence(evidence_id_version="2", document_ids=("  ",))

    def test_non_string_link_rejected(self) -> None:
        with pytest.raises(InvalidEvidence, match="must be strings"):
            normalise_link_ids([123], "document_ids")  # type: ignore[list-item]

    def test_normalise_is_pure(self) -> None:
        values = ("doc-b", "doc-a", "doc-b")
        assert normalise_link_ids(values, "document_ids") == ("doc-a", "doc-b")
        assert values == ("doc-b", "doc-a", "doc-b")


# ────────────────────────────────────────────────────────────────────
# Trust is imported, never recomputed
# ────────────────────────────────────────────────────────────────────


class TestTrustIsImported:
    def test_trust_must_be_the_evidence_subsystem_type(self) -> None:
        lookalike = type("TrustScore", (), {"overall": 0.5})
        with pytest.raises(InvalidEvidence, match="TrustScore from the evidence"):
            _evidence(
                evidence_id_version="2",
                trust_score=lookalike(),  # type: ignore[arg-type]
            )

    def test_trust_arrives_from_the_subsystems_own_scorer(self) -> None:
        score = _subsystem_trust_score()
        item = _evidence(evidence_id_version="2", trust_score=score)
        assert item.trust_score == score
        assert item.trust_score is score

    def test_nothing_is_recomputed(self) -> None:
        """A score whose factors do not add up must survive verbatim.

        If research ever recomposed ``overall`` from its factors, this
        is the assertion that catches it.
        """
        contradictory = TrustScore(
            overall=0.99,
            factors=[TrustFactor(name="authority", weight=0.3, value=0.1)],
            source_trust_level=SourceTrustLevel.THIRD_PARTY_UNVERIFIED,
            freshness_days=1000,
        )
        item = _evidence(evidence_id_version="2", trust_score=contradictory)

        assert item.trust_score is contradictory
        assert item.trust_score is not None
        assert item.trust_score.overall == 0.99
        assert item.trust_score.freshness_days == 1000
        assert item.trust_score.source_trust_level is (
            SourceTrustLevel.THIRD_PARTY_UNVERIFIED
        )

    def test_trust_survives_a_wire_round_trip_unchanged(self) -> None:
        score = _subsystem_trust_score()
        item = _evidence(evidence_id_version="2", trust_score=score)
        restored = Evidence.from_dict(json.loads(json.dumps(item.to_dict())))

        assert restored.trust_score is not None
        assert restored.trust_score.model_dump(mode="json") == score.model_dump(
            mode="json"
        )
        assert restored.trust_score.factors == score.factors

    def test_serialisation_dumps_without_rederiving(self) -> None:
        score = _subsystem_trust_score()
        assert trust_score_payload(score) == score.model_dump(mode="json")

    def test_absent_trust_is_none_not_zero(self) -> None:
        """'No trust reference' and 'trust of zero' must not collide."""
        untrusted = _evidence(
            evidence_id_version="2",
            trust_score=TrustScore(overall=0.0),
        )
        unlinked = _evidence(evidence_id_version="2")

        assert untrusted.evidence_id != unlinked.evidence_id
        assert trust_score_payload(None) is None
        assert unlinked.trust_score is None

    def test_identical_trust_produces_identical_identity(self) -> None:
        first = _evidence(
            evidence_id_version="2", trust_score=_subsystem_trust_score()
        )
        second = _evidence(
            evidence_id_version="2", trust_score=_subsystem_trust_score()
        )
        assert first.evidence_id == second.evidence_id


# ────────────────────────────────────────────────────────────────────
# Citations and document ids are read from the evidence subsystem
# ────────────────────────────────────────────────────────────────────


class TestCitationAndDocumentLinkage:
    def test_citation_id_is_prefixed_and_content_addressed(self) -> None:
        identifier = citation_id(_citation())
        assert identifier.startswith(CITATION_ID_PREFIX)
        assert len(identifier) == len(CITATION_ID_PREFIX) + RESEARCH_DIGEST_LENGTH

    def test_citation_id_is_stable(self) -> None:
        assert citation_id(_citation()) == citation_id(_citation())

    def test_citation_id_follows_content(self) -> None:
        assert citation_id(_citation(claim="A")) != citation_id(_citation(claim="B"))

    def test_citation_id_delegates_to_the_canonical_digest(self) -> None:
        """Proves there is no second encoding of citations in research."""
        citation = _citation()
        payload = citation.model_dump(mode="json")
        payload["citation_id_version"] = CITATION_ID_VERSION
        assert citation_id(citation) == content_digest(
            payload, length=RESEARCH_DIGEST_LENGTH, prefix=CITATION_ID_PREFIX
        )

    def test_citation_ids_are_sorted_and_deduplicated(self) -> None:
        duplicate = [_citation(), _citation(), _citation(claim="Other")]
        ids = citation_ids(duplicate)
        assert len(ids) == 2
        assert list(ids) == sorted(ids)

    def test_document_ids_are_read_from_citations(self) -> None:
        ids = document_ids_for(
            [
                _citation(document_ids=("doc-b", "doc-a")),
                _citation(document_ids=("doc-c", "doc-a")),
            ]
        )
        assert ids == ("doc-a", "doc-b", "doc-c")

    def test_document_ids_without_citations_are_empty(self) -> None:
        assert document_ids_for([]) == ()

    def test_grounded_claim_references_real_documents_and_citations(self) -> None:
        """The work item's end state: a claim that points at documents
        and carries the trust the evidence subsystem assigned them."""
        citations = [_citation()]
        score = _subsystem_trust_score()

        item = _evidence(
            evidence_id_version="2",
            document_ids=document_ids_for(citations),
            citation_ids=citation_ids(citations),
            trust_score=score,
        ).rehash(to_version="2")

        assert item.document_ids == ("doc-a", "doc-b")
        assert len(item.citation_ids) == 1
        assert item.citation_ids[0].startswith(CITATION_ID_PREFIX)
        assert item.trust_score is score

        restored = Evidence.from_dict(json.loads(json.dumps(item.to_dict())))
        assert restored.document_ids == item.document_ids
        assert restored.citation_ids == item.citation_ids
        assert restored.evidence_id == item.evidence_id


# ────────────────────────────────────────────────────────────────────
# Model families stay separate
# ────────────────────────────────────────────────────────────────────


class TestModelFamiliesRemainDistinct:
    def test_research_evidence_is_not_a_document(self) -> None:
        assert Evidence is not EvidenceDocument
        assert not issubclass(Evidence, EvidenceDocument)
        assert not issubclass(EvidenceDocument, Evidence)

    def test_research_provenance_is_not_document_provenance(self) -> None:
        """Graph-fact provenance and chain-of-custody are different jobs."""
        assert GraphProvenance is not ProvenanceRecord
        assert not issubclass(GraphProvenance, ProvenanceRecord)

    def test_document_linkage_does_not_add_document_fields(self) -> None:
        """A linked claim must not grow url/fetch/transport state."""
        linked = _evidence(
            evidence_id_version="2", document_ids=("doc-a",)
        ).to_dict()
        forbidden = {"url", "fetched_at", "http_status", "retrieval_method", "text"}
        assert set(linked).isdisjoint(forbidden)


# ────────────────────────────────────────────────────────────────────
# Structural guarantees that tests above would not catch
# ────────────────────────────────────────────────────────────────────


def _runtime_imported_modules(source: str) -> set[str]:
    """Module names imported outside a ``if TYPE_CHECKING:`` block."""
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
        for statement in _walk_imports(node):
            if statement.module:
                modules.add(statement.module)
    return modules


def _walk_imports(node: ast.AST) -> list[ast.ImportFrom]:
    found: list[ast.ImportFrom] = []
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.module:
            found.append(child)
    return found


class TestStructuralGuarantees:
    def test_bridge_depends_on_nothing_that_persists_or_fetches(self) -> None:
        """The provenance bridge reads values; it must not acquire the
        ability to store or retrieve them."""
        allowed = {
            "__future__",
            "collections.abc",
            "predictron_engine.canonical",
            "predictron_engine.research.exceptions",
            "typing",
        }
        assert _runtime_imported_modules(BRIDGE_SOURCE.read_text(encoding="utf-8")) <= allowed

    def test_research_models_do_not_import_http_clients(self) -> None:
        source = MODELS_SOURCE.read_text(encoding="utf-8")
        for module in ("httpx", "aiohttp", "requests", "urllib.request", "socket"):
            assert module not in source

    def test_research_does_not_call_the_trust_scorer(self) -> None:
        """``compute_document_trust`` is the evidence subsystem's.  A
        research-layer call would mean research formed its own opinion."""
        for source in (BRIDGE_SOURCE, MODELS_SOURCE):
            text = source.read_text(encoding="utf-8")
            assert "compute_document_trust" not in text, source.name
            assert "build_provenance_record" not in text, source.name

    def test_research_does_not_reimplement_trust_banding(self) -> None:
        """Band thresholds belong with the scorer that produced them."""
        for source in (BRIDGE_SOURCE, MODELS_SOURCE):
            text = source.read_text(encoding="utf-8")
            assert "SourceTrustLevel" not in text, source.name
            assert "TrustSummary" not in text, source.name
            assert "trust_summary" not in text, source.name

    def test_evidence_identity_version_constant_is_one(self) -> None:
        assert EVIDENCE_ID_VERSION == "1"
        assert GROUNDED_EVIDENCE_ID_VERSION == "2"
