# WI-3: Grounding Stage (Authoritative Contract)

**Status:** APPROVED and COMPLETE. This is the authoritative WI-3 contract,
now frozen. Implemented and verified in the same change. Behavior, APIs,
tests, baselines, and architecture are frozen; do not modify without a later
approved work item or the project's ADR process.

**Depends on:** WI-1 (canonicalization), WI-2 (versioned provenance/trust
linkage), WI-4 (persistence ports), WI-5 (`ResearchExecutorPort`).

**Provenance note.** The original WI-3 specification could not be
recovered from the repository, git history, or the working session. This
document was reconstructed from the frozen interfaces of WI-1 through
WI-5, proposed for review, and accepted with the decisions in Section 0.
Section 0 is binding; where the body still lists options, Section 0 wins.

---

## 0. Approved decisions

- **D7b adopted - research remains pure.** `DocumentSource` stays a
  protocol inside `predictron_engine/research/`. Live adapters, including
  `SyncEvidenceSource`, live **outside** the research package, in
  `predictron_engine/integrations/grounding.py`. `ReplayDocumentSource`
  stays inside `research` because it operates only on deterministic
  replay artifacts. Because no I/O is added to
  `predictron_engine/research/`, **no ADR is required**.
- **`ExtractionClaimProjector` delegates, it does not invent.** It runs
  the existing extraction pipeline (`CompositeExtractor`) over a bundle
  built from the retrieved documents and projects the resulting
  `ExtractedFeatures` into version-2 `Evidence`. It defines **no**
  topic-to-field mapping and **no** extraction logic of its own.
- **Binding non-goal.** Grounding does not modify planning, extraction
  heuristics, confidence computation, trust computation, evidence
  ranking, or orchestration semantics. It only replaces placeholder
  evidence generation with evidence projected from existing retrieved
  documents when grounding is enabled.

## 1. Scope

WI-3 introduces an **opt-in grounding stage** for the research subsystem.
Grounding replaces a topic's *placeholder* evidence with evidence *linked
to real documents* produced by the existing evidence subsystem.

In scope:

- New module `predictron_engine/research/grounding.py`.
- New module `predictron_engine/integrations/grounding.py` holding the
  live adapter `SyncEvidenceSource` (outside research).
- A new test module `tests/research/test_grounding.py`.
- Additive exports from `predictron_engine/research/__init__.py`.
- New factory `grounded_collector_registry` in `research/registry.py`.

Out of scope (explicitly unchanged):

- `EvidenceCollector`, `PlaceholderCollector`, the 15 default collectors.
- `COLLECTOR_REGISTRY`, `default_collector_registry` (extended only by a
  new factory function).
- `EvidenceCollectionEngine`, `Dispatcher`, `ResearchOrchestrator`,
  `ResearchLoop`.
- The evidence subsystem, replay corpus format, and
  `predictron_engine.canonical`.
- Any schema version and any existing content-addressed identifier.

## 2. Goals / Non-goals

### Goals

- **G1.** Let a research task yield grounded `Evidence` (identity version
  `"2"`) carrying `document_ids`, `citation_ids`, and an imported
  `TrustScore`.
- **G2.** Reuse existing subsystems rather than duplicate them: the
  evidence subsystem's document model and citation formatter, the offline
  replay corpus, and the extraction pipeline.
- **G3.** Keep grounding **opt-in**. The placeholder path stays
  byte-identical: same `evidence_id` values, same collection/report ids,
  same research baseline.
- **G4.** Preserve every WI-1/WI-2/WI-4/WI-5 invariant.
- **G5.** Provide a deterministic, offline replay path so grounding is
  testable without network access.
- **G6.** Leave planning, extraction, confidence, trust, ranking, and
  orchestration semantics byte-identical; grounding only substitutes the
  evidence source.

### Non-goals

- **N1.** No live crawling, search APIs, browsers, or LLMs added to the
  default research path.
- **N2.** No changes to the evidence subsystem's trust computation or
  document enrichment.
- **N3.** No new persistence format or schema version.
- **N4.** No automatic migration of stored version `"1"` evidence.
- **N5.** No async surface exposed to the collector contract.
- **N6.** No change to the validation or memory layers.
- **N7 (binding).** Grounding does not modify planning, extraction
  heuristics, confidence computation, trust computation, evidence
  ranking, or orchestration semantics. It only swaps placeholder evidence
  generation for document-grounded evidence when enabled.

## 3. Public interfaces

New module `predictron_engine/research/grounding.py`.

Additive exports from `predictron_engine/research/__init__.py`:

- `DocumentSource`, `ClaimProjector`
- `GroundingPolicy`, `MissingDocumentPolicy`, `ProjectedClaim`
- `ReplayDocumentSource`, `ExtractionClaimProjector`, `GroundedCollector`
- `grounded_collector_registry`

New module `predictron_engine/integrations/grounding.py`:

- `SyncEvidenceSource`

New factory in `research/registry.py`:

- `grounded_collector_registry(policy, *, source, projector, fallback=None)`

Symbol map:

| Symbol | Kind | Module |
| --- | --- | --- |
| `DocumentSource` | Protocol | `research/grounding.py` |
| `ClaimProjector` | Protocol | `research/grounding.py` |
| `GroundingPolicy` | frozen dataclass | `research/grounding.py` |
| `MissingDocumentPolicy` | `str` Enum | `research/grounding.py` |
| `ProjectedClaim` | frozen dataclass | `research/grounding.py` |
| `ReplayDocumentSource` | class | `research/grounding.py` |
| `ExtractionClaimProjector` | class | `research/grounding.py` |
| `GroundedCollector` | `EvidenceCollector` subclass | `research/grounding.py` |
| `SyncEvidenceSource` | class (live adapter) | `integrations/grounding.py` |
| `grounded_collector_registry` | function | `research/registry.py` |

## 4. Exact protocol signatures

```python
@runtime_checkable
class DocumentSource(Protocol):
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
    def project(
        self,
        task: ResearchTask,
        documents: tuple[EvidenceDocument, ...],
        *,
        company_name: str,
    ) -> tuple[ProjectedClaim, ...]:
        """Return the claims ``task`` can support from ``documents``."""
        ...
```

Both are company/task scoped (D1/D2): a source returns one document set
per company; a projector decides relevance per task and receives the
whole set. `research/grounding.py` imports concrete evidence/extraction
types only for implementations and uses `TYPE_CHECKING` for protocol
annotations, so the protocol surface stays free of runtime dependencies.

## 4.1 Value object: `ProjectedClaim`

```python
@dataclass(frozen=True)
class ProjectedClaim:
    category: str
    claim: str
    confidence: float
    document_ids: tuple[str, ...] = ()
```

Validation: `category`/`claim` non-empty after strip; `confidence` in
`[0, 1]`; `document_ids` stripped, de-duplicated, and sorted via
`provenance_bridge.normalise_link_ids`. A projector supplies no
`EvidenceReference`; the collector derives it from the supporting
documents (D3).

## 4.2 Value object: `MissingDocumentPolicy`

```python
class MissingDocumentPolicy(str, Enum):
    FALLBACK = "fallback"
    SKIP = "skip"
    ERROR = "error"
```

The collection engine treats an empty tuple as a *successfully executed
task that produced no evidence*, and `InvalidEvidence`/`UnsupportedTopic`
as a *failed task* (`execution.py`). The three policies map to three
distinct deterministic outcomes.

## 5. `GroundingPolicy` fields and defaults

```python
@dataclass(frozen=True)
class GroundingPolicy:
    max_documents: int = 5
    min_document_trust: float = 0.0
    min_claim_confidence: float = 0.0
    max_claims_per_task: int = 3
    on_missing_document: MissingDocumentPolicy = MissingDocumentPolicy.FALLBACK
```

| Field | Type | Default | Constraint |
| --- | --- | --- | --- |
| `max_documents` | `int` | `5` | `>= 1` |
| `min_document_trust` | `float` | `0.0` | `[0, 1]` |
| `min_claim_confidence` | `float` | `0.0` | `[0, 1]` |
| `max_claims_per_task` | `int` | `3` | `>= 1` |
| `on_missing_document` | `MissingDocumentPolicy` | `FALLBACK` | enum member |

Invalid construction raises `InvalidCollectionInputError`. The policy is
frozen scalars plus one enum, so it is JSON-serializable without a schema
version, and it deliberately holds **no callables** (D3b): dependencies
are injected into `GroundedCollector`. Defaults are conservative and
opt-in (D4); they are not load-bearing for correctness.

## 6. Registry behavior

```python
def grounded_collector_registry(
    policy: GroundingPolicy,
    *,
    source: DocumentSource,
    projector: ClaimProjector,
    fallback: EvidenceCollector | None = None,
) -> CollectorRegistry:
    ...
```

- Returns a **new** `CollectorRegistry`; `COLLECTOR_REGISTRY` and
  `DEFAULT_COLLECTORS` are never mutated.
- Registers exactly one `GroundedCollector` whose `supported_topics`
  cover every topic from `research.topics.all_topic_ids()`.
- Because `CollectorRegistry.resolve` returns the first supporting
  collector, a grounded registry resolves every research topic to the
  grounded collector (D5).
- Callers opt in by passing the registry to an engine:
  `EvidenceCollectionEngine(registry=...)`, or
  `ResearchOrchestrator(collection_engine=...)`, both of which already
  accept injection. No orchestrator change is required.

## 7. Placeholder semantics (default path)

Unchanged, and provably so.

- `grounding.py` and `integrations/grounding.py` are never imported by
  `collector.py`, `collectors.py`, `dispatcher.py`, `execution.py`,
  `orchestrator.py`, or `research_loop.py`.
- Adding names to `research/__init__.py` must not change an existing
  binding and must not perform I/O at import time.
- `COLLECTOR_REGISTRY` remains `CollectorRegistry(DEFAULT_COLLECTORS)`.
- Proof obligation: the research baseline digests and the V1 evidence
  golden are byte-identical before and after (Section 11).

## 8. Grounded semantics

### 8.1 `GroundedCollector`

```python
class GroundedCollector(EvidenceCollector):
    collector_id: ClassVar[str] = "grounded_collector"
    display_name: ClassVar[str] = "Grounded evidence"
    description: ClassVar[str] = (
        "Evidence grounded in documents collected by the evidence subsystem."
    )
    supported_topics: ClassVar[tuple[str, ...]] = tuple(all_topic_ids())

    def __init__(
        self,
        *,
        source: DocumentSource,
        projector: ClaimProjector,
        policy: GroundingPolicy,
        fallback: EvidenceCollector | None = None,
    ) -> None: ...

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]: ...
```

`collect` is synchronous and performs no trust computation and no I/O.

### 8.2 `collect` algorithm (normative)

Given `task`, `company_name`, `website`:

1. Validate input exactly as `PlaceholderCollector.collect`: reject a
   non-`ResearchTask` and a blank `company_name` with
   `InvalidCollectionInputError`; call `enforce_supported(self,
   task.topic_id)`.
2. `docs = source.documents(company_name=company_name, website=website)`.
3. Keep documents whose overall trust `>= min_document_trust` (`0.0` when
   a document has no metadata or no trust score); sort by document `id`;
   truncate to `max_documents`.
4. `claims = projector.project(task, documents, company_name=...)`.
5. Drop claims below `min_claim_confidence`. Keep only `document_ids`
   that are in the selected set; drop a claim whose cited ids are all
   unknown. A claim with no cited ids is linked to every selected
   document.
6. Sort claims by `(category, claim, confidence, document_ids)`; truncate
   to `max_claims_per_task`.
7. If no claims remain, apply `policy.on_missing_document`:
   `FALLBACK` calls the configured fallback collector (or `SKIP` when
   none is set); `SKIP` returns `()`; `ERROR` raises `InvalidEvidence`.
8. Otherwise build one grounded `Evidence` per claim (Section 8.4) and
   return the tuple sorted by `evidence_id`.

Determinism: for a fixed `(task, company_name, website, policy, source,
projector)`, the output tuple and every `evidence_id` are identical
across runs and processes.

### 8.3 `ExtractionClaimProjector` (delegating, no invented mapping)

```python
class ExtractionClaimProjector:
    def __init__(self, *, extractor: CompositeExtractor | None = None) -> None: ...

    def project(
        self,
        task: ResearchTask,
        documents: tuple[EvidenceDocument, ...],
        *,
        company_name: str,
    ) -> tuple[ProjectedClaim, ...]: ...
```

Behaviour (binding):

- Builds an `EvidenceBundle` from `documents`, minimal `Startup`
  (name = `company_name`, description = joined document text) and
  `CollectedData` shells, and calls
  `CompositeExtractor.extract(startup, collected_data, evidence_bundle)`
  exactly once per `project` call.
- Projects the resulting `ExtractedFeatures` into `ProjectedClaim`s by
  iterating `ExtractedFeatures.model_fields` in definition order and
  emitting one claim per **populated** field (a field whose value differs
  from its declared default; `None`/empty values are unpopulated).
  `category` is the field name; the claim text is a deterministic
  rendering of the value; `document_ids` is the full selected document
  set.
- Defines **no** topic-to-field mapping and **no** extraction heuristics.
  It does not score, rank, or recompute confidence or trust. It reads
  documents only as the values the source produced.

The projector is generic on purpose: because extraction is
company-scoped, any criterion that selected *which* features matter for a
topic would be an invented mapping, which the approved decision forbids.
Field selection is therefore "all populated fields", and
`GroundingPolicy.max_claims_per_task` bounds the result.

### 8.4 Building grounded `Evidence` (normative)

For a surviving `claim` supported by documents `supporting`:

1. Reference document = supporting document with the highest
  `metadata.trust_score.overall`, ties broken by document `id` (all
  selected documents are referenced when trust is absent).
2. `reference = EvidenceReference(source_category=<document_type or
   page_type>, source_identifier=<url host>, url=str(ref.url),
   description=f"Grounded in document {ref.id}.")`.
3. Import trust; never compute it:
   `trust = ref.metadata.trust_score` (a `TrustScore`) or `None`.
4. Build one `EvidenceCitation`:
   `claim=claim.claim`, `domain=task.topic_id`, `category=claim.category`,
   `source_document_ids=list(claim.document_ids)`,
   `source_urls=[str(d.url) for d in supporting]`,
   `best_trust_score=round(trust.overall, 4) if trust else 0.0`,
   `citation_text=format_citation(...)` (reused from
   `evidence.citation`), `provider=collector_id`.
5. Derive ids through the WI-2 bridge only:
   `ids = citation_ids((citation,))`,
   `doc_ids = document_ids_for((citation,))`.
6. Construct `Evidence(..., evidence_id_version=GROUNDED_EVIDENCE_ID_VERSION,
   document_ids=doc_ids, citation_ids=ids, trust_score=trust)`. The model
   validates the grounded/version invariant.

`GroundingPolicy` caps apply before construction, so identity is a
function of the surviving claims only.

### 8.4.1 Design decision D6 - synchronous bridging (settled)

The evidence orchestrator is async. The reviewed bridge
`predictron_engine/evidence/runner.py:collect_evidence_sync` exists for
this. **Decision:** the live adapter uses `collect_evidence_sync`; the
replay source does not run the orchestrator at all (Section 9). No
`asyncio.run` is introduced by this work item.

### 8.5 `SyncEvidenceSource` (live adapter, outside research)

Lives in `predictron_engine/integrations/grounding.py`.

```python
class SyncEvidenceSource:
    def __init__(
        self,
        orchestrator: EvidenceOrchestratorProtocol | EvidenceCollectorFactory,
    ) -> None: ...

    def documents(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[EvidenceDocument, ...]: ...
```

- Runs `collect_evidence_sync(...)` and returns `tuple(bundle.documents)`.
- Empty `website` returns `()` (the bridge already returns an empty
  bundle; no URL is fabricated).
- Documents are the orchestrator's enriched documents, so
  `metadata.trust_score` is populated by the evidence subsystem, not by
  research.
- It satisfies `DocumentSource` structurally and does not change the
  behavior of any existing module.

## 9. Replay requirements

`ReplayDocumentSource` (inside `research`) is the offline, deterministic
source and the primary test vehicle.

```python
class ReplayDocumentSource:
    def __init__(self, dataset: str | Path) -> None: ...

    def documents(
        self,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[EvidenceDocument, ...]: ...
```

- **R1.** Resolve `dataset` with
  `evidence.replay.dataset.resolve_dataset_path` for strings, or use a
  `Path` verbatim. No second resolver.
- **R2.** Perform **no network access**.
- **R3 (chosen).** Load with `load_corpus` + `rebuild_bundle` and return
  the bundle's already-enriched documents directly. The corpus stores the
  full post-enrichment bundle, so `metadata.trust_score` is present
  without running the orchestrator. This keeps the source offline and
  free of async bridging, preserving the purity of the research package.
  (The provider-plus-orchestrator alternative was rejected because it
  reintroduces asynchronous evidence collection inside `research`.)
- **R4.** Determinism: identical corpus and policy produce identical
  documents, claims, and evidence ids, across runs and processes.
- **R5.** A missing or malformed corpus surfaces as
  `EvidenceCorpusError` from the evidence subsystem.

## 10. Compatibility guarantees

- **C1.** No existing public signature changes.
- **C2.** New modules and exports are additive only.
- **C3.** No schema version changes; no new schema version introduced.
- **C4.** All existing content-addressed identifiers keep their values.
- **C5.** Version `"1"` `Evidence.to_dict()` stays byte-identical; new
  keys appear only for version `"2"` (WI-2).
- **C6.** Protocol modules import no concrete runtime types.
- **C7.** The research baseline and V1 evidence goldens are
  byte-identical in placeholder mode.
- **C8.** `tests/test_load.py` stays excluded from full runs; the rest of
  the suite passes.

## 11. Acceptance criteria

- **AC1.** `research/grounding.py`, `integrations/grounding.py`, and
  `grounded_collector_registry` exist and are importable as specified.
- **AC2.** `DocumentSource`/`ClaimProjector` are `runtime_checkable`
  Protocols; implementations satisfy them structurally.
- **AC3.** `GroundingPolicy` rejects invalid construction (`max_documents
  < 1`, `max_claims_per_task < 1`, out-of-range trust/confidence, wrong
  enum type).
- **AC4.** Grounded items have `evidence_id_version == "2"` with
  non-empty `document_ids` and `citation_ids` derived from the source.
- **AC5.** `trust_score` equals the supporting document's
  `metadata.trust_score` object, not a recomputed value.
- **AC6.** No file under `predictron_engine/research/` imports or calls
  `compute_document_trust`.
- **AC7.** `grounded_collector_registry` returns a new registry and never
  mutates `COLLECTOR_REGISTRY`.
- **AC8.** Placeholder-mode research baseline and V1 evidence golden
  compare equal.
- **AC9.** Replay-backed grounded output is deterministic.
- **AC10.** `on_missing_document` yields the three distinct outcomes.
- **AC11.** Reverting to the default registry reproduces baseline
  `collection_id`/`report_id`/`validation_id`.
- **AC12.** `ExtractionClaimProjector` defines no topic-to-field mapping
  and no extraction logic; it delegates to `CompositeExtractor` and
  projects populated `ExtractedFeatures` fields only.
- **AC13.** `ruff`, `mypy` (configured targets),
  `tools.check_canonical_invariant`, `pytest tests/research -q`, and full
  `pytest tests --ignore=tests/test_load.py -q` all pass.

## 12. Required tests

`tests/research/test_grounding.py`, covering: protocol structural
conformance; `GroundingPolicy` defaults and validation; `ProjectedClaim`
validation; `ReplayDocumentSource` offline enriched output and
determinism and corpus error; `ExtractionClaimProjector` delegation and
populated-field projection with document linkage; grounded evidence
version/ids/trust; determinism; every policy cap; unknown-document drop;
`FALLBACK`/`SKIP`/`ERROR`; registry isolation and full-topic resolution;
the no-trust-recompute guard; and placeholder-mode evidence remaining
version `"1"`.

## 13. Invariants

Inherited (must not be weakened): I1 canonical hashing; I2 versioned
identity; I3 research never computes trust; I4 structural ports; I5 loop
depends on `ResearchExecutorPort`; I6 determinism.

New (strengthenings, no ADR): I7 grounding is opt-in and the default
registry is immutable; I8 `GroundedCollector.collect` is synchronous and
side-effect-free w.r.t. research state; I9 grounded identity depends only
on task, company, website, policy, source documents, and projector
output.

## 14. ADR status

**No ADR is required.** With D7b adopted, `predictron_engine/research/`
performs no I/O: the live adapter is outside the package and the replay
source reads only committed, deterministic artifacts. The "no I/O / no
live evidence" invariant is preserved, not relaxed.

## 15. Alternatives considered (record)

| # | Question | Chosen | Alternative(s) | Reason |
| --- | --- | --- | --- | --- |
| D1 | Source granularity | company-scoped | task-scoped | matches evidence bundle shape |
| D2 | Projector scope | `(task, documents)` | single document | whole-set delegation |
| D3 | Claim carries reference | no | yes | keep projector minimal |
| D3b | Policy holds deps | no | yes | policy stays pure data |
| D5 | Registry shape | one collector, all topics | per-topic | `supported_topics` is a `ClassVar` |
| D5c | Extraction reuse | delegate, project populated fields | lexical; knowledge subsystem | no invented mapping (Section 0) |
| D5d | Citation build | direct `EvidenceCitation` | `build_citation` via `EvidenceItem` | avoids artificial `EvidenceItem` |
| D6 | Sync bridge | `collect_evidence_sync` | `asyncio.run` | reuses reviewed bridge |
| D7 | Replay mechanism | `rebuild_bundle` | provider + orchestrator | keeps research offline/pure |
| D7b | Live adapter location | outside `research` | inside (ADR) | preserves purity, no ADR |
