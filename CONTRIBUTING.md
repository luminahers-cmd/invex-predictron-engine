# Contributing to Predictron Engine

Thank you for your interest in contributing to Predictron Engine.

Predictron Engine is a modular, explainable venture intelligence system. Contributions should preserve that focus and keep changes easy to test, review, and maintain.

## Design Principles

All contributions should align with the project's core principles:

- Explainability
- Deterministic behavior
- Modular architecture
- Evidence-first reasoning
- Continuous evaluation

## Architectural Invariants

The following rules are **not negotiable style preferences**. Each one
exists because breaking it produced a bug that was invisible at the time
it was written — in every case, the code looked reasonable in review and
the defect only surfaced later as an unexplained discrepancy between two
runs.

CI enforces what it can mechanically. This section exists so that a
future contributor understands *why* a check exists, and can tell the
difference between a rule that should be satisfied and a rule that
should be reconsidered at the architectural level.

### 1. Content-addressed identifiers go through `predictron_engine.canonical`

Every content-addressed identifier produced anywhere in the repository
must ultimately be computed through `predictron_engine.canonical`
(`predictron_engine/canonical.py`). No module may introduce a new
canonical JSON serialization or digest implementation. Wrapper APIs are
permitted only if they delegate to `predictron_engine.canonical`.

Use `content_digest` for a structure, `text_digest` for a string you have
already encoded, and `canonical_json` when you need the bytes themselves.

**Why:** the repository once had eleven compact-canonical call sites
speaking three dialects at three digest widths. Nothing about that looked
wrong to a reader — each site was a reasonable `hashlib.sha256` over
`json.dumps(..., sort_keys=True)`. The failure was invisible: two
subsystems encoding the same logical fact produced two different digests,
and only a replay run comparing them would ever notice.

Two encoding details are load-bearing rather than cosmetic:

- **`ensure_ascii=True`.** `ensure_ascii=False` yields different bytes
  for the same content as soon as a payload contains a non-ASCII
  character. Two predictions differing only in an accented `company_name`
  would hash incompatibly with every other canonical id in the engine.
- **No `default=` fallback.** A digest that accepts a lossy encoding of
  its input is worse than one that refuses, because the persisted bytes
  and the hashed bytes can then disagree with no error anywhere. A
  non-JSON-native payload raises `TypeError`.

**Enforced by:** `python -m tools.check_canonical_invariant`, an AST check
that runs in CI alongside ruff and mypy.

**Any future exception requires all three of:**

1. a documented architectural justification,
2. an explicit `Suppression` with a reason in
   `tools/check_canonical_invariant.py`,
3. reviewer approval.

A suppression without a written reason is itself a build failure, because
the reason is the part a reviewer actually needs to read.

### 2. Replay determinism

Given the same inputs, the pipeline must produce byte-identical output.
Nothing in the core may depend on wall-clock time, network order,
iteration order of an unordered collection, locale, or process-random
salted values.

**Why:** replay is the only mechanism by which a stored identifier can be
proved reproducible. An identifier that cannot be re-derived is an
identifier that cannot be audited.

Practical consequences: sort before iterating where order matters, pass a
fixed anchor date in tests, and never let `datetime.now()` reach a
content-addressed payload. Timestamps belong in explicitly informational
fields. `tests/test_load.py` measures latency and is therefore the one
test file excluded from the determinism guarantee.

### 3. Protocol boundaries, not concrete dependencies

Subsystems depend on `Protocol` interfaces, not concrete
implementations. Persistence, execution, and transport are reached
through an injected port.

**Why:** the reason a change is hard to review is usually that you cannot
tell what a component actually touches. Naming the dependency narrows the
surface a reader must hold in their head, and it is what makes a backend
swappable without touching domain logic.

The `research` executor hook is the current example: it is resolved via
`getattr(self.orchestrator, "execute", None)` so the orchestration seam
can be tightened later without breaking existing callers.

### 4. Evidence, provenance, and trust stay distinct

These are three separate responsibilities and are not to be merged:

| Concept | Answers |
| --- | --- |
| `Evidence` (research) | what was claimed, and about what |
| `EvidenceDocument` (evidence) | what was fetched, from where, over which transport |
| `ProvenanceRecord` | chain of custody for one document |
| `TrustScore` | how much this source is worth |

**Why:** merging them makes it impossible to re-score trust without
re-collecting evidence, and impossible to re-fetch a document without
invalidating the claims that cited it. Coverage in particular must be
computed from *validated extracted claims with non-empty citations and
sufficient trust* — never from the mere presence of retrieved documents.
A system that counts a document as coverage will report high confidence
for a company it knows nothing about.

### 5. Typed, strict, and dependency-light

`mypy` runs in `strict` mode over `predictron_engine`, `app`,
`benchmarks`, `predictron_sdk`, and `tools`. Public boundaries are
explicit `Protocol` or frozen model types rather than untyped dictionaries
or `Any`.

**Why:** the strict-mode escape hatches (`# type: ignore`,
`disallow_untyped_defs = false`) are already in use for `app` and
`benchmarks`. Keeping the core clean is what stops that boundary from
eroding outward.

### 6. Additive, versioned changes to persisted artifacts

Anything that is serialized and persisted carries a `*_SCHEMA_VERSION`.
Identifiers are content-addressed, so a change to what goes *into* a
digest silently changes the resulting identifier.

**Why:** stored identifiers outlive the code that produced them. When a
digest's input changes, previously stored records stop verifying, and the
only safe recovery is the original payload. Change the input, and you must
treat the old records as written by a different system version.

### Evolving an invariant

Architectural invariants may only be **strengthened or clarified**.

*Strengthening* closes a gap the rule did not cover. *Clarifying* removes
ambiguity in what the rule already requires. Neither needs a record: the
invariant only becomes more specific, and the reasoning is additive.

**Relaxing or removing an invariant requires an Architecture Decision
Record.**

A relaxation is any change that makes an invariant *permissive* where it
was previously *binding* — a broader suppression, a narrowed scope, a
carved-out subsystem, or outright deletion. See
[`docs/adr/README.md`](docs/adr/README.md) for the process, and
[`docs/adr/TEMPLATE.md`](docs/adr/TEMPLATE.md) for the required
structure.

An ADR must explain:

- the motivation — the concrete problem the relaxation solves, not the
  cost of the invariant in the abstract;
- the alternatives considered, including at least one that preserved the
  invariant, and why it did not work;
- the migration strategy;
- the compatibility impact — what breaks, what stops verifying, what
  stored records become unreadable;
- why the previous invariant is no longer appropriate.

That last item is the one that matters. If the only honest answer is
that the invariant is inconvenient, the ADR should say so and the
reviewer should reject it — the better resolution in that case is to
strengthen or clarify the invariant instead.

This policy exists because invariants erode by accumulation rather than
by reversal. A rule nobody can remove quietly becomes a rule nobody can
explain, and an unexplained rule is enforced by whoever happens to be
reading the log that day.

## Development Workflow

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Add or update tests where appropriate.
5. Update documentation if the change affects behavior or usage.
6. Run the verified checks before opening a pull request:
	- `python -m pytest tests`
	- `python -m ruff check`
	- `python -m mypy`
	- `python -m tools.check_canonical_invariant`

## Coding Standards

Please:

- Write clear, maintainable code.
- Keep components modular.
- Favor explicit behavior over implicit assumptions.
- Preserve backward compatibility whenever practical.
- Document significant architectural decisions.
- Prefer adapting an existing abstraction over adding a parallel one. A
  second implementation of an existing concept is a request for
  reconciliation, and reconciliation is more expensive than reuse.

## Testing

Before opening a pull request, verify that:

- Unit tests pass successfully with `python -m pytest tests`.
- Ruff checks pass with `python -m ruff check`.
- mypy passes with `python -m mypy`.
- The canonical hashing invariant holds with
  `python -m tools.check_canonical_invariant`.
- Existing functionality remains unaffected.
- Documentation reflects the implemented changes where relevant.

When changing a content-addressed identifier or the payload that feeds
one, capture the previous values and assert they are unchanged. A test
that only checks a digest is still self-consistent will pass happily
while every stored identifier in the system silently moves.

## Pull Requests

Each pull request should include:

- A concise summary of the change.
- Motivation for the implementation.
- Relevant testing information.
- References to related issues where applicable.
- Explicit reviewer approval if the change adds or alters an
  architectural invariant, or requests a suppression.

## Code of Conduct

Please communicate respectfully and constructively.

The goal of the project is to encourage collaborative engineering and continuous improvement.
