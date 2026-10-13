# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs): short,
permanent records of a decision that affects the system's architecture.

## Why ADRs exist here

The repository has a set of [architectural invariants](../../CONTRIBUTING.md#architectural-invariants)
— non-negotiable rules such as content-addressed identifiers going
through `predictron_engine.canonical`, replay determinism, and protocol
boundaries. CI enforces several of them technically.

Technical enforcement alone is insufficient. A check can tell you *that*
something is forbidden; it cannot tell you what it would cost to stop
forbidding it. The ADR is where that second question gets a permanent,
reviewable answer.

## The evolution policy

**Architectural invariants may only be strengthened or clarified.**

Strengthening means covering a gap: an invariant that named one encoding
dialect now names three. Clarifying means removing ambiguity in what an
invariant already requires. Neither needs an ADR — the rule only ever
gets more specific, and the reasoning is additive.

**Relaxing or removing an invariant requires an ADR.**

If a change makes an invariant *permissive* where it was previously
*binding* — a broader suppression, a narrower scope, a carved-out
subsystem, or outright deletion — that is a relaxation, and it needs a
record. This is deliberate. Without it, an invariant erodes one
reasonable-looking exception at a time, and a year later nobody can
explain why the rule exists at all.

## Required content

An ADR must explain:

- **Motivation** — the concrete problem the relaxation solves. Not the
  cost of the invariant, but the cost of keeping it.
- **Alternatives considered** — at least one option that preserved the
  invariant, and why it did not work.
- **Migration strategy** — how existing code, data, and identifiers
  move to the relaxed regime.
- **Compatibility impact** — what breaks, what stops verifying, what
  stored records become unreadable.
- **Why the previous invariant is no longer appropriate** — the part
  that forces real engagement. If the answer is only that the invariant
  is inconvenient, the ADR should say so and the reviewer should reject
  it.

## Format

Copy [`TEMPLATE.md`](TEMPLATE.md) to `NNNN-short-title.md`, using the
next sequential number. ADRs are not renumbered when earlier ones are
superseded; a superseded ADR points forward to its replacement and stays
in place as the record of what was previously believed.

## Index

No ADRs yet. The invariants they would relax are all active.
