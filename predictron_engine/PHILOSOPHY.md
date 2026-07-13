# Predictron Engine — Design Philosophy

> **Version:** 0.6.5
> **Last updated:** 2026-07-12

This document captures the foundational beliefs that guide every design decision
in the Predictron Engine. It is not a coding standard — those live in
`ARCHITECTURE.md`. This is the *why* behind the structure.

---

## Core Belief

**Every judgment about a startup should be traceable to a specific fact,
explainable in plain language, and reversible when the facts change.**

This belief drives every architectural decision in the system. If a conclusion
cannot be traced to its source, it does not belong in the engine. If a judgment
cannot be explained to a non-technical stakeholder, it is not ready. If changing
one input does not cascade cleanly through the pipeline, the architecture has
failed.

---

## Principles

### 1. Intelligence Is a Pipeline, Not a Model

The engine does not have a single "brain." It has a sequence of independent
stages, each with a single responsibility. Normalization does not reason.
Extraction does not score. Scoring does not recommend. This separation is
not bureaucratic — it is what allows any stage to be replaced, upgraded, or
removed without breaking the others.

A pipeline of simple, replaceable stages will always be more maintainable
than a monolithic model that does everything at once.

### 2. Facts Before Opinions

The pipeline enforces a strict ordering: collect facts, then extract features,
then gather evidence, then reason, then evaluate, then score, then recommend.
You cannot score what you have not observed. You cannot recommend what you have
not scored. You cannot score what you have not evaluated. This ordering prevents
premature conclusions and ensures every opinion is grounded in verifiable data.

The reasoning layer produces *observations*, not *judgments*. An observation
says "the market presents these factors." A judgment says "this is a good
investment." The engine produces observations. Judgment is the investor's job.

The evaluation layer produces *assessments*, not *scores*. An assessment says
"here is what the observations collectively tell us about this dimension,
with supporting evidence and rationale." A score says "this dimension is 75/100."
The engine produces structured assessments. Numerical scoring is a separate,
replaceable layer that follows evaluation.

### 3. Replaceability Over Reusability

Every module is designed to be *replaceable*, not just reusable. The protocol-based
interfaces mean that a new implementation does not need to subclass anything,
inherit from anything, or even know about the existing implementation. It just
needs to satisfy the same method signature. This is structural subtyping — the
system cares about what things *do*, not what they *are*.

This matters because the system will evolve. Today's placeholder scorer will
become tomorrow's proprietary algorithm. Today's keyword classifier will become
tomorrow's NLP model. The architecture should welcome these changes, not resist
them.

### 4. Explainability Is Not Optional

Every observation carries its evidence. Every score carries its rationale. Every
recommendation carries its reasoning. The system does not produce black-box
outputs. If a stakeholder asks "why did you say this market is promising?" the
system can answer: "because the industry evidence includes factors X, Y, Z, and
the reasoning rule MarketContextRule produced this observation with confidence C."

This traceability is not a feature — it is a requirement. An unexplainable
conclusion is indistinguishable from a guess.

### 5. Deterministic First, Intelligent Later

Every stage begins as a deterministic placeholder. The keyword classifier
classifies the same input the same way every time. The scoring engine assigns
the same score for the same features. This determinism makes the system
testable, debuggable, and predictable.

Intelligence — machine learning, probabilistic reasoning, adaptive scoring —
will be added later, but always as a *replacement* for a deterministic stage,
never as a bypass of the pipeline architecture. The system proves it works
with simple rules first, then upgrades to smarter rules when they are ready.

### 6. Knowledge Is Data, Not Logic

The `knowledge/` package contains enumerations, keyword mappings, and lookup
tables. It defines what industries exist, what stages mean, and what keywords
signal a particular business model. It does not contain algorithms, scoring
formulas, or decision trees.

This separation means that changing a taxonomy definition (adding an industry,
renaming a stage) requires updating one file, not scattering changes across
extraction, reasoning, and scoring modules.

### 7. Infrastructure and Intelligence Are Separate Worlds

The InveX backend (FastAPI, SQLAlchemy, Docker) is infrastructure. The
Predictron Engine (reasoning, scoring, recommendations) is intelligence.
They communicate through a thin adapter layer. Neither knows about the
other's implementation.

This separation exists because infrastructure evolves at a different rate
for different reasons than intelligence. The API may move from REST to gRPC.
The engine may gain real-time learning capabilities. Neither change should
require modifying the other.

### 8. Every Component Accepts Its Dependencies

No module instantiates its own dependencies. Every module receives its
 collaborators through constructor injection. This makes the system
 testable (inject mocks), configurable (inject different implementations),
 and auditable (see exactly which components are in use).

The constructor is the system's wiring diagram. Reading it tells you
everything the module depends on.

---

## What We Are Not Building

- **We are not building a recommendation engine.** We are building an
  intelligence pipeline that produces recommendations as one of its outputs.
  Recommendations are domain-stratified, explainable, and deterministic —
  each strategy owns exactly one decision domain and produces structured
  output with traceable supporting observations and assessments.

- **We are not building a scoring model.** We are building a framework
  where scoring is one replaceable stage among many.

- **We are not building an evaluation engine.** We are building a framework
  where structured dimension assessments are one replaceable stage. Evaluation
  is about synthesis and explanation, not scoring or prediction.

- **We are not building a black box.** Every pipeline decision is traceable
  from final recommendation back to original startup data. The validation and
  explainability framework exists so that no output is accepted on faith —
  every recommendation can be audited through its full chain of contributors.

- **We are building for the next decade.** Every design decision should
  still make sense when the system has 10x more rules, 10x more evidence
  sources, and 10x more analysis dimensions.

- **We are not optimizing for speed.** We are optimizing for correctness,
  explainability, and maintainability. Speed will come from infrastructure
  when it is needed.

---

## The Contract

Every module in the Predictron Engine makes the same promise:

> *I will transform my specific input into my specific output. I will not
> touch anything that is not mine. If I fail, I will fail loudly and let
> someone else handle it.*

This is the contract that makes the system composable. Breaking any module's
contract breaks only that module, not the entire pipeline.

---

*This philosophy evolves as the system matures. When a principle conflicts
with a practical constraint, the principle wins — unless the principle
itself is wrong, in which case we update this document and explain why.*
