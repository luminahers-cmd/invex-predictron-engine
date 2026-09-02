# Predictron Engine Architecture

> **Version:** 0.12.1 (matches `predictron_engine/version.py`)
> **Status:** Active
> **Last updated:** 2026-09-02

This document is the internal engineering blueprint for the Predictron Engine.
It explains the **current** architecture of the system — the pipeline exactly as
it runs today in `predictron_engine/engine.py`. It is written for future
engineers, AI coding assistants, and contributors who will maintain and extend
this system.

The sections below document only what exists in the code. Nothing is aspirational
or invented.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline](#2-pipeline)
3. [Pipeline Stage Reference](#3-pipeline-stage-reference)
4. [Evidence Flow](#4-evidence-flow)
5. [Models](#5-models)
6. [Evidence Replay](#6-evidence-replay)
7. [Dataset Module](#7-dataset-module)
8. [Intelligence & Observability](#8-intelligence--observability)
9. [Module Map](#9-module-map)
10. [Dependency Rules](#10-dependency-rules)
11. [Design Principles](#11-design-principles)
12. [Extension Guide](#12-extension-guide)

---

## 1. Overview

Predictron Engine is the intelligence layer behind InveX AI. Given a
`StartupAnalysisRequest`, it runs a deterministic, explainable pipeline that
normalizes the input, collects and extracts features, gathers domain evidence,
reasons over that evidence, evaluates and scores each dimension, recommends
actions, computes confidence, arrives at an investment decision, calibrates
that confidence, synthesizes the result, and assembles a structured `Report`.

The engine is modular. Every stage implements a narrow, well-defined interface
and is injected into the `PredictronEngine` constructor. Any stage can be
replaced or extended without modifying the others. The engine is deterministic —
given the same input (and the same offline evidence bundle), it produces the same
output.

---

## 2. Pipeline

The canonical pipeline, declared in the `PredictronEngine` module docstring and
implemented in `analyze()`, is:

```
normalize -> collect -> collect_evidence -> extract -> evidence ->
reason -> evaluate -> score -> recommend -> confidence -> decide ->
calibrate -> synthesize -> build_report
```

The stage numbering used throughout the rest of this document refers to these
pipeline positions:

| # | Stage | Module | Output |
|---|-------|--------|--------|
| 1 | Normalize | `ingest/normalizer.py` | `Startup` |
| 2 | Collect | `collection/collector.py` | `CollectedData` |
| 3 | Collect website evidence | `evidence/runner.py` + `evidence/orchestrator.py` | `EvidenceBundle` |
| 4 | Extract | `extraction/composite.py` | `ExtractedFeatures` |
| 5 | Gather evidence | `evidence/evidence_engine.py` | `list[EvidenceItem]` |
| 6 | Reason | `reasoning/reasoning_engine.py` | `list[Observation]` |
| 7 | Evaluate | `evaluation/composite.py` | `list[DimensionAssessment]` |
| 8 | Score | `scoring/scoring_engine.py` | `list[ScoreResult]` |
| 8b | Investment readiness | `evaluation/investment_readiness.py` | `InvestmentReadiness` |
| 9 | Recommend | `recommendations/composite.py` | `list[Recommendation]` |
| 10 | Confidence | `confidence/confidence_engine.py` | `list[ConfidenceAssessment]` |
| 11 | Decide | `decision/decision_engine.py` | `InvestmentDecision` |
| 11b | Calibrate | `decision/calibration.py` | `DecisionConfidence`, `CalibrationSummary` |
| 11c | Synthesize | `synthesis/engine.py` | `DecisionSynthesis` |
| 12 | Build report | `report/report_builder.py` | `Report` |

### Entry points

`PredictronEngine` exposes two public methods:

- `analyze(request, request_id=None, evidence_bundle=None) -> Report`
  — the production path.
- `analyze_with_debug(request) -> tuple[Report, DebugReport]`
  — runs the identical production pipeline, then augments a `DebugReport` with
  the reasoning contradiction graph, a reasoning trace, and adaptive-budget
  diagnostics via the validation module.

When `evidence_bundle` is supplied to `analyze()`, Stage 3 (live website evidence
collection) is skipped and the supplied bundle is used directly. This is how
deterministic offline replay is achieved without altering any scoring or
reasoning logic.

---

## 3. Pipeline Stage Reference

### Stage 1 — Normalize

- **Module:** `ingest/normalizer.py` — `DefaultNormalizer`
- **Input:** `StartupAnalysisRequest`
- **Output:** `Startup`

Converts API-validated input into the engine's canonical internal `Startup`
model: strips and normalizes strings, converts typed URL objects to plain
strings, preserves raw input, and attaches a UTC normalization timestamp. It
performs no judgment, scoring, or network access.

### Stage 2 — Collect

- **Module:** `collection/collector.py` — `DefaultDataCollector`
- **Input:** `Startup`
- **Output:** `CollectedData`

Gathers and parses structured and unstructured data: URL parsing, description
tokenization, word counts, domain extraction, and metadata computation. Enrichment
signals from injected `DataSource` instances flow through
`CollectedData.enrichment_signals`. It returns `CollectedData` even when all
sources fail (graceful degradation).

### Stage 3 — Collect website evidence

- **Module:** `evidence/runner.py` (`EvidenceOrchestratorProtocol`,
  `EvidenceCollectorFactory`, `collect_evidence_sync`),
  `evidence/orchestrator.py` (`EvidenceOrchestrator`)
- **Input:** `Startup` (name, website)
- **Output:** `EvidenceBundle`

The engine bridges the async evidence collection into the synchronous pipeline via
`collect_evidence_sync()`. The collector passed to the engine is either:

- a **factory** (the production default: `_default_evidence_collector_factory`),
  which returns a fresh `EvidenceOrchestrator` per collection so each concurrent
  analysis gets its own providers/HTTP clients and event loop — no locking needed; or
- an **orchestrator instance**, serialized through a module-level singleton lock
  (used for injected/test instances).

The `EvidenceOrchestrator` runs applicable providers concurrently via
`asyncio.gather`:
- **WebsiteEvidenceProvider** (always) fetches and parses the startup website.
- **SearchEvidenceProvider** (Tavily) is active only when
  `EVIDENCE_SEARCH_ENABLED` is truthy and `TAVILY_API_KEY` is set.
- **ReplayEvidenceProvider** replaces all live providers when evidence replay is
  enabled (see [Evidence Replay](#6-evidence-replay)).

Results are merged in deterministic provider order, then enriched by the Document
Intelligence layer (classification, quality, trust, deduplication).

Evidence collection is best-effort and never fatal. If there is no website, or
collection raises, the engine returns `EvidenceBundle.empty(startup_name)` and
analysis proceeds.

### Stage 4 — Extract

- **Module:** `extraction/composite.py` — `CompositeExtractor`
- **Input:** `(Startup, CollectedData, EvidenceBundle)`
- **Output:** `ExtractedFeatures`

The `CompositeExtractor` orchestrates and merges the output of specialized domain
extractors plus deterministic derived-metric and quantitative inference. Each
extractor produces only its own fields; the composite merges them (scalar fields
overlay last-non-default; list fields union). The `EvidenceBundle` is threaded
through so extraction can consume website evidence.

### Stage 5 — Gather evidence

- **Module:** `evidence/evidence_engine.py` — `DefaultEvidenceEngine`
- **Input:** `ExtractedFeatures`
- **Output:** `list[EvidenceItem]`

Gathers objective domain knowledge providers (industry, business model, stage,
geography, etc.) into a list of `EvidenceItem`s for downstream reasoning.

### Stage 6 — Reason

- **Module:** `reasoning/reasoning_engine.py` — `DefaultReasoningEngine`
- **Input:** `(ExtractedFeatures, list[EvidenceItem], EvidenceBundle)`
- **Output:** `list[Observation]`

Generates explainable observations from features and evidence, threading the
collected `EvidenceBundle` so reasoning consumes trust, provenance, authority,
quality, and citations rather than only extracted features and static evidence.

The reasoning layer also builds a **contradiction graph** exactly once from the
observations and exposes it as `last_contradiction_graph`; this is reused by every
downstream stage (confidence, synthesis) to avoid duplicate contradiction
detection. A consistency summary is exposed as `last_consistency`.

### Stage 7 — Evaluate

- **Module:** `evaluation/composite.py` — `CompositeEvaluator`
- **Input:** `(ExtractedFeatures, list[Observation], list[EvidenceItem])`
- **Output:** `list[DimensionAssessment]`

Translates observations and evidence into structured, explainable dimension
assessments — one per analysis dimension. This is an interpretive synthesis, not
a score.

After evaluation, the engine annexes the matching `ScoreResult.score` into each
`DimensionAssessment.score` (`_annex_scores_to_assessments`), so assessment scores
reflect real scoring output and score-conditional rationale branches are reachable.

### Stage 8 — Score

- **Module:** `scoring/scoring_engine.py` — `DefaultScoringEngine`
- **Input:** `(ExtractedFeatures, list[Observation])`
- **Output:** `list[ScoreResult]`

Assigns a 0–100 score to each analysis dimension with a rationale and supporting
evidence. Scoring logic is fully replaceable via injected `DimensionScorer`s.

### Stage 8b — Investment readiness

- **Module:** `evaluation/investment_readiness.py` — `compute_investment_readiness`
- **Input:** `(ExtractedFeatures, list[Observation], list[ScoreResult], list[DimensionAssessment])`
- **Output:** `InvestmentReadiness`

A pure deterministic computation that weights per-dimension contributions (score
60%, observation confidence 25%, assessment confidence 15%), applies missing-data
calibration (dimensions absent from all three sources are excluded rather than
averaged as neutral), applies reinforcement bonuses/penalties from
`investment_thesis` cross-signal observations, and classifies the result into
levels (`needs_data`, `early`, `developing`, `moderate`, `strong`,
`investment_ready`).

## Stage 9 — Recommend

- **Module:** `recommendations/composite.py` — `CompositeRecommendationEngine`
- **Input:** `(ExtractedFeatures, list[Observation], list[ScoreResult], list[DimensionAssessment])`
- **Output:** `list[Recommendation]`

Produces actionable, domain-stratified recommendations. Each carries
title, description, rationale, supporting observations/assessments, confidence,
action items, and metadata.

### Stage 10 — Confidence

- **Module:** `confidence/confidence_engine.py` — `DefaultConfidenceEngine`
- **Input:** `(ExtractedFeatures, list[Observation], list[ScoreResult], list[DimensionAssessment], contradiction_graph)`
- **Output:** `list[ConfidenceAssessment]`

Assesses per-dimension confidence (0–1) based on data completeness, observation
strength, and assessment confidence, with the reasoning contradiction graph as an
input signal.

### Stage 11 — Decide

- **Module:** `decision/decision_engine.py` — `DefaultDecisionEngine`
- **Input:** `(features, observations, scores, confidence, assessments, signal_relationships, readiness_score, evidence_bundle)`
- **Output:** `InvestmentDecision`

Computes five weighted factors (score 40%, readiness 25%, confidence 15%,
evidence quality 10%, cross-signal 10%), applies data-quality and risk modifiers,
accumulates a composite score (0–100), classifies it into a `DecisionCategory`
(`STRONG_INVEST`, `INVEST`, `WATCH`, `INVESTIGATE_FURTHER`, `PASS`), and derives a
`ConvictionLevel` and a structured `DecisionRationale`. It consumes the canonical
readiness score from Stage 8b and the collected `EvidenceBundle` for genuine
evidence-quality weighting. Fully deterministic.

### Stage 11b — Calibrate

- **Module:** `decision/calibration.py`
- **Input:** existing pipeline outputs
- **Output:** `DecisionConfidence`, `CalibrationSummary`

Three public functions:

- `compute_decision_confidence(...)` builds two independent weighted breakdowns:
  a 7-factor confidence breakdown (evidence trust, evidence confidence, evidence
  agreement, reasoning confidence, evaluator agreement, feature completeness,
  evidence diversity) and a 5-driver uncertainty breakdown (missing evidence,
  conflicting evidence, low trust, low coverage, evaluator disagreement).
- `apply_recommendation_risk(recommendations, decision_confidence)` returns new
  `Recommendation` objects with `expected_confidence`, `expected_uncertainty`,
  and `recommended_action` fields populated.
- `build_calibration_summary(...)` produces a compact report-facing digest.

This stage combines existing pipeline outputs only — no new evidence, reasoning,
or evaluation work.

### Stage 11c — Synthesize

- **Module:** `synthesis/engine.py` — `DecisionSynthesisEngine`
- **Input:** all already-produced outputs (features, observations, assessments,
  scores, recommendations, readiness, decision, decision_confidence,
  calibration_summary, consistency, contradiction_graph)
- **Output:** `DecisionSynthesis`

Aggregates already-produced outputs with no recomputation. Builds:

- an executive summary (fixed-template paragraph + key points) — `summary.py`
- prioritized recommendations (deduplicated, globally ranked, capped at 8) — `priorities.py`
- trade-offs (per-dimension strength-vs-concern tensions, up to 5) — `tradeoffs.py`
- alternative scenarios (improve/worsen/information what-ifs, up to 6) — `scenarios.py`
- a unified risk register (up to 10) — `risks.py`
- a unified opportunity register (up to 8) — `opportunities.py`

and passes through `overall_confidence`, `uncertainty_score`, `confidence_level`,
and `recommended_action`.

### Stage 12 — Build report

- **Module:** `report/report_builder.py` — `DefaultReportBuilder`
- **Input:** all pipeline outputs plus `processing_time_ms`
- **Output:** `Report`

Assembles the final `Report`: startup, features, evidence, observations, dimension
assessments, scores, overall_score (mean), recommendations, confidence,
overall_confidence (mean), investment_readiness, investment_decision,
decision_confidence, calibration_summary, decision_synthesis, signal_relationships
(from readiness), and `AnalysisMetadata` (engine_version, pipeline stages
completed, processing time, timestamp, data completeness, evidence collection
metadata).

---

## 4. Evidence Flow

Two distinct evidence concepts travel through the pipeline:

1. **`EvidenceBundle`** (Stage 3 output, from website collection): raw collected
   website documents, sources, provider runs, and Document Intelligence /
   trust summaries. It is threaded through extraction (Stage 4), reasoning
   (Stage 6), decision (Stage 11), and calibration (Stage 11b).

2. **`EvidenceItem` list** (Stage 5 output, from knowledge gathering): objective
   domain facts used by reasoning and evaluation.

The `AnalysisContext` (in `predictron_engine/context.py`) is a frozen dataclass
carrying `request_id`, `startup`, `evidence_bundle`, and arbitrary `metadata`,
shared across stages for correlation and evidence access.

Evidence collection is first-class in the report: `EvidenceCollectionMetadata`
records website, pages discovered/fetched, successful/failed sources, and
collection time.

---

## 5. Models

All canonical pipeline output models live in `predictron_engine/models/` and are
re-exported from `predictron_engine/models/__init__.py`:
`AnalysisMetadata`, `CollectedData`, `ConfidenceAssessment`, `EvidenceItem`,
`ExtractedFeatures`, `Observation`, `Recommendation`, `Report`, `ScoreResult`,
`Startup`.

`models/report.py` defines the full pipeline surface, including:
- `DecisionCategory` enum: `STRONG_INVEST`, `INVEST`, `WATCH`,
  `INVESTIGATE_FURTHER`, `PASS`
- `ConvictionLevel` enum: `VERY_HIGH`, `HIGH`, `MODERATE`, `LOW`, `VERY_LOW`
- `InvestmentDecision` with `DecisionRationale`
- `InvestmentReadiness`, `SignalRelationship`
- `DecisionConfidence` (with `ConfidenceBreakdown` / `UncertaintyBreakdown`)
- `CalibrationSummary`
- `DecisionSynthesis` with `TradeOff`, `AlternativeScenario`, `RiskItem`,
  `OpportunityItem`, `SynthesisSeverity`
- `EvidenceCollectionMetadata`, `AnalysisMetadata`
- `EvidenceCitation`, `EvidenceItem`, `Observation`, `ScoreResult`,
  `Recommendation`, `ConfidenceAssessment`, `DimensionAssessment`,
  `EvaluationResult`

---

## 6. Evidence Replay

**Module:** `evidence/replay/` (`config.py`, `dataset.py`, `provider.py`)

A deterministic offline replay feature (Sprint P7) for regression testing and CI
without network access.

- **Config:** gated by `EVIDENCE_REPLAY_ENABLED` and `EVIDENCE_REPLAY_DATASET`.
  When both are set, a single `ReplayEvidenceProvider` replaces all live
  providers in the orchestrator.
- **Dataset:** versioned, deterministic JSON corpus files stored under
  `benchmarks/offline_evidence/`. Functions serialize, dump, and reload evidence
  bundles and raw documents.
- **Provider:** `ReplayEvidenceProvider` satisfies the `EvidenceProvider`
  protocol. On collection it loads the corpus, strips Document Intelligence
  metadata from raw documents (so the orchestrator re-enriches deterministically),
  and returns a `ProviderResult`. It never performs network access. The
  `build_replay_provider()` factory returns `None` when replay is disabled,
  preserving byte-identical production behavior.

The benchmark runner also supports offline replay via `--offline-replay` and
`load_case_evidence_bundle()` / `build_replay_engine()` / `load_all_case_evidence_bundles()`,
feeding `evidence_bundle` into `engine.analyze()`.

---

## 7. Dataset Module

**Module:** `predictron_engine/dataset/`

A standalone module for collecting, storing, and evaluating historical startup
prediction data — used for empirical validation of the engine without modifying
engine behavior. It consumes `PredictronEngine` (via `AnalysisPipeline`) but is
never imported by the engine itself.

Public API (exported via `predictron_engine/dataset/__init__.py`):
- `DatasetRecord`, `PredictionSummary` — historical prediction snapshots
- `StartupOutcome`, `OutcomeRecord`, `OutcomeStatus`, `OutcomeVerdict`,
  `FundingEvent` — ground-truth outcomes
- `ImportPipeline`, `ImportSource`, `ImportSourceRegistry`, `RawImportRecord`
  — pluggable data import
- `DatasetStore` — JSON storage backend
- `AnalysisPipeline`, `AnalysisResult`, `AnalysisRun` — runs the engine over records
- `EvaluationPipeline`, `PredictionEvaluation`, `EvaluationMetadata`,
  `EvaluationBatchResult` — evaluation of predictions against outcomes
- `EvaluationMetrics`, `BinaryLabel`, `compute_evaluation_metrics` — metrics
- `DatasetReportBuilder` — deterministic JSON reports
- `ValidationIssue`, `ValidationReport`, `validate_dataset` — integrity checks

A CLI (`predictron-dataset`) exposes subcommands: `import`, `analyze`, `evaluate`,
`verify`, `stats`, `export`. See `docs/dataset-builder.md`.

---

## 8. Intelligence & Observability

**Module:** `predictron_engine/intelligence/`

A self-evaluation/observability layer (Sprint 9) that sits *around* the
deterministic pipeline. It consumes already-produced `Report`s and computes
analytics: benchmark-driven evaluation, regression detection, historical
performance tracking, calibration monitoring (including the `calibration_sprint8`
ECE tooling), rule/recommendation effectiveness, confidence drift, engine quality
metrics, and performance dashboards.

It is **not** wired into the live analysis pipeline or the `app/` layer. It is a
standalone analytics module entered via `IntelligenceAnalyzer().dashboard(...)`
and exercised by `tests/engine/test_intelligence_sprint9.py`.

---

## 9. Module Map

```
predictron_engine/
├── engine.py                     # Pipeline orchestrator (single composition root)
├── version.py                    # Canonical ENGINE_VERSION (0.12.1)
├── context.py                    # AnalysisContext (frozen, shared across stages)
├── models/                       # Canonical models (leaf module)
├── ingest/normalizer.py          # Stage 1
├── collection/collector.py       # Stage 2
├── evidence/
│   ├── runner.py                 # Orchestrator/factory bridge + collect_evidence_sync
│   ├── orchestrator.py           # EvidenceOrchestrator (website + search providers)
│   ├── evidence_engine.py        # Stage 5 knowledge gathering
│   ├── providers/                # Website, search providers
│   └── replay/                   # Offline evidence replay (P7)
├── extraction/
│   ├── composite.py              # CompositeExtractor (Stage 4)
│   ├── extractors/               # Domain extractors
│   ├── derived/                  # Derived metric inference
│   └── quantitative/             # Quantitative feature extraction
├── reasoning/
│   ├── reasoning_engine.py       # Stage 6
│   └── rules/                    # Individual reasoning rules
├── evaluation/
│   ├── composite.py              # Stage 7 evaluators
│   └── investment_readiness.py   # Stage 8b
├── scoring/scoring_engine.py     # Stage 8
├── recommendations/
│   └── composite.py              # Stage 9 strategies
├── confidence/confidence_engine.py  # Stage 10
├── decision/
│   ├── decision_engine.py        # Stage 11
│   ├── calibration.py            # Stage 11b
│   └── models.py                 # DecisionConfidence etc.
├── synthesis/
│   ├── engine.py                 # Stage 11c
│   └── summary.py / priorities.py / tradeoffs.py / scenarios.py /
│       risks.py / opportunities.py
├── report/report_builder.py      # Stage 12
├── validation/                   # DebugReport, analyze_with_debug diagnostics
├── intelligence/                 # Offline analytics (not in live pipeline)
└── dataset/                      # Offline dataset/evaluation tooling (not in live pipeline)
```

---

## 10. Dependency Rules

The engine is a single composition root. Only `predictron_engine/engine.py`
orchestrates all stages. Module boundaries are enforced structurally:

- `models/` is a leaf module — nothing imports back into it except leaf
  definitions; no model imports an engine stage.
- `knowledge/` (taxonomies) is data-only and never imports logic.
- Pipeline stage modules resolve cross-stage dependencies through the engine's
  constructor, never by importing one another.
- `app/` and `predictron_engine/` meet only at the request/response boundary
  (`app.schemas.analysis.StartupAnalysisRequest`) and in the `app/` service layer
  which invokes `PredictronEngine`.

The `PredictronEngine` constructor accepts optional implementations for every
stage: `normalizer`, `collector`, `extractor`, `evidence_collector`, `evidence`,
`reasoning`, `evaluation`, `scoring`, `recommendations`, `confidence`, `decision`,
`report_builder`, and `synthesis`. When omitted, the default implementation is
used. This is the sole extension point for replacing any stage.

---

## 11. Design Principles

These principles are enforced by the code structure:

- **Modularity / single responsibility** — each stage does one thing and is
  independently replaceable and testable.
- **Dependency injection** — every stage receives collaborators via the
  constructor; the engine never hardcodes a stage.
- **Protocol-based interfaces** — stage contracts are `Protocol` classes
  (`@runtime_checkable`); implementations need not inherit.
- **Composition over inheritance** — the engine composes modules.
- **Strong typing** — full type annotations; `mypy --strict`; Pydantic at
  boundaries.
- **Explicit models** — every transformation produces a named Pydantic model.
- **Explainability** — every observation cites evidence, every score carries a
  rationale, every recommendation states why.
- **Deterministic first** — deterministic rules and heuristics; AI/ML can be
  injected behind the same interfaces.
- **Infrastructure/intelligence separation** — `app/` is infrastructure; the
  engine is intelligence.

---

## 12. Extension Guide

### Replacing a stage

Pass a protocol-compatible implementation to the `PredictronEngine` constructor:

```python
engine = PredictronEngine(scoring=MyCustomScoringEngine())
```

No existing code changes; the new stage is tested independently.

### Adding a reasoning rule

Define a class satisfying the `ReasoningRule` protocol and inject it through a
`DefaultReasoningEngine`:

```python
engine = PredictronEngine(
    reasoning=DefaultReasoningEngine(rules=[MyNewRule(), *DEFAULT_RULES])
)
```

### Enabling offline replay

Set `EVIDENCE_REPLAY_ENABLED` and `EVIDENCE_REPLAY_DATASET` to substitute the
live evidence providers with a `ReplayEvidenceProvider`, or pass an
`evidence_bundle` directly to `engine.analyze(request, evidence_bundle=bundle)`
to skip live collection.

### Adding a new analysis dimension

Extend the `AnalysisDimension` taxonomy in `knowledge/`, add its label, provide a
`DimensionScorer`, and inject it via `DefaultScoringEngine`. Update the
evaluation/readiness paths that consume dimension assessments.
