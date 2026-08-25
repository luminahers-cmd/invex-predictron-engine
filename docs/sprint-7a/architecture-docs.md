# Predictron Engine — Architecture Reference

**Date:** 2026-08-25
**Version:** 0.6.5 (Sprint 7A hardened)
**Codebase:** ~35k LOC production, 2726 tests

---

## Overview

The Predictron Engine is a deterministic, 12-stage pipeline that analyzes
startups and produces structured investment reports.  No LLM calls, no
probabilistic behaviour, no network I/O at runtime — all data flows through
Pydantic v2 models and is validated at each stage boundary.

```
normalize → collect → extract → evidence → reasoning → evaluate
   → score → recommend → confidence → decide → calibrate
   → synthesize → build_report
```

---

## Directory Structure

```
predictron_engine/
├── ingest/              # Stage 1 — request normalisation
├── collection/          # Stage 2 — website content collection
├── extraction/          # Stage 3 — feature extraction from collected data
├── evidence/            # Stage 4 — evidence gathering, trust, provenance
│   ├── providers/       #   per-domain evidence providers
│   ├── provenance.py    #   trust scoring & provenance records
│   └── retrieval.py     #   document retrieval & ranking
├── reasoning/           # Stage 5 — rule-based observation generation
│   ├── rules/           #   11 reasoning rules
│   ├── composite.py     #   composite reasoner orchestrator
│   ├── context.py       #   pre-indexed reasoning context
│   └── diagnostics.py   #   per-rule execution diagnostics
├── evaluation/          # Stage 6 — dimension assessment synthesis
│   ├── evaluators/      #   per-dimension evaluators
│   └── evaluation_models.py
├── scoring/             # Stage 7 — quantitative scoring
│   └── scoring_engine.py  # main scoring engine (~1800 lines)
├── recommendation/      # Stage 8 — actionable recommendations
│   └── strategies/      #   per-category recommendation strategies
├── confidence/          # Stage 9 — confidence assessment
│   └── confidence_engine.py
├── decision/            # Stage 10 — threshold-based classification
│   ├── calibration.py   #   calibration & synthesis helpers
│   └── synthesis.py     #   final decision synthesis
├── calibration/         # Stage 11 — score calibration
├── synthesis/           # Stage 12 — report assembly
├── validation/          # Post-pipeline validation layer
│   ├── validators/      #   completeness, consistency, pipeline, report
│   └── explainability.py  # structured stage explanations
├── models/              # Pydantic v2 data contracts
│   ├── startup.py
│   ├── extracted_features.py
│   └── report.py        # Observation, ScoreResult, Recommendation, etc.
├── interfaces/          # Protocol definitions (decorative, not enforced)
├── knowledge/           # Static taxonomies & domain lists
├── index.py             # Pipeline entry point
├── engine.py            # Pipeline orchestrator
└── config.py            # Engine configuration
```

---

## Pipeline Stages (Detail)

### Stage 1 — Normalize (`ingest/normalizer.py`)
Converts an `StartupAnalysisRequest` into a canonical `Startup` model.
Parses URLs, normalises website fields, and validates required inputs.

### Stage 2 — Collect (`collection/collector.py`)
Fetches website content.  Returns a `CollectedData` object with raw text,
structured data, and metadata per page.  A process-wide `_LOCK` serialises
concurrent collection to avoid rate-limit conflicts.

### Stage 3 — Extract (`extraction/`)
Runs domain-specific extractors over collected data to produce an
`ExtractedFeatures` model.  Each extractor writes to a shared features
object.  `data_completeness` is the fraction of populated feature fields.

### Stage 4 — Evidence (`evidence/`)
Gathers evidence items from internal knowledge bases, evidence providers,
and provenance records.  Each `EvidenceItem` carries a domain, category,
statement, source, trust score, and optional provenance record.  Trust
scoring is computed once per document via `provenance.py`.

### Stage 5 — Reasoning (`reasoning/`)
Evaluates 11 rules against features and evidence to produce
`Observation` objects.  The `ReasoningContext` pre-indexes evidence by
domain and documents by provider to avoid repeated scans.  Each rule
receives the context and returns a list of observations.

**Rules:** MarketContextRule, BusinessModelContextRule, StageExpectationRule,
TechnologyContextRule, TeamAssessmentRule, DataQualityRule, RiskIndicatorRule,
CompetitionAssessmentRule, QuantitativeSignalsRule, CrossSignalReasoningRule,
QuantitativeCrossSignalRule.

`DEFAULT_RULES` is a **tuple** (immutable) to prevent accidental mutation.

### Stage 6 — Evaluate (`evaluation/`)
Synthesises observations into `DimensionAssessment` objects — one per
analysed dimension.  Each assessment carries a summary, rationale,
confidence, and references to supporting observations and evidence.

### Stage 7 — Score (`scoring/scoring_engine.py`)
Converts assessments into numeric `ScoreResult` values (0–100) using
weighted aggregation, dampening factors, and quantitative signal scoring.
The scoring engine is the largest module (~1800 lines).

### Stage 8 — Recommend (`recommendation/`)
Generates `Recommendation` objects from scores, assessments, and
observations.  Strategies are category-specific (opportunity, risk, etc.)
and each produces prioritised actions with supporting references.

### Stage 9 — Confidence (`confidence/confidence_engine.py`)
Assesses reliability per dimension.  Combines data completeness,
observation confidence/diversity/coverage, and assessment confidence.
**Factor weights sum to exactly 1.0** (base 0.15 + 0.22 + 0.22 + 0.18
+ 0.12 + 0.11).

### Stage 10 — Decide (`decision/`)
Classifies each dimension into a band (strong/supportive/neutral/concern)
using configurable thresholds.  Produces a `DecisionResult` with
per-dimension classifications and an overall verdict.

### Stage 11 — Calibrate (`calibration/`)
Adjusts scores based on data quality, evidence diversity, and historical
calibration data.  Ensures final scores reflect true confidence levels.

### Stage 12 — Synthesize & Report (`synthesis/`, `engine.py`)
Assembles all stage outputs into a `Report` — the final deliverable.
The report includes metadata, features, observations, assessments,
scores, recommendations, confidence, decisions, and calibration info.

---

## Data Contracts (models/)

All inter-stage data flows use Pydantic v2 `BaseModel` subclasses:

| Model | Stage | Key Fields |
|---|---|---|
| `Startup` | 1→2 | name, website, description |
| `CollectedData` | 2→3 | pages, structured_data |
| `ExtractedFeatures` | 3→5 | industry, business_model, data_completeness |
| `EvidenceItem` | 4→5 | domain, category, statement, trust_score |
| `Observation` | 5→6 | dimension, category, confidence, evidence |
| `DimensionAssessment` | 6→8 | dimension, summary, rationale, confidence |
| `ScoreResult` | 7→8 | dimension, score (0–100) |
| `Recommendation` | 8→12 | category, action, priority, confidence |
| `ConfidenceAssessment` | 9→12 | dimension, confidence, factors |
| `DecisionResult` | 10→12 | per-dimension classifications |
| `Report` | 12 | all of the above |

---

## Validation Layer

Post-pipeline validators check structural integrity without modifying
outputs:

| Validator | What it checks |
|---|---|
| `PipelineValidator` | Required fields, data completeness thresholds |
| `CompletenessValidator` | Unused evidence/observations, dimension coverage gaps |
| `ConsistencyValidator` | Observation conflicts, assessment-score alignment |
| `ReportValidator` | Final report structural integrity |

All findings are `ValidationFinding` objects with severity, category,
and message.  Set iterations in validators are **sorted** for
deterministic output ordering.

---

## Explainability

`ExplanationBuilder` produces `StageExplanation` and `ConclusionTrace`
objects that describe what each stage did, why, and what evidence
supports each conclusion.  All set comprehensions in explanations are
**sorted** for deterministic output.

---

## Determinism Guarantees

1. No `random` module usage anywhere in production code
2. No `datetime.now()` or `time.time()` in data paths
3. `time.perf_counter()` used only for diagnostic wall-clock durations
4. All set comprehensions and set difference iterations are **sorted**
5. `DEFAULT_RULES` is an immutable tuple
6. Float accumulation order is fixed (lists iterated in insertion order)
7. `sorted()` keys use stable tie-breakers (e.g., `str(d.url)`)

---

## Hardening Changes (Sprint 7A)

| File | Change | Reason |
|---|---|---|
| `confidence/confidence_engine.py` | Weights rebalanced to sum to 1.0 | Bug fix: old sum was 1.05 |
| `reasoning/rules/__init__.py` | `DEFAULT_RULES` → tuple | Prevent accidental mutation |
| `validation/validators/completeness_validator.py` | Sorted set iterations | Deterministic finding order |
| `validation/validators/consistency_validator.py` | Sorted set iterations | Deterministic finding order |
| `validation/explainability.py` | Sorted set comprehensions | Deterministic explanation output |
| `confidence/confidence_engine.py` | Sorted category set | Deterministic diversity computation |
| `reasoning/context.py` | Cached `trusted_documents`, `best_source`, `average_trust` | Avoid redundant O(n) scans |
| `reasoning/diagnostics.py` | Removed dead `cited_claims` variable | Dead code elimination |

**Regression tests added:** 13 new tests across 5 test files covering
all hardening changes.

---

## Test Suite

```
2726 passed in 263.76s (0:04:23)
ruff 0.11.12: All checks passed!
```

Tests are organised by module under `tests/engine/`:
- `test_reasoning/` — rules, context, diagnostics, composite
- `test_validation/` — validators, explainability
- `test_confidence_engine.py` — confidence assessment
- `test_scoring_engine.py` — scoring
- `test_recommendation/` — strategies
- `test_decision/` — decision, calibration, synthesis
- `test_evidence/` — evidence providers, trust, provenance
- `test_extraction/` — feature extractors
- `test_ingest/` — normaliser
- `test_collection/` — collector
- `test_benchmarks/` — performance regression benchmarks

---

## Known Technical Debt

These items are documented but intentionally not addressed in Sprint 7A:

1. **Evidence duplication** — domain lists, path tables, host-matching
   exist 3–5× across evidence modules.  Needs consolidation pass.
2. **Scoring engine size** — `_apply_data_dampener` duplicated 5×,
   `_score_observations` duplicated 7×.  Needs refactor to shared helpers.
3. **Dead imports** — `DecisionSynthesis`, `recommendation_fingerprint`
   in synthesis module.  Low priority cleanup.
4. **Protocol enforcement** — 12 protocols in `interfaces/protocols.py`
   are decorative.  Could add runtime checks in debug mode.
5. **Knowledge taxonomies** — 5 dead symbols in knowledge module.
6. **Mutable `DEFAULT_LISTS`** in knowledge modules — same class of
   hazard as `DEFAULT_RULES` (now fixed).
