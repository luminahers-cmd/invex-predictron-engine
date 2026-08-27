# Architecture Verification Report — Predictron Engine (Sprints 4A–8)

**Type:** Read-only verification (code inspection only)
**Scope:** `predictron_engine/`, `app/`, `tests/`, `engine.py`
**Baseline doc:** `docs/architecture.md` (v0.6.5 design) — **stale relative to code**
**Actual engine version:** `ENGINE_VERSION = "0.12.1"` (in `engine.py:70` and `report_builder.py:41`)
**Inspected git history:** 100 commits, sprint tags 4A/4B/4C/5A/5B/5C/6A/7A/7B (6B/6C/8 work folded into 7A/7B, no standalone commits)
**Date:** 2026-08-27

> Note on sprint bookkeeping: git log has no explicit `6B`, `6C`, or `8` commits.
> The Sprint 6B calibration and 6C synthesis work is folded into the 7A commit;
> the "Sprint 8" calibration improvements live in `decision/calibration_sprint8.py`,
> added during 7A/7B. This file is **not wired into the pipeline** (see §5).

---

## 1. Current Pipeline (implemented flow)

Confirmed by direct inspection of `PredictronEngine.analyze()` (`engine.py:127`) and
`__init__` constructor defaults (`engine.py:96`). The *actual* stage order is:

```
Stage   Module (default impl)              Output object                     engine.py line
------  ---------------------------------  ---------------------------------  -------------
 1      DefaultNormalizer.normalize        Startup                           :150
 2      DefaultDataCollector.collect       CollectedData                     :153
 3      EvidenceOrchestrator (sync)        EvidenceBundle (web crawl)        :156
        -> AnalysisContext holds bundle    AnalysisContext                    :158
 4      CompositeExtractor.extract         ExtractedFeatures (bundle-aware)  :165
 5      DefaultEvidenceEngine.gather       EvidenceSet (knowledge providers) :170
 6      DefaultReasoningEngine.reason      list[Observation]                 :173
 7      CompositeEvaluator.evaluate        EvaluationResult                  :176
 8      DefaultScoringEngine.score         list[ScoreResult]                 :181
 8b     compute_investment_readiness       InvestmentReadiness               :185
 9      CompositeRecommendationEngine      list[Recommendation]              :190
10      DefaultConfidenceEngine.assess     list[ConfidenceAssessment]        :195
11      DefaultDecisionEngine.decide       InvestmentDecision                :200
11b     decision.calibration               DecisionConfidence + risk         :212
11c     DecisionSynthesisEngine.synthesize DecisionSynthesis                 :230
12      DefaultReportBuilder.build         Report                            :244
```

### Key architectural fact: two independent evidence pipelines
The web-crawled `EvidenceBundle` (Stage 3) and the knowledge-provider `EvidenceSet`
(Stage 5) are **separate and do not merge**:
- The **bundle** feeds only: extraction (`:165-167`, via `context.evidence_bundle`),
  decision-confidence calibration (`:213`), calibration summary (`:220`), evidence
  collection metadata (`:255`), and synthesis.
- The **evidence set** (`evidence_set.items`) is what flows into reasoning (`:173`),
  evaluation (`:177`), and the report (`:247`).
- Web citations are therefore **not** directly part of the reasoning evidence;
  they only influence features and calibration. This is intentional per code comments
  but is a notable separation a reader should be aware of.

### Integration point (app layer)
- `app/services/analysis.py:43` runs `engine.analyze(request)` in a thread pool via
  `asyncio.to_thread`, then `_report_to_response` maps `Report` → API response.
- `app/adapters/predictron_adapter.py` is a **stub/mock** that does HTTP to a
  `PREDICTRON_ENGINE_URL` and falls back to fixed mock scores. It is **not** wired to
  the local `predictron_engine` package; the real engine is invoked only through
  `AnalysisService` (`services/analysis.py`). This is a public-API inconsistency (§7).

---

## 2. Module Inventory

| Module | Purpose | Public API | Responsibility | Integration point | Deps |
|---|---|---|---|---|---|
| `engine.py` | Pipeline orchestrator | `PredictronEngine.analyze / analyze_with_debug` | Sequence stages, DI | Root | all stages |
| `models/report.py` | Canonical report models | `Report, Observation, ScoreResult, DecisionCategory, ConvictionLevel, ...` | Output schema | every stage | models |
| `models/extracted_features.py` | Extracted feature model (100+ fields) | `ExtractedFeatures` | Feature state | extraction→reasoning | models |
| `models/startup.py`, `models/collected_data.py` | Input models | `Startup, CollectedData` | Normalized inputs | normalize→extract | models |
| `ingest/normalizer.py` | Normalize request | `DefaultNormalizer.normalize` | Stage 1 | engine | models |
| `collection/collector.py` | Data collection | `DefaultDataCollector.collect` | Stage 2 | engine | models |
| `evidence/orchestrator.py`, `runner.py`, `models.py` | Web crawl | `EvidenceOrchestrator`, `collect_evidence_sync`, `EvidenceBundle` | Stage 3 | engine | httpx, models |
| `evidence/evidence_engine.py` | Knowledge gather | `DefaultEvidenceEngine.gather` | Stage 5 | engine | assessment, providers |
| `evidence/providers/*` | 6 knowledge providers | provider classes | EvidenceSet gen | evidence_engine | — |
| `extraction/composite.py` | Composite extractor | `CompositeExtractor.extract` | Stage 4 | engine | 10 extractors, `derived` |
| `extraction/extractors/*` | 10 domain extractors | Domain extractor classes | Feature extraction | composite | models |
| `extraction/derived/*` (DerivedMetricsEngine) | Deterministic inference | `derive` | Post-extraction metrics | composite | — |
| `extraction/feature_models.py` | Extract util/signatures | `call_extractor_with_evidence`, `NlpService` | Extractor glue | composite | — |
| `reasoning/composite.py` | Composite reasoner | `CompositeReasoner.reason/with_diagnostics/with_budget` | Rule orchestration | reasoning_engine | rules, context, evidence_backed |
| `reasoning/rules/*` | Reasoning rules | rule classes | Observations | composite | models |
| `reasoning/context.py` | ReasoningContext (6A) | `ReasoningContext` | Evidence+features bundle to rules | composite | models |
| `reasoning/evidence_backed.py` | Observation enrichment | `enrich_observation` | Trust/provenance metadata | composite | models |
| `reasoning/adaptive_budget.py` | Adaptive reasoning budget | `ReasoningBudget` | Skip rules | composite | — |
| `reasoning/contradiction_graph.py` | Contradiction detection | graph builder | Conflict analysis | rules | — |
| `reasoning/progressive_evidence.py` | Progressive evidence | module | Iterative reasoning | rules | — |
| `reasoning/diagnostics.py` | Rule diagnostics | `RuleDiagnostic`, `RuleTimer` | Metrics | composite | — |
| `reasoning/consistency.py` | Consistency report | `build_consistency_report` | Report | composite | — |
| `reasoning/reasoning_engine.py` | Default reasoning engine | `DefaultReasoningEngine.reason` | Stage 6 entry | engine | default rules |
| `evaluation/composite.py` | Composite evaluator | `CompositeEvaluator.evaluate` | Stage 7 | engine | 7 evaluators |
| `evaluation/evaluators/*` | 7 dimension evaluators | evaluator classes | DimensionAssessment | composite | models |
| `evaluation/investment_readiness.py` | Readiness computation | `compute_investment_readiness` | Stage 8b | engine + report_builder | models |
| `scoring/scoring_engine.py` | Scoring | `DefaultScoringEngine.score` | Stage 8 | engine | scorers |
| `recommendations/composite.py` | Recommendation engine | `CompositeRecommendationEngine.recommend` | Stage 9 | engine | strategies/helpers |
| `confidence/confidence_engine.py` | Confidence | `DefaultConfidenceEngine.assess` | Stage 10 | engine | models |
| `decision/decision_engine.py` | Decision synthesis | `DefaultDecisionEngine.decide` | Stage 11 | engine | models, readiness |
| `decision/calibration.py` | Calibration (6B) | `compute_decision_confidence`, `apply_recommendation_risk`, `build_calibration_summary` | Stage 11b | engine | models |
| `decision/calibration_sprint8.py` | ECE calibration (Sprint 8) | `compute_calibration_error`, `detect_overconfidence`, `build_calibration_report`, `CalibrationReport` | **Unused** | **none** | — |
| `synthesis/*` | Decision synthesis (6C) | `DecisionSynthesisEngine.synthesize` | Stage 11c | engine | models, decision |
| `report/report_builder.py` | Report assembly | `DefaultReportBuilder.build` | Stage 12 | engine | all models |
| `validation/*` | Debug/validation (debug path) | `ValidationEngine.validate_report` | `analyze_with_debug` | engine | models |
| `interfaces/protocols.py` | Protocol definitions | protocols | Contracts | all DI | — |

---

## 3. Data Flow (trace each transition)

| Transition | Where | Verdict |
|---|---|---|
| Startup → CollectedData | `engine.py:150` → `:153` | ✅ `normalize` returns Startup; `collect(startup)` returns CollectedData |
| CollectedData → ExtractedFeatures | `engine.py:165` `CompositeExtractor.extract(startup, collected_data, bundle)` | ✅ merged + derived inference |
| ExtractedFeatures → EvidenceSet | `engine.py:170` `evidence.gather(features)` | ✅ returns evidence_set |
| EvidenceSet → Observation | `engine.py:173` `reasoning.reason(features, evidence_set.items)` | ✅ enriched observations with evidence metadata |
| Observation → DimensionAssessment | `engine.py:176` `evaluation.evaluate(features, observations, items)` | ✅ assessments with per-dim confidence |
| DimensionAssessment → InvestmentDecision | `engine.py:200` `decision.decide(...)` | ✅ consumes assessments + readiness.signal_relationships |
| InvestmentDecision → DecisionConfidence | `engine.py:212` `compute_decision_confidence(bundle, observations, assessments, features, scores)` | ✅ calibration output |
| DecisionConfidence → DecisionSynthesis | `engine.py:230` `synthesis.synthesize(...)` | ✅ synthesis aggregates decision + confidence + calibration |
| DecisionSynthesis → Report | `engine.py:244` `report_builder.build(...)`; `:257-259` attaches decision_confidence, calibration_summary, decision_synthesis | ✅ all three attached post-build |

**All ten transitions exist and are executed in the correct order.** Every stage output
is threaded through to the next. Calibration/synthesis are pure aggregations (no recompute).

---

## 4. Verify Every Sprint

| Sprint | Claimed scope | Code state | Evidence |
|---|---|---|---|
| 4A | Core pipeline (normalize→collect→extract→reason→evaluate→score→report) | ✅ Implemented | `engine.py` stages 1,2,4,6,7,8,12 |
| 4B | Evidence collection framework (web) | ✅ Implemented | `evidence/orchestrator.py`, `runner.py`, `models.py`; `_collect_evidence` (:156) |
| 4C | Evidence-aware extraction | ✅ Implemented | `extraction/feature_models.py` `call_extractor_with_evidence`; composite passes bundle (#extract:166, composite.py) |
| 5A | Reasoning layer | ✅ Implemented | `reasoning/reasoning_engine.py`, `composite.py`, `rules/*` |
| 5B | Scoring layer | ✅ Implemented | `scoring/scoring_engine.py` + scorers |
| 5C | Evaluation layer (dimension assessments) | ✅ Implemented | `evaluation/composite.py` (7 evaluators), areas |
| 6A | Reasoning context + evidence-backed enrichment + diagnostics | ✅ Implemented | `reasoning/context.py`, `evidence_backed.py`, `diagnostics.py`; composite `_run` |
| 6B | Decision-confidence calibration | ✅ Implemented | `decision/calibration.py` = `compute_decision_confidence` etc.; engine :212-225 |
| 6C | Decision synthesis | ✅ Implemented | `synthesis/engine.py`, `DecisionSynthesisEngine`; engine :230 |
| 7A | Adaptive budget + contradiction graph + progressive evidence | ✅ Implemented | `reasoning/adaptive_budget.py`, `contradiction_graph.py`, `progressive_evidence.py`; tests present |
| 7B | Investment readiness + reliability/validation | ✅ Implemented | `evaluation/investment_readiness.py`; `validation/*`, `analyze_with_debug` |
| 8 (untracked tag) | ECE calibration / overconfidence | ⚠️ Partially — code exists but **not wired** | `decision/calibration_sprint8.py` imports nowhere; only tested standalone |

**Summary: 11/12 sprint scopes fully implemented in the live pipeline; the "Sprint 8"
ECE calibration is implemented as a standalone module (with its own test) but is dead code
w.r.t. the executed pipeline.**

---

## 5. Architecture Consistency

### Duplicated logic / constants
1. **`ENGINE_VERSION = "0.12.1"` duplicated** as independent literals in `engine.py:70`
   and `report_builder.py:41` (also `validation_engine.py`, `debug_report.py`). They happen
   to match today but are un-linked constants → drift risk (single-source-of-truth violation).
2. **`compute_investment_readiness` invoked from both `engine.py:185` and called again
   inside `report_builder.py`** — the engine computes it once (:185) to feed decision/
   synthesis, but the report builder recomputes it independently. Duplicated computation
   with no shared cache/single source.
3. **Two calibration modules with overlapping purpose** (`decision/calibration.py` vs
   `decision/calibration_sprint8.py`); only `calibration.py` is live.
4. **Two evidence models families**: `evidence/evidence_models.py` (EvidenceItem) vs
   `evidence/models.py` (EvidenceBundle) — intentionally separate but naming is easy to
   confuse; `composite.py` documents both under `TYPE_CHECKING`.

### Overlapping / redundant modules
- `reasoning/trace.py` **and** `reasoning/traceability.py` both trace reasoning;
  `traceability.py` appears redundant to `trace.py`.
- `reasoning/confidence.py` vs `confidence/confidence_engine.py` — two confidence concepts
  (reasoning-level vs conclusion-level). Intended but overlapping.

### Dead / unused code
1. **`decision/calibration_sprint8.py`** — public API (`compute_calibration_error`,
   `detect_overconfidence`, `build_calibration_report`, `CalibrationReport`) is **not
   imported anywhere** in the codebase (grep confirms zero references). Only
   `tests/engine/test_calibration_sprint8.py` exercises it.
2. **`app/adapters/predictron_adapter.py` HTTP path is dead/mock** — the local engine
   exists but the adapter still calls a remote URL / falls back to hardcoded mock scores.
   The real engine is only used via `services/analysis.py`.

### Unused / conflicting models
- `decision/models.py` — loaded via `decision/__init__.py` and `calibration.py`/`synthesis`,
  but the canonical decision types live in `models/report.py` (`InvestmentDecision`,
   `DecisionCategory`, `ConvictionLevel`). Duplicate decision model surface.

### Circular dependencies
- None detected in the live import graph (engine injects all stages; stages import models
  + protocols only).

### Stale comments vs implementation
- **Sprint numbering inconsistency**: docstrings mix two schemes — `composite.py` says
  "Sprint 6A"; `engine.py:227` says "Sprint 6C"; `report_builder.py:18` says "Sprint 13";
  `calibration_sprint8.py:1` says "Sprint 8"; `engine.py:209` says "Sprint 6B". No single
  canonical numbering.
- **`ARCHITECTURE.md` is v0.6.5 while code is ENGINE_VERSION 0.12.1** — the design doc
  describes an older, smaller feature model (`ExtractedFeatures` has evolved past v0.6.5;
  calibration/synthesis/readiness stages postdate the doc).

### Export inconsistencies
- Stage-relevant output classes are defined in `models/report.py` and re-exported through
  several `__init__.py` files; `EvaluationResult`/`DimensionAssessment` live under
  `evaluation/evaluation_models.py` but are consumed by `confidence`, `decision`, and
  `synthesis` — coupling to a non-top-level module path.

### Obsolete APIs
- `predictron_adapter.analyze`/`_call_engine`/`_mock_response` — obsolete given the local
  engine; retained as a stub.
- Legacy `Recommendation` default engine still referenced alongside
  `CompositeRecommendationEngine` (engine uses the composite; the legacy import path remains
  in `recommendations/`).

---

## 6. Test Coverage

`tests/` contains **112 test files**, organized 1:1 with modules:

| Area | Test files | Status |
|---|---|---|
| Engine (`engine.py`) | `test_engine.py`, `test_evidence_integration.py`, `test_explanation_consistency.py`, `test_quantitative_integration.py` | ✅ pipeline & integration covered |
| Normalization | `test_normalizer.py` | ✅ |
| Collection | `test_collector.py`, `test_evidence_collection/*` (cleaner, discover, fetcher, models, orchestrator) | ✅ |
| Evidence engine + providers | `test_evidence/*` (15 files: engine, models, providers, retrieval, provenance, citation, prioritization, search) | ✅ |
| Extraction | `test_extractor.py`, `test_extractors/*` (all 10 extractors + composite + quantitative) | ✅ |
| Derived metrics | `test_derived_metrics.py` | ✅ |
| Reasoning | `test_reasoning/*` (12 files incl. rules, context, consistency, diagnostics, evidence_backed, contradiction, adaptive_budget, progressive_evidence, trace) | ✅ |
| Evaluation | `test_evaluation/*` (composite, evaluators, helpers) | ✅ |
| Scoring | `test_scoring/*` (all scorers + integration) | ✅ |
| Confidence | `test_confidence_engine.py` | ✅ |
| Decision + calibration | `test_decision/*` (models, backward_compat, calibration_components, recommendation_risk, explanations, uncertainty), `test_calibration_sprint8.py` (⚠ dead-code module under test) | ✅ |
| Recommendations | `test_recommendation/*`, `test_recommendation_engine.py` | ✅ |
| Synthesis | `test_synthesis/*` (6 files) | ✅ |
| Validation | `test_validation/*` (engine, validators, trace, explainability, dataset) | ✅ |
| Report | `test_report_builder.py` | ✅ |
| App layer | `test_adapter`, `test_analyze`, `test_auth`, `test_integration_analysis`, `test_service_analysis`, `test_persistence`, `test_rate_limit`, `test_request_id`, `test_health`, `test_openapi`, `test_load`, `test_benchmark`, `test_readiness`, `test_structured_logging`, `test_exception_handling`, `test_config_validation` | ✅ |

**Gaps:**
- No test validates that `calibration_sprint8.py` integrates with the pipeline (correct,
  since it isn't wired — but untested-by-design dead code).
- Coverage of the `evidence_bundle → extraction` path is indirect (via
  `test_evidence_integration.py`); no dedicated test that web citations alter extraction
  output assertions beyond integration smoke.
- `predictron_adapter` HTTP path is not testable live; tests cover the service layer instead.

---

## 7. Public API Audit

- **Primary public entry:** `PredictronEngine.analyze(request)` — stable, used by
  `app/services/analysis.py`.
- **Secondary public:** `PredictronEngine.analyze_with_debug(request)`.
- **Backward compatibility:** `CompositeReasoner` preserves legacy `evaluate(features,
  evidence)` alongside new `evaluate_context(context)` (`composite.py:184`), with
  `test_backward_compat.py` for both reasoning and decision layers → **good**.
- **Deprecated / stale:** `app/adapters/predictron_adapter.py` (mock; the real local engine
  is not routed through it). Its `analyze`/`_call_engine`/`_mock_response` remain the named
  integration point per its docstring, contradicting actual usage.
- **Not wired (dead) public API:** `decision.calibration_sprint8` functions.
- **Engine version:** duplicated constant, not exported from a single canonical location.

---

## 8. Technical Debt

Only actual, code-observed debt (no speculation):

1. **Dead module `decision/calibration_sprint8.py`** — implemented, tested, but never
   wired into `engine.py`. Either integrate (apply `build_calibration_report` into the
   calibration stage) or remove.
2. **Duplicate `ENGINE_VERSION` constant** across 4+ files (single source-of-truth gap).
3. **Duplicate `compute_investment_readiness` computation** (engine + report builder).
4. **Mock/dead `app/adapters/predictron_adapter.py`** — two integration paths to the
   engine; one is a remote-HTTP mock.
5. **Stale `ARCHITECTURE.md` (v0.6.5)** vs code (0.12.1) — design doc no longer reflects
   calibration/synthesis/readiness stages or expanded feature model.
6. **Dual/non-canonical decision models** (`decision/models.py` vs `models/report.py`).
7. **Dual sprint numbering in comments** (4A/5C *and* 6A/6B/6C/8/13) — documentation drift.
8. **Redundant reasoning trace modules** (`trace.py` + `traceability.py`).

---

## 9. Architecture Scorecard

| Criterion | Score (1–10) | Rationale |
|---|---|---|
| **Modularity** | 9 | Every stage is an injected dependency; composable in isolation; clean package boundaries. Penalty for overlapping/duplicate modules (calibration, trace, decision models). |
| **Determinism** | 9 | Deterministic merge (`composite._merge`), derived-inference, calibration & synthesis are pure; rule order is fixed. Penalty: web-crawl evidence introduces run-to-run nondeterminism that flows into features/calibration. |
| **Maintainability** | 8 | Clear DI + protocols; but duplicated version/readiness logic, stale docs, and dead code raise maintenance cost. |
| **Testability** | 9 | Excellent 1:1 module↔test coverage (112 files), integration + backward-compat tests, injected dependencies. |
| **Separation of concerns** | 8 | Single-responsibility stages are clearly separated; weakened by the two overlapping evidence paths, duplicated computation, and redundant modules. |
| **Extensibility** | 9 | Adding an extractor/rule/evaluator requires no engine change (documented in `composite.py`); protocol-based DI. |
| **Consistency** | 6 | Lowest score: duplicated constants, two evidence models, dual sprint numbering, dead-but-tested module, adapter mocking the local engine, stale design doc. |

**Overall:** The implemented system matches the intended architecture across all live
sprints (4A–7B). The deviations are documentation drift, duplicate constants/computation,
and one fully-built-but-unwired Sprint-8 calibration module — all non-blocking, concrete,
and addressable.
