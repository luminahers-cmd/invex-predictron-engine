# Sprint 7A — Architecture Audit Report

**Date:** 2026-08-25
**Scope:** Complete Predictron Engine pipeline
**Method:** Full source read of every module in `predictron_engine/`, `app/`, `tests/`, `benchmarks/`
**Baseline:** 2713 tests pass, ruff clean

---

## Executive Summary

The Predictron Engine is a well-structured, deterministic startup analysis pipeline with
~35k lines of production code across 12 pipeline stages. The architecture follows
dependency injection throughout, uses Pydantic v2 models for all data contracts, and
maintains clean separation between extraction, reasoning, evaluation, scoring,
recommendation, decision, calibration, synthesis, and reporting.

**Overall grade: B+** — production-quality with identifiable hardening opportunities.

Key findings:
- No import cycles, no hidden randomness, no LLM dependencies
- Evidence layer has the most duplication (domain lists, path tables, URL matching)
- Scoring engine has massive code duplication (5x identical methods)
- Synthesis layer duplicates `_Register` plumbing across 2 files
- All 12 protocols in `interfaces/protocols.py` are decorative (never enforced)
- 5 dead symbols in knowledge taxonomies
- Mutable `DEFAULT_RULES` list is a latent hazard

---

## Stage-by-Stage Findings

### Stage 1: Ingest/Normalizer (`ingest/normalizer.py`)
- **Dead code:** `DefaultNormalizer.normalize()` — no issues found
- **Coupling:** Imports `app.schemas.analysis.StartupAnalysisRequest` at module level,
  creating a soft reverse dependency on the `app` package

### Stage 2: Collection (`collection/collector.py`)
- **Dead code:** `DataSource` class (lines 32-42) — never instantiated in prod or tests
- **Dead code:** `DiscoverError` and `FetchError` in `evidence/exceptions.py` — never raised or caught

### Stage 3: Evidence Collection (`evidence/`)
- **Duplicated domain lists:** `_REPUTABLE_DOMAINS`, `_SIMILAR_DOMAINS`,
  `_THIRD_PARTY_EVIDENCE`, `_THIRD_PARTY_TYPES`, `_AUTHORITY_DOMAINS` exist 5×
  across `ranking.py`, `prioritization.py`, `document_intelligence.py` with ~26 overlapping entries
- **Duplicated path tables:** `_PATH_BOOSTS`, `_PATH_TYPE_SIGNALS`, `_PATH_SIGNALS` exist 3×
  with the same `/about`, `/pricing`, `/docs` tokens but divergent weights
- **Duplicated host matching:** `host == d or host.endswith(f".{d}")` repeated 6×
  across `ranking.py`, `prioritization.py`, `document_intelligence.py`
- **Redundant URL sorting:** Documents sorted by `str(original_url)` 4× in pipeline
- **Dead code:** `retryable_statuses` in `fetcher.py` (never read)
- **Dead code:** `DocumentIntelligence` class used only in tests
- **Dead code:** `duplicate_heading_ratio` computed but excluded from quality formula
- **Dead code:** `EvidenceSource.retrieval_method` + `RetrievalMethod` enum (never set)
- **Dead code:** `EvidenceBundle.failures` (test-only)
- **Performance:** `doc.text.split()` executed ≥3× per document
- **Performance:** SHA-256 content hash computed twice per document
- **Performance:** `_trust_level()` called up to 4× per document
- **Hidden coupling:** Process-wide `_LOCK` serializes all evidence collection

### Stage 4: Extraction (`extraction/`)
- **Behavioral bug:** `extractors/company.py:72` — substring `"us"` matches inside
  business/industry/focus fields → `headquarters_region` almost always `north_america`
- **Duplicated regex:** `competition.py:109` — `\bcustom[\s-]built\b` appears twice
- **Duplicated helper:** `_extract_snippet` copied 4× across extractors
- **Duplicated import pattern:** `from ... import Observation as Obs` inside every method
  in 3 rule modules (~50 redundant import lookups per pass)
- **Dead param:** `evidence_confidence.py:127` — `documents` parameter never used
- **Dead code:** `evidence_agreement.py:172` — trailing ternary branch unreachable

### Stage 5: Evidence (`evidence/evidence_engine.py`)
- **No issues found** — clean, protocol-based, deterministic

### Stage 6: Reasoning (`reasoning/`)
- **Dead constants:** `_REINFORCING_WEIGHTS` and `_CONFLICT_WEIGHTS` in
  `cross_signal_reasoning.py:21-39` — defined but never referenced
- **Dead computation:** `cited_claims` set in `diagnostics.py:83` — populated but never read
- **Mutable hazard:** `DEFAULT_RULES` in `rules/__init__.py:48-60` is a mutable list
- **Redundant traversal:** `detect_missing_evidence` called twice per reasoning pass
  (once directly, once inside `build_consistency_report`)
- **Context not cached:** `trusted_documents()`, `best_source()`, `average_trust` re-scan
  `_documents` on every property access instead of caching at init time
- **Inconsistency:** reinforcement detection uses case-sensitive comparison but
  contradiction detection uses case-insensitive comparison

### Stage 7: Evaluation (`evaluation/`)
- **Dead fields:** `EvaluationResult.overall_summary`, `.overall_confidence`,
  `.dimensions_assessed`, `.metadata` are write-only in production
- **Dead field:** `SignalRelationship.source_observations` — written but never read
- **Redundant computation:** Investment readiness re-checks feature thresholds
  that reasoning rules already encode as observations

### Stage 8: Scoring (`scoring/scoring_engine.py`)
- **Massive duplication:** `_apply_data_dampener` copied 5× verbatim (lines 446, 652, 928, 1204, 1566)
- **Massive duplication:** `_score_observations` copied 7× with only multiplier differing
- **Dead code:** `PlaceholderDimensionScorer` never instantiated in production
- **Dead code:** `hasattr(o, "importance")` guard is always True (field has default)
- **Drift hazard:** `_score_engineering_strength` duplicated between `FounderQualityScorer`
  and `TeamExecutionScorer` with **drifted weights** (6.0/2.0 vs 8.0/3.0)
- **Confidence weight bug:** `confidence_engine.py` weights sum to 1.05, not 1.0
  (masked by clamp at line 95)

### Stage 9: Recommendations (`recommendations/`)
- **Legacy module:** `DefaultRecommendationEngine` is wired only in tests; production uses
  `CompositeRecommendationEngine` which has no dedup/prioritize
- **Duplicated blend formula:** `(conf + assess_conf) / 2` repeated 3× across strategies
- **Missing dedup:** `CompositeRecommendationEngine` doesn't deduplicate, allowing
  identical actions from different strategies to survive

### Stage 10: Confidence (`confidence/confidence_engine.py`)
- **Weight sum bug:** Weights sum to 1.05 instead of 1.0 (masked by clamp)
- **Dead code:** `hasattr` guards on always-present `Observation.importance`

### Stage 11: Decision (`decision/`)
- **Dead branch:** `decision_engine.py:818-821` — empty list comprehension, guard always false
- **Dead params:** `_build_rationale` accepts `conviction`, `signal_relationships`,
  `decision_factors` but never uses them
- **Triple-maintained threshold inventory:** Risk fields listed 3× in different functions
- **Duplicated thresholds:** Quantitative thresholds in `decision_engine.py` duplicate
  those in `opportunities.py` and `risks.py` — nothing enforces alignment
- **Hidden coupling:** `report_builder.py:20` imports `DimensionAssessment` from
  `evaluation.evaluation_models` (re-export) rather than `models/report.py`

### Stage 11b: Calibration (`decision/calibration.py`)
- **Duplicated evaluator dispersion:** Same mean/std/normalize math computed 2×
- **Duplicated conflict counting:** Same conflict tally computed 2×
- **Duplicated evidence trust:** `compute_evidence_trust(bundle)` called 2× per invocation

### Stage 11c: Synthesis (`synthesis/`)
- **Duplicated `_Register` class:** `opportunities.py:83-158` ≡ `risks.py:111-186`
- **Duplicated helpers:** `_normalize_label`, `_citations_from`, `_provenance_from` copy-pasted 3-4×
- **Duplicated keyword matching:** Dimension matching from statement text duplicated
  in `opportunities.py` and `tradeoffs.py`
- **Re-scanning:** Full `observations` list re-filtered per score/relationship/metric
  (~14 repeated list comprehensions)

### Stage 12: Report (`report/report_builder.py`)
- **Post-build mutation:** `engine.py:257-259` sets `decision_confidence`,
  `calibration_summary`, `decision_synthesis` after `build()` returns —
  builder contract is silently incomplete

### Validation (`validation/`)
- **No critical issues** — all validators have docstrings, findings are well-structured
- **Determinism risk:** Set-based dimension comparisons produce findings in arbitrary order
- **API inconsistency:** Four validators have different `validate()` signatures

### Models (`models/`)
- **No dead fields** — all 120+ `ExtractedFeatures` fields are consumed
- **All Report model fields consumed** — no orphaned models

### Interfaces (`interfaces/protocols.py`)
- **All 12 protocols are decorative** — defined and re-exported but never imported
  or enforced by any consumer

### Knowledge (`knowledge/`)
- **5 dead symbols:** `DEFAULT_DIMENSIONS`, `STAGE_KEYWORDS`, `CustomerType`,
  `Geography`, `INDUSTRY_KEYWORDS`, `MODEL_KEYWORDS` — defined but never imported

### Application Layer (`app/`)
- **Lazy import pattern** correctly breaks the engine→app cycle
- **Rate limiter and middleware** are clean, well-documented

---

## Top 10 Hardening Opportunities (Ranked by Impact)

1. **Extract shared `_apply_data_dampener` and `_score_observations`** into scorer helpers
2. **Cache `trusted_documents()`, `best_source()`, `average_trust`** in `ReasoningContext.__init__`
3. **Deduplicate calibration internals** — single compute, thread result
4. **Fix confidence weight sum** from 1.05 to 1.00
5. **Make `DEFAULT_RULES` a tuple** to prevent mutation
6. **Sort validator findings** for deterministic output ordering
7. **Delete dead code** — 30+ provably dead items identified
8. **Consolidate synthesis `_Register`** and helper duplication
9. **Single shared domain/path registry** for evidence layer
10. **Compute content hash once** in `enrich_documents`
