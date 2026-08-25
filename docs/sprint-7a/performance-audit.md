# Sprint 7A — Performance Audit Report

**Date:** 2026-08-25
**Baseline:** Full pipeline runs in ~3.5 seconds (2713 tests in 223s)

---

## Slowest Pipeline Stages (by relative cost)

| Stage | Estimated Cost | Reason |
|-------|---------------|--------|
| Evidence collection (Stage 3) | ~50-80% of wall time | Network I/O (httpx fetches) |
| Extraction (Stage 4) | ~10-15% | 10 extractors × document scan |
| Reasoning (Stage 6) | ~3-5% | 12+ rules × observation scan |
| Scoring (Stage 8) | ~2-3% | 7 scorers × observation filter |
| Synthesis (Stage 11c) | ~1-2% | Multi-pass observation scan |

**Note:** All stages after evidence collection are CPU-bound and sub-second.
Evidence collection is I/O-bound and guarded by timeout/retry.

---

## Redundant Computation Sites

### Evidence Layer

| Site | Lines | Issue | Impact |
|------|-------|-------|--------|
| `doc.text.split()` | `document_intelligence.py:368,791,650` | Split 3× per doc | Minor |
| SHA-256 hash | `document_intelligence.py:540,790` | Computed twice per doc | Minor |
| `_trust_level()` | `document_intelligence.py:776,777,809,821` | Called 4× per doc | Minor |
| Document sort | `website_provider.py:114` → `orchestrator.py:162` → `document_intelligence.py:604,835` | Same sort 4× | Trivial |
| Host matching | `ranking.py`, `prioritization.py`, `document_intelligence.py` | 6× `host.endswith(f".{d}")` | Trivial |
| Citation O(n×m) | `citation.py:121-133` per extractor call | Full doc pool scan per evidence item | Moderate at scale |

### Reasoning Layer

| Site | Lines | Issue | Impact |
|------|-------|-------|--------|
| `trusted_documents()` | `context.py:131-147` | Re-scans `_documents` per call | Moderate |
| `best_source()` | `context.py:149-166` | Re-scans `_documents` per call | Moderate |
| `average_trust` | `context.py:180-190` | Re-scans `_documents` per call | Moderate |
| `detect_missing_evidence` | `composite.py:111,140` | Called 2× per pass | Minor |
| `provenance_records()` JSON parsing | `context.py:217-236` | Parses on every call | Minor |

### Scoring Layer

| Site | Lines | Issue | Impact |
|------|-------|-------|--------|
| Observation filter | `scoring_engine.py` 7× | Same `[o for o in obs if o.dimension == d]` per scorer | Moderate |
| `adjustments` tuples | `scoring_engine.py` ~35× | `(str, float)` tuples allocated only for `len()` | Minor |

### Synthesis Layer

| Site | Lines | Issue | Impact |
|------|-------|-------|--------|
| Observation re-filter | `opportunities.py` 8×, `risks.py` 7× | Same list comprehension repeated | Moderate |
| Score sort | `decision_engine.py` 6× | Same list sorted for rationale builders | Minor |
| Average confidence | `decision_engine.py` 770,848 | Computed 2× in rationale | Minor |

### Calibration Layer

| Site | Lines | Issue | Impact |
|------|-------|-------|--------|
| Evidence trust | `calibration.py:318,491` | `compute_evidence_trust(bundle)` called 2× | Minor |
| Evaluator std | `calibration.py:431,571` | Same mean/std/normalize 2× | Minor |
| Conflict count | `calibration.py:547,715` | Same conflict tally 2× | Minor |

---

## Optimization Recommendations

All recommendations preserve identical outputs:

### 1. Cache ReasoningContext document accessors (Medium Impact)
Cache `_sorted_trusted`, `_average_trust`, `_best_source` computed once in `__init__`.
Eliminates O(N) re-scans per property access. Estimated 10-20% reduction in reasoning time.

### 2. Single-pass observation partitioning (Medium Impact)
Build `observations_by_dimension: dict[str, list[Observation]]` once in `engine.py:173`,
pass it to scorers, confidence engine, and synthesis. Eliminates 7+ redundant filter passes.

### 3. Single evidence trust computation (Low Impact)
Compute `compute_evidence_trust(bundle)` once in `calibration.py`, thread result through.
Eliminates one redundant bundle traversal.

### 4. Single content hash in enrich_documents (Low Impact)
Reuse hash from `detect_duplicates` instead of recomputing at `document_intelligence.py:790`.
Eliminates one SHA-256 per document.

### 5. Observation re-filter → dict lookup in synthesis (Medium Impact)
Build `obs_by_dim: dict[str, list[Observation]]` at top of `aggregate_risks` and
`aggregate_opportunities`. Replace ~14 list comprehensions with dict lookups.

---

## Performance Not Issues

- `scoring_engine.py` is 1790 lines but each scorer is O(1) per dimension (no loops)
- `quantitative_cross_signal.py` (1007 lines) and `quantitative_signals.py` (772 lines)
  are large but single-pass, O(N) in observations
- Evidence collection network I/O is properly guarded by timeouts
- No unbounded recursion, no quadratic algorithms in hot paths
