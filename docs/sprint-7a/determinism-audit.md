# Sprint 7A — Determinism Audit Report

**Date:** 2026-08-25
**Requirement:** Identical inputs must always produce identical outputs

---

## Determinism Verification by Component

### ✅ Deterministic (Verified)

| Component | Verification |
|-----------|-------------|
| **Normalizer** | Pure string transforms, no randomness |
| **Collector** | Returns structured data, no ordering issues |
| **Extraction** | All extractors are pure functions of input text |
| **Evidence gathering** | Knowledge base lookups are pure |
| **Evidence engine** | Deterministic — iterates fixed evidence list |
| **Reasoning** | All rules are pure functions of (features, evidence) |
| **Evaluation** | Evaluators are pure functions of (features, observations, evidence) |
| **Scoring** | All scorers are pure functions of (features, observations) |
| **Recommendations** | Strategies are pure functions of inputs |
| **Confidence** | Pure weighted sums of bounded inputs |
| **Decision** | Pure threshold comparisons on bounded scores |
| **Calibration** | Pure arithmetic on existing pipeline outputs |
| **Synthesis** | Pure aggregation, no new computation |
| **Report builder** | Assembles existing outputs |

### ⚠️ Time-Dependent (Metadata Only, Non-Breaking)

| Location | Field | Issue |
|----------|-------|-------|
| `models/report.py:569-572` | `AnalysisMetadata.timestamp` | `datetime.now(UTC)` — changes every run |
| `engine.py:261-262` | `processing_time_ms` | Wall-clock measurement |
| `document_intelligence.py:837` | `processing_duration_ms` | `time.monotonic()` delta |
| `evidence/fetcher.py:82` | `response_time_ms` | Network latency |
| `evidence/orchestrator.py:98,186` | `collected_at`, `duration_ms` | Wall-clock |

**Assessment:** All time-dependent fields are confined to `AnalysisMetadata` and
evidence collection metadata. They are intentionally diagnostic and excluded from
golden-output baselines. Core analysis outputs (scores, decisions, recommendations)
are fully deterministic.

### ⚠️ Hardcoded Time Coupling

| Location | Issue |
|----------|-------|
| `scoring_engine.py:1950` | `_score_recency` hardcodes `current_year = 2026` |

This is deterministic per build but silently changes behavior after year-end.
Low risk since the scorer compares against year-specific thresholds.

### ⚠️ Set Iteration Order

| Location | Risk Level | Assessment |
|----------|-----------|------------|
| `confidence.py:191-200` | Safe | Set used only for `len()` — order-independent |
| `diagnostics.py:82` | Safe | Set used only for `len()` |
| `tradeoffs.py:109` | Safe | Set used only for membership check |
| `validation/consistency_validator.py:87-92` | **Determinism Risk** | Set difference iterated for finding generation |
| `validation/completeness_validator.py:85-89` | **Determinism Risk** | Set difference iterated for finding generation |
| `explainability.py:147,187,233` | **Determinism Risk** | Set comprehensions in text output without sorting |

### ✅ Sort Stability Verified

All sorts use either:
- `sorted()` with deterministic keys (stable, same input → same output)
- `list.sort()` with bounded keys
- Single-element keys where ties don't occur

Key examples:
- `opportunities.py:154-157` — `sorted(total, key=(severity, dimension, label))`
- `risks.py:182-185` — same pattern
- `document_intelligence.py:604,835` — `sorted(str(original_url))`
- `scoring_engine.py` — score ordering follows strategy list order (fixed)

### ✅ No Randomness

Grep for `random`, `uuid4`, `secrets`, `hashlib`:
- `uuid4` only at `engine.py:146` — request_id, confined to log extras, not in Report
- `hashlib` only at `document_intelligence.py:485` — deterministic SHA-256 for content hashing
- No `random` or `secrets` usage anywhere in engine

### ✅ Float Accumulation

All float sums use Python's built-in `sum()` which follows insertion order.
Since input order is deterministic (controlled by pipeline stage ordering),
accumulation order is deterministic. No reordering of floating-point additions.

### ⚠️ Hardcoded Thresholds Drift Risk

Multiple modules maintain parallel copies of the same thresholds:
- `decision_engine.py`, `opportunities.py`, `risks.py` all define ARR≥10M, NRR≥110, etc.
- Nothing enforces alignment — a change in one silently diverges

**Not a determinism issue today** (all copies currently match), but a drift hazard.

---

## Determinism Recommendations

1. **Sort validator set iterations** — wrap `for dim in uncovered` with `for dim in sorted(uncovered)` in `completeness_validator.py` and `consistency_validator.py`
2. **Sort set comprehensions in explainability** — use `sorted()` before `join()` for domain/rule/dimension sets
3. **Inject `current_year`** — make `TeamExecutionScorer._score_recency` accept `year` parameter
4. **Enforce threshold alignment** — extract shared constants to a single module
