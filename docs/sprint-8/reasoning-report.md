# Sprint 8 — Adaptive Reasoning Report

**Date:** 2026-08-26
**Engine Version:** 0.14.0

---

## Overview

Sprint 8 transforms the Predictron Engine from a deterministic scoring
pipeline into an **adaptive reasoning engine** that dynamically allocates
computational effort based on uncertainty, evidence quality, and decision
impact. All optimizations are measurable and fully deterministic.

---

## New Modules

### 1. Adaptive Reasoning Budget (`reasoning/adaptive_budget.py`)

Dynamically allocates reasoning computation per analysis run.

- **`compute_reasoning_budget(features, evidence)`** — pure function, identical
  inputs always produce identical budgets.
- **`ReasoningBudget`** — frozen dataclass capturing budget fraction, skipped
  rules, confidence estimate, impact score, and rationale.
- **`BudgetReport`** — tracks actual vs. planned execution for benchmarking.
- **`should_skip_rule(rule_name, budget)`** — deterministic skip decision.

**Budget tiers:**

| Tier | Fraction | When |
|------|----------|------|
| Full | 1.00 | Low confidence + high impact (uncertain, complex cases) |
| Elevated | 0.85 | Moderate uncertainty, early-stage, limited data |
| Standard | 0.70 | Average confidence, normal cases |
| Reduced | 0.50 | High confidence, low impact, abundant evidence |
| Minimal | 0.35 | Very high confidence, mature company, full data |

**Skip priority order** (least critical first):

1. `QuantitativeCrossSignalRule`
2. `CrossSignalReasoningRule`
3. `QuantitativeSignalsRule`
4. `CompetitionAssessmentRule`
5. `RiskIndicatorRule`
6. `DataQualityRule`
7. `StageExpectationRule`
8. `BusinessModelContextRule`
9. `TechnologyContextRule`
10. `TeamAssessmentRule`
11. `MarketContextRule` (never skipped)

### 2. Progressive Evidence Evaluation (`reasoning/progressive_evidence.py`)

Evaluates evidence incrementally with early-stop when conclusive.

- **`evaluate_evidence_progressively(evidence, observations)`** — processes
  evidence items sequentially, stopping when thresholds are met.
- **`ProgressiveEvaluation`** — contains checkpoints, skipped evidence, and
  `computation_saved_fraction`.
- **`EvidenceCheckpoint`** — snapshot at each evaluation step.

**Early-stop triggers (any one suffices):**

| Trigger | Threshold |
|---------|-----------|
| Agreement ratio | >= 0.85 |
| Support ratio | >= 0.80 |
| Consecutive support | >= 3 items |
| Cumulative confidence | >= 0.92 |
| Max items | >= 20 |

### 3. Contradiction Graph (`reasoning/contradiction_graph.py`)

Explicitly models tension between supporting and conflicting evidence.

- **`build_contradiction_graph(observations)`** — pure function, O(n^2) pair-wise
  analysis of observation relationships.
- **`ContradictionEdge`** — supporting, conflicting, or neutral with intensity.
- **`DominantConflict`** — highest-intensity conflict with explanation.
- **`DimensionContradictionSummary`** — per-dimension conflict statistics.

**Conflict classification requires 2+ signals:**

| Signal | Condition |
|--------|-----------|
| Category tension | `("strength","risk")`, `("growth","decline")`, etc. |
| Confidence divergence | One > 0.7, other < 0.3 |
| Importance divergence | One > 0.7, other < 0.3 |

### 4. Reasoning Trace (`reasoning/trace.py`)

Structured explainability for the reasoning process.

- **`build_reasoning_trace(observations, evidence, scores)`** — builds complete
  reasoning trace from pipeline outputs.
- **`ReasoningTrace`** — contains strongest supporting/opposing evidence,
  confidence evolution, dominant conflict, and final rationale.
- **`TraceEntry`** — single reasoning observation ranked by importance x confidence.
- **`ConfidenceEvolution`** — confidence snapshot at each pipeline stage.

**Confidence evolution stages:**

1. `post_extraction` — data completeness proxy
2. `post_reasoning` — observation confidence mean
3. `post_evaluation` — assessment confidence proxy
4. `post_scoring` — final score-based confidence proxy

### 5. Calibration Improvements (`decision/calibration_sprint8.py`)

Extends Sprint 6B calibration with validation and overconfidence detection.

- **`compute_calibration_error(predictions, outcomes)`** — Expected Calibration
  Error (ECE) with configurable bin count.
- **`detect_overconfidence(bins)`** — finds bins where predicted confidence
  exceeds actual accuracy by > 0.15.
- **`build_calibration_report(confidence, completeness, outcomes)`** — complete
  calibration validation report.
- **`validate_pipeline_confidence(results)`** — multi-analysis calibration.

**Calibration quality tiers:**

| ECE | Quality |
|-----|---------|
| <= 0.03 | excellent |
| <= 0.07 | good |
| <= 0.12 | fair |
| <= 0.20 | poor |
| > 0.20 | very_poor |

---

## Engine Integration

### `PredictronEngine.reason_adaptive(features, evidence, bundle)`

Runs reasoning with adaptive budget allocation. Returns:

```python
{
    "budget": ReasoningBudget,
    "budget_report": BudgetReport,
    "contradiction_graph": ContradictionGraph,
    "observations": list[Observation],
}
```

### `PredictronEngine.reason_with_trace(observations, evidence, scores)`

Generates reasoning trace for existing observations. Returns `ReasoningTrace`.

### `CompositeReasoner.reason_with_budget(features, evidence, bundle, budget)`

Runs only the rules allowed by the given budget. Skipped rules are logged
at DEBUG level.

---

## Benchmark Metrics (New)

5 new metrics added to `BenchmarkMetrics`:

| Metric | Description |
|--------|-------------|
| `avg_reasoning_depth` | Average observations per case (reasoning depth proxy) |
| `adaptive_budget_savings` | Estimated savings fraction for straightforward cases |
| `contradiction_detection_accuracy` | Accuracy of contradiction detection across cases |
| `explanation_generation_latency` | Latency of reasoning trace generation |
| `calibration_error` | Expected Calibration Error across cases |

---

## Determinism Guarantees

Every new module satisfies:

1. **Identical inputs produce identical outputs** — verified by test suite
2. **No external dependencies** — pure Python, no network calls
3. **Every optimization is measurable** — budget reports, progressive checkpoints
4. **Existing pipeline unchanged** — `analyze()` behaviorally identical

---

## Test Coverage

6 new test files (75+ test cases):

| Test File | Coverage |
|-----------|----------|
| `test_adaptive_budget.py` | Budget computation, determinism, skip rules, report |
| `test_progressive_evidence.py` | Early-stop, computation savings, determinism, checkpoints |
| `test_contradiction_graph.py` | Graph building, edges, determinism, per-dimension |
| `test_reasoning_trace.py` | Trace generation, confidence evolution, rationale, serialization |
| `test_explanation_consistency.py` | Determinism across runs, bounded confidence, rankings |
| `test_calibration_sprint8.py` | ECE computation, overconfidence detection, reports |

All 2804 tests pass with zero regressions.

---

## Running Tests

```bash
# Sprint 8 adaptive reasoning tests
pytest tests/engine/test_adaptive_budget.py tests/engine/test_contradiction_graph.py \
       tests/engine/test_progressive_evidence.py tests/engine/test_reasoning_trace.py \
       tests/engine/test_explanation_consistency.py tests/engine/test_calibration_sprint8.py -v

# Full regression suite
pytest tests/ -x -q

# Specific new tests
pytest tests/engine/test_adaptive_budget.py -v
pytest tests/engine/test_progressive_evidence.py -v
pytest tests/engine/test_reasoning_trace.py -v
pytest tests/engine/test_calibration_sprint8.py -v
```
