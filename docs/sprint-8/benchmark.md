# Sprint 8 — Benchmark Suite Report

**Date:** 2026-08-26
**Engine Version:** 0.14.0
**Test Framework:** pytest 8.4.0

---

## Benchmark Suite Overview

The Predictron Engine benchmark suite consists of 10 deterministic startup
cases across diverse industries, processed through the complete 13-stage
analysis pipeline. Sprint 8 adds adaptive reasoning budget allocation,
progressive evidence evaluation, contradiction detection, and explanation
generation.

### Cases

| Case ID | Industry | Business Model | Stage |
|---------|----------|----------------|-------|
| `b2b_saas` | Enterprise SaaS | Subscription | Series A |
| `healthcare_ai` | HealthTech | Licensing | Seed |
| `fintech` | FinTech | Transactional | Series B |
| `devtools` | DevTools | Open-source + SaaS | Seed |
| `marketplace` | Marketplace | Two-sided marketplace | Series A |
| `consumer_app` | Consumer Tech | Freemium | Pre-Seed |
| `climate_tech` | Climate Tech | SaaS | Series A |
| `robotics` | Hardware / Robotics | RaaS | Series B |
| `enterprise_software` | Enterprise Software | SaaS | Series C |
| `ai_infrastructure` | AI / ML | PaaS | Seed |

---

## Benchmark Metrics Computed

The `BenchmarkMetrics` engine now computes **22 deterministic metrics**:

### Core Metrics (1–17, Sprint 7B)

1. **score_mean** — Mean overall score
2. **score_std_dev** — Score standard deviation
3. **score_range** — Score spread (max - min)
4. **confidence_mean** — Mean overall confidence
5. **confidence_std_dev** — Confidence standard deviation
6. **confidence_calibration** — Pearson correlation (confidence vs data completeness)
7. **reasoning_consistency** — Cross-case scoring consistency
8. **explainability_coverage** — Fraction of outputs with explanatory content
9. **recommendation_quality** — Action completeness, confidence, priority, diversity
10. **extraction_completeness** — Average data completeness
11. **processing_time** — Mean/min/max/P50 processing time
12. **coverage_distribution** — Industry category distribution
13. **throughput** — Analyses per second (NPS analog)
14. **avg_observations_per_case** — Observations per analysis
15. **avg_evidence_per_case** — Evidence items per analysis
16. **avg_branching_factor** — Observations per scored dimension
17. **avg_recommendations_per_case** — Recommendations per analysis

### Sprint 8 Metrics (18–22)

18. **avg_reasoning_depth** — Average observations per case (reasoning depth proxy). Higher values indicate deeper reasoning on complex cases; adaptive budget should maintain or improve this on high-uncertainty cases while reducing it on straightforward cases.

19. **adaptive_budget_savings** — Estimated adaptive budget savings fraction (0–1). Measures how much reasoning computation could be saved on straightforward cases (high data completeness, abundant evidence). Directly measurable via `BudgetReport.savings_fraction`.

20. **contradiction_detection_accuracy** — Measures that the contradiction graph correctly detects contradictions when evidence conflicts exist and does not produce false contradictions. Score is fraction of cases correctly classified.

21. **explanation_generation_latency** — Latency of reasoning trace generation in milliseconds. Must remain bounded (< 5ms) to avoid impacting pipeline throughput.

22. **calibration_error** — Expected Calibration Error (ECE) across all cases. Lower is better. A score of 0.0 means perfect calibration; > 0.10 triggers investigation.

---

## Adaptive Reasoning Behavior

### Budget Allocation Distribution

Expected distribution across benchmark cases:

| Case | Expected Tier | Rationale |
|------|---------------|-----------|
| `b2b_saas` | Reduced | High completeness, strong evidence |
| `healthcare_ai` | Standard | Moderate uncertainty, licensing model |
| `fintech` | Reduced | Strong data, transactional model |
| `devtools` | Elevated | Limited revenue data, seed stage |
| `marketplace` | Standard | Moderate evidence, network effects |
| `consumer_app` | Elevated | Pre-seed, limited traction data |
| `climate_tech` | Standard | Moderate completeness |
| `robotics` | Elevated | Hardware complexity, limited data |
| `enterprise_software` | Minimal | High completeness, mature stage |
| `ai_infrastructure` | Standard | Seed stage but good data signals |

### Progressive Evidence Savings

- Cases with 5+ evidence items of consistent direction should trigger
  early-stop after evaluating only 3–4 items.
- Expected computation savings: 15–40% on clear cases.

### Contradiction Detection

- Cases with both risk and opportunity signals should show 1–3 conflicting edges.
- Mature cases (enterprise_software, fintech) should show 0 conflicting edges.

---

## Running Benchmarks

```bash
# Run all benchmark cases
python -m benchmarks.benchmark_runner

# Run with profiling
python -m benchmarks.profile_pipeline --iterations 10

# Run specific case
python -m benchmarks.benchmark_runner --case b2b_saas

# Generate JSON output
python -m benchmarks.benchmark_runner --json

# Save snapshot for regression tracking
python -m benchmarks.benchmark_runner --save-snapshot --version 0.14.0
```

---

## Regression Tracking

Benchmarks are designed for cross-version regression detection:

1. Save baseline snapshot: `python -m benchmarks.benchmark_runner --save-snapshot --version 0.14.0`
2. Apply changes
3. Run benchmarks against new version
4. Compare: `python -m benchmarks.benchmark_report --diff 0.13.0 0.14.0`

### What to watch for in regressions

- **Score changes** — Should only change when scoring logic changes
- **Observation count changes** — Adaptive budget may reduce counts on simple cases; this is expected
- **Processing time** — Should not regress; adaptive budget should improve on simple cases
- **Determinism** — Same input must always produce identical output
- **Adaptive budget savings** — Should be positive (> 0.0) across mixed-complexity cases
- **Calibration error** — Should remain <= 0.10

---

## Summary

| Metric | Sprint 7B | Sprint 8 (Expected) |
|--------|-----------|---------------------|
| Test count | 2729 | 2804 |
| Benchmark metrics | 17 | 22 |
| Deterministic | Yes | Yes |
| External dependencies | None | None |
| Pipeline stages | 13 | 13 (unchanged) |
