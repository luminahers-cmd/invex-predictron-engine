# Sprint 7B — Benchmark Suite Report

**Date:** 2026-08-26
**Engine Version:** 0.12.1
**Test Framework:** pytest 8.4.0

---

## Benchmark Suite Overview

The Predictron Engine benchmark suite consists of 10 deterministic startup cases across diverse industries, processed through the complete 13-stage analysis pipeline.

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

## Benchmark Results (Post-Optimization)

### Processing Time

| Metric | Value |
|--------|-------|
| Mean processing time | 37.23ms |
| Median processing time | 26.44ms |
| Min processing time | 20.60ms |
| Max processing time | 631.10ms |
| Engine-only time (mean) | 25.81ms |

### Throughput

| Metric | Value |
|--------|-------|
| Analyses per second (NPS) | ~27 per second |
| Total benchmark suite time | ~2.2s (10 cases × 3 iterations) |

### Pipeline Stage Breakdown

| Stage | Mean (ms) | Median (ms) | % of Total |
|-------|-----------|-------------|------------|
| normalize | 0.048 | 0.038 | 0.1% |
| collect | 0.082 | 0.072 | 0.2% |
| extract | 23.092 | 23.404 | 62.0% |
| evidence | 0.131 | 0.120 | 0.4% |
| reasoning | 1.157 | 1.112 | 3.1% |
| evaluation | 0.347 | 0.323 | 0.9% |
| scoring | 0.154 | 0.134 | 0.4% |
| investment_readiness | 0.139 | 0.128 | 0.4% |
| recommendations | 0.232 | 0.148 | 0.6% |
| confidence | 0.099 | 0.089 | 0.3% |
| decision | 0.176 | 0.160 | 0.5% |
| report_builder | 0.050 | 0.040 | 0.1% |

### Key Ratios

| Ratio | Value |
|-------|-------|
| Engine as % of total | 69.3% |
| Extraction as % of total | 62.0% |
| Serialization as % of total | 0.1% |
| Persistence model prep as % of total | 1.9% |

---

## Quality Metrics

### Score Distribution

| Metric | Value |
|--------|-------|
| Score mean | ~55-65 |
| Score std dev | ~10-15 |
| Score range | ~30 |

### Reasoning Consistency

| Metric | Value |
|--------|-------|
| Reasoning consistency score | 0.95+ |
| Dimensions measured | 7 |

### Explainability Coverage

| Metric | Value |
|--------|-------|
| Scores with rationale | 100% |
| Assessments with summary | 100% |
| Observations with source rule | 100% |
| Recommendations with title | 100% |

### Extraction Completeness

| Metric | Value |
|--------|-------|
| Average data completeness | 0.85+ |
| Key feature population rate | 90%+ |

---

## Determinism Validation

### Test: `test_same_input_produces_identical_output`
- Same request processed twice through the engine
- Verified: identical scores, observations, assessments, investment decision, and node count
- **Result: PASS**

### Test: `test_all_benchmark_cases_are_deterministic`
- All 10 benchmark cases processed twice
- Verified: identical overall_score and observation count for each case
- **Result: PASS**

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
python -m benchmarks.benchmark_runner --save-snapshot --version 0.12.1
```

### Running Performance Profiler

```bash
# Default: 10 iterations across all cases
python -m benchmarks.profile_pipeline

# Custom iterations
python -m benchmarks.profile_pipeline --iterations 50

# Specific case
python -m benchmarks.profile_pipeline --case b2b_saas

# JSON output
python -m benchmarks.profile_pipeline --json
```

---

## Benchmark Metrics Computed

The `BenchmarkMetrics` engine computes 17 deterministic metrics:

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

---

## Regression Tracking

Benchmarks are designed for cross-version regression detection:

1. Save baseline snapshot: `python -m benchmarks.benchmark_runner --save-snapshot --version 0.12.1`
2. Apply changes
3. Run benchmarks against new version
4. Compare: `python -m benchmarks.benchmark_report --diff 0.12.1 0.13.0`

### What to watch for in regressions

- **Score changes** — Should only change when scoring logic changes
- **Observation count changes** — Should only change when reasoning rules change
- **Processing time** — Should not regress significantly
- **Determinism** — Same input must always produce identical output
