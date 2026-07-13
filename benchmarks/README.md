# Predictron Engine Benchmark & Regression Framework

Benchmark suite for evaluating the consistency, explainability, and stability of the Predictron Engine across versions.

## Purpose

This framework validates engine quality by running deterministic startup cases through the complete analysis pipeline and capturing structured outputs. It is designed to:

- **Detect regressions** when engine internals change
- **Track improvements** across version upgrades
- **Ensure explainability** by capturing every pipeline stage output
- **Maintain stability** across identical inputs over time

## Design Principles

- No hardcoded "correct" venture decisions
- No proprietary heuristics
- No machine learning or LLMs
- Focus on consistency, explainability, and stability

## Directory Structure

```
benchmarks/
  __init__.py
  benchmark_runner.py       # Executes cases through the engine
  benchmark_report.py       # Generates reports and regression diffs
  startup_cases/
    __init__.py
    cases.py                # Deterministic benchmark startup cases
  expected_outputs/
    __init__.py
    snapshot_v0.6.5.json    # Version snapshots for regression comparison
  README.md
```

## Quick Start

```bash
# Run all benchmark cases
python -m benchmarks.benchmark_runner

# Run a specific case
python -m benchmarks.benchmark_runner --case b2b_saas

# List all available cases
python -m benchmarks.benchmark_runner --list-cases

# Save a snapshot for regression tracking
python -m benchmarks.benchmark_runner --save-snapshot --version 0.6.5

# Generate a full report
python -m benchmarks.benchmark_report

# Compare two versions
python -m benchmarks.benchmark_report --diff 0.6.5 0.7.0
```

## Available Benchmark Cases

| Case ID | Industry | Business Model |
|---------|----------|----------------|
| `b2b_saas` | Enterprise SaaS | Subscription |
| `healthcare_ai` | HealthTech | Licensing |
| `fintech` | FinTech | Transactional |
| `devtools` | DevTools | Open-source + SaaS |
| `marketplace` | Marketplace | Two-sided marketplace |
| `consumer_app` | Consumer Tech | Freemium |
| `climate_tech` | Climate Tech | SaaS |
| `robotics` | Hardware / Robotics | RaaS |
| `enterprise_software` | Enterprise Software | SaaS |
| `ai_infrastructure` | AI / ML | PaaS |

## How to Add a Benchmark Case

### 1. Define the case in `startup_cases/cases.py`

Add a new entry to the `BENCHMARK_CASES` list:

```python
{
    "id": "your_case_id",
    "label": "Human-readable description",
    "request": {
        "startup_name": "Example Startup",
        "website": "https://example.com",
        "description": (
            "A detailed description of the startup that gives the engine "
            "enough information to extract features across all dimensions. "
            "Include industry signals, business model indicators, traction "
            "data, team information, and geographic context."
        ),
        "pitch_deck_url": "https://example.com/deck.pdf",  # or None
        "founder_linkedin_urls": [                          # or empty list
            "https://linkedin.com/in/founder",
        ],
    },
    "expected_features": {
        "industry": "expected_industry_value",
        "business_model": "expected_model_value",
        "customer_type": "b2b",
        "has_revenue": True,
        "has_pitch_deck": True,
        "founder_profile_count": 1,
    },
}
```

### 2. Case Design Guidelines

- **Deterministic**: The same input must always produce the same output for a given engine version
- **Realistic**: Represent genuine startup profiles with plausible data
- **Diverse**: Cover different industries, business models, funding stages, and geographies
- **Rich**: Provide enough description text for feature extraction to work across all dimensions
- **Edge cases**: Include cases with missing data (no pitch deck, no founders, pre-revenue) to test graceful degradation

## How to Run the Benchmark Suite

```bash
# Full run with summary table
python -m benchmarks.benchmark_runner

# JSON output (for programmatic consumption)
python -m benchmarks.benchmark_runner --json

# Run specific cases only
python -m benchmarks.benchmark_runner --case b2b_saas fintech

# Generate a detailed report and save snapshot
python -m benchmarks.benchmark_report --version 0.6.5
```

## How Future Engine Versions Should Be Evaluated

### Before releasing a new version

1. **Save a snapshot of the current version** (baseline):
   ```bash
   python -m benchmarks.benchmark_runner --save-snapshot --version 0.6.5
   ```

2. **Apply engine changes** (new scorers, extractors, reasoning rules, etc.)

3. **Run benchmarks against the new version**:
   ```bash
   python -m benchmarks.benchmark_runner
   ```

4. **Save a snapshot of the new version**:
   ```bash
   python -m benchmarks.benchmark_runner --save-snapshot --version 0.7.0
   ```

5. **Compare versions**:
   ```bash
   python -m benchmarks.benchmark_report --diff 0.6.5 0.7.0
   ```

6. **Review the diff output**:
   - Score changes per dimension
   - Confidence changes
   - Feature extraction changes
   - Observation count changes
   - Recommendation count changes
   - Processing time changes

### What to look for in regression diffs

- **Improvements**: Score increases with stable confidence indicate better intelligence
- **Regressions**: Score decreases or confidence drops on unchanged inputs warrant investigation
- **Stability**: Feature extraction should remain consistent unless extractors were intentionally modified
- **Performance**: Processing time should not regress significantly
- **Coverage**: Observation and recommendation counts should not drop without reason

### Version progression

```
v0.6.5  →  v0.7  →  v0.8  →  v1.0
  ↓          ↓         ↓        ↓
snapshot   snapshot  snapshot  snapshot
```

Each version's snapshot captures the full engine state for any future comparison.

## Snapshot Format

Snapshots are JSON files stored in `expected_outputs/`:

```json
{
  "engine_version": "0.6.5",
  "benchmark_version": "1.0.0",
  "total_cases": 10,
  "successful_cases": 10,
  "failed_cases": 0,
  "results": [
    {
      "case_id": "b2b_saas",
      "case_label": "B2B SaaS — Cloud analytics platform",
      "success": true,
      "processing_time_ms": 12.5,
      "engine_version": "0.6.5",
      "extracted_features": { ... },
      "evidence": [ ... ],
      "observations": [ ... ],
      "dimension_assessments": [ ... ],
      "scores": [ ... ],
      "overall_score": 60.0,
      "recommendations": [ ... ],
      "confidence": [ ... ],
      "overall_confidence": 0.55,
      "stage_timings": { ... }
    }
  ]
}
```

## Extending the Framework

### Adding new pipeline stage capture

To capture output from a new pipeline stage, modify the `CaseResult.to_dict()` method in `benchmark_runner.py` and add the corresponding fields to the snapshot format.

### Adding new benchmark dimensions

To test a new analysis dimension:
1. Add benchmark cases that exercise the dimension
2. The runner automatically captures all output from every pipeline stage
3. The regression diff will surface any changes to the new dimension's outputs

### Custom engine configurations

To benchmark custom engine configurations, pass a configured `PredictronEngine` instance to `run_benchmark()`:

```python
from predictron_engine.engine import PredictronEngine
from benchmarks.benchmark_runner import run_benchmark

engine = PredictronEngine(scoring=MyCustomScorer())
results = run_benchmark(engine=engine)
```
