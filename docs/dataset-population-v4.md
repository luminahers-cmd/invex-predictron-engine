# Production Population Pipeline — Project V4

This document describes the production population orchestrator that populates
the Predictron historical dataset with real companies at scale.

Everything here is **additive infrastructure** layered on top of the existing
acquisition framework (`AcquisitionPipeline`, `ImportPipeline`,
`DatasetStore`, source connectors). It does **not** change the prediction
engine, scoring, reasoning, benchmarks, or existing APIs.

---

## Overview

The `PopulationOrchestrator` drives a population run end to end:

1. selects which configured sources to populate (all, or a named subset),
2. acquires records from each source using the existing acquisition framework,
3. performs record-granular resume of interrupted sources,
4. collects per-source acquisition results and aggregates metrics,
5. runs quality checks over the affected records,
6. writes a deterministic JSON report (and an optional report file),
7. records import history so each run is auditable.

The core entry point lives in `predictron_engine/dataset/population.py`.

```python
from predictron_engine.dataset import (
    PopulateOptions,
    PopulationConfig,
    PopulationOrchestrator,
    SourceConfig,
)
from predictron_engine.dataset.store import DatasetStore

store = DatasetStore("./my_dataset")
store.initialize()

config = PopulationConfig(
    sources=[
        SourceConfig(name="csv_export", file_paths=["./data/yc.csv"]),
    ]
)

report = PopulationOrchestrator(store, config=config).populate(
    PopulateOptions(resume=True)
)
print(report.counts)  # processed / imported / skipped / duplicates / failed
```

## Modules

| Module | Purpose |
|--------|---------|
| `population.py` | `PopulationOrchestrator`, `PopulateOptions`, record-granular resume |
| `population_config.py` | `SourceConfig`, `PopulationConfig`, `load_config`, `build_default_config` |
| `population_metrics.py` | `PopulationRunMetrics`, `compute_population_metrics`, `load_import_history` |
| `population_report.py` | `PopulationReport`, `build_population_report`, `FailureReason` |
| `quality.py` | `DatasetQualityChecker`, `QualityReport`, `generate_quality_report` |

All modules reuse the existing acquisition framework and are deterministic —
they never fabricate company data.

---

## Configuration

Sources are configured through a `PopulationConfig` of ordered `SourceConfig`
objects. Each source names a registered connector (`csv_export`, `yc_oss`,
`sec_edgar`, `companies_house`) and the files or directories to acquire from.

```json
{
  "sources": [
    {
      "name": "csv_export",
      "file_paths": ["./data/yc_top_companies.csv"],
      "directories": ["./data/llm_export/"],
      "enable_resume": true
    }
  ],
  "idempotent": true,
  "batch_size": 1000,
  "checkpoint_every": 1
}
```

Load it from Python or the CLI:

```python
from predictron_engine.dataset.population_config import load_config
config = load_config("./population.json")
```

```bash
predictron-dataset populate --dataset ./my_dataset --config ./population.json
```

`build_default_config()` produces a `PopulationConfig` with one source per
registered connector (file paths empty, to be supplied by the operator).

### Selecting sources

- No `--source` and no `--all`: every configured source is populated.
- `--source <name>` (repeatable): only the named sources are populated.
- `--all`: every configured source is populated (same as the default).

---

## CLI

```bash
# Populate from a config file
predictron-dataset populate --dataset ./my_dataset --config ./population.json

# Populate a single source
predictron-dataset populate --dataset ./my_dataset --config ./population.json \
    --source csv_export

# Dry-run (report what would happen; imports nothing)
predictron-dataset populate --dataset ./my_dataset --config ./population.json --dry-run

# Resume interrupted sources at record granularity
predictron-dataset populate --dataset ./my_dataset --config ./population.json --resume

# Write the JSON report to a file
predictron-dataset populate --dataset ./my_dataset --config ./population.json \
    --report ./reports/run.json
```

The command exits `0` when nothing failed and `1` otherwise.

---

## Record-granular resume

The acquisition framework's own resume is **file-granular**: a partially
processed file is re-read and every still-pending record is re-imported, which
re-imports records that were already committed before the interruption.

The orchestrator's `--resume` path (and `PopulateOptions(resume=True)`) is
**record-granular**. For each file that has no completed batch, every raw
record is matched against the store with the existing `match_record_to_store`
dedup logic before import, so:

- records already in the store are **not** re-imported (they count as skipped),
- only genuinely missing records are imported,
- the pending checkpoint is cleared once the file is handled.

This reuses the existing import pipeline (preserving provenance and outcomes)
without changing the engine or the framework's own semantics.

---

## Quality checks

After every population run, the orchestrator runs `DatasetQualityChecker` over
the affected records and embeds the findings in the report. Checks include:

- missing required fields (`startup_name`, `website`, `engine_version`),
- invalid URLs,
- duplicate companies and duplicate websites,
- conflicting identifiers,
- missing provenance,
- malformed outcomes.

```python
from predictron_engine.dataset.quality import generate_quality_report
report = generate_quality_report(store)
print(report.is_clean(), report.by_check())
```

Checks are non-destructive and deterministic.

---

## Metrics and reports

`compute_population_metrics` derives operational metrics from acquisition
results and store counts:

- total processed / imported / skipped / duplicated / failed,
- import rate (records/second) and records per hour,
- validation success rate and duplicate rate,
- per-source contribution,
- dataset growth (end minus start record count),
- import history from prior completed runs.

`build_population_report` serializes counts, metrics, source breakdown,
failure reasons, and quality findings into a `PopulationReport` with a stable
`run_id`. With a pinned `generated_at` timestamp, identical runs produce
byte-identical reports (determinism is testable).

```python
report.write("./reports/run.json")
data = report.to_dict()
print(data["counts"], data["metrics"], data["quality"]["clean"])
```

---

## Guarantees

- Additive: reuses the acquisition framework; the engine, scoring, reasoning,
  benchmarks, and existing APIs are unchanged.
- Deterministic: no network calls, no fabricated data, stable report output.
- Auditable: import history and per-run reports persist to the store.
- Resume-safe: interrupted runs are completed without re-importing records.
