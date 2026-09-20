# Ground Truth Evaluation & Venture Benchmark Platform

## Overview

The ground-truth evaluation platform measures how well the Predictron
Engine's *fresh* predictions match *verified, real-world outcomes* on
curated **golden datasets**. It is a quality-assurance harness, not a
competition: every run is deterministic, append-only-recorded in history,
and comparable across engine versions.

Two distinct measurement modes are supported:

- **Replay benchmarks** (`benchmarks/benchmark_runner.py`): deterministic
  regression of the engine against recorded *historical predictions*.
- **Ground-truth evaluation** (`benchmarks/ground_truth_eval/`): pairs a
  fresh prediction with a *verified outcome*, computes binary/calibration
  metrics, and tracks drift over time. This document covers the latter.

## Layout

```text
benchmarks/ground_truth_eval/
  models.py    Golden dataset & run models (GoldenDataset, GoldenEntry,
               BenchmarkRun, RunEntryOutput, dataset fingerprint)
  dataset.py   Loaders, canonical JSON, validation, dataset discovery
  runner.py    BenchmarkRunner: deterministic engine replay per entry
  metrics.py   Build samples, confusion/ranking/calibration/coverage metrics
  drift.py     Score/decision/metric drift detection between runs
  history.py   Append-only on-disk history store (run.json files)
  reports.py   Deterministic report bodies by kind (executive, trend, ...)
  export.py    Canonical JSON export/import of datasets, runs, metrics,
               bundles, and reports
  cli.py       predictron-benchmark command-line interface
benchmarks/cohort/
  manifest.py  Cohort manifest schema, hashing, loading (deterministic)
  build.py     Deterministic cohort -> golden dataset builder
benchmarks/golden_datasets/example_v1.json   Committed example dataset
benchmarks/golden_datasets/starter_v1.json   Committed starter cohort dataset
benchmarks/cohort_sources/starter_v1.json    Committed starter cohort manifest
benchmarks/gen_example_golden_dataset.py     Reproducible generator
predictron_engine/dataset/prediction.py      Time-scope shim + look-ahead guard
scripts/build_cohort.py                      Cohort builder CLI (V1.4)
tests/ground_truth_eval/                     Test suite
tests/api/test_benchmark_accuracy.py         Read-only accuracy API tests
```

## Concepts & Pipeline

```text
Golden dataset ──► BenchmarkRunner ──► BenchmarkRun
                                   │
                                   ├─► compute_metrics(dataset, run) ──► GroundTruthMetrics
                                   ├─► history.append(run, metrics)    ──► history dir
                                   └─► export bundle / reports
```

- **Golden dataset**: a JSON file of startup cases, each carrying the
  *request* to replay and a *verified outcome* (exit, ARR milestone, or
  unresolved status). Outcomes must come from verifiable sources; the
  platform never fabricates labels.
- **Run**: one deterministic replay of every entry through the current
  engine. `result_hash` covers a canonical payload minus wall-clock timing,
  so identical inputs always produce identical hashes.
- **Metrics**: computed over pairs of (*predicted_positive*, *outcome*).
  `None` outcomes are un-scoreable and excluded — never coerced.
- **History**: append-only `benchmarks/history/` directory, one
  `<run_id>.json` file per run; run ids must match `^[A-Za-z0-9._-]+$`.
- **Reports & exports**: fully deterministic JSON documents, safe to diff
  and checksum.

## Golden Datasets

A dataset is a JSON document:

```json
{
  "dataset_name": "example_v1",
  "benchmark_version": "2.0.0",
  "schema_version": 1,
  "created_at": "...",
  "entries": [{
    "company_id": "...",
    "startup_name": "...",
    "startup_website": "...",
    "industry_category": "...",
    "geography": "...",
    "request": { "...": "the replay request" },
    "verified_outcome": {
      "actual_status": "acquired | shutdown | operating | unknown",
      "exit_value_usd": 120000000,
      "outcome_events": [...],
      "outcome_observation_date": "...",
      "evaluation_horizon_days": 180,
      "sources": [{"kind": "...", "url": "...", "retrieved_at": "...", "note": "..."}]
    },
    "historical_prediction": { "score": 78.12, "confidence": 0.6 },
    "provenance": { "is_example": true }
  }]
}
```

- Unsupported `schema_version` is rejected with a `GoldenDatasetError`.
- Multiple entries for one company are **not** a single golden case — each
  entry may carry its own historical prediction; duplicate `company_id`
  within a dataset is a validation error.
- Validation (`validate_golden_dataset`) reports duplicates, missing
  names/descriptions, empty datasets, and unscoreable outcomes.
- `provenance.is_example = true` marks *illustrative* labels so they are
  never mistaken for real ground truth.

### Example dataset

`benchmarks/golden_datasets/example_v1.json` is generated deterministically
by `benchmarks/gen_example_golden_dataset.py`. Historical predictions come
from the committed engine snapshot
`benchmarks/expected_outputs/snapshot_v0.13.0.json`; the *outcomes are
fictional and flagged `is_example`*. 13 entries, 11 scoreable.

## Time-Scoped Cohorts (Milestone V1.4)

Golden datasets in the platform are **time-anchored** benchmarks. Each entry
may carry two optional temporal anchors:

- `analysis_timestamp` — the UTC instant the prediction is pinned to. The
  **look-ahead guard** (`predictron_engine/dataset/prediction.py`) drops all
  evidence whose retrieval timestamp (`fetched_at`) is *after* this instant
  before the engine runs, so a recorded prediction can never be informed by
  documents that were not yet knowable at analysis time. Ties at
  `fetched_at == analysis_timestamp` are retained.
- `evaluation_horizon_days` — minimum days between analysis and outcome
  observation. Dataset validation errors on outcomes *before* the analysis
  and warns on windows below the horizon; naive timestamps are rejected.

**Cohort manifests** (`benchmarks/cohort/`) are the deterministic input for
building real datasets:

```text
cohort manifest ──► build_cohort_dataset()  ──► golden dataset
   (company inputs,      ├─► time-scope evidence (look-ahead guard)
    evidence corpus,      ├─► one pinned engine run per company
    analysis anchor,      ├─► best-known outcome verbatim
    best-known outcome)   └─► provenance (is_example when unverified)
```

Rules:

- Outcomes are stored **verbatim**; `outcome.verified=false` only sets
  `provenance.is_example=true` — nothing is fabricated.
- A record without a documented outcome is emitted as *pending ground
  truth* and excluded from the dataset (never scored).
- The build is deterministic: the same manifest always produces the same
  `dataset_hash`, `manifest_hash`, and per-company scope stats (stored in
  `entry.metadata["evidence_scope"]`).
- Ignore-worthy records are excluded rather than repaired.

Commands:

```text
python scripts/build_cohort.py make-manifest --out benchmarks/cohort_sources/starter_v1.json
python scripts/build_cohort.py build \
    --manifest benchmarks/cohort_sources/starter_v1.json \
    --out benchmarks/golden_datasets/starter_v1.json
```

The committed **starter cohort** (`starter_v1.json`) exercises the full
pipeline against the real offline corpora: 13 illustrative entries (flagged
`is_example`) and 5 pending-ground-truth records.

**Read-only accuracy view**: `GET /api/v1/evaluation/benchmark` returns the
accuracy report of the latest recorded benchmark run (confusion, F0.5, base
rates, ROC-AUC, average precision, calibration, per-company replay results)
straight from the append-only history — never recomputed, never fabricated.
`404` until at least one `predictron-benchmark run` has been stored.

## Metrics

`metrics.py` pairs every successful run entry with its verified outcome
into `ScoredSample`s:

- **Confusion**: TP/FP/TN/FN, accuracy, precision, recall, specificity,
  F1, **F0.5**, balanced accuracy, FPR/FNR over scoreable samples.
- **Base rates**: observed success/failure prevalence over verified samples
  (never the engine's label).
- **Ranking**: precision@k, recall@k, top-decile precision, investment hit
  rate (and at threshold 60), plus **ROC-AUC** and **average precision**.
- **Calibration**: Expected Calibration Error (ECE), Maximum Calibration
  Error (MCE), Brier score, reliability-diagram bins, overconfidence flag.
- **Coverage**: dataset vs replayed vs scoreable counts, per
  sector/stage/country.

Rules honored everywhere: `None` outcomes are excluded before any binary
metric; neutral predictions (watch / investigate further) are excluded from
TP/FP/TN/FN; ties in ranking break on ascending `company_id`; every metric
is a pure, deterministic function of `(dataset, run)`.

Segment accuracy by sector/stage/country uses the same scoreable samples.

## Drift

`drift.detect(run_a, run_b)` compares two runs of the same dataset:

- score drift (per-company delta, threshold `SCORE_DRIFT_EPSILON`)
- decision drift (category/confidence changes)
- metric drift (≤1% of samples + metric change > threshold)

`total_count` / `affected_count` give signal magnitude, e.g. 1 of 3
companies affected ≈ 0.33 magnitude.

## History

`BenchmarkHistory(directory)` is an append-only store:

```text
history/
  <run_id>.json    # run envelope + optional metrics block, AtomicWriter
```

- Storing an existing run id raises `HistoryError` (runs are immutable).
- `list_summaries()` lists stored runs; `verify_all()` re-checks hashes.
- `--verify` on the CLI recomputes and confirms stored result hashes.

## Reports

`build_report(kind, ...)` produces one deterministic JSON document.
Kinds: `executive`, `accuracy`, `calibration`, `sectors`, `countries`,
`stages`, `coverage`, `trend`, `engine_drift`, `leaderboard`.

- Metric kinds compute metrics lazily from `dataset`/`run` if not passed.
- `trend` requires `history=`; `engine_drift` requires `run_a=`/`run_b=`.
- Header carries `run_id`, engine/benchmark version, dataset name/hash.

## Exports

`export.py` writes canonical, byte-deterministic JSON:

```text
<run_id>.run.json       # run payload
<run_id>.metrics.json   # metric block
<run_id>.bundle.json    # dataset + run + metrics + reports combined
<run_id>.reports.json   # all report kinds
```

`benchmark-run --export-dir` writes the full bundle; `benchmark-export`
writes just the run and metrics files.

## CLI

```text
predictron-benchmark benchmark-run    --dataset <path|name> [--history DIR]
                                      [--run-id ID] [--no-store] [--export-dir DIR]
predictron-benchmark benchmark-report --run ID [--kind KIND] [--history DIR] [--out FILE]
predictron-benchmark benchmark-history [--history DIR] [--dataset NAME] [--verify]
predictron-benchmark benchmark-compare --run-a ID --run-b ID [--history DIR] [--out FILE]
predictron-benchmark benchmark-export  --run ID [--history DIR] [--out DIR]
```

Example:

```text
predictron-benchmark benchmark-run --dataset example_v1 --run-id baseline
predictron-benchmark benchmark-report --run baseline --kind executive
predictron-benchmark benchmark-compare --run-a baseline --run-b retrained
predictron-benchmark benchmark-export --run retrained --out artifacts/
```

## Quality Gates

The platform and its tests pass the project gates:

- `ruff check` (select `E,F,I,N,W,UP`) on the package and tests.
- `mypy --strict` on `predictron_engine`, `app`, `benchmarks`.
- `pytest` — full suite is green; the ground-truth suite contains 329
  tests covering models, dataset validation, runner determinism, metrics,
  drift, history, reports, exports/CLI, the example dataset, and
  backwards-compat contracts.

## Adding a Real Dataset

1. Curate verified outcomes with sources (never `is_example`).
2. Add the dataset JSON under `benchmarks/golden_datasets/`.
3. Validate: `validate_golden_dataset()`.
4. Run a baseline: `predictron-benchmark benchmark-run --dataset <name>`.
5. Commit the dataset; subsequent engine changes can be measured with
   `benchmark-compare` and `benchmark-report --kind trend`.