# Company Signals & Temporal Intelligence

## Overview

The Company Signals layer (Project E4) turns the static dataset into companies
that evolve over time. `predictron_engine/dataset/signals/` defines an
immutable, evidence-backed event model (**CompanySignal**), a chronologically
ordered per-company view (**CompanyTimeline**), a deterministic trend engine,
aggregation metrics, statistics, report builders, validation, import, and JSON
persistence under the dataset store's `signals/` directory.

Signals are **grounded**: extraction and imports never fabricate a timestamp,
source, or payload. Every signal carries the record(s) that produced it, and
every signal ID is a deterministic SHA-256 digest of its canonical payload, so
duplicate prevention and merging are stable across runs and machines.

**This module does NOT modify any engine scoring, reasoning, confidence,
recommendation, or threshold behavior. Nothing here predicts.** Every trend
and metric is derived from recorded history, and all reporting accepts an
explicit `as_of` reference timestamp so reads are reproducible.

## Architecture

```
predictron_engine/dataset/signals/
├── model.py       # SignalType vocabulary, CompanySignal, EvidenceReference, IDs
├── timeline.py    # CompanyTimeline (immutable, deduped, sorted) + TimelineStore
├── validation.py  # validate_signal / validate_timeline / SignalValidationReport
├── persistence.py # Versioned JSON timeline contract (write/read helpers)
├── builder.py     # OutcomeSignalExtractor, CompanySignalBuilder
├── importers.py   # import_signals_from_json, parse_signal_record
├── aggregate.py   # momentum, recent activity, funding/growth cadence, freshness
├── trends.py      # TrendEngine, TrendResult, compute_*_trend functions
├── metrics.py     # SignalStatistics, compute_signal_statistics
├── reports.py     # deterministic JSON report builders
└── integration.py # SignalDatasetManager — ties signals to the dataset store
```

## Signal Vocabulary

`CompanySignal` is an immutable frozen dataclass with:

- `company_id` — a company identity, aligned with the knowledge graph
  (`company:<sorted record ids>`).
- `signal_type` — one of the sixteen canonical `SignalType` values.
- `timestamp` — a **tz-aware** datetime, normalized to UTC on construction.
- `source` — where the observation came from (literal, never invented).
- `confidence` — a float in `[0, 1]`.
- `evidence` / `sources` — ground truth references: `EvidenceReference` objects
  (kind, value, references) plus the record IDs that produced the signal.
- `metadata` — optional payload (e.g. `amount_usd`, `count`, `name`).

The sixteen canonical signal types:

| Signal Type | Meaning |
|-------------|---------|
| `funding_round` | A dated funding round with an amount |
| `hiring_growth` | Headcount increases from an employee history |
| `layoffs` | Headcount decreases from an employee history |
| `acquisition` | The company was acquired |
| `ipo` | The company went public |
| `bankruptcy` | Bankruptcy proceeding observed |
| `shutdown` | The company shut down |
| `founder_change` | A change in the founding team |
| `investor_added` | A new investor observed in a round |
| `valuation_update` | A dated valuation observation |
| `revenue_milestone` | A revenue threshold crossed |
| `arr_milestone` | An ARR threshold crossed |
| `employee_milestone` | An employee-count threshold crossed |
| `product_launch` | A product launch |
| `partnership` | A announced partnership |
| `regulatory_approval` | A regulatory approval/clearance |

`EXTERNAL_ONLY_SIGNAL_TYPES` lists five types that are produced **only** by
external import — the deterministic builder cannot ground them in stored data:
`founder_change`, `product_launch`, `partnership`, `regulatory_approval`,
`revenue_milestone`. Note that `layoffs` is deliberately **not** in that set:
the builder emits grounded `layoffs` signals from dated headcount decreases.

## Determinism & Deduplication

- `signal_canonical_dict(signal)` serializes the canonical payload (type,
  timestamp, company, confidence, evidence, metadata) with sorted keys.
- `compute_signal_id(signal)` is the SHA-256 of that canonical JSON; the same
  observation in any order of ingestion produces the same ID.
- `sort_and_dedupe(signals)` orders by `(timestamp, signal_id)` and keeps one
  signal per ID, making timeline assembly order-independent.

## Timelines

`CompanyTimeline` is an immutable, chronologically ordered sequence of signals
for one company:

- `build(company_id, signals)` / `append(*new_signals)` / `merge(other)` — all
  return **new** timelines; history is never mutated.
- `signal_count`, `has(signal_id)`, `find(signal_id)`, `first`, `last`,
  `span_days`, `types` (a `Counter`), `by_type`, `signals_since`.
- `to_dict()` / `from_dict()` and, in `persistence.py`, file helpers
  `write_timeline` / `read_timeline` under a versioned schema
  (`SIGNALS_SCHEMA_VERSION = "1.0.0"`). Timeline files live in the store's
  `signals/` directory, encoded by company ID, alongside the zero-write
  `list_signal_company_ids` helper.

`TimelineStore` is the in-memory companion: `put`, `get`, `remove`,
`list_companies`, `count`, `all`, `total_signals`.

## Grounded Building

`builder.py` derives signals only from stored records and outcomes:

- **`OutcomeSignalExtractor`** reads `OutcomeRecord` funding rounds,
  valuations, ARR milestones, employee history, and terminal events and emits
  `funding_round`, `investor_added`, `valuation_update`, `arr_milestone`,
  `employee_milestone`, `hiring_growth`, `layoffs`, `acquisition`, and so on.
  Funding storms and milestone waves are staggered by their recorded dates.
- **`CompanySignalBuilder.build(records, outcomes=None)`** runs the same
  `EntityResolver` used by the knowledge graph, so signal `company_id` values
  match graph company nodes (`record_to_company_ids` maps record IDs to those
  IDs), then extracts per-company timelines and returns a `SignalBuildResult`
  with a `SignalBuildReport` (timeline/signal counts per type).

Because company IDs are shared with Entity Resolution (E2) and the graph (E3),
a signal lookup and a graph query for the same company always agree.

## Import

`importers.py` ingests external JSON:

- `parse_signal_record(record, company_id=None)` — validates a signal record
  (`company_id`, `signal_type`, `timestamp`, `source`) and returns either a
  parsed `CompanySignal` or a concrete error.
- `import_signals_from_json(path, company_id=None)` — returns a
  `SignalImportReport` with counts for requested / imported / rejected /
  deduplicated signals plus per-error breakdowns. Signal IDs make repeated
  imports idempotent: already-seen payloads are counted but not re-added.

`SignalDatasetManager` (in `integration.py`) binds all of this to the dataset:

- `company_id_for(key)` / `company_map()` / `distinct_startup_count()` —
  identity resolution against stored records.
- `build()` — resolve + extract + persist timelines into the store.
- `import_file(path)` — import external signals and merge into timelines.
- `timeline(company_id)` / `all_timelines()` — read views (None when absent).
- `aggregate(company_id, as_of=None)` / `trends(company_id, as_of=None)`.
- `validate()` / `statistics(as_of=None)` / `report(as_of=None)`.
- `profile_highlights(company_id, as_of=None)` —
  at most 5 human-readable highlights,
  and `graph_overview()` — per-company signal counts and type breakdown.

## Trends

`trends.py` implements `TrendEngine` with a fully deterministic,
non-forecasting model. Each trend returns a `TrendResult` with normalized
`value`, `direction` (`accelerating` / `stable` / `declining` / their
variants), `available` flag, and a human-readable `explanation`.

| Trend | Meaning |
|-------|---------|
| `compute_funding_velocity` | USD per year across funding rounds (or rounds per year when amounts are missing) |
| `compute_funding_acceleration` | First-half vs second-half funding rates, annualized |
| `compute_hiring_trend` | Slope of dated headcount observations, annualized by 365.25-day year |
| `compute_growth_consistency` | Coefficient of variation over interval counts |
| `compute_stagnation` | How long activity has been quiet; empty timelines report `value=1.0`, `direction="stagnant"`, `available=True` |
| `compute_decline` | Share of recent decline indicators discovered in the history |

Every trend is explainable: results carry the inputs and intervals behind the
computed value, and an empty/insufficient timeline makes a trend
`available=False` instead of inventing a number.

## Aggregation & Metrics

`aggregate.py`:

- `recent_activity(timeline, as_of=None)` — signal counts by recency window.
- `momentum_score(timeline, as_of=None)` / `momentum_summary` /
  `momentum_ranking` — a bounded, explainable activity momentum index.
- `funding_cadence` / `growth_cadence` / `signal_frequency` /
  `signal_freshness` (with `aggregate_all` bundling everything).

`metrics.py`:

- `SignalStatistics` — companies with signals, total signals, coverage ratio,
  signals per company, funding event count / total USD, type counts,
  confidence histogram, freshness histogram, and an activity heatmap by period.
- `compute_signal_statistics(timelines, as_of=None)` — deterministic across
  input order.

## Reports

`reports.py` builds deterministic, JSON-serializable documents:

- `build_signal_dataset_report` — full dataset view (statistics, distribution,
  coverage, momentum statistics, freshness).
- `build_timeline_report` — one company's timeline and derived views.
- `build_signal_distribution`, `build_coverage_report`,
  `build_momentum_statistics`, `build_dataset_freshness_report`,
  `build_activity_heatmap`, `build_trend_summary_report`.

All report functions accept `as_of` so output is reproducible.

## Validation

`validation.py` checks every invariant:

- timestamps are tz-aware (reject naive datetimes),
- `signal_type` is canonical,
- `confidence` is within `[0, 1]`,
- `company_id` and `source` are non-empty,
- `signal_id` matches the recomputed canonical ID,
- evidence/sources reference real record IDs.

`validate_signal` / `validate_signals` / `validate_timeline` return a
`SignalValidationReport` with `is_valid`, `issue_count`, and a list of
`SignalValidationIssue` records (kind + message), so a clean store is
verifiable programmatically.

## Persistence

Timelines persist as versioned JSON documents under the dataset store's
`signals/` directory:

```
root/signals/{encoded_company_id}.json
```

`timeline_to_dict` / `timeline_from_dict` plus `write_timeline` /
`read_timeline` provide a strict round-trip against
`SIGNALS_SCHEMA_VERSION = "1.0.0"`. `DatasetStore` gains additive methods
(`save_timeline`, `load_timeline`, `save_signal`, `list_signal_company_ids`,
`count_signal_timelines`) and an eagerly-created (but empty) `signals/`
directory during `initialize()` — the original record/outcome/evaluation/run
APIs are untouched.

## CLI

Commands are exposed on the existing `predictron-dataset` console script:

```
predictron-dataset signal-import <file.json> [--dataset PATH] [--company ID]
    Import external signals; print imported/rejected/deduplicated counts.

predictron-dataset signal-report [--dataset PATH] [--as-of ISO]

predictron-dataset timeline <company-id> [--dataset PATH]
    Show a company's chronological signal timeline (JSON).

predictron-dataset trend-report [--company ID] [--dataset PATH]
    [--as-of ISO] [--output FILE]
    Emit the deterministic trend report.

predictron-dataset signal-validate [--dataset PATH]
    Validate all stored timelines; exit 0 when valid, 1 otherwise.
```

All commands are read-only except `signal-import` (which merges into the
store) and write JSON to stdout (or `--output`).

## Relation to Entity Resolution and the Knowledge Graph

- Company IDs are inherited from Entity Resolution (E2): the same
  `EntityResolver` produces the `company:<sorted record ids>` strings used as
  graph company nodes, so E2 → E3 → E4 views stay aligned.
- The signal builder consumes the same `OutcomeRecord` stream the graph does;
  outcome-derived investors/rounds become signals rather than graph edges, and
  both layers treat outcomes as *observed* facts, never predictions.
- The signal layer is strictly additive: importing `predictron_engine.dataset
  .signals` and extending `DatasetStore` changes nothing about the original
  engine, graph, or dataset behaviors (guarded by a backwards-compatibility
  test suite, e.g. the graph keeps exactly 12 node and 12 edge types).

## Recommended Next Milestones

1. **Sector/cohort trend aggregation** — trend statistics bucketed by
   industry/country from the graph, remaining deterministic.
2. **Timeline lineage** — persist per-build manifest (input record/outcome
   counts) alongside timelines for reproducible rebuilds.
3. **Richer decline analysis** — compose `compute_decline` with a wider set of
   documented indicators while keeping `explanation` human-readable.
4. **Exported datasets** — materialize a long-format `company, date, signal`
   table for downstream consumers via the existing `export` contract.