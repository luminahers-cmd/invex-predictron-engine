# Historical Dataset Population — Project V2

This document describes how to populate the historical startup dataset with
real companies so the Predictron Engine can be evaluated empirically.

Everything here is **additive infrastructure** — it does not change the
prediction engine, scoring, reasoning, benchmarks, or APIs.

---

## Supported Sources

The dataset builder supports importing from publicly available datasets.
Each source is implemented as an independent adapter under
`predictron_engine/dataset/sources/`.

| Source | Adapter | Source name | Format | Notes |
|--------|---------|-------------|--------|-------|
| JSON file | `JsonFileSource` | `json_file` | JSON | Array of record objects |
| CSV file | `CsvFileSource` | `csv_file` | CSV | Flexible column mapping |
| SEC EDGAR | `SecEdgarSource` | `sec_edgar` | JSON | Company filings / full-text search exports |
| YC OSS | `YcOssSource` | `yc_oss` | CSV | Y Combinator open-source startup dataset |
| Companies House (UK) | `CompaniesHouseSource` | `companies_house` | CSV | UK open company data |
| US state registries | `UsStateRegistrySource` | `us_state_registry` | CSV | Incorporation registry exports |
| Government registries (aggregate) | `GovRegistrySource` | `gov_registries` | CSV | Sniffs Companies House vs. US state format |

All adapters are **independent** and stateless. They only parse **local
files** — they never scrape websites, make network requests, or download
data. Downloading/preparing source files is a separate operator step.

### Registering all sources

```python
from predictron_engine.dataset import ImportSourceRegistry

registry = ImportSourceRegistry.default()
print(registry.list_sources())
# ['json_file', 'csv_file', 'sec_edgar', 'yc_oss', 'gov_registries']
```

---

## Import Workflow

### CLI

```bash
# Import a JSON file (JSON record array)
predictron-dataset import data.json --dataset ./my_dataset

# Import a CSV file (source inferred from extension)
predictron-dataset import data.csv --dataset ./my_dataset

# Explicitly choose a public source adapter
predictron-dataset import companies.csv --source companies_house --dataset ./my_dataset
predictron-dataset import edgar.json --source sec_edgar --dataset ./my_dataset
predictron-dataset import yc_oss.csv --source yc_oss --dataset ./my_dataset
```

The import command:

1. Reads raw records with the selected source adapter.
2. Validates each raw record (incomplete records are rejected).
3. Normalizes valid records into `DatasetRecord` / `OutcomeRecord`.
4. Attaches field-level provenance to each imported record.
5. Persists the records and outcomes to the store.

Rejected records are reported on stderr with the reason, and the summary
JSON reports `imported` / `rejected` counts.

### Python API

```python
from predictron_engine.dataset import (
    ImportPipeline,
    ImportSourceRegistry,
    DatasetStore,
)

store = DatasetStore("./my_dataset")
store.initialize()

registry = ImportSourceRegistry.default()
source = registry.get("sec_edgar")
pipeline = ImportPipeline(source)
result = pipeline.run("edgar.json")

for record in result.imported_records:
    store.save_record(record)
for outcome in result.imported_outcomes:
    store.save_outcome(outcome)
```

### Post-import analysis

```bash
# Validate dataset integrity (duplicates, chronology, orphans)
predictron-dataset verify --dataset ./my_dataset

# Report duplicate startup records
predictron-dataset dedup --dataset ./my_dataset

# Print statistics (sectors, stages, years, countries, missing, duplicates)
predictron-dataset statistics --dataset ./my_dataset
```

---

## Validation Rules

Import-time validation runs on each raw record before normalization.  A
record with any validation error is rejected and never stored.

### Required fields

| Source | Required fields |
|--------|-----------------|
| `json_file` / `csv_file` | `startup_name`, `website` |
| `sec_edgar` | `company_name` |
| `yc_oss` | company name |
| `companies_house` / `us_state_registry` / `gov_registries` | company name |

> Note: government registries (Companies House, US state registries) do
> **not** publish websites or prediction fields.  Records imported from
> these sources are intentionally flagged as incomplete (missing website)
> unless enriched from another source.  They are still accepted at import
> time.

### Field-level validation (`validate_record_fields`)

Applied to a `DatasetRecord` to reject incomplete or inconsistent records:

- `startup_name` non-empty
- `website` non-empty with an `http(s)` scheme
- `engine_version` non-empty
- `analysis_date` not in the future
- `analysis_date` year >= 2000
- `prediction.confidence` in `[0, 1]`
- `prediction.composite_score` in `[0, 100]`
- each `dimension_score` in `[0, 100]`

### Outcome validation (`validate_outcome_fields`)

- `record_id` non-empty
- `latest_verification_date` not in the future
- `shutdown_date` not in the future
- `bankruptcy_date >= shutdown_date` (when both present)
- `acquisition_price_usd >= 0`
- `total_funding_usd >= 0`
- each funding round `amount_usd >= 0`
- `time_horizon_days >= 0`

### Store-level validation (`validate_dataset`)

Complements the existing integrity validation:

- duplicate record/outcome/evaluation IDs
- missing predictions / outcomes
- orphan outcomes / evaluations
- future timestamps
- invalid chronology (`recorded_at` before `analysis_date`)

Validation never silently repairs data — it reports issues deterministically.

---

## Temporal Integrity Requirements

The dataset enforces strict temporal consistency so that a prediction and
its ground truth are unambiguous about timing:

- **`analysis_date`** — when the engine analysed the startup (UTC). Must
  not be in the future relative to the import time.
- **`analysis_metadata.recorded_at`** — when the record was created. Must
  be **at or after** `analysis_date`; an earlier `recorded_at` is flagged
  as `invalid_chronology`.
- **`outcome.latest_verification_date`** — when the outcome state was last
  verified. Must not be in the future.
- **`OutcomeRecord.time_horizon_days`** — days between `analysis_date` and
  outcome observation. Must be non-negative when present.
- **`shutdown_date` / `bankruptcy_date`** — must not be in the future, and
  `bankruptcy_date` must not precede `shutdown_date`.

Rule: **No date may precede the analysis date for the same startup's
prediction context, and no date may be in the future.** The validation
utilities report violations but never correct them.

---

## Provenance Tracking (Part D)

Every imported field records its source and retrieval date.  Provenance is
stored under `analysis_metadata["provenance"]` as a mapping of
`field_name -> {source, retrieval_date}`.

```python
from predictron_engine.dataset.provenance import ProvenanceTracker

# Provenance is attached automatically by ImportPipeline
tracker = ProvenanceTracker()          # uses now() by default
provenance = tracker.build_provenance(raw, ["startup_name", "website"])
```

Retrieval date can be pinned for reproducibility:

```python
result = pipeline.run("data.json", retrieval_date=datetime(2024, 1, 1, tzinfo=UTC))
```

`ProvenanceTracker.extract_from_record(record)` retrieves a record's
stored provenance.

---

## Deduplication (Part E)

Deduplication uses website, company name, and normalized identifiers.

Normalization rules:

- **Website**: strip scheme, `www.` prefix, trailing slash, query and
  fragment; lowercase the host.
- **Company name**: lowercase, strip legal suffixes (`inc`, `llc`, `corp`,
  `ltd`, `gmbh`, etc.), collapse to alphanumeric characters.
- **Identifiers**: any of `sec_cik`, `company_number`, `crunchbase_url`,
  `normalized_identifier` from `analysis_metadata`.

Matching priority: normalized identifier > website > company name.

```python
from predictron_engine.dataset import find_duplicates, match_record_to_store

report = find_duplicates(records)          # groups of duplicates
matched, method = match_record_to_store(record, existing_records)
```

The dedup report includes group count, duplicate record count, and which
method triggered each group.  Deduplication is non-destructive — it only
identifies duplicates; the operator decides how to handle them.

---

## Dataset Statistics (Part F)

```python
from predictron_engine.dataset import compute_dataset_stats

stats = compute_dataset_stats(store)
stats.to_dict()
```

Statistics include:

- **startups imported** (record count and distinct startup count)
- **sectors** (from `metadata.sector`)
- **stages** (from `metadata.funding_stage_at_analysis`)
- **years** (from `analysis_date.year`)
- **countries** (from `metadata.country`)
- **missing-field report** (counts per missing field, via the completeness
  validator)
- **duplicate report** (groups, counts, and IDs)

---

## Constraints

This module does NOT:

- Modify `PredictronEngine`
- Modify `ReportBuilder` or scoring/reasoning/confidence
- Modify benchmarks or regenerate snapshots
- Modify any existing public API
- Fabricate historical outcomes
- Scrape websites or make network requests
- Change any prediction behavior

All existing tests remain green. Ruff and Mypy are clean.
