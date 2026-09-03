# Historical Startup Dataset Builder

## Overview

The Historical Startup Dataset Builder is a standalone module within the
Predictron Engine that enables collecting, storing, and evaluating historical
startup prediction data. Its purpose is to build the infrastructure for
empirical validation of the engine's predictions.

**This module does NOT modify any engine behavior.** It is purely data
collection infrastructure.

## Architecture

```
predictron_engine/dataset/
├── __init__.py           # Public API surface
├── models.py             # Part A: Historical dataset record schema
├── outcomes.py           # Part B: Outcome tracking models
├── imports.py            # Part C: Pluggable import pipeline
├── csv_import.py         # Part A: CSV import adapter
├── evaluation.py         # Part D: Immutable evaluation storage
├── store.py              # JSON-file persistence backend
├── evaluation_pipeline.py # Part C: Evaluation orchestration
├── metrics.py            # Part D: Aggregate evaluation metrics
├── analysis.py           # Part B: Engine integration (AnalysisPipeline)
├── analysis_support.py   # Request building / report extraction helpers
├── validation.py         # Dataset integrity validation
├── validation_utils.py   # Field-level validation utilities (Part B)
├── provenance.py         # Field-level provenance tracking (Part D)
├── dedup.py              # Deduplication logic (Part E)
├── statistics.py         # Dataset statistics generation (Part F)
├── reports.py            # Deterministic dataset JSON reports
├── sources/              # Public dataset source adapters (Part C)
│   ├── sec_edgar.py      # SEC EDGAR adapter
│   ├── yc_oss.py         # Y Combinator OSS adapter
│   └── gov_registries.py # Companies House / US state registry adapters
└── cli.py                # predictron-dataset command-line interface
```

## Part A — Historical Dataset Schema

### DatasetRecord

The atomic unit of the dataset. One record per analysis run.

| Field | Type | Description |
|-------|------|-------------|
| `record_id` | `str (UUID)` | Unique identifier |
| `startup_name` | `str` | Startup name as submitted |
| `website` | `str` | Startup website URL |
| `analysis_date` | `datetime` | UTC timestamp of the analysis |
| `engine_version` | `str` | Engine version used |
| `benchmark_version` | `str | None` | Benchmark version, if any |
| `evidence_bundle_reference` | `str | None` | Evidence bundle identifier |
| `prediction` | `PredictionSummary` | Engine prediction snapshot |
| `funding_stage_at_analysis` | `FundingStage` | Known funding stage |
| `analysis_metadata` | `dict` | Arbitrary preserved metadata |
| `tags` | `list[str]` | User-defined tags |
| `source` | `str` | Origin: 'direct', 'import', 'manual' |

### PredictionSummary

Immutable snapshot of engine output at analysis time.

| Field | Type | Description |
|-------|------|-------------|
| `decision` | `DecisionLabel` | Engine decision |
| `confidence` | `float (0-1)` | Overall confidence |
| `composite_score` | `float (0-100)` | Composite venture score |
| `dimension_scores` | `dict[str, float]` | Per-dimension scores |
| `investment_readiness_score` | `float | None` | Readiness score |
| `recommendation_count` | `int` | Number of recommendations |

## Part B — Outcome Tracking

### OutcomeRecord

Links a DatasetRecord to its observed future outcome.

| Field | Type | Description |
|-------|------|-------------|
| `outcome_id` | `str (UUID)` | Unique outcome identifier |
| `record_id` | `str` | Linked dataset record |
| `outcome` | `StartupOutcome` | Observed outcome data |
| `verdict` | `OutcomeVerdict` | Derived categorical verdict |
| `verdict_reasoning` | `str` | How verdict was determined |
| `time_horizon_days` | `int | None` | Days between analysis and outcome |
| `status` | `OutcomeStatus` | Verification status |

### StartupOutcome

Observable outcome data. All fields allow None or UNKNOWN.

| Field | Type | Description |
|-------|------|-------------|
| `funding_rounds` | `list[FundingEvent]` | Funding history |
| `total_funding_usd` | `float | None` | Total funding raised |
| `acquisition` | `str | None` | Acquirer name |
| `shutdown` | `bool | None` | Shutdown status |
| `bankruptcy` | `bool | None` | Bankruptcy status |
| `arr_milestones` | `list[dict]` | ARR snapshots |
| `employee_count_history` | `list[dict]` | Headcount snapshots |
| `valuation_history` | `list[dict]` | Valuation snapshots |
| `investors` | `list[str]` | Known investors |
| `exit_date` | `datetime | None` | Exit date |
| `exit_type` | `str | None` | Exit type |
| `latest_verification_date` | `datetime | None` | Last verified date |

**Rule: Never fabricate labels.** All outcome fields default to UNKNOWN
and are populated only through verified imports or manual verification.

## Part C — Import Pipeline

### Interfaces

#### ImportSource Protocol

```python
class ImportSource(Protocol):
    @property
    def source_name(self) -> str: ...
    def read(self, path: str) -> list[RawImportRecord]: ...
    def validate(self, record: RawImportRecord) -> list[str]: ...
```

#### ImportSourceRegistry

Registry of pluggable source adapters:

```python
registry = ImportSourceRegistry.default()  # includes JsonFileSource
registry.register(MyCustomSource())        # add new adapters
source = registry.get("json_file")
pipeline = ImportPipeline(source)
result = pipeline.run("/path/to/data.json")
```

#### RawImportRecord

Intermediate representation for normalization:

```python
@dataclass
class RawImportRecord:
    startup_name: str
    website: str
    analysis_date: datetime | None
    engine_version: str
    prediction_data: dict[str, object]
    outcome_data: dict[str, object]
    metadata: dict[str, object]
    tags: list[str]
    source: str
```

#### Built-in Sources

- `JsonFileSource` (`json_file`): Reads JSON arrays of record objects
- `CsvFileSource` (`csv_file`): Reads CSV files with flexible column mapping
- `SecEdgarSource` (`sec_edgar`): Parses SEC EDGAR company filing exports
- `YcOssSource` (`yc_oss`): Parses Y Combinator OSS startup dataset CSV
- `GovRegistrySource` (`gov_registries`): Parses government registry CSV exports (Companies House UK, US state registries)

### Creating a Custom Source

Implement the `ImportSource` protocol:

```python
class MyCustomSource:
    @property
    def source_name(self) -> str:
        return "my_custom"

    def read(self, path: str) -> list[RawImportRecord]:
        # Parse source and yield RawImportRecords
        ...

    def validate(self, record: RawImportRecord) -> list[str]:
        errors = []
        if not record.startup_name:
            errors.append("startup_name is required")
        return errors

# Register with the pipeline:
from predictron_engine.dataset import ImportPipeline, ImportSourceRegistry

registry = ImportSourceRegistry.default()
registry.register(MyCustomSource())
source = registry.get("my_custom")
pipeline = ImportPipeline(source)
result = pipeline.run("/path/to/my_data")
```

## Part D — Evaluation Storage

### PredictionEvaluation

Immutable comparison of a prediction against an outcome.

```python
evaluation = PredictionEvaluation.from_records(
    dataset_record=record,
    outcome_record=outcome,
)
```

The prediction is **frozen** (copied) at evaluation creation time
and never modified. The evaluation derives:

- `verdict`: CORRECT / PARTIALLY_CORRECT / INCORRECT / INCONCLUSIVE
- `alignment`: STRONG_MATCH / PARTIAL_MATCH / MISMATCH / NEUTRAL

### EvaluationMetadata

Records when and how the evaluation was performed:

```python
class EvaluationMetadata(BaseModel):
    evaluated_at: datetime
    evaluator_version: str
    evaluation_config: dict[str, Any]
    notes: str
```

## Storage Layout

```
dataset_store/
├── records/
│   ├── {record_id_1}.json
│   ├── {record_id_2}.json
│   └── ...
├── outcomes/
│   ├── {outcome_id_1}.json
│   ├── {outcome_id_2}.json
│   └── ...
├── evaluations/
│   ├── {evaluation_id_1}.json
│   ├── {evaluation_id_2}.json
│   └── ...
└── manifest.json
```

## Workflow

### 1. Recording a Prediction

```python
from predictron_engine.dataset import DatasetRecord, PredictionSummary, DatasetStore

record = DatasetRecord(
    startup_name="Acme Corp",
    website="https://acme.com",
    engine_version="0.12.1",
    prediction=PredictionSummary(
        decision="invest",
        confidence=0.75,
        composite_score=68.5,
    ),
)

store = DatasetStore("./my_dataset")
store.initialize()
store.save_record(record)
```

### 2. Importing Historical Data

```python
from predictron_engine.dataset import ImportPipeline, ImportSourceRegistry

registry = ImportSourceRegistry.default()
source = registry.get("json_file")
pipeline = ImportPipeline(source)
result = pipeline.run("/path/to/historical_data.json")

for record in result.imported_records:
    store.save_record(record)
for outcome in result.imported_outcomes:
    store.save_outcome(outcome)
```

### 3. Evaluating Predictions

```python
from predictron_engine.dataset import PredictionEvaluation

record = store.load_record("some-record-id")
outcome = store.find_outcome_by_record("some-record-id")

if outcome:
    evaluation = PredictionEvaluation.from_records(record, outcome)
    store.save_evaluation(evaluation)
```

### 4. Querying Results

```python
# Find all records for a startup
records = store.find_records_by_startup("Acme Corp")

# Find outcome for a specific record
outcome = store.find_outcome_by_record("record-id")

# Find evaluation for a specific record
evaluation = store.find_evaluation_by_record("record-id")

# List all stored IDs
record_ids = store.list_records()
outcome_ids = store.list_outcomes()
evaluation_ids = store.list_evaluations()
```

## Validation Process

1. **Import validation**: Each ImportSource validates raw records before normalization
2. **Model validation**: Pydantic models enforce field constraints at creation time
3. **Outcome derivation**: `OutcomeRecord.derive_verdict()` deterministically derives verdicts from outcome data
4. **Evaluation derivation**: `PredictionEvaluation.from_records()` deterministically derives evaluation verdicts

## Update Policy

- Outcome records are updated when new verified information becomes available
- Dataset records are immutable once created (prediction snapshots)
- Evaluations can be re-derived but individual evaluations are immutable
- The `updated_at` field on OutcomeRecord tracks when outcomes were last verified
- The `latest_verification_date` on StartupOutcome tracks when the outcome data itself was last verified

## Provenance

Every record tracks its provenance:

- `source` field: 'direct' (live analysis), 'import' (historical data), 'manual' (hand-entered)
- `engine_version`: Which engine version produced the prediction
- `benchmark_version`: Which benchmark version was used
- `evidence_bundle_reference`: Which evidence bundle was referenced
- `analysis_metadata`: Arbitrary preserved metadata from the analysis run
- `evaluation_metadata`: When and how evaluations were performed

## Temporal Integrity

- `analysis_date`: When the analysis was performed (UTC)
- `outcome.created_at` / `outcome.updated_at`: When the outcome record was created/updated
- `outcome.outcome.latest_verification_date`: When the outcome data was last verified
- `evaluation.created_at`: When the evaluation was performed
- `evaluation.evaluation_metadata.evaluated_at`: When the evaluation was performed
- `time_horizon_days`: Computed gap between analysis and outcome observation

## Constraints

This module does NOT:

- Modify `PredictronEngine`
- Modify `ReportBuilder`
- Modify scoring, reasoning, confidence, recommendations, or thresholds
- Modify benchmarks or regenerate snapshots
- Fabricate historical outcomes
- Introduce external services
- Change any prediction behavior

All existing engine tests pass. Ruff and Mypy are clean.
