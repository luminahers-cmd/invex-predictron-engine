# Feature Store

The Feature Store is a deterministic, explainable, additive feature layer for
the Predictron Engine. It converts raw company data (dataset records, signal
timelines, knowledge graph, benchmarks) into reusable venture intelligence
features that can be consumed by `PredictronEngine` and future analytics.

## Design Principles

- **Deterministic** — the same input data always produces the same feature
  values. No ML, embeddings, LLMs, or randomized logic.
- **Explainable** — every feature carries `evidence_references` pointing back
  to the exact source records, signals, graph nodes, or benchmark context.
- **Grounded** — no fabricated values. A feature is `None` when its source
  data is unavailable.
- **Immutable & Additive** — old feature snapshots are never overwritten;
  registered features can only be extended, never silently removed or changed.
- **Backwards Compatible** — existing engine decision logic is untouched. New
  feature versions or fields degrade gracefully.
- **Production-quality** — strict typing (`mypy --strict`), linting (`ruff`),
  and 300+ tests (`pytest`).

## Architecture

```
predictron_engine/feature_store/
  models.py        Core data models (FeatureSnapshot, CompanyFeatureSet, ...)
  registry.py      FeatureRegistry — registrations, discovery, dependency topo sort
  engine.py        FeatureEngine — deterministic computation with dependency resolution
  store.py         FeatureStore — JSON persistence, snapshots, history, lookup, import/export
  validation.py    FeatureValidator — 7 validation checks on computed features
  reports.py       FeatureReportBuilder — coverage, freshness, dependency, stat reports
  cli.py           Command-line interface (predictron-feature)
  features/        Feature definitions + computors by category
    company.py        5 features (company_age, funding_stage, ...)
    growth.py         5 features (funding_velocity, momentum_score, ...)
    founder.py        3 features (founder_count, repeat_founder_indicator, ...)
    funding.py        5 features (total_funding, investor_count, ...)
    knowledge_graph.py 4 features (graph_degree, graph_density, ...)
    signals.py        4 features (signal_frequency, signal_diversity, ...)
    benchmark.py      3 features (benchmark_similarity, historical_success_rate, ...)
```

### Feature Categories (29 total)

| Category        | Count | Examples |
|-----------------|-------|----------|
| Company         | 5     | `company_age`, `funding_stage`, `employee_band`, `operating_country`, `industry` |
| Growth          | 5     | `funding_velocity`, `hiring_velocity`, `milestone_frequency`, `growth_consistency`, `momentum_score` |
| Founder         | 3     | `founder_count`, `repeat_founder_indicator`, `founder_change_count` |
| Funding         | 5     | `total_funding`, `funding_round_count`, `average_round_size`, `investor_count`, `funding_recency` |
| Knowledge Graph | 4     | `graph_degree`, `graph_density`, `connected_component_size`, `ecosystem_connections` |
| Signals         | 4     | `signal_frequency`, `signal_diversity`, `signal_recency`, `activity_score` |
| Benchmark       | 3     | `benchmark_similarity`, `historical_success_rate`, `sector_accuracy_reference` |

## Key Models

- **`FeatureSnapshot`** — the value of one feature for one company at a point in
  time: value, `ValueType`, `FeatureStatus`, computation version, evidence.
- **`CompanyFeatureSet`** — a company's computed features plus `built_at`.
- **`FeatureStoreSnapshot`** — a versioned, immutable snapshot of the whole store.
- **`FeatureDefinition`** — registered metadata: id, name, category, description,
  value type, dependencies, source fields, tags, min/max.
- **`EvidenceReference`** — `{source_type, source_id, source_field}` showing
  exactly where a value came from.

## Usage

### Python API

```python
from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.features import ALL_FEATURES
from predictron_engine.feature_store.registry import FeatureRegistry

registry = FeatureRegistry()
registry.register_all(ALL_FEATURES)
engine = FeatureEngine(registry)

feature_set = engine.build_company_features(
    record,
    timeline=timeline,          # optional timeline of signals
    knowledge_graph=graph,      # optional KG
    graph_queries=queries,      # optional KG query adapters
    outcome=outcome,            # optional outcome data
    benchmark_context={...},    # optional benchmark context
    as_of=datetime.now(UTC),    # reference point for time-based features
)
```

### Persistence

```python
from predictron_engine.feature_store.store import FeatureStore

store = FeatureStore("data/features")
store.initialize()
store.save_company_features(feature_set)          # latest + history, append-only
loaded = store.load_company_features("company_id")
history = store.get_company_history("company_id")
snap = store.lookup_feature("company_id", "total_funding")
data = store.export_all()                          # deterministic export dict
```

### CLI

```
predictron-feature feature-build   Compute features from the dataset store
predictron-feature feature-report  Generate a feature store report
predictron-feature feature-validate Validate computed features
predictron-feature feature-history Show feature value history
predictron-feature feature-export  Export feature store data
predictron-feature feature-registry List registered features
```

## Validation

`FeatureValidator` runs 7 checks over a computed `CompanyFeatureSet`:

1. Missing features
2. Unknown feature IDs
3. Value range violations (against `min_value`/`max_value`)
4. Staleness (older than `max_age_days`)
5. Computation version mismatch
6. Dependency integrity (dependencies present & resolved)
7. Evidence presence for non-null values

## Reports

`FeatureReportBuilder` produces:

- Feature coverage (computed / missing / failed)
- Category distribution
- Freshness statistics
- Dependency graph summary
- Computation statistics
- Evidence statistics

## Development Checklist

Ruff, strict mypy, and tests are configured in `pyproject.toml`:

```bash
ruff check .
mypy predictron_engine app benchmarks --strict
pytest tests/feature_store/ -q
```