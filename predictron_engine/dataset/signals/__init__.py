"""Company Signals & Temporal Intelligence (Project E4).

A deterministic layer that turns the static dataset into companies that
evolve over time.  The package provides:

- :class:`CompanySignal` — an immutable, dated, evidence-backed observation
  about a company (16 canonical signal types).
- :class:`CompanyTimeline` — an immutable, chronologically ordered sequence
  of signals per company; append/merge always return new timelines so
  history is never mutated.
- :class:`TrendEngine` — deterministic, explainable trends (funding
  velocity/acceleration, hiring trend, growth consistency, stagnation,
  decline) with no forecasting.
- Aggregation metrics: recent activity, momentum, funding/growth cadence,
  signal frequency, and freshness.
- Statistics and report builders for signal distributions, activity
  heatmaps, momentum statistics, dataset freshness, and coverage.
- :class:`SignalDatasetManager` — ties signals into the dataset store,
  entity resolution, knowledge graph, profiles, outcomes, and CLI.
- Deterministic JSON persistence under a store's ``signals/`` directory.

Design rules
------------
- Signals are immutable and grounded: extraction and imports never
  fabricate a timestamp, source, or payload.
- ``signal_id`` is a SHA-256 digest of the canonical signal payload, so
  duplicate prevention and merging are deterministic.
- Nothing here predicts; every value is derived from recorded history.
- All reporting accepts an explicit ``as_of`` reference timestamp.

Public API
----------
model : SignalType, CompanySignal, EvidenceReference, serialization helpers.
timeline : CompanyTimeline, TimelineStore.
validation : validate_signal / validate_timeline / SignalValidationReport.
builder : OutcomeSignalExtractor, CompanySignalBuilder, record_to_company_ids.
importers : import_signals_from_json, SignalImportReport.
aggregate : recent_activity, momentum_score, funding_cadence, ...
trends : TrendEngine, TrendResult, compute_*_trend functions.
metrics : SignalStatistics, compute_signal_statistics.
reports : build_timeline_report, build_signal_dataset_report, ...
persistence : timeline_to_dict / timeline_from_dict, write/read helpers.
integration : SignalDatasetManager.
"""

from predictron_engine.dataset.signals.aggregate import (
    aggregate_all,
    funding_cadence,
    growth_cadence,
    momentum_ranking,
    momentum_score,
    momentum_summary,
    recent_activity,
    signal_frequency,
    signal_freshness,
)
from predictron_engine.dataset.signals.builder import (
    CompanySignalBuilder,
    OutcomeSignalExtractor,
    SignalBuildReport,
    SignalBuildResult,
    record_to_company_ids,
)
from predictron_engine.dataset.signals.importers import (
    SignalImportReport,
    import_signals_from_json,
    parse_signal_record,
)
from predictron_engine.dataset.signals.integration import SignalDatasetManager
from predictron_engine.dataset.signals.metrics import (
    SignalStatistics,
    compute_signal_statistics,
)
from predictron_engine.dataset.signals.model import (
    EXTERNAL_ONLY_SIGNAL_TYPES,
    SIGNAL_TYPES,
    CompanySignal,
    EvidenceReference,
    SignalType,
    compute_signal_id,
    signal_from_dict,
    signal_to_dict,
)
from predictron_engine.dataset.signals.persistence import (
    SIGNALS_SCHEMA_VERSION,
    list_signal_company_ids,
    read_timeline,
    timeline_from_dict,
    timeline_to_dict,
    write_timeline,
)
from predictron_engine.dataset.signals.reports import (
    build_activity_heatmap,
    build_coverage_report,
    build_dataset_freshness_report,
    build_momentum_statistics,
    build_signal_dataset_report,
    build_signal_distribution,
    build_timeline_report,
    build_trend_summary_report,
)
from predictron_engine.dataset.signals.timeline import (
    CompanyTimeline,
    TimelineStore,
    sort_and_dedupe,
)
from predictron_engine.dataset.signals.trends import (
    TrendEngine,
    TrendResult,
    compute_decline,
    compute_funding_acceleration,
    compute_funding_velocity,
    compute_growth_consistency,
    compute_hiring_trend,
    compute_stagnation,
)
from predictron_engine.dataset.signals.validation import (
    SignalValidationIssue,
    SignalValidationReport,
    validate_signal,
    validate_signals,
    validate_timeline,
)

__all__ = [
    "CompanySignal",
    "CompanySignalBuilder",
    "CompanyTimeline",
    "EXTERNAL_ONLY_SIGNAL_TYPES",
    "EvidenceReference",
    "OutcomeSignalExtractor",
    "SIGNALS_SCHEMA_VERSION",
    "SIGNAL_TYPES",
    "SignalBuildReport",
    "SignalBuildResult",
    "SignalDatasetManager",
    "SignalImportReport",
    "SignalStatistics",
    "SignalType",
    "SignalValidationIssue",
    "SignalValidationReport",
    "TimelineStore",
    "TrendEngine",
    "TrendResult",
    "aggregate_all",
    "build_activity_heatmap",
    "build_coverage_report",
    "build_dataset_freshness_report",
    "build_momentum_statistics",
    "build_signal_dataset_report",
    "build_signal_distribution",
    "build_timeline_report",
    "build_trend_summary_report",
    "compute_decline",
    "compute_funding_acceleration",
    "compute_funding_velocity",
    "compute_growth_consistency",
    "compute_hiring_trend",
    "compute_signal_id",
    "compute_signal_statistics",
    "compute_stagnation",
    "funding_cadence",
    "growth_cadence",
    "import_signals_from_json",
    "list_signal_company_ids",
    "momentum_ranking",
    "momentum_score",
    "momentum_summary",
    "parse_signal_record",
    "read_timeline",
    "recent_activity",
    "record_to_company_ids",
    "signal_frequency",
    "signal_freshness",
    "signal_from_dict",
    "signal_to_dict",
    "sort_and_dedupe",
    "timeline_from_dict",
    "timeline_to_dict",
    "validate_signal",
    "validate_signals",
    "validate_timeline",
    "write_timeline",
]
