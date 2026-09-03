"""Historical Startup Dataset Builder.

Standalone module for collecting, storing, and evaluating historical
startup prediction data.  Designed to eventually enable empirical
validation of the Predictron Engine without modifying any engine
behavior.

Modules
-------
models : Historical dataset record schema (Part A).
outcomes : Future outcome tracking models (Part B).
imports : Pluggable import pipeline interfaces (Part C).
evaluation : Immutable prediction evaluation storage (Part D).
store : JSON-based dataset storage backend.
analysis : Engine integration pipeline (Part B).
evaluation_pipeline : Evaluation pipeline (Part C).
metrics : Aggregate evaluation metrics (Part D).
reports : Dataset JSON reports (Part E).
validation : Dataset validation (Part F).
cli : Command-line interface (Part A).
"""

from predictron_engine.dataset.analysis import (
    AnalysisPipeline,
    AnalysisResult,
    AnalysisRun,
)
from predictron_engine.dataset.dedup import (
    DeduplicationReport,
    find_duplicates,
)
from predictron_engine.dataset.evaluation import (
    EvaluationMetadata,
    PredictionEvaluation,
)
from predictron_engine.dataset.evaluation_pipeline import (
    EvaluationBatchResult,
    EvaluationPipeline,
)
from predictron_engine.dataset.imports import (
    ImportPipeline,
    ImportSource,
    ImportSourceRegistry,
    RawImportRecord,
)
from predictron_engine.dataset.metrics import (
    BinaryLabel,
    EvaluationMetrics,
    compute_evaluation_metrics,
)
from predictron_engine.dataset.models import (
    DatasetRecord,
    PredictionSummary,
)
from predictron_engine.dataset.outcomes import (
    FundingEvent,
    OutcomeRecord,
    OutcomeStatus,
    OutcomeVerdict,
    StartupOutcome,
)
from predictron_engine.dataset.provenance import (
    FieldProvenance,
    ProvenanceTracker,
)
from predictron_engine.dataset.reports import DatasetReportBuilder
from predictron_engine.dataset.statistics import (
    DatasetStats,
    MissingFieldReport,
    compute_dataset_stats,
)
from predictron_engine.dataset.store import DatasetStore
from predictron_engine.dataset.validation import (
    ValidationIssue,
    ValidationReport,
    validate_dataset,
)
from predictron_engine.dataset.validation_utils import (
    validate_outcome_fields,
    validate_record_completeness,
    validate_record_fields,
)

__all__ = [
    "AnalysisPipeline",
    "AnalysisResult",
    "AnalysisRun",
    "BinaryLabel",
    "DatasetRecord",
    "DatasetReportBuilder",
    "DatasetStats",
    "DatasetStore",
    "DeduplicationReport",
    "EvaluationBatchResult",
    "EvaluationMetadata",
    "EvaluationMetrics",
    "EvaluationPipeline",
    "FieldProvenance",
    "FundingEvent",
    "ImportPipeline",
    "ImportSource",
    "ImportSourceRegistry",
    "MissingFieldReport",
    "OutcomeRecord",
    "OutcomeStatus",
    "OutcomeVerdict",
    "PredictionEvaluation",
    "PredictionSummary",
    "ProvenanceTracker",
    "RawImportRecord",
    "StartupOutcome",
    "ValidationIssue",
    "ValidationReport",
    "compute_dataset_stats",
    "compute_evaluation_metrics",
    "find_duplicates",
    "validate_dataset",
    "validate_outcome_fields",
    "validate_record_completeness",
    "validate_record_fields",
]
