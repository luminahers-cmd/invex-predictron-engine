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
company_name : Company name normalization / suffix stripping (Project E2).
fuzzy : Deterministic string-similarity functions (Project E2).
identity : Canonical company identity model (Project E2).
entity_resolution : Multi-stage entity resolution (Project E2).
duplicate_review : Duplicate review queue (Project E2).
enrichment : Company profile enrichment (Project E1).
validation : Dataset validation (Part F).
cli : Command-line interface (Part A).
"""

from predictron_engine.dataset.acquisition import (
    AcquireOptions,
    AcquisitionManager,
    AcquisitionMetrics,
    AcquisitionResult,
    AcquisitionScheduler,
    SourceConnectorRegistry,
)
from predictron_engine.dataset.analysis import (
    AnalysisPipeline,
    AnalysisResult,
    AnalysisRun,
)
from predictron_engine.dataset.company_name import (
    canonical_name_key,
    core_name,
    strip_corporate_suffixes,
)
from predictron_engine.dataset.dedup import (
    DeduplicationReport,
    find_duplicates,
)
from predictron_engine.dataset.duplicate_review import (
    DuplicateReviewReport,
    build_review_report,
    classify_clusters,
)
from predictron_engine.dataset.enrichment import (
    EnrichmentReport,
    EnrichmentService,
    enrich_record,
)
from predictron_engine.dataset.entity_resolution import (
    EntityResolver,
    MatchResult,
    ResolutionResult,
)
from predictron_engine.dataset.evaluation import (
    EvaluationMetadata,
    PredictionEvaluation,
)
from predictron_engine.dataset.evaluation_pipeline import (
    EvaluationBatchResult,
    EvaluationPipeline,
)
from predictron_engine.dataset.fuzzy import (
    jaccard,
    jaro,
    jaro_winkler,
    levenshtein,
    name_similarity,
)
from predictron_engine.dataset.graph import (
    COMPANY_TO_COMPANY_EDGE_TYPES,
    SYMMETRIC_EDGE_TYPES,
    CompanyKnowledgeGraphBuilder,
    CompanyNameIndex,
    DegreeDistribution,
    EdgeType,
    GraphBuildReport,
    GraphBuildResult,
    GraphEdge,
    GraphMetrics,
    GraphNode,
    GraphQueries,
    KnowledgeGraph,
    NodeType,
    build_graph_report,
    build_graph_statistics,
    build_relationship_summary,
    compute_graph_metrics,
    graph_key,
    read_graph,
    write_graph,
)
from predictron_engine.dataset.identity import (
    CompanyIdentity,
    CompanyIdentityBuilder,
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
    CompanyProfile,
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
from predictron_engine.dataset.population import (
    PopulateOptions,
    PopulationOrchestrator,
)
from predictron_engine.dataset.population_config import (
    PopulationConfig,
    SourceConfig,
)
from predictron_engine.dataset.population_metrics import (
    PopulationRunMetrics,
    compute_population_metrics,
)
from predictron_engine.dataset.population_report import (
    PopulationReport,
    build_population_report,
)
from predictron_engine.dataset.provenance import (
    FieldProvenance,
    ProvenanceTracker,
)
from predictron_engine.dataset.quality import (
    DatasetQualityChecker,
    QualityFinding,
    QualityReport,
    generate_quality_report,
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
    "COMPANY_TO_COMPANY_EDGE_TYPES",
    "SYMMETRIC_EDGE_TYPES",
    "AcquireOptions",
    "AcquisitionManager",
    "AcquisitionMetrics",
    "AcquisitionResult",
    "AcquisitionScheduler",
    "AnalysisPipeline",
    "AnalysisResult",
    "AnalysisRun",
    "BinaryLabel",
    "CompanyIdentity",
    "CompanyIdentityBuilder",
    "CompanyKnowledgeGraphBuilder",
    "CompanyNameIndex",
    "CompanyProfile",
    "DatasetRecord",
    "DatasetReportBuilder",
    "DatasetStats",
    "DatasetStore",
    "DeduplicationReport",
    "DuplicateReviewReport",
    "DatasetQualityChecker",
    "DegreeDistribution",
    "EdgeType",
    "EnrichmentReport",
    "EnrichmentService",
    "EntityResolver",
    "EvaluationBatchResult",
    "EvaluationMetadata",
    "EvaluationMetrics",
    "EvaluationPipeline",
    "FieldProvenance",
    "FundingEvent",
    "GraphBuildReport",
    "GraphBuildResult",
    "GraphEdge",
    "GraphMetrics",
    "GraphNode",
    "GraphQueries",
    "ImportPipeline",
    "ImportSource",
    "ImportSourceRegistry",
    "KnowledgeGraph",
    "MatchResult",
    "MissingFieldReport",
    "NodeType",
    "OutcomeRecord",
    "OutcomeStatus",
    "OutcomeVerdict",
    "PopulateOptions",
    "PopulationConfig",
    "PopulationOrchestrator",
    "PopulationReport",
    "PopulationRunMetrics",
    "PredictionEvaluation",
    "PredictionSummary",
    "ProvenanceTracker",
    "QualityFinding",
    "QualityReport",
    "RawImportRecord",
    "ResolutionResult",
    "SourceConfig",
    "SourceConnectorRegistry",
    "StartupOutcome",
    "ValidationIssue",
    "ValidationReport",
    "build_graph_report",
    "build_graph_statistics",
    "build_population_report",
    "build_relationship_summary",
    "build_review_report",
    "canonical_name_key",
    "classify_clusters",
    "compute_dataset_stats",
    "compute_evaluation_metrics",
    "compute_graph_metrics",
    "compute_population_metrics",
    "core_name",
    "enrich_record",
    "find_duplicates",
    "generate_quality_report",
    "graph_key",
    "jaccard",
    "jaro",
    "jaro_winkler",
    "levenshtein",
    "name_similarity",
    "read_graph",
    "strip_corporate_suffixes",
    "validate_dataset",
    "validate_outcome_fields",
    "validate_record_completeness",
    "validate_record_fields",
    "write_graph",
]
