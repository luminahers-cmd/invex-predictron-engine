"""Automated historical dataset acquisition framework.

Extensible acquisition system that continuously grows the Predictron
historical dataset from public sources.  Additive infrastructure
that does not modify the prediction engine, scoring, reasoning,
benchmarks, or APIs.

Modules
-------
manager : Top-level orchestrator for acquisition runs.
pipeline : Batch processing with retry, rate limiting, and progress.
state : State persistence, checkpoints, and import history.
scheduler : Scheduled import management.
cache : Local file caching for fetched sources.
sources : Pluggable source connectors.
"""

from predictron_engine.dataset.acquisition.cache import AcquisitionCache
from predictron_engine.dataset.acquisition.manager import (
    AcquireOptions,
    AcquisitionManager,
)
from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionMetrics,
    AcquisitionPipeline,
    AcquisitionResult,
    RetryPolicy,
)
from predictron_engine.dataset.acquisition.scheduler import (
    AcquisitionScheduler,
    ImportSchedule,
)
from predictron_engine.dataset.acquisition.sources import (
    BaseSource,
    CompaniesHouseConnector,
    CsvExportConnector,
    FetchResult,
    SecEdgarConnector,
    SourceConnectorRegistry,
    SourceDescriptor,
    YcOssConnector,
)
from predictron_engine.dataset.acquisition.state import (
    AcquisitionBatch,
    AcquisitionRecord,
    AcquisitionStateManager,
    CheckpointData,
    compute_file_hash,
    generate_batch_id,
)

__all__ = [
    "AcquireOptions",
    "AcquisitionBatch",
    "AcquisitionCache",
    "AcquisitionManager",
    "AcquisitionMetrics",
    "AcquisitionPipeline",
    "AcquisitionRecord",
    "AcquisitionResult",
    "AcquisitionScheduler",
    "AcquisitionStateManager",
    "BaseSource",
    "CheckpointData",
    "CompaniesHouseConnector",
    "CsvExportConnector",
    "FetchResult",
    "ImportSchedule",
    "RetryPolicy",
    "SecEdgarConnector",
    "SourceConnectorRegistry",
    "SourceDescriptor",
    "YcOssConnector",
    "compute_file_hash",
    "generate_batch_id",
]
