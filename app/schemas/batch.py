"""Batch Processing API schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Batch job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class BatchJobRequest(BaseModel):
    """Request to submit a batch of analyses."""

    job_name: str = Field(
        default="",
        max_length=255,
        description="Optional human-readable job name",
    )
    items: list[dict[str, object]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "List of analysis items. Each item is a dict with at least "
            "startup_name, description keys matching StartupAnalysisRequest."
        ),
    )


# ---------------------------------------------------------------------------
# Response — Job Status
# ---------------------------------------------------------------------------


class BatchJobSummary(BaseModel):
    """Summary of a batch job."""

    job_id: str
    job_name: str = ""
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


class BatchJobListResponse(BaseModel):
    """List of batch jobs."""

    jobs: list[BatchJobSummary] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class BatchJobDetailResponse(BaseModel):
    """Detailed batch job status with results."""

    job_id: str
    job_name: str = ""
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    results: list[BatchJobResultItem] = Field(default_factory=list)
    cursor: str | None = Field(
        default=None, description="Pagination cursor for next page"
    )


class BatchJobResultItem(BaseModel):
    """Single item result within a batch job."""

    index: int = 0
    status: JobStatus = JobStatus.PENDING
    startup_name: str = ""
    result_id: str | None = None
    error: str | None = None
    processing_time_ms: float | None = None


class BatchJobProgressResponse(BaseModel):
    """Lightweight progress-only response for streaming."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    progress_pct: float = Field(default=0.0)
    current_item: str | None = None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class BatchStreamEvent(BaseModel):
    """Event emitted during batch processing for streaming."""

    event_type: str = Field(
        default="progress",
        description="Event type: progress, item_completed, item_failed, job_completed, job_failed",
    )
    job_id: str = ""
    data: dict[str, object] = Field(default_factory=dict)
    timestamp: str | None = None
