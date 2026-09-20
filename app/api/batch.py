"""Batch Processing API — async job management for batch analyses."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.auth.jwt import get_current_user_optional
from app.schemas.batch import (
    BatchJobDetailResponse,
    BatchJobListResponse,
    BatchJobProgressResponse,
    BatchJobRequest,
    BatchJobSummary,
)
from app.services.batch import (
    cancel_job,
    get_job_progress,
    get_job_status,
    list_jobs,
    submit_batch_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch-processing"])


@router.post(
    "",
    response_model=BatchJobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch job",
    description=(
        "Submit a batch of startup analyses for async processing. "
        "Returns a job ID that can be used to track progress."
    ),
    responses={
        202: {"description": "Job accepted"},
        422: {"description": "Validation error"},
    },
)
async def submit_job(
    request: BatchJobRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> BatchJobSummary:
    engine = raw_request.app.state.predictron_engine
    return submit_batch_job(request, engine)


@router.get(
    "",
    response_model=BatchJobListResponse,
    status_code=status.HTTP_200_OK,
    summary="List batch jobs",
    description="List all batch jobs with pagination.",
)
async def list_batch_jobs(
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
) -> BatchJobListResponse:
    return list_jobs(offset=offset, limit=limit)


@router.get(
    "/{job_id}",
    response_model=BatchJobDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get job status",
    description="Get detailed status of a batch job including results.",
)
async def job_status(
    job_id: str,
    offset: int = Query(default=0, ge=0, description="Results pagination offset"),
    limit: int = Query(default=100, ge=1, le=1000, description="Results page size"),
) -> BatchJobDetailResponse:
    result = get_job_status(job_id, offset=offset, limit=limit)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )
    return result


@router.get(
    "/{job_id}/progress",
    response_model=BatchJobProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Job progress",
    description="Lightweight progress endpoint suitable for polling or streaming.",
)
async def job_progress(job_id: str) -> BatchJobProgressResponse:
    result = get_job_progress(job_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )
    return result


@router.post(
    "/{job_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel batch job",
    description="Cancel a pending or running batch job.",
    responses={
        200: {"description": "Job cancelled"},
        404: {"description": "Job not found"},
        409: {"description": "Job already completed or failed"},
    },
)
async def cancel_batch_job(job_id: str) -> dict[str, str]:
    success = cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found or already terminal",
        )
    return {"status": "cancelled", "job_id": job_id}
