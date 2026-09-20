"""Batch Processing service — async job management for batch analyses."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from app.schemas.batch import (
    BatchJobDetailResponse,
    BatchJobListResponse,
    BatchJobProgressResponse,
    BatchJobRequest,
    BatchJobResultItem,
    BatchJobSummary,
    JobStatus,
)

logger = logging.getLogger(__name__)

_jobs: dict[str, dict[str, Any]] = {}


def submit_batch_job(
    request: BatchJobRequest,
    engine: Any,
) -> BatchJobSummary:
    """Submit a new batch analysis job."""
    job_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    job: dict[str, Any] = {
        "job_id": job_id,
        "job_name": request.job_name,
        "status": JobStatus.PENDING,
        "total_items": len(request.items),
        "completed_items": 0,
        "failed_items": 0,
        "items": request.items,
        "results": [],
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "error_message": None,
    }

    _jobs[job_id] = job

    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(_process_job(job_id, engine))
    else:
        asyncio.ensure_future(_process_job(job_id, engine))

    return BatchJobSummary(
        job_id=job_id,
        job_name=request.job_name,
        status=JobStatus.PENDING,
        total_items=len(request.items),
        created_at=now,
    )


def get_job_status(
    job_id: str,
    offset: int = 0,
    limit: int = 100,
) -> BatchJobDetailResponse | None:
    """Get detailed status of a batch job."""
    job = _jobs.get(job_id)
    if job is None:
        return None

    results = job["results"]
    total = len(results)
    page = results[offset : offset + limit]
    progress_pct = (
        (job["completed_items"] + job["failed_items"]) / job["total_items"] * 100
        if job["total_items"] > 0
        else 0.0
    )

    result_items = [
        BatchJobResultItem(
            index=r.get("index", 0),
            status=r.get("status", JobStatus.PENDING),
            startup_name=r.get("startup_name", ""),
            result_id=r.get("result_id"),
            error=r.get("error"),
            processing_time_ms=r.get("processing_time_ms"),
        )
        for r in page
    ]

    cursor = None
    if offset + limit < total:
        cursor = f"{offset + limit}"

    return BatchJobDetailResponse(
        job_id=job["job_id"],
        job_name=job["job_name"],
        status=job["status"],
        total_items=job["total_items"],
        completed_items=job["completed_items"],
        failed_items=job["failed_items"],
        progress_pct=progress_pct,
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error_message=job["error_message"],
        results=result_items,
        cursor=cursor,
    )


def list_jobs(
    offset: int = 0,
    limit: int = 20,
) -> BatchJobListResponse:
    """List all batch jobs with pagination."""
    all_jobs = list(_jobs.values())
    all_jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)

    total = len(all_jobs)
    page = all_jobs[offset : offset + limit]

    summaries = [
        BatchJobSummary(
            job_id=j["job_id"],
            job_name=j.get("job_name", ""),
            status=j["status"],
            total_items=j["total_items"],
            completed_items=j["completed_items"],
            failed_items=j["failed_items"],
            created_at=j.get("created_at"),
            started_at=j.get("started_at"),
            completed_at=j.get("completed_at"),
            error_message=j.get("error_message"),
        )
        for j in page
    ]

    return BatchJobListResponse(jobs=summaries, total=total)


def get_job_progress(job_id: str) -> BatchJobProgressResponse | None:
    """Get lightweight progress for streaming."""
    job = _jobs.get(job_id)
    if job is None:
        return None

    current = None
    if job["results"]:
        last = job["results"][-1]
        current = last.get("startup_name")

    return BatchJobProgressResponse(
        job_id=job["job_id"],
        status=job["status"],
        total_items=job["total_items"],
        completed_items=job["completed_items"],
        failed_items=job["failed_items"],
        progress_pct=(
            (job["completed_items"] + job["failed_items"]) / job["total_items"] * 100
            if job["total_items"] > 0
            else 0.0
        ),
        current_item=current,
    )


def cancel_job(job_id: str) -> bool:
    """Cancel a pending or running batch job."""
    job = _jobs.get(job_id)
    if job is None:
        return False
    if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return False

    job["status"] = JobStatus.CANCELLED
    job["completed_at"] = datetime.now(UTC).isoformat()
    return True


async def _process_job(job_id: str, engine: Any) -> None:
    """Process all items in a batch job."""
    job = _jobs.get(job_id)
    if job is None:
        return

    job["status"] = JobStatus.RUNNING
    job["started_at"] = datetime.now(UTC).isoformat()

    for idx, item in enumerate(job["items"]):
        if job["status"] == JobStatus.CANCELLED:
            break

        startup_name = str(item.get("startup_name", f"item_{idx}"))
        description = str(item.get("description", f"Batch item {idx}"))
        website_url = item.get("website_url")

        result_entry: dict[str, Any] = {
            "index": idx,
            "status": JobStatus.PENDING,
            "startup_name": startup_name,
            "result_id": None,
            "error": None,
            "processing_time_ms": None,
        }

        try:
            from pydantic import HttpUrl

            from app.schemas.analysis import StartupAnalysisRequest as _Req

            request = _Req(
                startup_name=startup_name,
                website_url=HttpUrl(website_url) if website_url else None,
                description=description,
            )

            item_start = time.monotonic()
            await asyncio.to_thread(engine.analyze, request)
            elapsed = (time.monotonic() - item_start) * 1000

            result_entry["status"] = JobStatus.COMPLETED
            result_entry["processing_time_ms"] = elapsed
            result_entry["result_id"] = f"{job_id}_{idx}"

            job["completed_items"] += 1

        except Exception as exc:
            logger.warning("Batch item %d failed: %s", idx, exc)
            result_entry["status"] = JobStatus.FAILED
            result_entry["error"] = str(exc)[:500]
            job["failed_items"] += 1

        job["results"].append(result_entry)

    all_done = (
        job["completed_items"] + job["failed_items"] >= job["total_items"]
        or job["status"] == JobStatus.CANCELLED
    )

    if all_done:
        if job["status"] != JobStatus.CANCELLED:
            job["status"] = (
                JobStatus.FAILED
                if job["failed_items"] == job["total_items"]
                else JobStatus.COMPLETED
            )
        job["completed_at"] = datetime.now(UTC).isoformat()
