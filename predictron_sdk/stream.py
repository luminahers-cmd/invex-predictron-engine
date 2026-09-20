"""Streaming helpers for the Predictron SDK.

The Predictron API does not expose raw long-lived SSE streams on the public
surface; batch progress is served through pollable endpoints. ``progress_events``
and :class:`BatchJobStream` translate deterministic polling into an iterator of
:class:`~predictron_sdk.models.BatchEvent` objects, giving callers a streaming
interface over long-running batch jobs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import UTC

from predictron_sdk.batch import (
    DEFAULT_POLL_INTERVAL,
    TERMINAL_STATUSES,
    SleepFn,
)
from predictron_sdk.models import BatchEvent, BatchJobProgress, JobStatus

__all__ = ["progress_events", "BatchJobStream", "event_from_progress"]


def event_from_progress(
    progress: BatchJobProgress,
    *,
    event_type: str,
) -> BatchEvent:
    """Build a :class:`BatchEvent` from a progress snapshot."""
    return BatchEvent(
        event_type=event_type,
        job_id=progress.job_id,
        data=progress.model_dump(mode="json"),
        timestamp=_now_iso(),
    )


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def progress_events(
    *,
    poll: Callable[[], BatchJobProgress],
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    sleep_fn: SleepFn = time.sleep,
) -> Iterator[BatchEvent]:
    """Yield batch lifecycle events by polling the progress endpoint.

    Emits ``job_started``, ``progress``, ``item_completed`` and
    ``item_failed`` events while the job runs, then a single
    ``job_completed``/``job_failed``/``job_cancelled`` terminal event.
    """
    if poll_interval < 0:
        raise ValueError("poll_interval must be >= 0")
    last_completed = 0
    last_failed = 0
    emitted_any = False
    while True:
        progress = poll()

        if not emitted_any:
            yield event_from_progress(progress, event_type="job_started")
            emitted_any = True

        for _ in range(last_completed, progress.completed_items):
            yield event_from_progress(progress, event_type="item_completed")
        for _ in range(last_failed, progress.failed_items):
            yield event_from_progress(progress, event_type="item_failed")

        last_completed = progress.completed_items
        last_failed = progress.failed_items

        yield event_from_progress(progress, event_type="progress")

        if progress.status in TERMINAL_STATUSES:
            terminal_type = _terminal_event_type(progress.status)
            yield event_from_progress(progress, event_type=terminal_type)
            return

        sleep_fn(poll_interval)


def _terminal_event_type(status: JobStatus) -> str:
    if status == JobStatus.COMPLETED:
        return "job_completed"
    if status == JobStatus.FAILED:
        return "job_failed"
    return "job_cancelled"


class BatchJobStream:
    """Streaming interface over a pollable batch job.

    ``poll`` must return the latest
    :class:`~predictron_sdk.models.BatchJobProgress` for the job.
    """

    def __init__(
        self,
        poll: Callable[[], BatchJobProgress],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        self._poll = poll
        self._poll_interval = poll_interval
        self._sleep_fn = sleep_fn

    def __iter__(self) -> Iterator[BatchEvent]:
        yield from progress_events(
            poll=self._poll,
            poll_interval=self._poll_interval,
            sleep_fn=self._sleep_fn,
        )

    def events(self) -> Iterator[BatchEvent]:
        return iter(self)
