"""Tests for batch streaming helpers."""

from __future__ import annotations

import pytest

from predictron_sdk.models import BatchEvent, BatchJobProgress, JobStatus
from predictron_sdk.stream import (
    BatchJobStream,
    event_from_progress,
    progress_events,
)


def make_progress(**overrides) -> BatchJobProgress:
    defaults = {"job_id": "j1", "status": JobStatus.RUNNING}
    defaults.update(overrides)
    return BatchJobProgress(**defaults)


def event_types(events) -> list[str]:
    return [e.event_type for e in events]


def test_progress_events_success_sequence() -> None:
    polls = iter(
        [
            make_progress(),
            make_progress(completed_items=2, progress_pct=50.0),
            make_progress(
                status=JobStatus.COMPLETED, completed_items=3, progress_pct=100.0
            ),
        ]
    )

    events = list(
        progress_events(
            poll=lambda: next(polls),
            poll_interval=0.0,
            sleep_fn=lambda _: None,
        )
    )
    assert event_types(events) == [
        "job_started",
        "progress",
        "item_completed",
        "item_completed",
        "progress",
        "item_completed",
        "progress",
        "job_completed",
    ]
    assert all(e.job_id == "j1" for e in events)
    assert events[-2].data["progress_pct"] == 100.0


def test_progress_events_failure_sequence() -> None:
    polls = iter(
        [
            make_progress(),
            make_progress(status=JobStatus.FAILED, failed_items=1),
        ]
    )
    events = list(
        progress_events(
            poll=lambda: next(polls),
            poll_interval=0.0,
            sleep_fn=lambda _: None,
        )
    )
    assert event_types(events) == [
        "job_started",
        "progress",
        "item_failed",
        "progress",
        "job_failed",
    ]


def test_progress_events_cancelled() -> None:
    events = list(
        progress_events(
            poll=lambda: make_progress(status=JobStatus.CANCELLED),
            poll_interval=0.0,
            sleep_fn=lambda _: None,
        )
    )
    assert event_types(events) == ["job_started", "progress", "job_cancelled"]


def test_progress_events_never_yields_duplicate_completed() -> None:
    polls = iter(
        [
            make_progress(completed_items=2),
            make_progress(completed_items=2, status=JobStatus.COMPLETED),
        ]
    )
    events = list(
        progress_events(
            poll=lambda: next(polls),
            poll_interval=0.0,
            sleep_fn=lambda _: None,
        )
    )
    completed = [e for e in events if e.event_type == "item_completed"]
    assert len(completed) == 2


def test_progress_events_negative_interval_raises() -> None:
    with pytest.raises(ValueError):
        list(
            progress_events(
                poll=lambda: make_progress(),
                poll_interval=-1.0,
            )
        )


def test_progress_events_sleep_called_between_polls() -> None:
    sleeps: list[float] = []
    polls = iter(
        [
            make_progress(),
            make_progress(status=JobStatus.COMPLETED),
        ]
    )
    list(
        progress_events(
            poll=lambda: next(polls),
            poll_interval=2.0,
            sleep_fn=sleeps.append,
        )
    )
    assert sleeps == [2.0]


def test_batch_job_stream_iter() -> None:
    polls = iter(
        [
            make_progress(),
            make_progress(status=JobStatus.COMPLETED),
        ]
    )
    stream = BatchJobStream(
        poll=lambda: next(polls),
        poll_interval=0.0,
        sleep_fn=lambda _: None,
    )
    events = [e for e in stream]
    assert event_types(events) == [
        "job_started",
        "progress",
        "progress",
        "job_completed",
    ]


def test_batch_job_stream_events() -> None:
    polls = iter(
        [
            make_progress(status=JobStatus.COMPLETED),
        ]
    )
    stream = BatchJobStream(
        poll=lambda: next(polls),
        poll_interval=0.0,
        sleep_fn=lambda _: None,
    )
    events = list(stream.events())
    assert event_types(events) == ["job_started", "progress", "job_completed"]


def test_batch_job_stream_negative_interval_raises() -> None:
    stream = BatchJobStream(poll=lambda: make_progress(), poll_interval=-1.0)
    with pytest.raises(ValueError):
        next(iter(stream))


def test_event_from_progress() -> None:
    progress = make_progress(progress_pct=33.0)
    event: BatchEvent = event_from_progress(progress, event_type="progress")
    assert event.event_type == "progress"
    assert event.job_id == "j1"
    assert event.data["progress_pct"] == 33.0
    assert event.timestamp is not None


def test_event_from_progress_timestamp_format() -> None:
    event = event_from_progress(make_progress(), event_type="job_started")
    assert "T" in event.timestamp  # ISO-8601 style


def test_events_carry_full_snapshot() -> None:
    polls = iter(
        [
            make_progress(status=JobStatus.COMPLETED, completed_items=9, failed_items=1),
        ]
    )
    final = list(
        progress_events(
            poll=lambda: next(polls),
            poll_interval=0.0,
            sleep_fn=lambda _: None,
        )
    )[-1]
    assert final.data["completed_items"] == 9
    assert final.data["failed_items"] == 1


def test_progress_events_running_even_when_no_items_done() -> None:
    polls = iter([make_progress(status=JobStatus.COMPLETED, progress_pct=0.0)])
    events = list(
        progress_events(
            poll=lambda: next(polls),
            poll_interval=0.0,
            sleep_fn=lambda _: None,
        )
    )
    final = events[-1]
    assert final.event_type == "job_completed"
    assert final.data["completed_items"] == 0
