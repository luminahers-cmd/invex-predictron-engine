"""Batch operations for the Predictron SDK.

Provides payload builders for batch submissions plus deterministic polling
helpers. The SDK never re-implements business logic — it only assembles the
request payloads the API expects and advances the job lifecycle via polling.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from predictron_sdk.errors import SDKError
from predictron_sdk.models import (
    AnalyzeRequest,
    BatchItem,
    BatchJobDetail,
    BatchJobProgress,
    JobStatus,
    VentureRequest,
    to_payload,
)

__all__ = [
    "TERMINAL_STATUSES",
    "build_batch_items",
    "items_from_requests",
    "batch_items_from_names",
    "wait_for_completion",
    "BatchTimeoutError",
    "DEFAULT_POLL_INTERVAL",
]

DEFAULT_POLL_INTERVAL = 2.0

#: Statuses after which a job will no longer transition.
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)

SleepFn = Callable[[float], None]


class BatchTimeoutError(SDKError):
    """Raised when :func:`wait_for_completion` exceeds its deadline."""


def batch_items_from_names(
    startup_names: Sequence[str],
    *,
    descriptions: Mapping[str, str] | None = None,
    website_urls: Mapping[str, str | None] | None = None,
    pitch_deck_urls: Mapping[str, str | None] | None = None,
    founder_linkedin_urls: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build batch item payloads from a list of startup names."""
    descriptions = descriptions or {}
    website_urls = website_urls or {}
    pitch_deck_urls = pitch_deck_urls or {}
    founder_linkedin_urls = founder_linkedin_urls or {}
    items: list[dict[str, Any]] = []
    for name in startup_names:
        item: dict[str, Any] = {"startup_name": name}
        if name in descriptions:
            item["description"] = descriptions[name]
        website = website_urls.get(name)
        if website:
            item["website_url"] = website
        pitch_deck = pitch_deck_urls.get(name)
        if pitch_deck:
            item["pitch_deck_url"] = pitch_deck
        linkedin = founder_linkedin_urls.get(name)
        if linkedin:
            item["founder_linkedin_urls"] = list(linkedin)
        items.append(item)
    return items


def build_batch_items(
    requests: Sequence[BatchItem],
) -> list[dict[str, Any]]:
    """Serialize typed batch items into raw payload dicts."""
    return [item.to_payload() for item in requests]


def items_from_requests(
    requests: Sequence[AnalyzeRequest | VentureRequest],
) -> list[dict[str, Any]]:
    """Convert analyze/venture request models into batch item payloads."""
    items: list[dict[str, Any]] = []
    for request in requests:
        if isinstance(request, AnalyzeRequest | VentureRequest):
            items.append(to_payload(request))
        else:
            raise TypeError(
                "Unsupported request type: expected AnalyzeRequest or "
                f"VentureRequest, got {type(request).__name__}"
            )
    return items


def wait_for_completion(
    *,
    poll: Callable[[], BatchJobProgress | BatchJobDetail],
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> BatchJobProgress | BatchJobDetail:
    """Deterministically poll a batch job until it reaches a terminal state.

    Raises :class:`BatchTimeoutError` if ``timeout`` seconds elapse before the
    job reaches a terminal state.
    """
    if poll_interval < 0:
        raise ValueError("poll_interval must be >= 0")
    deadline = time.monotonic() + timeout if timeout is not None else None
    while True:
        progress = poll()
        if progress.status in TERMINAL_STATUSES:
            return progress
        if deadline is not None and time.monotonic() >= deadline:
            raise BatchTimeoutError(
                f"Batch job {progress.job_id} did not complete within "
                f"{timeout} seconds (last status: {progress.status.value})"
            )
        sleep_fn(poll_interval)


def is_terminal(status: JobStatus) -> bool:
    """Return ``True`` when a job status is terminal."""
    return status in TERMINAL_STATUSES
