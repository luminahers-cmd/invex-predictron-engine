"""Tests for the batch resource and batch helper functions."""

from __future__ import annotations

import json

import pytest

from predictron_sdk.batch import (
    TERMINAL_STATUSES,
    BatchTimeoutError,
    batch_items_from_names,
    build_batch_items,
    items_from_requests,
    wait_for_completion,
)
from predictron_sdk.models import (
    AnalyzeRequest,
    BatchItem,
    BatchJobProgress,
    JobStatus,
    VentureRequest,
)
from tests.sdk.conftest import json_response


def test_batch_submit(make_client, batch_summary_payload) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/batch"
        body = json.loads(request.content)
        assert body["job_name"] == "demo"
        assert body["items"][0]["startup_name"] == "Acme AI"
        return json_response(200, batch_summary_payload)

    client = make_client(handler)
    summary = client.batch.submit(
        [{"startup_name": "Acme AI", "description": "A sufficiently long description."}],
        job_name="demo",
    )
    assert summary.job_id == "job-1"
    assert summary.total_items == 2


def test_batch_submit_typed(make_client, batch_summary_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["items"][0]["startup_name"] == "Baked"
        return json_response(200, batch_summary_payload)

    client = make_client(handler)
    client.batch.submit_typed(
        [BatchItem(startup_name="Baked", description="A sufficiently long description.")],
        job_name="typed",
    )


def test_batch_evaluate(make_client, batch_summary_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert [item["startup_name"] for item in body["items"]] == ["A", "B"]
        return json_response(200, batch_summary_payload)

    client = make_client(handler)
    summary = client.batch.evaluate(["A", "B"])
    assert summary.job_id == "job-1"


def test_batch_evaluate_with_metadata(make_client, batch_summary_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        items = {i["startup_name"]: i for i in body["items"]}
        assert items["A"]["website_url"] == "https://a.example.com"
        assert items["B"]["description"] == "B desc"
        return json_response(200, batch_summary_payload)

    client = make_client(handler)
    client.batch.evaluate(
        ["A", "B"],
        descriptions={"B": "B desc"},
        website_urls={"A": "https://a.example.com"},
    )


def test_batch_from_requests(make_client, batch_summary_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["items"][1]["startup_name"] == "B"
        return json_response(200, batch_summary_payload)

    client = make_client(handler)
    client.batch.from_requests(
        [
            VentureRequest(startup_name="A", description="A sufficiently long description."),
            AnalyzeRequest(startup_name="B", description="A sufficiently long description."),
        ]
    )


def test_batch_list_jobs(make_client) -> None:
    payload = {"jobs": [{"job_id": "j1", "status": "running"}], "total": 1}

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/batch"
        params = dict(request.url.params)
        assert params["limit"] == "5"
        return json_response(200, payload)

    client = make_client(handler)
    listing = client.batch.list_jobs(limit=5)
    assert listing.total == 1
    assert listing.jobs[0].status is JobStatus.RUNNING


def test_batch_status(make_client) -> None:
    payload = {
        "job_id": "j1",
        "status": "completed",
        "total_items": 1,
        "completed_items": 1,
        "progress_pct": 100.0,
        "results": [{"index": 0, "status": "completed", "startup_name": "A"}],
    }

    def handler(request):
        assert request.url.path == "/api/v1/batch/j1"
        return json_response(200, payload)

    client = make_client(handler)
    detail = client.batch.status("j1")
    assert detail.status is JobStatus.COMPLETED
    assert detail.progress_pct == 100.0
    assert detail.results[0].index == 0


def test_batch_progress(make_client) -> None:
    payload = {
        "job_id": "j1",
        "status": "running",
        "total_items": 10,
        "completed_items": 4,
        "failed_items": 1,
        "progress_pct": 40.0,
    }

    def handler(request):
        assert request.url.path == "/api/v1/batch/j1/progress"
        return json_response(200, payload)

    client = make_client(handler)
    progress = client.batch.progress("j1")
    assert progress.progress_pct == 40.0
    assert progress.completed_items == 4


def test_batch_cancel(make_client) -> None:
    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/batch/j1/cancel"
        return json_response(200, {"job_id": "j1", "status": "cancelled"})

    client = make_client(handler)
    result = client.batch.cancel("j1")
    assert result.job_id == "j1"
    assert result.status == "cancelled"


def test_batch_wait_for_completion(make_client) -> None:
    progresses = [
        {"job_id": "j1", "status": "pending"},
        {"job_id": "j1", "status": "running", "completed_items": 1, "progress_pct": 50.0},
        {"job_id": "j1", "status": "completed", "progress_pct": 100.0},
    ]
    detail_payload = {
        "job_id": "j1",
        "status": "completed",
        "progress_pct": 100.0,
        "results": [],
    }
    state = {"i": 0}

    def handler(request):
        if request.url.path.endswith("/progress"):
            payload = progresses[min(state["i"], len(progresses) - 1)]
            state["i"] += 1
            return json_response(200, payload)
        return json_response(200, detail_payload)

    client = make_client(handler)
    detail = client.batch.wait_for_completion("j1", poll_interval=0.01, timeout=5.0)
    assert detail.status is JobStatus.COMPLETED


def test_batch_wait_timeout(make_client) -> None:
    def handler(request):
        return json_response(
            200, {"job_id": "j1", "status": "running", "progress_pct": 10.0}
        )

    client = make_client(handler)
    with pytest.raises(BatchTimeoutError):
        client.batch.wait_for_completion("j1", poll_interval=0.01, timeout=0.05)


def test_batch_result_pages(make_client) -> None:
    first = {
        "job_id": "j1",
        "status": "running",
        "total_items": 3,
        "results": [{"index": 0, "startup_name": "A"}],
        "cursor": "1",
    }
    second = {
        "job_id": "j1",
        "status": "running",
        "total_items": 3,
        "results": [{"index": 1, "startup_name": "B"}, {"index": 2, "startup_name": "C"}],
        "cursor": None,
    }
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        return json_response(200, first if state["calls"] == 1 else second)

    client = make_client(handler)
    results = client.batch.results("j1")
    assert [r.startup_name for r in results] == ["A", "B", "C"]


def test_batch_payload_cloned_per_item(make_client, batch_summary_payload) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert all("result" not in item for item in body["items"])
        return json_response(200, batch_summary_payload)

    client = make_client(handler)
    client.batch.submit(
        [{"startup_name": "A", "description": "A sufficiently long description."}],
        job_name="x",
    )


def test_terminal_statuses_set() -> None:
    assert JobStatus.COMPLETED in TERMINAL_STATUSES
    assert JobStatus.FAILED in TERMINAL_STATUSES
    assert JobStatus.CANCELLED in TERMINAL_STATUSES
    assert JobStatus.RUNNING not in TERMINAL_STATUSES


def test_batch_items_from_names() -> None:
    items = batch_items_from_names(
        ["A", "B"],
        descriptions={"B": "b-desc"},
        website_urls={"A": "https://a.example.com"},
    )
    assert items[0]["startup_name"] == "A"
    assert items[0]["website_url"] == "https://a.example.com"
    assert items[1]["description"] == "b-desc"


def test_build_batch_items() -> None:
    items = build_batch_items(
        [BatchItem(startup_name="A", description="A sufficiently long description.")]
    )
    assert items[0]["description"].startswith("A")


def test_items_from_requests() -> None:
    items = items_from_requests(
        [VentureRequest(startup_name="A", description="A sufficiently long description.")]
    )
    assert items[0]["startup_name"] == "A"
    assert "overall_score" not in items[0]


def test_wait_for_completion_negative_poll() -> None:
    with pytest.raises(ValueError):
        wait_for_completion(
            poll=lambda: BatchJobProgress(job_id="j"),
            poll_interval=-1.0,
        )


def test_wait_for_completion_polls_until_terminal() -> None:
    states = iter(
        [
            BatchJobProgress(job_id="j", status=JobStatus.PENDING),
            BatchJobProgress(job_id="j", status=JobStatus.RUNNING, progress_pct=50.0),
            BatchJobProgress(job_id="j", status=JobStatus.COMPLETED, progress_pct=100.0),
        ]
    )
    sleeps: list[float] = []
    result = wait_for_completion(
        poll=lambda: next(states),
        poll_interval=0.01,
        sleep_fn=sleeps.append,
    )
    assert result.status is JobStatus.COMPLETED
    assert sleeps == [0.01, 0.01]


def test_wait_for_completion_timeout_raises() -> None:
    with pytest.raises(BatchTimeoutError):
        wait_for_completion(
            poll=lambda: BatchJobProgress(job_id="j", status=JobStatus.RUNNING),
            poll_interval=0.01,
            timeout=0.02,
            sleep_fn=lambda _: None,
        )
