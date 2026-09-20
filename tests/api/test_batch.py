"""Batch Processing API tests."""

from __future__ import annotations

import pytest


def _batch_payload(n=3):
    return {
        "job_name": "test-batch",
        "items": [
            {
                "startup_name": f"BulkCo{i}",
                "description": f"BulkCo{i} builds enterprise solutions for global markets.",
            }
            for i in range(n)
        ],
    }


class TestBatchSubmit:
    """Tests for POST /api/v1/batch."""

    @pytest.mark.asyncio
    async def test_batch_submit_returns_202(self, client):
        resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_batch_submit_returns_job_id(self, client):
        resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        data = resp.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    @pytest.mark.asyncio
    async def test_batch_submit_initial_status(self, client):
        resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        data = resp.json()
        assert data["status"] in ("pending", "running", "completed")

    @pytest.mark.asyncio
    async def test_batch_submit_total_items(self, client):
        resp = await client.post("/api/v1/batch", json=_batch_payload(5))
        data = resp.json()
        assert data["total_items"] == 5

    @pytest.mark.asyncio
    async def test_batch_submit_empty_rejected(self, client):
        resp = await client.post("/api/v1/batch", json={"items": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_submit_with_job_name(self, client):
        resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        data = resp.json()
        assert data["job_name"] == "test-batch"


class TestBatchListJobs:
    """Tests for GET /api/v1/batch."""

    @pytest.mark.asyncio
    async def test_list_jobs_returns_200(self, client):
        resp = await client.get("/api/v1/batch")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_jobs_structure(self, client):
        resp = await client.get("/api/v1/batch")
        data = resp.json()
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(self, client):
        resp = await client.get("/api/v1/batch?offset=0&limit=10")
        data = resp.json()
        assert len(data["jobs"]) <= 10

    @pytest.mark.asyncio
    async def test_list_jobs_after_submit(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        job_id = submit_resp.json()["job_id"]

        list_resp = await client.get("/api/v1/batch")
        data = list_resp.json()
        job_ids = [j["job_id"] for j in data["jobs"]]
        assert job_id in job_ids


class TestBatchJobStatus:
    """Tests for GET /api/v1/batch/{job_id}."""

    @pytest.mark.asyncio
    async def test_job_status_returns_200(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        job_id = submit_resp.json()["job_id"]

        resp = await client.get(f"/api/v1/batch/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_job_status_not_found(self, client):
        resp = await client.get("/api/v1/batch/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_job_status_has_results_list(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        job_id = submit_resp.json()["job_id"]

        resp = await client.get(f"/api/v1/batch/{job_id}")
        data = resp.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_job_status_progress_pct(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        job_id = submit_resp.json()["job_id"]

        resp = await client.get(f"/api/v1/batch/{job_id}")
        data = resp.json()
        assert 0 <= data["progress_pct"] <= 100

    @pytest.mark.asyncio
    async def test_job_status_cursor_pagination(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(3))
        job_id = submit_resp.json()["job_id"]

        resp = await client.get(f"/api/v1/batch/{job_id}?limit=1")
        data = resp.json()
        assert len(data["results"]) <= 1


class TestBatchJobProgress:
    """Tests for GET /api/v1/batch/{job_id}/progress."""

    @pytest.mark.asyncio
    async def test_progress_returns_200(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        job_id = submit_resp.json()["job_id"]

        resp = await client.get(f"/api/v1/batch/{job_id}/progress")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_progress_not_found(self, client):
        resp = await client.get("/api/v1/batch/fake/progress")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_progress_structure(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(2))
        job_id = submit_resp.json()["job_id"]

        resp = await client.get(f"/api/v1/batch/{job_id}/progress")
        data = resp.json()
        assert "job_id" in data
        assert "status" in data
        assert "progress_pct" in data


class TestBatchJobCancel:
    """Tests for POST /api/v1/batch/{job_id}/cancel."""

    @pytest.mark.asyncio
    async def test_cancel_returns_200(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(50))
        job_id = submit_resp.json()["job_id"]

        resp = await client.post(f"/api/v1/batch/{job_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, client):
        resp = await client.post("/api/v1/batch/fake-id/cancel")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_result_structure(self, client):
        submit_resp = await client.post("/api/v1/batch", json=_batch_payload(50))
        job_id = submit_resp.json()["job_id"]

        resp = await client.post(f"/api/v1/batch/{job_id}/cancel")
        data = resp.json()
        assert "job_id" in data
        assert data["job_id"] == job_id
