"""Load testing suite for the Predictron API.

Tests:
  - concurrent analysis requests
  - concurrent authenticated retrieval
  - mixed read/write workload
  - rate-limited scenarios

Measures:
  - throughput
  - average latency
  - P95 latency
  - P99 latency
  - error rate

Usage:
    python -m pytest tests/test_load.py -v --tb=short
    python -m pytest tests/test_load.py::TestLoadScenarios -v --tb=short
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

VALID_PAYLOAD = {
    "startup_name": "LoadTestCo",
    "website": "https://loadtest.example.com",
    "description": "A load test startup for benchmarking API performance under concurrent load.",
}

TEST_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJsb2FkX3Rlc3RfdXNlciJ9."
    "dGVzdF9zaWduYXR1cmU"
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _compute_percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


async def _single_analyze(client: AsyncClient, payload: dict | None = None) -> dict:
    resp = await client.post("/api/v1/analyze", json=payload or VALID_PAYLOAD)
    resp.raise_for_status()
    return resp.json()


async def _single_list(client: AsyncClient, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    resp = await client.get("/api/v1/analyze", headers=headers)
    resp.raise_for_status()
    return resp.json()


async def _single_analyze_error(client: AsyncClient) -> int:
    resp = await client.post("/api/v1/analyze", json={"startup_name": ""})
    return resp.status_code


def _summarize_latencies(
    latencies: list[float],
    label: str = "",
) -> dict:
    if not latencies:
        return {"label": label, "count": 0}

    sorted_lats = sorted(latencies)
    mean = statistics.mean(latencies)
    p95 = _compute_percentile(sorted_lats, 95)
    p99 = _compute_percentile(sorted_lats, 99)
    throughput = len(latencies) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0

    return {
        "label": label,
        "count": len(latencies),
        "mean_ms": round(mean, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "min_ms": round(sorted_lats[0], 2),
        "max_ms": round(sorted_lats[-1], 2),
        "throughput_req_per_sec": round(throughput, 1),
    }


class TestLoadScenarios:
    @pytest.mark.anyio
    async def test_concurrent_analysis_requests(self, client):
        concurrency = 10
        n_requests = 30

        payloads = [
            {
                "startup_name": f"LoadTestCo-{i}",
                "website": "https://loadtest.example.com",
                "description": f"Load test request {i} for concurrent benchmark.",
            }
            for i in range(n_requests)
        ]

        latencies: list[float] = []
        errors = 0

        async def _run(i: int) -> None:
            nonlocal errors
            start = time.perf_counter()
            try:
                await _single_analyze(client, payloads[i % len(payloads)])
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                errors += 1

        semaphore = asyncio.Semaphore(concurrency)

        async def _throttled(i: int) -> None:
            async with semaphore:
                await _run(i)

        tasks = [_throttled(i) for i in range(n_requests)]
        await asyncio.gather(*tasks)

        stats = _summarize_latencies(latencies, "concurrent_analysis")
        stats["error_rate"] = round(errors / n_requests * 100, 2)

        assert errors == 0, f"Errors occurred: {errors}/{n_requests}"
        assert latencies, "No successful requests"
        assert stats["p95_ms"] < 5000, f"P95 too high: {stats['p95_ms']}ms"

    @pytest.mark.anyio
    async def test_concurrent_authenticated_retrieval(self, client):
        n_requests = 20
        latencies: list[float] = []
        errors = 0

        async def _run() -> None:
            nonlocal errors
            start = time.perf_counter()
            try:
                await _single_list(client, TEST_TOKEN)
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                errors += 1

        tasks = [_run() for _ in range(n_requests)]
        await asyncio.gather(*tasks)

        stats = _summarize_latencies(latencies, "concurrent_list")
        stats["error_rate"] = round(errors / n_requests * 100, 2)

    @pytest.mark.anyio
    async def test_mixed_read_write_workload(self, client):
        n_writes = 10
        n_reads = 10
        latencies: list[float] = []
        errors = 0

        async def _write(i: int) -> None:
            nonlocal errors
            start = time.perf_counter()
            try:
                await _single_analyze(
                    client,
                    {
                        "startup_name": f"MixedLoad-{i}",
                        "website": "https://mixed.example.com",
                        "description": f"Mixed workload test request {i}.",
                    },
                )
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                errors += 1

        async def _read() -> None:
            nonlocal errors
            start = time.perf_counter()
            try:
                await _single_list(client, TEST_TOKEN)
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                errors += 1

        tasks = [_write(i) for i in range(n_writes)] + [_read() for _ in range(n_reads)]
        await asyncio.gather(*tasks)

        stats = _summarize_latencies(latencies, "mixed_read_write")
        stats["error_rate"] = round(errors / (n_writes + n_reads) * 100, 2)

    @pytest.mark.anyio
    async def test_rate_limited_scenario(self, client):
        from app.core.config import get_settings
        from app.middleware.rate_limit import RateLimitMiddleware

        RateLimitMiddleware.reset_windows()

        settings = get_settings()
        was_enabled = settings.RATE_LIMIT_ENABLED
        settings.RATE_LIMIT_ENABLED = True
        orig_limit = settings.RATE_LIMIT_REQUESTS
        orig_window = settings.RATE_LIMIT_WINDOW_SECONDS
        settings.RATE_LIMIT_REQUESTS = 5
        settings.RATE_LIMIT_WINDOW_SECONDS = 60
        RateLimitMiddleware.reset_windows()

        latencies: list[float] = []
        statuses: list[int] = []

        async def _run() -> None:
            start = time.perf_counter()
            resp = await client.post("/api/v1/analyze", json=VALID_PAYLOAD)
            latencies.append((time.perf_counter() - start) * 1000)
            statuses.append(resp.status_code)

        for _ in range(8):
            await _run()

        allowed = sum(1 for s in statuses if s == 200)
        blocked = sum(1 for s in statuses if s == 429)

        settings.RATE_LIMIT_ENABLED = was_enabled
        settings.RATE_LIMIT_REQUESTS = orig_limit
        settings.RATE_LIMIT_WINDOW_SECONDS = orig_window
        RateLimitMiddleware.reset_windows()

        stats = _summarize_latencies(latencies, "rate_limited")
        stats["allowed"] = allowed
        stats["blocked"] = blocked

        assert blocked >= 2, f"Expected at least 2 blocked, got {blocked}"
        assert allowed <= 5, f"Expected <=5 allowed, got {allowed}"

    @pytest.mark.anyio
    async def test_throughput_under_concurrent_load(self, client):
        n_requests = 50
        concurrency = 10

        payloads = [
            {
                "startup_name": f"Throughput-{i}",
                "website": "https://throughput.example.com",
                "description": f"Throughput test request {i}.",
            }
            for i in range(n_requests)
        ]

        latencies: list[float] = []
        errors = 0

        semaphore = asyncio.Semaphore(concurrency)

        async def _run(i: int) -> None:
            nonlocal errors
            async with semaphore:
                start = time.perf_counter()
                try:
                    await _single_analyze(client, payloads[i])
                    latencies.append((time.perf_counter() - start) * 1000)
                except Exception:
                    errors += 1

        all_start = time.perf_counter()
        tasks = [_run(i) for i in range(n_requests)]
        await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - all_start

        stats = _summarize_latencies(latencies, "throughput")
        stats["wall_time_s"] = round(wall_time, 2)
        stats["overall_throughput"] = round(n_requests / wall_time, 1)
        stats["error_rate"] = round(errors / n_requests * 100, 2)

        assert errors == 0, f"Errors: {errors}/{n_requests}"
        assert stats["overall_throughput"] > 0
