# Predictron SDK — Official Python Client (Project E9)

The official Python SDK for the Predictron API. It wraps every public API
endpoint behind strongly-typed models, typed exceptions, deterministic retries,
automatic pagination, and first-class batch streaming.

Everything is deterministic. There is **no ML, no LLM, and no embeddings** in
the SDK — it only calls the API.

---

## Installation

The SDK ships inside this repository as the `predictron_sdk` package. Its only
runtime dependencies are `httpx` and `pydantic` (both already present in the
backend environment).

```bash
pip install -e .          # from the repository root
# or, when the SDK is published as a standalone package:
pip install predictron-sdk
```

## Quickstart

```python
from predictron_sdk import PredictronClient

client = PredictronClient(
    api_key="your-api-key",            # or set PREDICTRON_API_KEY
    base_url="https://predictron.invex.ai",  # or set PREDICTRON_BASE_URL
)

# Analyze a single startup
analysis = client.venture.evaluate(
    startup_name="Acme AI",
    description="Acme AI builds enterprise machine learning tooling.",
    website_url="https://acme.example.com",
)
print(analysis.overall_score)          # 72.4
print(analysis.decision_category)      # "invest"
for dim in analysis.dimension_scores:
    print(dim.dimension, dim.score, dim.rationale)

# Search the dataset
results = client.search(query="robotics", search_type="all", limit=20)
for item in results.results:
    print(item.name, item.score)

# Compare companies
comparison = client.compare(company_names=["Acme AI", "Beta Labs"])
for diff in comparison.feature_diffs.feature_diffs:
    print(diff.feature, diff.company_a, diff.company_b)

# Due diligence report
report = client.due_diligence.generate(
    startup_name="Acme AI",
    description="Acme AI builds enterprise machine learning tooling.",
)
print(report.executive_summary.headline)
```

## Authentication

The SDK supports three authentication styles:

| Style | How | When |
|---|---|---|
| Bearer token | `PredictronClient(api_key=...)` or `BearerTokenAuth(token)` | Default for the Predictron API |
| API key header | `ApiKeyAuth("k-123", header_name="X-API-Key")` | Key-based gateways |
| Custom headers | `StructuredHeaderAuth("X-Tenant", "acme")` | Tenant / context headers |

Custom headers always win over provider-injected headers.

```python
from predictron_sdk import PredictronClient
from predictron_sdk.auth import ApiKeyAuth

client = PredictronClient(
    auth=ApiKeyAuth("k-123", header_name="X-Api-Key"),
    base_url="https://predictron.invex.ai",
)
```

Environment variables are read automatically when no explicit value is given:

| Variable | Purpose |
|---|---|
| `PREDICTRON_API_KEY` | Default API key / bearer token |
| `PREDICTRON_BASE_URL` | Default base URL |
| `PREDICTRON_TIMEOUT` | Request timeout (seconds) |
| `PREDICTRON_MAX_RETRIES` | Maximum retries before giving up |

## Client options

```python
from predictron_sdk import PredictronClient, RetryPolicy

client = PredictronClient(
    api_key="...",
    base_url="...",
    timeout=30.0,                      # per-request timeout (seconds)
    max_retries=4,                     # retry count override
    retry_policy=RetryPolicy(          # full deterministic control
        max_retries=4,
        base_delay=1.0,
        max_delay=32.0,
        backoff_factor=2.0,
        retry_statuses={429, 502, 503, 504},
    ),
    headers={"X-Team": "platform"},    # extra headers sent on every request
)

# Custom httpx session (connection pooling, proxies, mTLS ...)
import httpx
client = PredictronClient(api_key="...", http_client=httpx.Client(proxies="http://proxy:8080"))
```

Use `client.close()` or the context manager to release the connection pool.

## Endpoint coverage

The SDK mirrors the API 1:1. No business logic is duplicated — every method
performs exactly one API call.

| Client accessor | Endpoint(s) |
|---|---|
| `client.venture` | `/venture`, `/venture/explain`, `/venture/trace`, `/venture/features`, `/venture/benchmark`, `/venture/knowledge-graph`, `/venture/signals/{id}`, `/venture/signals/{id}/trends`, `/venture/signals/{id}/aggregation` |
| `client.portfolio` | `/portfolio`, `/portfolio/similarity` |
| `client.compare(...)` | `/compare` |
| `client.due_diligence` | `/due-diligence` |
| `client.search(...)` / `client.search_resource` | `/search` (POST, GET), `/search/by-type/{type}` |
| `client.batch` | `/batch` (submit/list), `/batch/{job}` (status/progress/cancel) |
| `client.analyze` | `/analyze` (POST/GET), `/analyze/{id}` |
| `client.health` | `/health`, `/health/readiness` |

```python
# Venture intelligence
client.venture.evaluate(startup_name=..., description=...)
client.venture.explain(startup_name=..., description=...)
client.venture.trace(startup_name=..., description=...)
client.venture.features(startup_name=..., description=...)
client.venture.benchmark(startup_name=..., description=...)
client.venture.knowledge_graph()
client.venture.signals("company-1")
client.venture.signal_trends("company-1")
client.venture.signal_aggregation("company-1")

# Portfolio intelligence
client.portfolio.analyze(company_names=["A", "B"], descriptions={"A": "...", "B": "..."})
client.portfolio.similarity(company_names=["A", "B", "C"])

# Due diligence
client.due_diligence.generate(startup_name=..., description=...)

# Legacy analysis
client.analyze.analyze(startup_name=..., description=...)
page = client.analyze.list_analyses(offset=0, limit=20)
detail = client.analyze.get("analysis-123")

# Health checks
client.health.check()        # == client.health_status()
client.health.readiness()
```

Typed request models can be passed instead of keyword arguments:

```python
from predictron_sdk import PredictronClient
from predictron_sdk.models import VentureRequest, PortfolioRequest

client.venture.evaluate(request=VentureRequest(startup_name="A", description="Desc."))
client.portfolio.analyze(request=PortfolioRequest(company_names=["A", "B"]))
```

## Pagination

List endpoints are paginated automatically. Use `Paginator` — a standard
iterator — with the current offset and a page size:

```python
# Every page is a typed Page
for page in client.search_resource.paginate(query="robotics", limit=20):
    for item in page.items:
        print(item.name)

# Or flatten pages into individual items
items = client.search_resource.paginate(query="vehicles", limit=20).all()
```

Persisted analyses and batch results paginate the same way:

```python
for summary in client.analyze.paginate(limit=20).items():
    print(summary.startup_name, summary.venture_score)

results = client.batch.results(job_id="job-123")   # cursor-paginated
for item in results:
    print(item.index, item.status, item.startup_name)
```

Each `Page` exposes `items`, `total`, `limit` and `has_more`.

## Batch operations

Submit a batch, monitor progress, wait for completion, and stream events —

all without re-implementing business logic.

```python
client.batch.submit(
    [
        {"startup_name": "A", "description": "Alpha builds tooling."},
        {"startup_name": "B", "description": "Beta does analytics."},
    ],
    job_name="tuesdays-batch",
)

# Typed items
from predictron_sdk.models import BatchItem
client.batch.submit_typed([BatchItem(startup_name="A", description="...")])

# Convenience from names (+ optional per-company metadata)
client.batch.evaluate(
    ["A", "B"],
    descriptions={"A": "Alpha builds tooling.", "B": "Beta does analytics."},
)

# Lifecycle
summary  = ...                    # returned by submit: job_id, status
detail   = client.batch.status(summary.job_id)
progress = client.batch.progress(summary.job_id)
job_list = client.batch.list_jobs(offset=0, limit=20)

# Wait until the job reaches a terminal state (deterministic polling)
final = client.batch.wait_for_completion(summary.job_id, poll_interval=2.0, timeout=600.0)

# Cancel a pending/running job
cancelled = client.batch.cancel(summary.job_id)
```

### Streaming batch progress

`client.batch.stream` turns polling into an iterator of typed `BatchEvent`s:

```python
from predictron_sdk.models import JobStatus

for event in client.batch.stream(job_id="job-123", poll_interval=2.0):
    print(event.event_type, event.job_id)   # progress / item_completed / job_completed ...
    if event.event_type == "job_completed":
        break
```

## Errors

Every failure is a typed exception derived from `SDKError`.

| Exception | Meaning |
|---|---|
| `AuthenticationError` | HTTP 401 — missing/invalid/expired credentials |
| `NotFoundError` | HTTP 404 — resource not found or access denied |
| `ConflictError` | HTTP 409 — request conflicts with resource state |
| `ValidationError` | HTTP 422 — invalid request payload |
| `RateLimitError` | HTTP 429 — rate limit exceeded (has `retry_after`) |
| `ServerError` | HTTP 5xx — server-side failure |
| `TimeoutError` | The server did not respond in time |
| `SerializationError` | An API response could not be parsed |
| `APIError` | Any other HTTP-level error |
| `SDKError` | Base class for all of the above |

```python
from predictron_sdk import (
    AuthenticationError,
    RateLimitError,
    ServerError,
    SDKError,
)

try:
    client.venture.evaluate(startup_name="A", description="...")
except RateLimitError as err:
    retry_in = err.retry_after
    print("Back off for", retry_in)
except AuthenticationError as err:
    print("Bad credentials:", err.status_code, err.message)
except ServerError as err:
    print("Server said:", err.response_body)
except SDKError as err:
    print("Other SDK failure:", err)
```

All `APIError` subclasses expose `status_code`, `code`, `method`, `url`,
`headers`, `response_body`, and `message`.

## Retries

The SDK retries **only** transient failures — HTTP `429`, `502`, `503`, `504`,
timeouts and connection errors — with deterministic exponential backoff. The
schedule is fully deterministic (no jitter), so identical failures retry on an
identical schedule. `Retry-After` is honoured for `429` responses when the
`use_retry_after` flag is on.

```python
from predictron_sdk import RetryPolicy

client = PredictronClient(
    api_key="...",
    retry_policy=RetryPolicy(
        max_retries=6,
        base_delay=0.5,
        max_delay=16.0,
        backoff_factor=2.0,
        retry_on_timeout=True,
        retry_on_connect_error=True,
    ),
)
```

## Serialization

Responses are deserialized into pydantic models — never raw dicts. Access the
full schema of every model through rich objects:

```python
analysis = client.venture.evaluate(startup_name="A", description="...")
analysis.overall_score          # float
analysis.overall_confidence     # float
analysis.dimension_scores       # list[DimensionScore]
report   = client.due_diligence.generate(startup_name="A", description="...")
report.strengths[0].title       # str
report.benchmark_context.percentile_rank
```

Request models double as payload builders via `to_payload(model)`, which emits
the exact JSON body the API expects.

## CLI / one-liners

```python
from predictron_sdk import make_bearer_client

client = make_bearer_client("token-123", base_url="https://predictron.invex.ai")
with client:
    print(client.health_status().status)
```

## Running the SDK tests

```bash
python -m pytest tests/sdk -q
python -m ruff check predictron_sdk
python -m mypy --strict predictron_sdk
```

## Design notes

- **No duplicated business logic** — the SDK only serializes requests and
  deserializes responses. All intelligence stays in the Predictron API.
- **Typed everywhere** — models, pagination, exceptions and resources are fully
  annotated (mypy strict).
- **Sync-first** — the current surface is synchronous and built on `httpx`.
  Async support can be layered on without changing any public signature.
- **Backwards compatibility** — the SDK is additive; no existing API was
  modified.